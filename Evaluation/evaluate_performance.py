import os
import json
import numpy as np
import pandas as pd
from collections import defaultdict
from dataclasses import fields

import mlflow
import hydra
from hydra.utils import to_absolute_path
from omegaconf import OmegaConf, DictConfig

import utils.evaluation as eval_tools

@hydra.main(config_path='configs/eval', config_name='run3')
def main(cfg: DictConfig) -> None:
    mlflow.set_tracking_uri(f"file://{to_absolute_path(cfg.path_to_mlflow)}")

    # setting paths
    # path_to_weights_taus = to_absolute_path(cfg.path_to_weights_taus) if cfg.path_to_weights_taus is not None else None
    # path_to_weights_vs_type = to_absolute_path(cfg.path_to_weights_vs_type) if cfg.path_to_weights_vs_type is not None else None
    path_to_artifacts = to_absolute_path(f'{cfg.path_to_mlflow}/{cfg.experiment_id}/{cfg.run_id}/artifacts/')
    output_json_path = f'{path_to_artifacts}/performance.json'

    # init Discriminator() class from filtered input configuration
    field_names = set(f_.name for f_ in fields(eval_tools.Discriminator))
    init_params = {k:v for k,v in cfg.discriminator.items() if k in field_names}
    if 'wp_thresholds' in init_params:
        if not (isinstance(init_params['wp_thresholds'], DictConfig) or isinstance(init_params['wp_thresholds'], dict)):
            if isinstance(init_params['wp_thresholds'], str): # assume that it's the filename to read WPs from
                with open(f"{path_to_artifacts}/{init_params['wp_thresholds']}", 'r') as f:
                    wp_thresholds = json.load(f)
                init_params['wp_thresholds'] = wp_thresholds[cfg['vs_type']] # pass laoded dict with thresholds to Discriminator() class
            else:
                raise RuntimeError(f"Expect `wp_thresholds` argument to be either dict-like or str, but got the type: {type(init_params['wp_thresholds'])}")
    else:
        wp_thresholds = None
    discriminator = eval_tools.Discriminator(**init_params)

    # construct branches to be read from input files
    input_branches = OmegaConf.to_object(cfg.input_branches)
    if ((_b:=discriminator.pred_column) is not None) and (cfg.path_to_pred is None):
        input_branches.append(_b)
    if (discriminator.wp_column) is not None:
        if 'wp_name_to_index_map' in cfg['discriminator']: # append all branches for multiclass WP models
            for tau_type in cfg['discriminator']['wp_name_to_index_map'].keys():
                input_branches.append(cfg['discriminator']['wp_column_prefix'] + tau_type)
        else: # append only wp_column branch for binary WP model
            input_branches.append(discriminator.wp_column)

    # loop over input samples
    df_list = []
    print()
    for sample_alias, tau_types in cfg.input_samples.items():
        input_files, pred_files, target_files = eval_tools.prepare_filelists(sample_alias, cfg.path_to_input, cfg.path_to_pred, cfg.path_to_target, path_to_artifacts)

        # loop over all input files per sample with associated predictions/targets (if present) and combine together into df
        print(f'[INFO] Creating dataframe for sample: {sample_alias}')
        for input_file, pred_file, target_file in zip(input_files, pred_files, target_files):
            df = eval_tools.create_df(input_file, input_branches, pred_file, target_file, None, # weights functionality is WIP
                                            cfg.discriminator.pred_column_prefix, cfg.discriminator.target_column_prefix)
            gen_selection = ' or '.join([f'(gen_{tau_type}==1)' for tau_type in tau_types]) # gen_* are constructed in `add_targets()`
            df = df.query(gen_selection)
            df_list.append(df)
    df_all = pd.concat(df_list)

    # apply selection
    if cfg['cuts'] is not None:
        df_all = df_all.query(cfg.cuts)
    if cfg['WPs_to_require'] is not None:
        for wp_vs_type, wp_name in cfg['WPs_to_require'].items():
            if cfg['discriminator']['wp_from']=='wp_column':
                wp = cfg['discriminator']['wp_name_to_index_map'][wp_vs_type][wp_name]
                wp_column = f"{cfg['discriminator']['wp_column_prefix']}{wp_vs_type}"
                flag = 1 << wp
                df_all = df_all[np.bitwise_and(df_all[wp_column], flag) != 0]
            else:
                if wp_thresholds is not None: # take thresholds from previously loaded json
                    wp_thr = wp_thresholds[wp_vs_type][wp_name]
                elif cfg['discriminator']['wp_thresholds_map'] is not None: # take thresholds from discriminator cfg
                    wp_thr = cfg['discriminator']['wp_thresholds_map'][wp_vs_type][wp_name]
                else:
                    raise RuntimeError('WP thresholds either from wp_column, or wp_thresholds_map, or via input json file are not provided.')
                wp_cut = f"{cfg['discriminator']['pred_column_prefix']}{wp_vs_type} > {wp_thr}"
                df_all = df_all.query(wp_cut)
        

    # # inverse scaling
    # df_all['tau_pt'] = df_all.tau_pt*(1000 - 20) + 20
    
    # dump curves' data into json file
    json_exists = os.path.exists(output_json_path)
    json_open_mode = 'r+' if json_exists else 'w'
    with open(output_json_path, json_open_mode) as json_file:
        if json_exists: # read performance data to append additional info 
            performance_data = json.load(json_file)
        else: # create dictionary to fill with data
            performance_data = {'name': discriminator.name, 'metrics': defaultdict(list)}

        # loop over pt bins
        print(f'\n{discriminator.name}')
        for dm_bin in cfg.dm_bins:
         for eta_index, (eta_min, eta_max) in enumerate(cfg.eta_bins):
          for pt_index, (pt_min, pt_max) in enumerate(cfg.pt_bins):
            # apply pt/eta/dm bin selection
            df_cut = df_all.query(f'tau_pt >= {pt_min} and tau_pt < {pt_max} and abs(tau_eta) >= {eta_min} and abs(tau_eta) < {eta_max} and tau_decayMode in {dm_bin}')
            if df_cut.shape[0] == 0:
                print("Warning: bin with pt ({}, {}) and eta ({}, {}) and DMs {} is empty.".format(pt_min, pt_max, eta_min, eta_max, dm_bin))
                continue
            print(f'\n-----> pt bin: [{pt_min}, {pt_max}], eta bin: [{eta_min}, {eta_max}], DM bin: {dm_bin}')
            print('[INFO] counts:\n', df_cut[['gen_tau', f'gen_{cfg.vs_type}']].value_counts())

            # create roc curve and working points
            roc, wp_roc = discriminator.create_roc_curve(df_cut)
            if roc is not None:
                # prune the curve
                roc = roc.prune(tpr_decimals=cfg['roc_prune_decimal'][cfg['vs_type']])
                if roc.auc_score is not None:
                    print(f'[INFO] ROC curve done, AUC = {roc.auc_score:.6f}')

            # loop over [ROC curve, ROC curve WP] for a given discriminator and store its info into dict
            for curve_type, curve in zip(['roc_curve', 'roc_wp'], [roc, wp_roc]):
                if curve is None: continue
                if json_exists and curve_type in performance_data['metrics'] \
                                and (existing_curve := eval_tools.select_curve(performance_data['metrics'][curve_type], 
                                                                                pt_min=pt_min, pt_max=pt_max, eta_min=eta_min, eta_max=eta_max, dm_bin=dm_bin, vs_type=cfg.vs_type,
                                                                                dataset_alias=cfg.dataset_alias)) is not None:
                    print(f'[INFO] Found already existing curve (type: {curve_type}) in json file for a specified set of parameters: will overwrite it.')
                    performance_data['metrics'][curve_type].remove(existing_curve)

                curve_data = {
                    'pt_min': pt_min, 'pt_max': pt_max, 
                    'eta_min': eta_min, 'eta_max': eta_max, 
                    'dm_bin': list(dm_bin), 
                    'vs_type': cfg.vs_type,
                    'dataset_alias': cfg.dataset_alias,
                    'auc_score': curve.auc_score,
                    'false_positive_rate': eval_tools.FloatList(curve.pr[0, :].tolist()),
                    'true_positive_rate': eval_tools.FloatList(curve.pr[1, :].tolist()),
                }
                if curve.thresholds is not None:
                    curve_data['thresholds'] = eval_tools.FloatList(curve.thresholds.tolist())
                if curve.pr_err is not None:
                    curve_data['false_positive_rate_up'] = eval_tools.FloatList(curve.pr_err[0, 0, :].tolist())
                    curve_data['false_positive_rate_down'] = eval_tools.FloatList(curve.pr_err[0, 1, :].tolist())
                    curve_data['true_positive_rate_up'] = eval_tools.FloatList(curve.pr_err[1, 0, :].tolist())
                    curve_data['true_positive_rate_down'] = eval_tools.FloatList(curve.pr_err[1, 1, :].tolist())

                # append data for a given curve_type and pt bin
                if curve_type not in performance_data['metrics']:
                    performance_data['metrics'][curve_type] = []
                performance_data['metrics'][curve_type].append(curve_data)

        json_file.seek(0) 
        json_file.write(json.dumps(performance_data, indent=4, cls=eval_tools.CustomJsonEncoder))
        json_file.truncate()
    print()
    
if __name__ == '__main__':
    main()
