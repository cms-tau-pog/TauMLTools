import os
import math
import pandas as pd
import numpy as np
import json
from collections import defaultdict
from dataclasses import fields

import mlflow
import hydra
from hydra.utils import to_absolute_path
from omegaconf import OmegaConf, DictConfig, ListConfig

import eval_tools

@hydra.main(config_path='configs', config_name='run3')
def main(cfg: DictConfig) -> None:
    mlflow.set_tracking_uri(f"file://{to_absolute_path('mlruns')}")
    experiment = mlflow.get_experiment_by_name(cfg.experiment_name)
    experiment_id = experiment.experiment_id

    input_taus = to_absolute_path(cfg.input_taus)
    input_vs_type = to_absolute_path(cfg.input_vs_type)
    predictions = predictions_path if os.path.exists(predictions_path:=to_absolute_path(f'mlruns/{experiment_id}/{cfg.run_id}/artifacts/predictions'))  else None
    output_json_path = to_absolute_path(f'mlruns/{experiment_id}/{cfg.run_id}/artifacts/performance.json')
    weights = to_absolute_path(cfg.weights) if cfg.weights is not None else None

    # init Discriminator() class from filtered input configuration
    field_names = set(f_.name for f_ in fields(eval_tools.Discriminator))
    init_params = {k:v for k,v in cfg.discriminator.items() if k in field_names}
    discriminator = eval_tools.Discriminator(**init_params)
    
    # init PlotSetup() class from filtered input configuration
    field_names = set(f_.name for f_ in fields(eval_tools.PlotSetup))
    init_params = {k:v for k,v in cfg.plot_setup.items() if k in field_names}
    plot_setup = eval_tools.PlotSetup(**init_params)

    # construct branches to be read from input files
    read_branches = OmegaConf.to_object(cfg.read_branches)
    if discriminator.from_tuple:
        read_branches.append(discriminator.column)
        if discriminator.wp_column != discriminator.column:
            read_branches.append(discriminator.wp_column)

    # read original data and corresponging predictions into DataFrame 
    if input_vs_type is None:
        df_all = eval_tools.create_df(input_taus, predictions, cfg.discriminator.column_prefix, read_branches, weights, ['tau', cfg.vs_type])
    else:
        df_taus = eval_tools.create_df(input_taus, predictions, cfg.discriminator.column_prefix, read_branches, weights, ['tau'])
        df_vs_type = eval_tools.create_df(input_vs_type, predictions, cfg.discriminator.column_prefix, read_branches, weights, [cfg.vs_type])
        df_all = df_taus.append(df_vs_type)

    # apply selection cuts
    df_all = df_all.query(cfg.cuts)
    
    # dump curves' data into json file
    json_exists = os.path.exists(output_json_path)
    json_open_mode = 'r+' if json_exists else 'w'
    with open(output_json_path, json_open_mode) as json_file:
        if json_exists: # read performance data to append additional info 
            performance_data = json.load(json_file)
        else: # create dictionary to fill with data
            performance_data = {'name': discriminator.name, 'period': cfg.period, 'metrics': defaultdict(list)}

        # loop over pt bins
        for pt_index, (pt_min, pt_max) in enumerate(zip(cfg.pt_bins[:-1], cfg.pt_bins[1:])):
            # apply pt bin selection
            df_cut = df_all.query(f'tau_pt >= {pt_min} and tau_pt < {pt_max}')
            if df_cut.shape[0] == 0:
                print("Warning: pt bin ({}, {}) is empty.".format(pt_min, pt_max))
                continue

            # create roc curve and working points
            roc, wp_roc = discriminator.CreateRocCurve(df_cut)
            
            if roc is not None:
                # prune the curve
                lim = getattr(plot_setup,  'xlim')
                x_range = lim[1] - lim[0] if lim is not None else 1
                roc = roc.Prune(tpr_decimals=max(0, round(math.log10(1000 / x_range))))
                if roc.auc_score is not None:
                    print('\n[{}, {}] {} roc_auc = {}'.format(pt_min, pt_max, discriminator.name,
                                                            roc.auc_score))

            # loop over [ROC curve, ROC curve WP] for a given discriminator and store its info into dict
            for curve_type, curve in zip(['roc_curve', 'roc_wp'], [roc, wp_roc]):
                if curve is None: continue
                if json_exists and curve_type in performance_data['metrics'] \
                                and (existing_curve := eval_tools.select_curve(performance_data['metrics'][curve_type], 
                                                                        pt_min=pt_min, pt_max=pt_max, vs_type=cfg.vs_type,
                                                                        sample_tau=cfg.sample_tau, sample_vs_type=cfg.sample_vs_type)) is not None:
                    print(f'[INFO] Found already existing curve (type: {curve_type}) in json file for a specified set of parameters: will overwrite it.')
                    performance_data['metrics'][curve_type].remove(existing_curve)

                curve_data = {
                    'pt_min': pt_min, 'pt_max': pt_max, 
                    'vs_type': cfg.vs_type,
                    'sample_tau': cfg.sample_tau, 'sample_vs_type': cfg.sample_vs_type,
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

                # plot setup for the curve
                curve_data['plot_setup'] = {
                    'color': curve.color,
                    'dots_only': curve.dots_only,
                    'dashed': curve.dashed,
                    'marker_size': curve.marker_size
                }
                curve_data['plot_setup']['ratio_title'] = 'MVA/DeepTau' if cfg.vs_type != 'mu' else 'cut based/DeepTau'

                # plot setup for the curve
                for lim_name in [ 'x', 'y', 'ratio_y' ]:
                    lim = getattr(plot_setup, lim_name + 'lim')
                    if lim is not None:
                        lim = OmegaConf.to_object(lim[pt_index]) if isinstance(lim[0], (list, ListConfig)) else lim
                        curve_data['plot_setup'][lim_name + '_min'] = lim[0]
                        curve_data['plot_setup'][lim_name + '_max'] = lim[1]
                for param_name in [ 'ylabel', 'yscale', 'ratio_yscale', 'legend_loc', 'ratio_ylabel_pad']:
                    val = getattr(plot_setup, param_name)
                    if val is not None:
                        curve_data['plot_setup'][param_name] = val

                # plot setup for the curve
                if cfg.plot_setup.public_plots:
                    if pt_max == 1000:
                        pt_text = r'$p_T > {}$ GeV'.format(pt_min)
                    elif pt_min == 20:
                        pt_text = r'$p_T < {}$ GeV'.format(pt_max)
                    else:
                        pt_text = r'$p_T\in ({}, {})$ GeV'.format(pt_min, pt_max)
                    curve_data['plot_setup']['pt_text'] = pt_text
                else:
                    if cfg.plot_setup.inequality_in_title and (pt_min == 20 or pt_max == 1000) \
                            and not (pt_min == 20 and pt_max == 1000):
                        if pt_min == 20:
                            title_str = 'tau vs {}. pt < {} GeV'.format(cfg.vs_type, pt_max)
                        else:
                            title_str = 'tau vs {}. pt > {} GeV'.format(cfg.vs_type, pt_min)
                    else:
                        title_str = 'tau vs {}. pt range ({}, {}) GeV'.format(cfg.vs_type, pt_min,
                                                                                pt_max)
                    curve_data['plot_setup']['pt_text'] = title_str

                # append data for a given curve_type and pt bin
                performance_data['metrics'][curve_type].append(curve_data)

        json_file.seek(0) 
        json_file.write(json.dumps(performance_data, indent=4, cls=eval_tools.CustomJsonEncoder))
        json_file.truncate()
       
if __name__ == '__main__':
    main()