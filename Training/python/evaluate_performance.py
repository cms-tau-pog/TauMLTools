import os
import math
import pandas as pd
import numpy as np
import json
import eval_tools
from collections import defaultdict
from dataclasses import fields

TAU_TYPES = [ 'e', 'mu', 'tau', 'jet' ]

def AddPredictionsToDataFrame(df, file_name, label = ''):
    df_pred = pd.read_hdf(file_name)
    for tau_type in TAU_TYPES:
        if tau_type != 'tau':
            prob_tau = df_pred['deepId_tau'].values
            prob_vs_type = df_pred['deepId_' + tau_type].values
            tau_vs_other_type = np.where(prob_tau > 0, prob_tau / (prob_tau + prob_vs_type), np.zeros(prob_tau.shape))
            df['deepId{}_vs_{}'.format(label, tau_type)] = pd.Series(tau_vs_other_type, index=df.index)
        df['deepId{}_{}'.format(label, tau_type)] = pd.Series(df_pred['deepId_' + tau_type].values, index=df.index)
    return df

def AddWeightsToDataFrame(df, file_name):
    df_weights = pd.read_hdf(file_name)
    df['weight'] = pd.Series(df_weights.weight.values, index=df.index)
    return df

def CreateDF(file_name, deep_results, read_branches, weights, tau_types):

    # TODO: add branching creation on the fly
    
    df = eval_tools.ReadBrancesToDataFrame(file_name, 'taus', read_branches)
    base_name = os.path.basename(file_name)
    pred_file_name = os.path.splitext(base_name)[0] + '_pred.h5'
    if deep_results is not None:
        AddPredictionsToDataFrame(df, os.path.join(deep_results, pred_file_name))
    if weights is not None:
        weight_file_name = os.path.splitext(base_name)[0] + '_weights.h5'
        AddWeightsToDataFrame(df, os.path.join(weights, weight_file_name))
    else:
        df['weight'] = pd.Series(np.ones(df.shape[0]), index=df.index)

    # inverse linear scaling 
    df['tau_pt'] = pd.Series(df.tau_pt *(1000 - 20) + 20, index=df.index)

    # gen match selection of given tau types
    gen_selection = ' or '.join([f'(gen_{tau_type}==1)' for tau_type in tau_types])
    df = df.query(gen_selection)
    return df

import mlflow
import hydra
from hydra.utils import to_absolute_path
from omegaconf import OmegaConf, DictConfig, ListConfig

@hydra.main(config_path='configs', config_name='run3')
def main(cfg: DictConfig) -> None:
    mlflow.set_tracking_uri(f"file://{to_absolute_path('mlruns')}")
    experiment = mlflow.get_experiment_by_name(cfg.experiment_name)
    experiment_id = experiment.experiment_id

    input_taus = to_absolute_path(cfg.input_taus)
    input_vs_type = to_absolute_path(cfg.input_vs_type)
    deep_results = to_absolute_path(f'mlruns/{experiment_id}/{cfg.run_id}/artifacts/predictions') if cfg.discriminator.name=='DeepTau' else None
    output_json_path = to_absolute_path(f'{cfg.output_folder}/{cfg.output_name}.json')
    weights = to_absolute_path(cfg.weights) if cfg.weights is not None else None

    # init Discriminator() class from filtered input configuration
    field_names = set(f_.name for f_ in fields(eval_tools.Discriminator))
    init_params = {k:v for k,v in cfg.discriminator.items() if k in field_names}
    discriminator = eval_tools.Discriminator(**init_params)
    discr_json = {
        'name': f'{discriminator.name} vs. {cfg.vs_type}', 'period': cfg.period, 'metrics': defaultdict(list), 
    }
    
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
        df_all = CreateDF(input_taus, deep_results, read_branches, weights, ['tau', cfg.vs_type])
    else:
        df_taus = CreateDF(input_taus, deep_results, read_branches, weights, ['tau'])
        df_vs_type = CreateDF(input_vs_type, deep_results, read_branches, weights, [cfg.vs_type])
        df_all = df_taus.append(df_vs_type)

    # apply selection cuts
    df_all = df_all.query(cfg.cuts)

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
                print('[{}, {}] {} roc_auc = {}'.format(pt_min, pt_max, discriminator.name,
                                                        roc.auc_score))

        # loop over [ROC curve, ROC curve WP] for a given discriminator and store its info into dict
        for curve_type, curve in zip(['roc_curve', 'roc_wp'], [roc, wp_roc]):
            if curve is None: continue

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
            discr_json['metrics'][curve_type].append(curve_data)

    # dump curves' data into json file
    with open(output_json_path, 'w') as json_file:
        json_file.write(json.dumps(discr_json, indent=4, cls=eval_tools.CustomJsonEncoder))

    # log json to corresponding mlflow run
    with mlflow.start_run(experiment_id=experiment.experiment_id, run_id=cfg.run_id) as run:
        mlflow.log_artifact(output_json_path)
       
if __name__ == '__main__':
    main()