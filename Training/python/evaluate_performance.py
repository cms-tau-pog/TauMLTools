#!/usr/bin/env python

import argparse
parser = argparse.ArgumentParser(description='Apply training and store results.')
parser.add_argument('--input-taus', required=True, type=str, help="Input file with taus")
parser.add_argument('--input-other', required=False, default=None, type=str, help="Input file with non-taus")
parser.add_argument('--other-type', required=True, type=str, help="Type of non-tau objects")
parser.add_argument('--deep-results', required=True, type=str, help="Directory with deepId results")
parser.add_argument('--setup', required=True, type=str, help="Path to the file with the plot setup definition")
parser.add_argument('--setup-args', required=False, default='', type=str,
                    help="Comma separated arguments for the plot setup module. E.g. arg1=value1,arg2=value2 etc.")
parser.add_argument('--weights', required=False, default=None, type=str,
                    help="Directory with weights to correct the spectrum")
parser.add_argument('--prev-deep-results', required=False, default=None, type=str,
                    help="Directory with previous deepId results")
parser.add_argument('--deep-results-label', required=False, default='', type=str,
                    help="Label for deepId results")
parser.add_argument('--prev-deep-results-label', required=False, default='', type=str,
                    help="Label for deepId results")
parser.add_argument('--output', required=True, type=str, help="Output pdf file")
parser.add_argument('--inequality-in-title', action="store_true",
                    help="Use inequality in the title to define pt range, instead of an interval")
parser.add_argument('--public-plots', action="store_true", help="Apply public plot styles")

args = parser.parse_args()

import os
import sys
import math
import pandas
import numpy as np
import json
import eval_tools
from collections import defaultdict

def AddPredictionsToDataFrame(df, file_name, label = ''):
    df_pred = pandas.read_hdf(file_name)
    for out in match_suffixes:
        if out != 'tau':
            prob_tau = df_pred['deepId_tau'].values
            prob_other = df_pred['deepId_' + out].values
            tau_vs_other = np.where(prob_tau > 0, prob_tau / (prob_tau + prob_other), np.zeros(prob_tau.shape))
            df['deepId{}_vs_{}'.format(label, out)] = pandas.Series(tau_vs_other, index=df.index)
        df['deepId{}_{}'.format(label, out)] = pandas.Series(df_pred['deepId_' + out].values, index=df.index)
    return df

def AddWeightsToDataFrame(df, file_name):
    df_weights = pandas.read_hdf(file_name)
    df['weight'] = pandas.Series(df_weights.weight.values, index=df.index)
    return df

def CreateDF(file_name, tau_types, setup_provider):
    df = eval_tools.ReadBrancesToDataFrame(file_name, 'taus', all_branches)
    base_name = os.path.basename(file_name)
    pred_file_name = os.path.splitext(base_name)[0] + '_pred.h5'
    AddPredictionsToDataFrame(df, os.path.join(args.deep_results, pred_file_name))
    if args.weights is not None:
        weight_file_name = os.path.splitext(base_name)[0] + '_weights.h5'
        AddWeightsToDataFrame(df, os.path.join(args.weights, weight_file_name))
    else:
        df['weight'] = pandas.Series(np.ones(df.shape[0]), index=df.index)
    has_prev_results = len(args.prev_deep_results_label) > 0 and 'None' not in args.prev_deep_results_label
    if has_prev_results:
        AddPredictionsToDataFrame(df, os.path.join(args.prev_deep_results, pred_file_name),
                                  args.prev_deep_results_label)
    df['tau_pt'] = pandas.Series(df.tau_pt *(1000 - 20) + 20, index=df.index)
    if hasattr(setup_provider, "DefineBranches"):
        df = setup_provider.DefineBranches(df, tau_types)
    sel = None
    for tau_type in tau_types:
        tau_sel = df['gen_{}'.format(tau_type)] == 1
        if sel is None:
            sel = tau_sel
        else:
            sel = sel | tau_sel
    if sel is not None:
        df = df[sel]
    return df

if sys.version_info.major > 2:
    import importlib.util
    spec = importlib.util.spec_from_file_location('setup_provider', args.setup)
    setup_provider = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(setup_provider)
else:
    import imp
    setup_provider = imp.load_source('setup_provider', args.setup)

setup_args = {}
setup_args_list = [ s.strip() for s in args.setup_args.split(',') if len(s.strip()) > 0 ]
for setup_arg in setup_args_list:
    split_arg = setup_arg.split('=')
    if len(split_arg) != 2:
        raise RuntimeError('Invalid setup argumetn = "{}".'.format(setup_arg))
    setup_args[split_arg[0]] = split_arg[1]

setup_provider.Initialize(eval_tools, setup_args)

discriminator = setup_provider.GetDiscriminators(args.other_type, args.deep_results_label,
                                                  args.prev_deep_results_label) # wip: here the function is changed to return only one discr


match_suffixes = [ 'e', 'mu', 'tau', 'jet' ]
core_branches = [ 'tau_pt', 'tau_decayModeFinding', 'tau_decayMode', 'gen_{}'.format(args.other_type), 'gen_tau',
                  'tau_charge', 'lepton_gen_charge' ]

all_branches = []
all_branches.extend(core_branches)
if hasattr(setup_provider, 'setup_branches'):
    all_branches.extend(setup_provider.setup_branches)

if discriminator.from_tuple:
    all_branches.append(discriminator.column)
    if discriminator.wp_column != discriminator.column:
        all_branches.append(discriminator.wp_column)

if args.input_other is None:
    df_all = CreateDF(args.input_taus, ['tau', args.other_type], setup_provider)
else:
    df_taus = CreateDF(args.input_taus, ['tau'], setup_provider)
    df_other = CreateDF(args.input_other, [args.other_type], setup_provider)
    df_all = df_taus.append(df_other)
if hasattr(setup_provider, 'ApplySelection'):
    df_all = setup_provider.ApplySelection(df_all, args.input_other)
    
plot_setup = setup_provider.GetPlotSetup(args.other_type)
discr_json = {
    'name': discriminator.name, 'period': '2017 (13 TeV)', 'metrics': defaultdict(list), 
}

pt_bins = setup_provider.GetPtBins()
for pt_index in range(len(pt_bins) - 1):
    df_tx = df_all[(df_all.tau_pt > pt_bins[pt_index]) & (df_all.tau_pt < pt_bins[pt_index + 1])]
    if df_tx.shape[0] == 0:
        print("Warning: pt bin ({}, {}) is empty.".format(pt_bins[pt_index], pt_bins[pt_index + 1]))
        continue

    # create roc curve and working points
    roc, wp_roc = discriminator.CreateRocCurve(df_tx)
    
    if roc is not None:
        # prune the curve
        lim = getattr(plot_setup,  'xlim')
        x_range = lim[1] - lim[0] if lim is not None else 1
        roc = roc.Prune(tpr_decimals=max(0, round(math.log10(1000 / x_range))))
        if roc.auc_score is not None:
            print('[{}, {}] {} roc_auc = {}'.format(pt_bins[pt_index], pt_bins[pt_index + 1], discriminator.name,
                                                    roc.auc_score))

    # loop over [ROC curve, ROC curve WP] for a given discriminator and store its info into dict
    for curve_type, curve in zip(['roc_curve', 'roc_wp'], [roc, wp_roc]):
        if curve is None: continue

        curve_data = {
            'pt_min': pt_bins[pt_index], 'pt_max': pt_bins[pt_index + 1], 
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
        curve_data['plot_setup']['ratio_title'] = 'MVA/DeepTau' if args.other_type != 'mu' else 'cut based/DeepTau'

        for lim_name in [ 'x', 'y', 'ratio_y' ]:
            lim = getattr(plot_setup, lim_name + 'lim')
            if lim is not None:
                lim = lim[pt_index] if type(lim[0]) == list else lim
                curve_data['plot_setup'][lim_name + '_min'] = lim[0]
                curve_data['plot_setup'][lim_name + '_max'] = lim[1]

        for param_name in [ 'ylabel', 'yscale', 'ratio_yscale', 'legend_loc', 'ratio_ylabel_pad']:
            val = getattr(plot_setup, param_name)
            if val is not None:
                curve_data['plot_setup'][param_name] = val

        if args.public_plots:
            if pt_bins[pt_index + 1] == 1000:
                pt_text = r'$p_T > {}$ GeV'.format(pt_bins[pt_index])
            elif pt_bins[pt_index] == 20:
                pt_text = r'$p_T < {}$ GeV'.format(pt_bins[pt_index + 1])
            else:
                pt_text = r'$p_T\in ({}, {})$ GeV'.format(pt_bins[pt_index], pt_bins[pt_index + 1])
            curve_data['plot_setup']['pt_text'] = pt_text
        else:
            if args.inequality_in_title and (pt_bins[pt_index] == 20 or pt_bins[pt_index + 1] == 1000) \
                    and not (pt_bins[pt_index] == 20 and pt_bins[pt_index + 1] == 1000):
                if pt_bins[pt_index] == 20:
                    title_str = 'tau vs {}. pt < {} GeV'.format(args.other_type, pt_bins[pt_index + 1])
                else:
                    title_str = 'tau vs {}. pt > {} GeV'.format(args.other_type, pt_bins[pt_index])
            else:
                title_str = 'tau vs {}. pt range ({}, {}) GeV'.format(args.other_type, pt_bins[pt_index],
                                                                        pt_bins[pt_index + 1])
            curve_data['plot_setup']['pt_text'] = title_str

        discr_json['metrics'][curve_type].append(curve_data)

with open(args.output, 'w') as json_file:
    json_file.write(json.dumps(discr_json, indent=4, cls=eval_tools.CustomJsonEncoder))
