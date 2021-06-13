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
parser.add_argument('--draw-wp', action="store_true", help="Draw working points for raw discriminators")
parser.add_argument('--store-json', action="store_true", help="Store ROC curves in JSON format")
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
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import eval_tools
import common

def AddPredictionsToDataFrame(df, file_name, label = ''):
    df_pred = pandas.read_hdf(file_name)
    for out in common.match_suffixes:
        if out != 'tau':
            tau_vs_other = common.TauLosses.tau_vs_other(df_pred['deepId_tau'].values, df_pred['deepId_' + out].values)
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

discriminators = setup_provider.GetDiscriminators(args.other_type, args.deep_results_label,
                                                  args.prev_deep_results_label)

core_branches = [ 'tau_pt', 'tau_decayModeFinding', 'tau_decayMode', 'gen_{}'.format(args.other_type), 'gen_tau',
                  'tau_charge', 'lepton_gen_charge' ]

all_branches = []
all_branches.extend(core_branches)
if hasattr(setup_provider, 'setup_branches'):
    all_branches.extend(setup_provider.setup_branches)
for disc in discriminators:
    if disc.from_tuple:
        all_branches.append(disc.column)
        if disc.wp_column != disc.column:
            all_branches.append(disc.wp_column)

if args.input_other is None:
    df_all = CreateDF(args.input_taus, ['tau', args.other_type], setup_provider)
else:
    df_taus = CreateDF(args.input_taus, ['tau'], setup_provider)
    df_other = CreateDF(args.input_other, [args.other_type], setup_provider)
    df_all = df_taus.append(df_other)
if hasattr(setup_provider, 'ApplySelection'):
    df_all = setup_provider.ApplySelection(df_all, args.input_other)

pt_bins = setup_provider.GetPtBins()

plot_setup = setup_provider.GetPlotSetup(args.other_type)

roc_json = []

with PdfPages(args.output) as pdf:
    for pt_index in range(len(pt_bins) - 1):
        df_tx = df_all[(df_all.tau_pt > pt_bins[pt_index]) & (df_all.tau_pt < pt_bins[pt_index + 1])]
        if df_tx.shape[0] == 0:
            print("Warning: pt bin ({}, {}) is empty.".format(pt_bins[pt_index], pt_bins[pt_index + 1]))
            continue
        n_discr = len(discriminators)
        rocs = [None] * n_discr
        wp_rocs = [None] * n_discr
        names = [ disc.name for disc in discriminators ]

        roc_json_entry = {
            'pt_min': pt_bins[pt_index], 'pt_max': pt_bins[pt_index + 1], 'discriminators': [], 'plot_setup': { },
        }

        for param_name in [ 'ylabel', 'yscale', 'ratio_yscale', 'legend_loc', 'ratio_ylabel_pad']:
            val = getattr(plot_setup, param_name)
            if val is not None:
                roc_json_entry['plot_setup'][param_name] = val

        x_range = 1
        for lim_name in [ 'x', 'y', 'ratio_y' ]:
            lim = getattr(plot_setup, lim_name + 'lim')
            if lim is not None:
                lim = lim[pt_index] if type(lim[0]) == list else lim
                roc_json_entry['plot_setup'][lim_name + '_min'] = lim[0]
                roc_json_entry['plot_setup'][lim_name + '_max'] = lim[1]
                if lim_name == 'x':
                    x_range = lim[1] - lim[0]

        for n in reversed(range(n_discr)):
            ref_roc = rocs[-1]
            rocs[n], wp_rocs[n] = discriminators[n].CreateRocCurve(df_tx, ref_roc)
            if rocs[n].auc_score is not None:
                #target_prs = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99, 0.995 ]
                #thrs = [ find_threshold(rocs[n].pr[1, :], rocs[n].thresholds, pr) for pr in target_prs ]
                print('[{}, {}] {} roc_auc = {}'.format(pt_bins[pt_index], pt_bins[pt_index + 1], names[n],
                                                        rocs[n].auc_score))
                #print(thrs)
            #print(discriminators[n].name)
            name_suffix = ''
            for roc in [ rocs[n].Prune(tpr_decimals=max(0, round(math.log10(1000 / x_range)))), wp_rocs[n] ]:
                if roc is None: continue

                discr_data = {
                    'name': discriminators[n].name + name_suffix,
                    'false_positive_rate': eval_tools.FloatList(roc.pr[0, :].tolist()),
                    'true_positive_rate': eval_tools.FloatList(roc.pr[1, :].tolist()),
                    'is_ref': n == n_discr - 1,
                    'color': roc.color,
                    'auc_score': roc.auc_score,
                    'dots_only': roc.dots_only,
                    'dashed': roc.dashed,
                    'marker_size': roc.marker_size,
                }
                if roc.thresholds is not None:
                    discr_data['thresholds'] = eval_tools.FloatList(roc.thresholds.tolist())
                if roc.pr_err is not None:
                    discr_data['false_positive_rate_up'] = eval_tools.FloatList(roc.pr_err[0, 0, :].tolist())
                    discr_data['false_positive_rate_down'] = eval_tools.FloatList(roc.pr_err[0, 1, :].tolist())
                    discr_data['true_positive_rate_up'] = eval_tools.FloatList(roc.pr_err[1, 0, :].tolist())
                    discr_data['true_positive_rate_down'] = eval_tools.FloatList(roc.pr_err[1, 1, :].tolist())
                roc_json_entry['discriminators'].insert(0, discr_data)
                name_suffix = ' WP'


        fig, (ax, ax_ratio) = plt.subplots(2, 1, figsize=(7, 7), sharex=True,
                                           gridspec_kw = {'height_ratios':[3, 1]})

        plot_entries = []
        for n in range(n_discr):
            entry = rocs[n].Draw(ax, ax_ratio)
            plot_entries.append(entry)
        for n in range(n_discr):
            if wp_rocs[n] is not None:
                wp_rocs[n].Draw(ax, ax_ratio)

        ratio_title = 'MVA/DeepTau' if args.other_type != 'mu' else 'cut based/DeepTau'
        plot_setup.Apply(names, plot_entries, pt_index, ratio_title, ax, ax_ratio)

        roc_json_entry['plot_setup']['ratio_title'] = ratio_title
        roc_json_entry['period'] = '2017 (13 TeV)'
        if args.public_plots:
            header_y = 1.02
            # ax.text(0.03, 0.90, r'$p_T\in ({}, {})$ GeV'.format(pt_bins[pt_index], pt_bins[pt_index + 1]),
            #         fontsize=14, transform=ax.transAxes)
            if pt_bins[pt_index + 1] == 1000:
                pt_text = r'$p_T > {}$ GeV'.format(pt_bins[pt_index])
            elif pt_bins[pt_index] == 20:
                pt_text = r'$p_T < {}$ GeV'.format(pt_bins[pt_index + 1])
            else:
                pt_text = r'$p_T\in ({}, {})$ GeV'.format(pt_bins[pt_index], pt_bins[pt_index + 1])
            roc_json_entry['pt_text'] = pt_text
            ax.text(0.03, 0.92 - n_discr * 0.10, pt_text, fontsize=14, transform=ax.transAxes)
            ax.text(0.01, header_y, 'CMS', fontsize=14, transform=ax.transAxes, fontweight='bold',
                    fontfamily='sans-serif')
            ax.text(0.12, header_y, 'Simulation Preliminary', fontsize=14, transform=ax.transAxes, fontstyle='italic',
                    fontfamily='sans-serif')
            ax.text(0.73, header_y, '2017 (13 TeV)', fontsize=13, transform=ax.transAxes, fontweight='bold',
                    fontfamily='sans-serif')
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
            roc_json_entry['pt_text'] = title_str
            ax.set_title(title_str, fontsize=18, y=1.04)
        plt.subplots_adjust(hspace=0)
        pdf.savefig(fig, bbox_inches='tight')
        roc_json.append(roc_json_entry)

if args.store_json:
    with open(os.path.splitext(args.output)[0] + '.json', 'w') as json_file:
        json_file.write(json.dumps(roc_json, indent=4, cls=eval_tools.CustomJsonEncoder))
