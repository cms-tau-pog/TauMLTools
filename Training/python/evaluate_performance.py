#!/usr/bin/env python

import argparse
parser = argparse.ArgumentParser(description='Apply training and store results.')
parser.add_argument('--input-taus', required=True, type=str, help="Input file with taus")
parser.add_argument('--input-other', required=True, type=str, help="Input file with non-taus")
parser.add_argument('--other-type', required=True, type=str, help="Type of non-tau objects")
parser.add_argument('--deep-results', required=True, type=str, help="Directory with deepId results")
parser.add_argument('--prev-deep-results', required=False, default=None, type=str,
                    help="Directory with previous deepId results")
parser.add_argument('--output', required=True, type=str, help="Output pdf file")
#parser.add_argument('--apply-loose', action="store_true", help="Submission dryrun.")
args = parser.parse_args()

import os
import pandas
import numpy as np
import uproot
from sklearn import metrics
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from statsmodels.stats.proportion import proportion_confint
from scipy import interpolate
from common import *
matplotlib.use('Agg')

class DiscriminatorWP:
    VVVLoose = 0
    VVLoose = 1
    VLoose = 2
    Loose = 3
    Medium = 4
    Tight = 5
    VTight = 6
    VVTight = 7
    VVVTight = 8

    @staticmethod
    def GetName(wp):
        names = [ "VVVLoose", "VVLoose", "VLoose", "Loose", "Medium", "Tight", "VTight", "VVTight", "VVVTight" ]
        return names[wp]

class RocCurve:
    def __init__(self, n_points, color, has_errors, dots_only = False):
        self.pr = np.zeros((2, n_points))
        self.color = color
        if has_errors:
            self.pr_err = np.zeros((2, 2, n_points))
        else:
            self.pr_err = None
        self.ratio = None
        self.thresholds = None
        self.auc_score = None
        self.dots_only = dots_only

    def Draw(self, ax, ax_ratio = None):
        if self.pr_err is not None:
            ax.errorbar(self.pr[1], self.pr[0], xerr=self.pr_err[1], yerr=self.pr_err[0], color=self.color,
                        fmt='--o', markersize='3', linewidth=1)
        else:
            if self.dots_only:
                ax.errorbar(self.pr[1], self.pr[0], color=self.color, fmt='o', markersize='3')
            else:
                ax.errorbar(self.pr[1], self.pr[0], color=self.color)
        if self.ratio is not None and ax_ratio is not None:
            ax_ratio.plot(self.ratio[1], self.ratio[0], color=self.color, linewidth=1)

class PlotSetup:
    def __init__(self, xlim = None, ylim = None, ratio_ylim = None, ylabel = None, yscale='log', ratio_yscale='linear'):
        self.xlim = xlim
        self.ylim = ylim
        self.ratio_ylim = ratio_ylim
        self.ylabel = ylabel
        self.yscale = yscale
        self.ratio_yscale = ratio_yscale

    def Apply(self, names, range_index, ax, ax_ratio = None):
        if self.xlim is not None:
            xlim = self.xlim[range_index] if type(self.xlim[0]) == list else self.xlim
            ax.set_xlim(xlim)

        if self.ylim is not None:
            ylim = self.ylim[range_index] if type(self.ylim[0]) == list else self.ylim
            ax.set_ylim(ylim)

        ax.set_yscale(self.yscale)
        ax.set_ylabel(self.ylabel, fontsize=16)
        ax.tick_params(labelsize=14)
        ax.grid(True)
        ax.legend(names, fontsize=14, loc='upper left')

        if ax_ratio is not None:
            if self.ratio_ylim is not None:
                ylim = self.ratio_ylim[range_index] if type(self.ratio_ylim[0]) == list else self.ratio_ylim
                ax_ratio.set_ylim(ylim)

            ax_ratio.set_yscale(self.ratio_yscale)
            ax_ratio.set_xlabel('Tau ID efficiency', fontsize=16)
            ax_ratio.set_ylabel('id/deepTau v2', fontsize=14, labelpad=20)
            ax_ratio.tick_params(labelsize=10)

            ax_ratio.grid(True, which='both')

def find_threshold(pr, thresholds, target_pr):
    min_delta_index = 0
    min_delta = abs(pr[0] - target_pr)
    for n in range(len(pr)):
        delta = abs(pr[n] - target_pr)
        if delta < min_delta:
            min_delta = delta
            min_delta_index = n
    if min_delta > 0.01:
        return None
    return thresholds[min_delta_index]

class Discriminator:
    def __init__(self, name, column, raw, from_tuple, color, working_points = [], wp_column = None,
                 working_points_thrs = None):
        self.name = name
        self.column = column
        self.raw = raw
        self.from_tuple = from_tuple
        self.color = color
        self.working_points = working_points
        self.wp_column = wp_column if wp_column is not None else column
        self.working_points_thrs = working_points_thrs


    def CountPassed(self, df, wp):
        if self.from_tuple:
            flag = 1 << wp
            return np.count_nonzero(np.bitwise_and(df[self.wp_column], flag))
        if self.working_points_thrs is None:
            raise RuntimeError("Working points are not specified")
        wp_thr = self.working_points_thrs[DiscriminatorWP.GetName(wp)]
        return np.count_nonzero(df[self.column] > wp_thr)

    def CreateRocCurve(self, df, ref_roc = None):
        n_wp = len(self.working_points)
        wp_roc = None
        if self.raw:
            fpr, tpr, thresholds = metrics.roc_curve(df['gen_tau'].values, df[self.column].values)
            roc = RocCurve(len(fpr), self.color, False)
            roc.pr[0, :] = fpr
            roc.pr[1, :] = tpr
            roc.thresholds = thresholds
            roc.auc_score = metrics.roc_auc_score(df['gen_tau'].values, df[self.column].values)
        if n_wp > 0:
            wp_roc = RocCurve(n_wp, self.color, not self.raw, self.raw)
            for n in range(n_wp):
                for kind in [0, 1]:
                    df_x = df[df['gen_tau'] == kind]
                    n_passed = self.CountPassed(df_x, self.working_points[n])
                    n_total = df_x.shape[0]
                    eff = float(n_passed) / n_total
                    wp_roc.pr[kind, n_wp - n - 1] = eff
                    if not self.raw:
                        ci_low, ci_upp = proportion_confint(n_passed, n_total, alpha=1-0.68, method='beta')
                        wp_roc.pr_err[kind, 1, n_wp - n - 1] = ci_upp - eff
                        wp_roc.pr_err[kind, 0, n_wp - n - 1] = eff - ci_low
        if not self.raw:
            roc = wp_roc
            wp_roc = None
        if ref_roc is not None:
            roc.ratio = create_roc_ratio(roc.pr[1], roc.pr[0], ref_roc.pr[1], ref_roc.pr[0])
        elif roc.pr[1].shape[0] > 0:
            roc.ratio = np.array([ [1, 1], [ roc.pr[1][0], roc.pr[1][-1] ] ])

        return roc, wp_roc

def ReadBrancesToDataFrame(file_name, tree_name, branches):
    if file_name.endswith('.root'):
        with uproot.open(file_name) as file:
            tree = file[tree_name]
            df = tree.arrays(branches, outputtype=pandas.DataFrame)
        return df
    elif file_name.endswith('.h5') or file_name.endswith('.hdf5'):
        return pandas.read_hdf(file_name, tree_name, columns=branches)
    raise RuntimeError("Unsupported file type.")

core_branches = [ 'tau_pt', 'tau_decayModeFinding', 'tau_decayMode', 'gen_tau', 'againstElectronMVA6',
                  'againstMuon3', 'byIsolationMVArun2017v2DBoldDMwLT2017' ]

# deep_wp_thrs = {
#     "e": {
#         "VVVLoose": 0.1950947,
#         "VVLoose": 0.3743916,
#         "VLoose": 0.6877663,
#         "Loose": 0.8877130,
#         "Medium": 0.9692497,
#         "Tight": 0.9871012,
#         "VTight": 0.9935400,
#         "VVTight": 0.9964663,
#     },
#     "mu": {
#         "VLoose": 0.2817462,
#         "Loose": 0.6823795,
#         "Medium": 0.9315399,
#         "Tight": 0.9627454,
#     },
#     "jet": {
#         "VVVLoose": 0.4356258,
#         "VVLoose": 0.6066188,
#         "VLoose": 0.7895795,
#         "Loose": 0.8857221,
#         "Medium": 0.9322081,
#         "Tight": 0.9580242,
#         "VTight": 0.9737815,
#         "VVTight": 0.9840425,
#     }
# }

deep_wp_thrs = {
    "e": {
        "VVVLoose": 0.0630386,
        "VVLoose": 0.1686942,
        "VLoose": 0.3628130,
        "Loose": 0.6815435,
        "Medium": 0.8847544,
        "Tight": 0.9675541,
        "VTight": 0.9859251,
        "VVTight": 0.9928449,
    },
    "mu": {
        "VLoose": 0.1058354,
        "Loose": 0.2158633,
        "Medium": 0.5551894,
        "Tight": 0.8754835,
    },
    "jet": {
        "VVVLoose": 0.2599605,
        "VVLoose": 0.4249705,
        "VLoose": 0.5983682,
        "Loose": 0.7848675,
        "Medium": 0.8834768,
        "Tight": 0.9308689,
        "VTight": 0.9573137,
        "VVTight": 0.9733927,
    },
}

all_discriminators = {
    'e': [
        Discriminator('MVA6', 'againstElectronMVA6', False, True, 'green',
                      [ DiscriminatorWP.VLoose, DiscriminatorWP.Loose, DiscriminatorWP.Medium, DiscriminatorWP.Tight,
                        DiscriminatorWP.VTight ] ),
        Discriminator('MVA6 2018', 'againstElectronMVA62018', False, True, 'red',
                      [ DiscriminatorWP.VLoose, DiscriminatorWP.Loose, DiscriminatorWP.Medium, DiscriminatorWP.Tight,
                        DiscriminatorWP.VTight ] ),
        Discriminator('deepTau 2017v1', 'byDeepTau2017v1VSeraw', True, True, 'blue',
                      [ DiscriminatorWP.VVVLoose, DiscriminatorWP.VVLoose, DiscriminatorWP.VLoose,
                        DiscriminatorWP.Loose, DiscriminatorWP.Medium, DiscriminatorWP.Tight, DiscriminatorWP.VTight,
                        DiscriminatorWP.VVTight ],
                      'byDeepTau2017v1VSe'),
        Discriminator('deepTau 2017v2 e2', 'deepIde2_vs_e', True, False, 'black'),
        Discriminator('deepTau 2017v2 e6', 'deepId_vs_e', True, False, 'orange',
                      [ DiscriminatorWP.VVVLoose, DiscriminatorWP.VVLoose, DiscriminatorWP.VLoose,
                        DiscriminatorWP.Loose, DiscriminatorWP.Medium, DiscriminatorWP.Tight, DiscriminatorWP.VTight,
                        DiscriminatorWP.VVTight ], working_points_thrs = deep_wp_thrs['e'])
    ],
    'mu': [
        Discriminator('againstMuon3', 'againstMuon3', False, True, 'green',
                      [ DiscriminatorWP.Loose, DiscriminatorWP.Tight] ),
        Discriminator('deepTau 2017v1', 'byDeepTau2017v1VSmuraw', True, True, 'blue',
                      [ DiscriminatorWP.VVVLoose, DiscriminatorWP.VVLoose, DiscriminatorWP.VLoose,
                        DiscriminatorWP.Loose, DiscriminatorWP.Medium, DiscriminatorWP.Tight, DiscriminatorWP.VTight,
                        DiscriminatorWP.VVTight ],
                      'byDeepTau2017v1VSmu'),
        Discriminator('deepTau 2017v2 e2', 'deepIde2_vs_mu', True, False, 'black'),
        Discriminator('deepTau 2017v2 e6', 'deepId_vs_mu', True, False, 'orange',
                      [ DiscriminatorWP.VLoose, DiscriminatorWP.Loose, DiscriminatorWP.Medium, DiscriminatorWP.Tight ],
                      working_points_thrs = deep_wp_thrs['mu'])
    ],
    'jet': [
        Discriminator('MVA 2017v2', 'byIsolationMVArun2017v2DBoldDMwLT2017raw', True, True, 'green',
                      [ DiscriminatorWP.VVLoose, DiscriminatorWP.VLoose, DiscriminatorWP.Loose, DiscriminatorWP.Medium,
                        DiscriminatorWP.Tight, DiscriminatorWP.VTight, DiscriminatorWP.VVTight ],
                      'byIsolationMVArun2017v2DBoldDMwLT2017'),
        Discriminator('MVA 2017v2 newDM', 'byIsolationMVArun2017v2DBnewDMwLT2017raw', True, True, 'red',
                      [ DiscriminatorWP.VVLoose, DiscriminatorWP.VLoose, DiscriminatorWP.Loose, DiscriminatorWP.Medium,
                        DiscriminatorWP.Tight, DiscriminatorWP.VTight, DiscriminatorWP.VVTight ],
                      'byIsolationMVArun2017v2DBnewDMwLT2017'),
        Discriminator('DPF 2016v0', 'byDpfTau2016v0VSallraw', True, True, 'magenta', [ DiscriminatorWP.Tight ],
                      'byDpfTau2016v0VSall'),
        Discriminator('deepTau 2017v1', 'byDeepTau2017v1VSjetraw', True, True, 'blue',
                      [ DiscriminatorWP.VVVLoose, DiscriminatorWP.VVLoose, DiscriminatorWP.VLoose,
                        DiscriminatorWP.Loose, DiscriminatorWP.Medium, DiscriminatorWP.Tight, DiscriminatorWP.VTight,
                        DiscriminatorWP.VVTight ],
                      'byDeepTau2017v1VSjet'),
        Discriminator('deepTau 2017v2 e2', 'deepIde2_vs_jet', True, False, 'black'),
        Discriminator('deepTau 2017v2 e6', 'deepId_vs_jet', True, False, 'orange',
                      [ DiscriminatorWP.VVVLoose, DiscriminatorWP.VVLoose, DiscriminatorWP.VLoose,
                        DiscriminatorWP.Loose, DiscriminatorWP.Medium, DiscriminatorWP.Tight, DiscriminatorWP.VTight,
                        DiscriminatorWP.VVTight ], working_points_thrs = deep_wp_thrs['jet'])
    ]
}

if args.other_type not in all_discriminators:
    raise RuntimeError("Unknown other_type")

discriminators = all_discriminators[args.other_type]
all_branches = []
all_branches.extend(core_branches)
for disc in discriminators:
    if disc.from_tuple:
        all_branches.append(disc.column)
        if disc.wp_column != disc.column:
            all_branches.append(disc.wp_column)

def AddPredictionsToDataFrame(df, file_name, label = ''):
    df_pred = pandas.read_hdf(file_name)
    for out in match_suffixes:
        if out != 'tau':
            tau_vs_other = TauLosses.tau_vs_other(df_pred['deepId_tau'].values, df_pred['deepId_' + out].values)
            df['deepId{}_vs_{}'.format(label, out)] = pandas.Series(tau_vs_other, index=df.index)
        df['deepId{}_{}'.format(label, out)] = pandas.Series(df_pred['deepId_' + out].values, index=df.index)
    return df

def CreateDF(file_name):
    df = ReadBrancesToDataFrame(file_name, 'taus', all_branches)
    base_name = os.path.basename(file_name)
    pred_file_name = os.path.splitext(base_name)[0] + '_pred.h5'
    AddPredictionsToDataFrame(df, os.path.join(args.deep_results, pred_file_name))
    if args.prev_deep_results is not None:
        AddPredictionsToDataFrame(df, os.path.join(args.prev_deep_results, pred_file_name), 'e2')
    df['tau_pt'] = pandas.Series(df.tau_pt *(1000 - 20) + 20, index=df.index)
    return df

df_taus = CreateDF(args.input_taus)
df_other = CreateDF(args.input_other)
df_all = df_taus.append(df_other)

apply_standard_cuts = False
apply_deep_cuts = True
if apply_standard_cuts:
    if args.other_type == 'e':
        df_all = df_all[ \
            (np.bitwise_and(df_all['byIsolationMVArun2017v2DBoldDMwLT2017'], 1 << DiscriminatorWP.VVLoose) > 0) \
            & (np.bitwise_and(df_all['againstMuon3'], 1 << DiscriminatorWP.Loose) > 0) \
            & (df_all['tau_decayMode'] != 5) & (df_all['tau_decayMode'] != 6) ]
    elif args.other_type == 'mu':
        df_all = df_all[ \
            (np.bitwise_and(df_all['byIsolationMVArun2017v2DBoldDMwLT2017'], 1 << DiscriminatorWP.VVLoose) > 0) \
            & (np.bitwise_and(df_all['againstElectronMVA6'], 1 << DiscriminatorWP.VLoose) > 0) \
            & (df_all['tau_decayMode'] != 5) & (df_all['tau_decayMode'] != 6) ]
    elif args.other_type == 'jet':
        df_all = df_all[ (np.bitwise_and(df_all['againstElectronMVA6'], 1 << DiscriminatorWP.VLoose) > 0) \
                         & (np.bitwise_and(df_all['againstMuon3'], 1 << DiscriminatorWP.Loose) > 0) \
                         & (df_all['tau_decayMode'] != 5) & (df_all['tau_decayMode'] != 6) ]
elif apply_deep_cuts:
    if args.other_type == 'e':
        df_all = df_all[(df_all['deepId_vs_jet'] > deep_wp_thrs['jet']['VVVLoose']) \
            & (df_all['deepId_vs_mu'] > deep_wp_thrs['mu']['VLoose']) ]
    elif args.other_type == 'mu':
        df_all = df_all[(df_all['deepId_vs_e'] > deep_wp_thrs['e']['VVVLoose']) \
            & (df_all['deepId_vs_jet'] > deep_wp_thrs['jet']['VVVLoose'])]
    elif args.other_type == 'jet':
        df_all = df_all[ (df_all['deepId_vs_e'] > deep_wp_thrs['e']['VVVLoose']) \
            & (df_all['deepId_vs_mu'] > deep_wp_thrs['mu']['VLoose'])]


pt_bins = [ 20, 30, 40, 50, 70, 100, 150, 200, 300, 500, 1000 ]
#pt_bins = [ 20, 30 ]

plot_setups = {
    'e': PlotSetup(ylabel='Electron mis-id probability', xlim=[0.4, 1], ratio_yscale='log',
                   ylim=[2e-5, 1], ratio_ylim=[0.4, 20]),
    'mu': PlotSetup(ylabel='Muon mis-id probability', xlim=[0.98, 1], ratio_yscale='log',
                    ratio_ylim=[ [0.5, 20], [0.5, 20], [0.5, 20], [0.5, 20], [0.5, 20],
                                 [0.5, 20], [0.5, 20], [0.5, 50], [0.5, 50], [0.5, 50] ] ),
    'jet': PlotSetup(ylabel='Jet mis-id probability', xlim=[0.2, 1],
                     ylim=[ [2e-4, 1], [8e-5, 1], [8e-5, 1], [2e-5, 1], [2e-5, 1],
                            [5e-6, 1], [5e-6, 1], [5e-6, 1], [5e-6, 1], [2e-6, 1] ],
                     ratio_ylim=[ [0.5, 2], [0.5, 2.5], [0.5, 2.5], [0.5, 2.5], [0.5, 2.5],
                                  [0.5, 3.5], [0.5, 3.5], [0.5, 3.5], [0.5, 10], [0.5, 10] ] )
}

def create_roc_ratio(x1, y1, x2, y2):
    idx_min = np.argmax((x2 >= x1[0]) & (y2 > 0))
    if x2[-1] <= x1[-1]:
        idx_max = x2.shape[0]
    else:
         idx_max = np.argmax(x2 > x1[-1])
    sp = interpolate.interp1d(x1, y1)
    x1_upd = x2[idx_min:idx_max]
    y1_upd = sp(x1_upd)
    ratio = np.empty((2, x1_upd.shape[0]))
    ratio[0, :] = y1_upd / y2[idx_min:idx_max]
    ratio[1, :] = x1_upd
    return ratio

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

        for n in reversed(range(n_discr)):
            ref_roc = rocs[-1]
            rocs[n], wp_rocs[n] = discriminators[n].CreateRocCurve(df_tx, ref_roc)
            if rocs[n].auc_score is not None:
                #target_prs = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99, 0.995 ]
                #thrs = [ find_threshold(rocs[n].pr[1, :], rocs[n].thresholds, pr) for pr in target_prs ]
                print('[{}, {}] {} roc_auc = {}'.format(pt_bins[pt_index], pt_bins[pt_index + 1], names[n],
                                                        rocs[n].auc_score))
                #print(thrs)

        fig, (ax, ax_ratio) = plt.subplots(2, 1, figsize=(7,6), sharex=True, gridspec_kw = {'height_ratios':[3, 1]})
        for n in range(n_discr):
            rocs[n].Draw(ax, ax_ratio)
        for n in range(n_discr):
            if wp_rocs[n] is not None:
                wp_rocs[n].Draw(ax, ax_ratio)

        plot_setups[args.other_type].Apply(names, pt_index, ax, ax_ratio)

        ax.set_title('tau vs {}. pt range ({}, {}) GeV'.format(args.other_type, pt_bins[pt_index],
                     pt_bins[pt_index + 1]), fontsize=18, y=1.04)
        plt.subplots_adjust(hspace=0)
        pdf.savefig(fig, bbox_inches='tight')
