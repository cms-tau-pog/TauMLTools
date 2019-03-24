#!/usr/bin/env python

import argparse
parser = argparse.ArgumentParser(description='Apply training and store results.')
parser.add_argument('--input-taus', required=True, type=str, help="Input file with taus")
parser.add_argument('--input-other', required=True, type=str, help="Input file with non-taus")
parser.add_argument('--other-type', required=True, type=str, help="Type of non-tau objects")
parser.add_argument('--deep-results', required=True, type=str, help="Directory with deepId results")
parser.add_argument('--output', required=True, type=str, help="Output pdf file")
#parser.add_argument('--apply-loose', action="store_true", help="Submission dryrun.")
args = parser.parse_args()

import os
import pandas
import numpy as np
import uproot
from sklearn import metrics
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from statsmodels.stats.proportion import proportion_confint
from scipy import interpolate
from common import *

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

class RocCurve:
    def __init__(self, n_points, color, has_errors):
        self.pr = np.zeros((2, n_points))
        self.color = color
        if has_errors:
            self.pr_err = np.zeros((2, 2, n_points))
        else:
            self.pr_err = None
        self.ratio = None

    def Draw(self, ax, ax_ratio = None):
        if self.pr_err is not None:
            ax.errorbar(self.pr[1], self.pr[0], xerr=self.pr_err[1], yerr=self.pr_err[0], color=self.color,
                        fmt='--o', markersize='3', linewidth=1)
        else:
            ax.errorbar(self.pr[1], self.pr[0], color=self.color)
        if self.ratio is not None and ax_ratio is not None:
            ax_ratio.plot(self.ratio[1], self.ratio[0], color=self.color, linewidth=1)

class PlotSetup:
    def __init__(self, xlim = None, ylim = None, ratio_ylim = None, ylabel = None):
        self.xlim = xlim
        self.ylim = ylim
        self.ratio_ylim = ratio_ylim
        self.ylabel = ylabel

    def Apply(self, names, ax, ax_ratio = None):
        if self.xlim is not None:
            ax.set_xlim(self.xlim)
        if self.ylim is not None:
            ax.set_ylim(self.ylim)

        ax.set_yscale('log')
        ax.set_ylabel(self.ylabel, fontsize=16)
        ax.tick_params(labelsize=14)
        ax.grid(True)
        ax.legend(names, fontsize=14, loc='upper left')

        if ax_ratio is not None:
            if self.ratio_ylim is not None:
                ax_ratio.set_ylim(self.ratio_ylim)

            ax_ratio.set_xlabel('Tau ID efficiency', fontsize=16)
            ax_ratio.set_ylabel('id/deepId', fontsize=14)
            ax_ratio.tick_params(labelsize=10)

            ax_ratio.grid(True)


class Discriminator:
    def __init__(self, name, column, raw, from_tuple, color, working_points = []):
        self.name = name
        self.column = column
        self.raw = raw
        self.from_tuple = from_tuple
        self.color = color
        self.working_points = working_points

    def CountPassed(self, df, wp):
        flag = 1 << wp
        return np.count_nonzero(np.bitwise_and(df[self.column], flag))

    def CreateRocCurve(self, df, ref_roc = None):
        if self.raw:
            fpr, tpr, thresholds = metrics.roc_curve(df['gen_tau'].values, df[self.column].values)
            roc = RocCurve(len(fpr), self.color, False)
            roc.pr[0, :] = fpr
            roc.pr[1, :] = tpr
        else:
            n_wp = len(self.working_points)
            roc = RocCurve(n_wp, self.color, True)
            for n in range(n_wp):
                for kind in [0, 1]:
                    df_x = df[df['gen_tau'] == kind]
                    n_passed = self.CountPassed(df_x, self.working_points[n])
                    n_total = df_x.shape[0]
                    ci_low, ci_upp = proportion_confint(n_passed, n_total, alpha=1-0.68, method='beta')
                    eff = float(n_passed) / n_total
                    roc.pr[kind, n_wp - n - 1] = eff
                    roc.pr_err[kind, 1, n_wp - n - 1] = ci_upp - eff
                    roc.pr_err[kind, 0, n_wp - n - 1] = eff - ci_low
        if ref_roc is not None:
            roc.ratio = create_roc_ratio(roc.pr[1], roc.pr[0], ref_roc.pr[1], ref_roc.pr[0])

        return roc


        fpr[n][0], tpr[n][0], thresholds = metrics.roc_curve(df_tx['gen_tau'], df_tx[discriminators[n].column].values)
        if n != n_discr - 1:
            tpr[n][1], fpr[n][1] = create_roc_ratio(tpr[n][0], fpr[n][0], tpr[-1][0], fpr[-1][0])


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

all_discriminators = {
    'e': [
        Discriminator('MVA6', 'againstElectronMVA6', False, True, 'green',
                      [ DiscriminatorWP.VLoose, DiscriminatorWP.Loose, DiscriminatorWP.Medium, DiscriminatorWP.Tight,
                        DiscriminatorWP.VTight ] ),
        Discriminator('MVA6 2018', 'againstElectronMVA62018', False, True, 'red',
                      [ DiscriminatorWP.VLoose, DiscriminatorWP.Loose, DiscriminatorWP.Medium, DiscriminatorWP.Tight,
                        DiscriminatorWP.VTight ] ),
        Discriminator('deepTau 2017v1', 'byDeepTau2017v1VSeraw', True, True, 'blue'),
        Discriminator('new deepTau', 'deepId_vs_e', True, False, 'yellow')
    ],
    'mu': [
        Discriminator('againstMuon3', 'againstMuon3', False, True, 'green',
                      [ DiscriminatorWP.Loose, DiscriminatorWP.Tight] ),
        Discriminator('deepTau 2017v1', 'byDeepTau2017v1VSmuraw', True, True, 'blue'),
        Discriminator('new deepTau', 'deepId_vs_mu', True, False, 'yellow')
    ],
    'jet': [
        Discriminator('MVA 2017v2', 'byIsolationMVArun2017v2DBoldDMwLT2017raw', True, True, 'green'),
        Discriminator('MVA 2017v2 newDM', 'byIsolationMVArun2017v2DBnewDMwLT2017raw', True, True, 'red'),
        Discriminator('DPF 2016v0', 'byDpfTau2016v0VSallraw', True, True, 'magenta'),
        Discriminator('deepTau 2017v1', 'byDeepTau2017v1VSjetraw', True, True, 'blue'),
        Discriminator('new deepTau', 'deepId_vs_jet', True, False, 'yellow')
    ]
}

plot_setups = {
    'e': PlotSetup(xlim=[0.4, 1], ratio_ylim=[0, 2], ylabel='Electron mis-id probability'),
    'mu': PlotSetup(xlim=[0.9, 1], ratio_ylim=[0, 2], ylabel='Muon mis-id probability'),
    'jet': PlotSetup(xlim=[0.2, 1], ratio_ylim=[0, 2], ylabel='Jet mis-id probability')
}

if args.other_type not in all_discriminators:
    raise RuntimeError("Unknown other_type")

discriminators = all_discriminators[args.other_type]
all_branches = core_branches + [ disc.column for disc in discriminators if disc.from_tuple == True ]

def CreateDF(file_name):
    df = ReadBrancesToDataFrame(file_name, 'taus', all_branches)
    base_name = os.path.basename(file_name)
    pred_file_name = os.path.splitext(base_name)[0] + '_pred.h5'
    df_pred = pandas.read_hdf(os.path.join(args.deep_results, pred_file_name))
    #tau_vs_other = - df_pred['deepId_' + args.other_type].values
    tau_vs_other = TauLosses.tau_vs_other(df_pred['deepId_tau'].values, df_pred['deepId_' + args.other_type].values)
    tau_vs_other = tau_vs_other * (df_pred['deepId_tau'].values > 0.5) + \
                   df_pred['deepId_tau'].values * (df_pred['deepId_tau'].values <= 0.5)
    df['deepId_vs_' + args.other_type] = pandas.Series(tau_vs_other, index=df.index)
    df['tau_pt'] = pandas.Series(df.tau_pt *(1000 - 20) + 20, index=df.index)
    return df

df_taus = CreateDF(args.input_taus)
df_other = CreateDF(args.input_other)
df_all = df_taus.append(df_other)

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

pt_bins = [ 20, 30, 40, 50, 70, 100, 150, 200, 300, 500, 1000 ]

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

        for n in reversed(range(n_discr)):
            ref_roc = rocs[-1]
            rocs[n] = discriminators[n].CreateRocCurve(df_tx, ref_roc)

        fig, (ax, ax_ratio) = plt.subplots(2, 1, figsize=(7,6), sharex=True, gridspec_kw = {'height_ratios':[3, 1]})
        for n in range(n_discr):
            rocs[n].Draw(ax, ax_ratio)

        names = [ disc.name for disc in discriminators ]
        plot_setups[args.other_type].Apply(names, ax, ax_ratio)

        ax.set_title('tau vs {}. pt range ({}, {}) GeV'.format(args.other_type, pt_bins[pt_index],
                     pt_bins[pt_index + 1]), fontsize=18, y=1.04)
        plt.subplots_adjust(hspace=0)
        pdf.savefig(fig, bbox_inches='tight')
