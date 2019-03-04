#!/usr/bin/env python

import argparse
parser = argparse.ArgumentParser(description='Apply training and store results.')
parser.add_argument('--input-taus', required=True, type=str, help="Input file with taus")
parser.add_argument('--input-other', required=True, type=str, help="Input file with non-taus")
parser.add_argument('--other-type', required=True, type=str, help="Type of non-tau objects")
parser.add_argument('--deep-results', required=True, type=str, help="Directory with deepId results")
#parser.add_argument('--apply-loose', action="store_true", help="Submission dryrun.")
args = parser.parse_args()

import os
import pandas
import numpy as np
import uproot
from sklearn import metrics
import matplotlib.pyplot as plt
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
        return np.count_nonzero(np.bitwise_and(df[self.name], flag))

def ReadBrancesToDataFrame(file_name, tree_name, branches):
    with uproot.open(file_name) as file:
        tree = file[tree_name]
        df = tree.arrays(branches, outputtype=pandas.DataFrame)
    return df

core_branches = [ 'tau_pt', 'tau_decayModeFinding', 'tau_decayMode', 'gen_tau' ]

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

if args.other_type not in all_discriminators:
    raise RuntimeError("Unknown other_type")

discriminators = all_discriminators[args.other_type]
all_branches = core_branches + [ disc.column for disc in discriminators if disc.from_tuple == True ]

def CreateDF(file_name):
    df = ReadBrancesToDataFrame(file_name, 'taus', all_branches)
    base_name = os.path.basename(file_name)
    pred_file_name = os.path.splitext(base_name)[0] + '_pred.h5'
    df_pred = pandas.read_hdf(os.path.join(args.deep_results, pred_file_name))
    tau_vs_other = TauLosses.tau_vs_other(df_pred['deepId_tau'].values, df_pred['deepId_' + args.other_type].values)
    #print(file_name, np.amin(df_pred['deepId_jet'].values), np.amax(df_pred['deepId_jet'].values))
    df['deepId_vs_' + args.other_type] = pandas.Series(tau_vs_other, index=df.index)
    return df

df_taus = CreateDF(args.input_taus)
df_other = CreateDF(args.input_other)
df_all = df_taus.append(df_other)

pt_bins = [ 20, 30, 40, 50, 70, 100 ]

def create_roc_ratio(x1, y1, x2, y2):
    idx_min = np.argmax((x2 >= x1[0]) & (y2 > 0))
    if x2[-1] <= x1[-1]:
        idx_max = x2.shape[0]
    else:
         idx_max = np.argmax(x2 > x1[-1])
    sp = interpolate.interp1d(x1, y1)
    x1_upd = x2[idx_min:idx_max]
    y1_upd = sp(x1_upd)
    return x1_upd, y1_upd / y2[idx_min:idx_max]

for pt_index in range(len(pt_bins) - 1):
    df_tx = df_all[(df_all.tau_pt > pt_bins[pt_index]) & (df_all.tau_pt < pt_bins[pt_index + 1])]
    if df_tx.shape[0] == 0:
        print("Warning: pt bin ({}, {}) is empty.".format(pt_bins[pt_index], pt_bins[pt_index + 1]))
    n_discr = len(discriminators)
    fpr = [[None, None] for n in range(n_discr)]
    tpr = [[None, None] for n in range(n_discr)]

    for n in reversed(range(n_discr)):
        #print(discriminators[n].column, df_tx['gen_tau'].shape, df_tx[discriminators[n].column].shape)
        fpr[n][0], tpr[n][0], thresholds = metrics.roc_curve(df_tx['gen_tau'], df_tx[discriminators[n].column].values)
        if n != n_discr - 1:
            tpr[n][1], fpr[n][1] = create_roc_ratio(tpr[n][0], fpr[n][0], tpr[-1][0], fpr[-1][0])

    fig, (ax, ax_ratio) = plt.subplots(2, 1, figsize=(7,6), sharex=True, gridspec_kw = {'height_ratios':[3, 1]})
    for n in range(n_discr):
        ax.semilogy(tpr[n][0], fpr[n][0], color=discriminators[n].color)
        if tpr[n][1] is not None:
            ax_ratio.plot(tpr[n][1], fpr[n][1], color=discriminators[n].color, linewidth=1)

    #plt.ylim([0,0.2])
    #plt.xlim([.4, 0.8])
    ax_ratio.set_ylim([0, 3.5])
    ax.set_ylabel('Mis-id probability', fontsize=16)
    ax_ratio.set_xlabel('Tau ID efficiency', fontsize=16)
    ax_ratio.set_ylabel('id/deepId', fontsize=14)
    ax.tick_params(labelsize=14)
    ax_ratio.tick_params(labelsize=10)

    ax.grid(True)
    ax_ratio.grid(True)

    names = [ disc.name for disc in discriminators ]
    ax.legend(names, fontsize=14, loc='lower right')

    plt.subplots_adjust(hspace=0)
    #plt.show()
    fig.savefig('tau_vs_{}_pt-{}_{}.pdf'.format(args.other_type, pt_bins[pt_index], pt_bins[pt_index + 1]),
                bbox_inches='tight')
