#!/usr/bin/env python

import argparse
parser = argparse.ArgumentParser(description='Plot ROC curves.')
parser.add_argument('--input', required=True, type=str, help="Input JSON file with ROC curves")
parser.add_argument('--output', required=True, type=str, help="Output PDF file")

args = parser.parse_args()

import json
import numpy as np
from scipy import interpolate
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

def create_roc_ratio(x1, y1, x2, y2):
    sp = interpolate.interp1d(x2, y2)
    y2_upd = sp(x1)
    y2_upd_clean = y2_upd[y2_upd > 0]
    x1_clean = x1[y2_upd > 0]
    y1_clean = y1[y2_upd > 0]
    ratio = np.empty((2, x1_clean.shape[0]))
    ratio[0, :] = y1_clean / y2_upd_clean
    ratio[1, :] = x1_clean
    return ratio

def cm2inch(*tupl):
    inch = 2.54
    if isinstance(tupl[0], tuple):
        return tuple(i/inch for i in tupl[0])
    return tuple(i/inch for i in tupl)

class RocCurve:
    def __init__(self, data, ref_roc=None):
        fpr = np.array(data['false_positive_rate'])
        n_points = len(fpr)
        self.pr = np.empty((2, n_points))
        self.pr[0, :] = fpr
        self.pr[1, :] = data['true_positive_rate']

        self.color = data['color']
        if 'false_positive_rate_up' in data:
            self.pr_err = np.empty((2, 2, n_points))
            self.pr_err[0, 0, :] = data['false_positive_rate_up']
            self.pr_err[0, 1, :] = data['false_positive_rate_down']
            self.pr_err[1, 0, :] = data['true_positive_rate_up']
            self.pr_err[1, 1, :] = data['true_positive_rate_down']
        else:
            self.pr_err = None

        if 'threasholds' in data:
            self.thresholds = np.empty(n_points)
            self.thresholds[:] = data['threashods']
        else:
            self.thresholds = None

        self.auc_score = data.get('auc_score')
        self.dots_only = data['dots_only']
        self.dashed = data['dashed']
        self.marker_size = data.get('marker_size', 5)

        if ref_roc is None:
            ref_roc = self
        self.ratio = create_roc_ratio(self.pr[1], self.pr[0], ref_roc.pr[1], ref_roc.pr[0])

    def Draw(self, ax, ax_ratio = None):
        main_plot_adjusted = False
        if self.pr_err is not None:
            x = self.pr[1]
            y = self.pr[0]
            entry = ax.errorbar(x, y, xerr=self.pr_err[1], yerr=self.pr_err[0], color=self.color,
                        fmt='o', markersize=self.marker_size, linewidth=1)
        else:
            if self.dots_only:
                entry = ax.errorbar(self.pr[1], self.pr[0], color=self.color, fmt='o', markersize=self.marker_size)
            else:
                fmt = '--' if self.dashed else ''
                x = self.pr[1]
                y = self.pr[0]
                if x[-1] - x[-2] > 0.01:
                    x = x[:-1]
                    y = y[:-1]
                    x_max_main = x[-1]
                    main_plot_adjusted = True
                entry = ax.errorbar(x, y, color=self.color, fmt=fmt)
        if self.ratio is not None and ax_ratio is not None:
            if self.pr_err is not None:
                x = self.ratio[1]
                y = self.ratio[0]
                ax_ratio.errorbar(x, y, color=self.color, fmt='o', markersize='5', linewidth=1)
            else:
                linestyle = 'dashed' if self.dashed else None
                x = self.ratio[1]
                y = self.ratio[0]
                if main_plot_adjusted:
                    n = 0
                    while x[n] < x_max_main and n < len(x):
                        n += 1
                    x = x[:n]
                    y = y[:n]
                ax_ratio.plot(x, y, color=self.color, linewidth=1, linestyle=linestyle)
        return entry

class PlotSetup:
    def __init__(self, data):
        for lim_name in [ 'x', 'y', 'ratio_y' ]:
            if lim_name + '_min' in data and lim_name + '_max' in data:
                lim_value = (data[lim_name + '_min'], data[lim_name + '_max'])
            else:
                lim_value = None
            setattr(self, lim_name + 'lim', lim_value)
        self.ylabel = data.get('ylabel')
        self.yscale = data.get('yscale', 'log')
        self.ratio_yscale = data.get('ratio_yscale', 'linear')
        self.legend_loc = data.get('legend_loc', 'upper left')
        self.ratio_title = data.get('ratio_title', 'ratio')
        self.ratio_ylabel_pad = data.get('ratio_ylabel_pad', 20)

    def Apply(self, names, entries, ax, ax_ratio = None):
        if self.xlim is not None:
            ax.set_xlim(self.xlim)
        if self.ylim is not None:
            ax.set_ylim(self.ylim)

        ax.set_yscale(self.yscale)
        ax.set_ylabel(self.ylabel, fontsize=16)
        ax.tick_params(labelsize=14)
        ax.grid(True)
        ax.legend(entries, names, fontsize=14, loc=self.legend_loc)

        if ax_ratio is not None:
            if self.ratio_ylim is not None:
                ax_ratio.set_ylim(self.ratio_ylim)

            ax_ratio.set_yscale(self.ratio_yscale)
            ax_ratio.set_xlabel('Tau ID efficiency', fontsize=16)
            ax_ratio.set_ylabel(self.ratio_title, fontsize=14, labelpad=self.ratio_ylabel_pad)
            ax_ratio.tick_params(labelsize=10)
            ax_ratio.grid(True, which='both')



def create_discr_list(all_discr):
    discr_map = { discr["name"] : discr for discr in all_discr }
    discriminators = []
    names = []
    ref_index = -1
    ordered_indices = []
    for discr in all_discr:
        name = discr['name']
        if name.endswith(' WP'): continue
        discr_wp = discr_map.get(name + ' WP')
        if discr['is_ref']:
            ref_index = len(names)
        discriminators.append((discr, discr_wp))
        names.append(name)
    if ref_index >= 0:
        ordered_indices.append(ref_index)
    for index in range(len(names)):
        if index != ref_index:
            ordered_indices.append(index)
    return discriminators, names, ref_index, ordered_indices


with open(args.input, 'r') as f:
    data = json.load(f)

with PdfPages(args.output) as pdf:
    for plot_data in data:
        plot_setup = PlotSetup(plot_data["plot_setup"])

        discriminators, names, ref_index, ordered_indices = create_discr_list(plot_data["discriminators"])
        n_discr = len(discriminators)
        rocs = [None] * n_discr
        wp_rocs = [None] * n_discr

        for n in ordered_indices:
            ref_roc = rocs[ref_index] if ref_index >= 0 and n != ref_index else None
            rocs[n] = RocCurve(discriminators[n][0], ref_roc)
            if discriminators[n][1] is not None:
                wp_rocs[n] = RocCurve(discriminators[n][1], ref_roc)

        fig, (ax, ax_ratio) = plt.subplots(2, 1, figsize=(7, 7), sharex=True, gridspec_kw = {'height_ratios':[3, 1]})

        plot_entries = []
        for n in range(n_discr):
            plot_entry = rocs[n].Draw(ax, ax_ratio)
            plot_entries.append(plot_entry)
        for n in range(n_discr):
            if wp_rocs[n] is not None:
                wp_rocs[n].Draw(ax, ax_ratio)

        plot_setup.Apply(names, plot_entries, ax, ax_ratio)

        header_y = 1.02
        ax.text(0.03, 0.92 - n_discr * 0.10, plot_data['pt_text'], fontsize=14, transform=ax.transAxes)
        ax.text(0.01, header_y, 'CMS', fontsize=14, transform=ax.transAxes, fontweight='bold', fontfamily='sans-serif')
        ax.text(0.12, header_y, 'Simulation Preliminary', fontsize=14, transform=ax.transAxes, fontstyle='italic',
                fontfamily='sans-serif')
        ax.text(0.73, header_y, plot_data['period'], fontsize=13, transform=ax.transAxes, fontweight='bold',
                fontfamily='sans-serif')
        plt.subplots_adjust(hspace=0)
        pdf.savefig(fig, bbox_inches='tight')
