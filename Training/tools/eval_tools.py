import numpy as np
import pandas
import uproot
import math
import copy
from sklearn import metrics
from scipy import interpolate
from _ctypes import PyObj_FromPtr
import json
import re
import sys

if sys.version_info.major > 2:
    from statsmodels.stats.proportion import proportion_confint

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
    def __init__(self, n_points, color, has_errors, dots_only = False, dashed = False):
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
        self.dashed = dashed
        self.marker_size = '5'

    def Draw(self, ax, ax_ratio = None):
        main_plot_adjusted = False
        if self.pr_err is not None:
            x = self.pr[1]
            y = self.pr[0]
            entry = ax.errorbar(x, y, xerr=self.pr_err[1], yerr=self.pr_err[0], color=self.color,
                        fmt='o', markersize=self.marker_size, linewidth=1)
            # sp = interpolate.interp1d(x, y, kind='linear', fill_value="extrapolate")
            # x_fine = np.arange(x[0], x[-1], step=(x[-1] - x[0]) / 1000)
            # y_fine = sp(x_fine)
            # ax.errorbar(x_fine, y_fine, color=self.color, linewidth=1, fmt='--')

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
                # sp = interpolate.interp1d(x, y, kind='cubic', fill_value="extrapolate")
                # x_fine = np.arange(x[0], x[-1], step=(x[-1] - x[0]) / 1000)
                # y_fine = sp(x_fine)
                # ax_ratio.errorbar(x_fine, y_fine, color=self.color, linewidth=1, fmt='--')
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

    def Prune(self, tpr_decimals=3):
        pruned = copy.deepcopy(self)
        rounded_tpr = np.round(self.pr[1, :], decimals=tpr_decimals)
        unique_tpr, idx = np.unique(rounded_tpr, return_index=True)
        idx = np.sort(idx)
        n_points = len(idx)
        pruned.pr = np.zeros((2, n_points))
        pruned.pr[0, :] = self.pr[0, idx]
        pruned.pr[1, :] = self.pr[1, idx]
        if self.thresholds is not None:
            pruned.thresholds = self.thresholds[idx]
        if self.pr_err is not None:
            pruned.pr_err = np.zeros((2, 2, n_points))
            pruned.pr_err[:, :, :] = self.pr_err[:, :, idx]
        return pruned

class PlotSetup:
    def __init__(self, xlim = None, ylim = None, ratio_ylim = None, ylabel = None, yscale='log',
                 ratio_yscale='linear', legend_loc='upper left', ratio_ylabel_pad=20):
        self.xlim = xlim
        self.ylim = ylim
        self.ratio_ylim = ratio_ylim
        self.ylabel = ylabel
        self.yscale = yscale
        self.ratio_yscale = ratio_yscale
        self.legend_loc = legend_loc
        self.ratio_ylabel_pad = ratio_ylabel_pad

    def Apply(self, names, entries, range_index, ratio_title, ax, ax_ratio = None):
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
        ax.legend(entries, names, fontsize=14, loc=self.legend_loc)
        #ax.legend(names, fontsize=14, loc='lower right')

        if ax_ratio is not None:
            if self.ratio_ylim is not None:
                ylim = self.ratio_ylim[range_index] if type(self.ratio_ylim[0]) == list else self.ratio_ylim
                ax_ratio.set_ylim(ylim)

            ax_ratio.set_yscale(self.ratio_yscale)
            ax_ratio.set_xlabel('Tau ID efficiency', fontsize=16)
            #ratio_title = 'MVA/DeepTau' if args.other_type != 'mu' else 'cut based/DeepTau'
            ax_ratio.set_ylabel(ratio_title, fontsize=14, labelpad=self.ratio_ylabel_pad)
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
                 working_points_thrs = None, dashed=False, draw_wp=True):
        self.name = name
        self.column = column
        self.raw = raw
        self.from_tuple = from_tuple
        self.color = color
        self.working_points = working_points
        self.wp_column = wp_column if wp_column is not None else column
        self.working_points_thrs = working_points_thrs
        self.dashed = dashed
        if not draw_wp and self.raw:
            self.working_points = []

    def CountPassed(self, df, wp):
        if self.from_tuple:
            flag = 1 << wp
            passed = (np.bitwise_and(df[self.wp_column], flag) != 0).astype(int)
            return np.sum(passed * df.weight.values)
        if self.working_points_thrs is None:
            raise RuntimeError('Working points are not specified for discriminator "{}"'.format(self.name))
        wp_thr = self.working_points_thrs[DiscriminatorWP.GetName(wp)]
        return np.sum(df[df[self.column] > wp_thr].weight.values)

    def CreateRocCurve(self, df, ref_roc = None):
        n_wp = len(self.working_points)
        wp_roc = None
        if self.raw:
            fpr, tpr, thresholds = metrics.roc_curve(df['gen_tau'].values, df[self.column].values,
                                                     sample_weight=df.weight.values)
            roc = RocCurve(len(fpr), self.color, False, dashed=self.dashed)
            roc.pr[0, :] = fpr
            roc.pr[1, :] = tpr
            roc.thresholds = thresholds
            roc.auc_score = metrics.roc_auc_score(df['gen_tau'].values, df[self.column].values,
                                                  sample_weight=df.weight.values)
        if n_wp > 0:
            wp_roc = RocCurve(n_wp, self.color, not self.raw, self.raw)
            for n in range(n_wp):
                for kind in [0, 1]:
                    df_x = df[df['gen_tau'] == kind]
                    n_passed = self.CountPassed(df_x, self.working_points[n])
                    n_total = np.sum(df_x.weight.values)
                    eff = float(n_passed) / n_total
                    wp_roc.pr[kind, n_wp - n - 1] = eff
                    if not self.raw:
                        if sys.version_info.major > 2:
                            ci_low, ci_upp = proportion_confint(n_passed, n_total, alpha=1-0.68, method='beta')
                        else:
                            err = math.sqrt(eff * (1 - eff) / n_total)
                            ci_low, ci_upp = eff - err, eff + err
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

# def create_roc_ratio(x1, y1, x2, y2):
#     idx_min = np.argmax((x2 >= x1[0]) & (y2 > 0))
#     if x2[-1] <= x1[-1]:
#         idx_max = x2.shape[0]
#     else:
#          idx_max = np.argmax(x2 > x1[-1])
#     sp = interpolate.interp1d(x1, y1, kind='cubic')
#     x1_upd = x2[idx_min:idx_max]
#     y1_upd = sp(x1_upd)
#     ratio = np.empty((2, x1_upd.shape[0]))
#     ratio[0, :] = y1_upd / y2[idx_min:idx_max]
#     ratio[1, :] = x1_upd
#     return ratio

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

class FloatList(object):
    def __init__(self, value):
        self.value = value

class CustomJsonEncoder(json.JSONEncoder):
    placeholder = '@@{}@@'
    placeholder_re = re.compile(placeholder.format(r'(\d+)'))

    def __init__(self, **kwargs):
        self.__sort_keys = kwargs.get('sort_keys', None)
        super(CustomJsonEncoder, self).__init__(**kwargs)

    def default(self, obj):
        if isinstance(obj, FloatList):
            return self.placeholder.format(id(obj))
        return super(CustomJsonEncoder, self).default(obj)

    def encode(self, obj):
        json_repr = super(CustomJsonEncoder, self).encode(obj)
        for match in self.placeholder_re.finditer(json_repr):
            id = int(match.group(1))
            float_list = PyObj_FromPtr(id)
            json_obj_repr = '[ ' + ', '.join([ '{:.4e}'.format(x) for x in float_list.value ]) + ' ]'
            json_repr = json_repr.replace('"{}"'.format(self.placeholder.format(id)), json_obj_repr)

        return json_repr
