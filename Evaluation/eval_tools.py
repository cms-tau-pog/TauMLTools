import numpy as np
import pandas as pd
import uproot
import math
import copy
from sklearn import metrics
from scipy import interpolate
from _ctypes import PyObj_FromPtr
import os
import h5py
import json
import re
import sys
from dataclasses import dataclass, field

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

@dataclass
class PlotSetup:
    xlim: list = None
    ylim: list = None
    ratio_ylim: list = None
    ylabel: str = None
    yscale: str = 'log'
    ratio_yscale: str = 'linear'
    legend_loc: str = 'upper left'
    ratio_ylabel_pad: int = 20

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

@dataclass
class Discriminator:
    name: str
    pred_column: str
    raw: bool
    from_tuple: bool
    color: str
    working_points: list = field(default_factory=list)
    wp_column: str = None
    working_points_thrs: dict = None 
    dashed: bool = False 
    draw_wp: bool = True

    def __post_init__(self):
        if not self.draw_wp and self.raw:
            self.working_points = []
        if self.wp_column is None:
            self.wp_column = self.pred_column

    def CountPassed(self, df, wp):
        if self.from_tuple:
            flag = 1 << wp
            passed = (np.bitwise_and(df[self.wp_column], flag) != 0).astype(int)
            return np.sum(passed * df.weight.values)
        if self.working_points_thrs is None:
            raise RuntimeError('Working points are not specified for discriminator "{}"'.format(self.name))
        wp_thr = self.working_points_thrs[DiscriminatorWP.GetName(wp)]
        return np.sum(df[df[self.pred_column] > wp_thr].weight.values)

    def CreateRocCurve(self, df, ref_roc = None):
        n_wp = len(self.working_points)
        roc, wp_roc = None, None
        if self.raw:
            fpr, tpr, thresholds = metrics.roc_curve(df['gen_tau'].values, df[self.pred_column].values,
                                                     sample_weight=df.weight.values)
            roc = RocCurve(len(fpr), self.color, False, dashed=self.dashed)
            roc.pr[0, :] = fpr
            roc.pr[1, :] = tpr
            roc.thresholds = thresholds
            roc.auc_score = metrics.roc_auc_score(df['gen_tau'].values, df[self.pred_column].values,
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

        return roc, wp_roc

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

def select_curve(curve_list, **selection):
    filter_func = lambda x: all([x[k]==v if k in x else False for k,v in selection.items()])
    filtered_curves = list(filter(filter_func, curve_list))
    if len(filtered_curves)==0:
        return None
    elif len(filtered_curves)==1:
        return filtered_curves[0]
    else:
        raise Exception(f"Failed to find a single curve for selection: {[f'{k}=={v}' for k,v in selection.items()]}")

def create_df(path_to_input_file, input_branches, path_to_pred_file, pred_column_prefix, path_to_weights):
    def read_branches(path_to_file, tree_name, branches):
        if path_to_file.endswith('.root'):
            with uproot.open(path_to_file) as f:
                tree = f[tree_name]
                df = tree.arrays(branches, library='pd')
            return df
        elif path_to_file.endswith('.h5') or path_to_file.endswith('.hdf5'):
            return pd.read_hdf(path_to_file, tree_name, columns=branches)
        raise RuntimeError("Unsupported file type.")

    def add_predictions(df, path_to_pred_file, pred_column_prefix):
        with h5py.File(path_to_pred_file, 'r') as f:
            file_keys = list(f.keys())
        if 'predictions' in file_keys: 
            df_pred = pd.read_hdf(path_to_pred_file, 'predictions')
        else:
            df_pred = pd.read_hdf(path_to_pred_file)
        prob_tau = df_pred[f'{pred_column_prefix}tau'].values
        for node_column in df_pred.columns:
            if not node_column.startswith(pred_column_prefix): continue # assume prediction column name to be "{pred_column_prefix}{tau_type}"
            tau_type = node_column.split(f'{pred_column_prefix}')[-1] 
            if tau_type != 'tau':
                prob_vs_type = df_pred[pred_column_prefix + tau_type].values
                tau_vs_other_type = np.where(prob_tau > 0, prob_tau / (prob_tau + prob_vs_type), np.zeros(prob_tau.shape))
                df[pred_column_prefix + tau_type] = pd.Series(tau_vs_other_type, index=df.index)
        return df

    def add_targets(df, path_to_target_file, pred_column_prefix):
        with h5py.File(path_to_target_file, 'r') as f:
            file_keys = list(f.keys())
        if 'targets' in file_keys: 
            df_targets = pd.read_hdf(path_to_target_file, 'targets')
        else:
            return df
        for node_column in df_targets.columns:
            if not node_column.startswith(pred_column_prefix): continue # assume prediction column name to be "{pred_column_prefix}{tau_type}"
            tau_type = node_column.split(f'{pred_column_prefix}')[-1]
            df[f'gen_{tau_type}'] = df_targets[node_column]
        return df
    
    def add_weights(df, path_to_weight_file):
        df_weights = pd.read_hdf(path_to_weight_file)
        df['weight'] = pd.Series(df_weights.weight.values, index=df.index)
        return df

    # TODO: add on the fly branching creation for uproot

    df = read_branches(path_to_input_file, 'taus', input_branches)
    if path_to_pred_file is not None:
        add_predictions(df, path_to_pred_file, pred_column_prefix)
        add_targets(df, path_to_pred_file, pred_column_prefix)
    else:
        print('[INFO] No predictions found in mlflow artifacts for this run, will proceed without them.')
    if path_to_weights is not None:
        add_weights(df, path_to_weights)
    else:
        df['weight'] = pd.Series(np.ones(df.shape[0]), index=df.index)
    return df

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
