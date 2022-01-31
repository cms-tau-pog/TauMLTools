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
from glob import glob
from dataclasses import dataclass, field
from hydra.utils import to_absolute_path

if sys.version_info.major > 2:
    from statsmodels.stats.proportion import proportion_confint

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
    color: str
    dashed: bool = False 
    wp_from: str = None
    wp_column: str = None
    wp_name_to_index: dict = None
    working_points: list = field(default_factory=list)
    working_points_thrs: dict = None 

    def __post_init__(self):
        if self.wp_from is None:
            self.working_points = []

    def count_passed(self, df, wp_name):
        if self.wp_from == 'wp_column':
            assert self.wp_column in df.columns
            wp = self.wp_name_to_index[wp_name]
            flag = 1 << wp
            passed = (np.bitwise_and(df[self.wp_column], flag) != 0).astype(int)
            return np.sum(passed * df.weight.values)
        elif self.wp_from == 'pred_column':
            if self.working_points_thrs is not None:
                assert self.pred_column in df.columns
                wp_thr = self.working_points_thrs[wp_name]
                return np.sum(df[df[self.pred_column] > wp_thr].weight.values)
            else:
                raise RuntimeError('Working points thresholds are not specified for discriminator "{}"'.format(self.name))
        else:
            raise RuntimeError(f'count_passed() behaviour not defined for: wp_from={self.wp_from}')
        
    def create_roc_curve(self, df):
        roc, wp_roc = None, None
        if self.raw: # construct ROC curve
            fpr, tpr, thresholds = metrics.roc_curve(df['gen_tau'].values, df[self.pred_column].values, sample_weight=df.weight.values)
            roc = RocCurve(len(fpr), self.color, False, dashed=self.dashed)
            roc.pr[0, :] = fpr
            roc.pr[1, :] = tpr
            roc.thresholds = thresholds
            roc.auc_score = metrics.roc_auc_score(df['gen_tau'].values, df[self.pred_column].values, sample_weight=df.weight.values)
        else:
            print('[INFO] raw=False, will skip creating ROC curve')        
        
        # construct WPs
        if self.wp_from in ['wp_column', 'pred_column']:  
            if (n_wp:=len(self.working_points)) > 0:
                wp_roc = RocCurve(n_wp, self.color, not self.raw, self.raw)
                for wp_i, wp_name in enumerate(self.working_points):
                    for kind in [0, 1]:
                        df_x = df[df['gen_tau'] == kind]
                        n_passed = self.count_passed(df_x, wp_name)
                        n_total = np.sum(df_x.weight.values)
                        eff = float(n_passed) / n_total
                        wp_roc.pr[kind, n_wp - wp_i - 1] = eff
                        if not self.raw:
                            if sys.version_info.major > 2:
                                ci_low, ci_upp = proportion_confint(n_passed, n_total, alpha=1-0.68, method='beta')
                            else:
                                err = math.sqrt(eff * (1 - eff) / n_total)
                                ci_low, ci_upp = eff - err, eff + err
                            wp_roc.pr_err[kind, 1, n_wp - wp_i - 1] = ci_upp - eff
                            wp_roc.pr_err[kind, 0, n_wp - wp_i - 1] = eff - ci_low
            else:
                raise RuntimeError('No working points specified')
        elif self.wp_from is None:
            print('[INFO] wp_from=None, will skip creating WP')
        else:
            raise RuntimeError(f'create_roc_curve() behaviour not defined for: wp_from={self.wp_from}')
        return roc, wp_roc

def create_roc_ratio(x1, y1, x2, y2, wp):
    if not wp:
      sp1 = interpolate.interp1d(x1, y1)
      sp2 = interpolate.interp1d(x2, y2)
      x_comb = np.unique(np.sort(np.concatenate((x1, x2))))
      x_sub = x_comb[np.all([ x_comb >= max(x1[0], x2[0]) , x_comb <= min(x1[-1], x2[-1]) ], axis=0)]
      y1_upd = sp1(x_sub)
      y2_upd = sp2(x_sub)
      y1_upd_clean = y1_upd[y2_upd > 0]
      y2_upd_clean = y2_upd[y2_upd > 0]
      x_clean = x_sub[y2_upd > 0]
      ratio = np.empty((2, x_clean.shape[0]))
      ratio[0, :] = y1_upd_clean / y2_upd_clean
      ratio[1, :] = x_clean
    else:
      sp = interpolate.interp1d(x1, y1)
      x2_sub = x2[np.all([ x2 >= max(x1[0], x2[0]) , x2 <= min(x1[-1], x2[-1]) ], axis=0)]
      y2_sub = y2[np.all([ x2 >= max(x1[0], x2[0]) , x2 <= min(x1[-1], x2[-1]) ], axis=0)]
      y1_upd = sp(x2_sub)
      y1_upd_clean = y1_upd[y2_sub > 0]
      x2_clean = x2_sub[y2_sub > 0]
      y2_clean = y2_sub[y2_sub > 0]
      ratio = np.empty((2, x2_clean.shape[0]))
      ratio[0, :] = y1_upd_clean / y2_clean
      ratio[1, :] = x2_clean

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

def create_df(path_to_input_file, input_branches, path_to_pred_file, path_to_target_file, path_to_weights, pred_column_prefix=None, target_column_prefix=None):
    def read_branches(path_to_file, tree_name, branches):
        if not os.path.exists(path_to_input_file):
            raise RuntimeError(f"Specified file for inputs ({path_to_input_file}) does not exist")
        if path_to_file.endswith('.root'):
            with uproot.open(path_to_file) as f:
                tree = f[tree_name]
                df = tree.arrays(branches, library='pd')
            return df
        elif path_to_file.endswith('.h5') or path_to_file.endswith('.hdf5'):
            return pd.read_hdf(path_to_file, tree_name, columns=branches)
        raise RuntimeError("Unsupported file type.")

    def add_group(df, group_name, path_to_file, group_column_prefix):
        if not os.path.exists(path_to_file):
            raise RuntimeError(f"Specified file {path_to_file} for {group_name} does not exist")
        with h5py.File(path_to_file, 'r') as f:
            file_keys = list(f.keys())
        if group_name in file_keys: 
            group_df = pd.read_hdf(path_to_file, group_name)
        else:
            group_df = pd.read_hdf(path_to_file)
        
        # weight case
        if group_name == 'weights':
            group_df = pd.read_hdf(path_to_file)
            df['weight'] = pd.Series(group_df['weight'].values, index=df.index)
            return df
        elif group_name == 'predictions': 
            prob_tau = group_df[f'{group_column_prefix}tau'].values
        elif group_name != 'targets':
            raise ValueError(f'group_name should be one of [predictions, targets, weights], got {group_name}')
        
        # add columns for predictions/targets case
        for node_column in group_df.columns:
            if not node_column.startswith(group_column_prefix): continue # assume prediction column name to be "{group_column_prefix}{tau_type}"
            tau_type = node_column.split(f'{group_column_prefix}')[-1] 
            if group_name == 'predictions':
                if tau_type != 'tau':
                    prob_vs_type = group_df[group_column_prefix + tau_type].values
                    tau_vs_other_type = np.where(prob_tau > 0, prob_tau / (prob_tau + prob_vs_type), np.zeros(prob_tau.shape))
                    df[group_column_prefix + tau_type] = pd.Series(tau_vs_other_type, index=df.index)
            elif group_name == 'targets':
                df[f'gen_{tau_type}'] = group_df[node_column]
        return df

    # TODO: add on the fly branching creation for uproot
    df = read_branches(path_to_input_file, 'taus', input_branches)
    if path_to_pred_file is not None:
        add_group(df, 'predictions', path_to_pred_file, pred_column_prefix)
    else:
        print(f'[INFO] path_to_pred_file=None, will proceed without reading predictions from there')
    if path_to_target_file is not None:
        add_group(df, 'targets', path_to_target_file, target_column_prefix)
    else:
        print(f'[INFO] path_to_target_file=None, will proceed without reading targets from there')
    if path_to_weights is not None:
        add_group(df, 'weights', path_to_weights, None)
    else:
        df['weight'] = pd.Series(np.ones(df.shape[0]), index=df.index)
    return df

def prepare_filelists(sample_alias, path_to_input, path_to_pred, path_to_target, path_to_artifacts):
    def path_splitter(path):
        basename = os.path.splitext(os.path.basename(path))[0]
        if basename.endswith('_pred'):
            i = basename.split('_')[-2]
        else:
            i = basename.split('_')[-1]
        return int(i)
        
    # prepare list of files with inputs
    if path_to_input is not None:
        path_to_input = os.path.abspath(to_absolute_path(fill_placeholders(path_to_input, {"{sample_alias}": sample_alias})))
        input_files = sorted(glob(path_to_input), key=path_splitter)
    else:
        input_files = []

    # prepare list of files with inputs/predictions
    if path_to_pred is not None:
        path_to_pred = os.path.abspath(to_absolute_path(fill_placeholders(path_to_pred, {"{sample_alias}": sample_alias})))
        if f'artifacts/predictions/{sample_alias}' in path_to_pred: # fetch corresponding input files from mlflow logs
            json_filemap_name = f'{path_to_artifacts}/predictions/{sample_alias}/pred_input_filemap.json'
            if os.path.exists(json_filemap_name):
                with open(json_filemap_name, 'r') as json_file:
                    pred_input_map = json.load(json_file)
                    pred_files, input_files = zip(*sorted(pred_input_map.items(), key=lambda item: path_splitter(item[1])))  # sort by values (input files)
            else:
                raise FileNotFoundError(f'File {json_filemap_name} does not exist. Please make sure that input<->pred file mapping is stored in mlflow run artifacts.')
        else: # use paths from cfg 
            pred_files = sorted(glob(path_to_pred), key=path_splitter)
            if len(pred_files) != len(input_files):
                raise Exception(f'Number of input files ({len(input_files)}) not equal to number of files with predictions ({len(pred_files)})')
    else: # will assume that predictions are present in input files
        assert len(input_files)>0
        pred_files = [None]*len(input_files)
    
    # prepare list of files with target labels
    if path_to_target is not None:
        path_to_target = os.path.abspath(to_absolute_path(fill_placeholders(path_to_target, {"{sample_alias}": sample_alias})))
        target_files = sorted(glob(path_to_target), key=path_splitter) 
        if len(target_files) != len(input_files):
            raise Exception(f'Number of input files ({len(input_files)}) not equal to number of files with labels ({len(target_files)})')
    else: # will assume that target branches "gen_*" are present in input files
        assert len(input_files)>0
        target_files = [None]*len(input_files)
        
    return input_files, pred_files, target_files

def fill_placeholders(string, placeholder_to_value):
    for placeholder, value in placeholder_to_value.items():
        string = string.replace(placeholder, str(value))
    return string

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
