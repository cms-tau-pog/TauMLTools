import numpy as np
import pandas as pd
import uproot
import copy
from sklearn import metrics
from scipy import interpolate
from statsmodels.stats.proportion import proportion_confint
from _ctypes import PyObj_FromPtr
import os
import h5py
import json
import re
from glob import glob
from dataclasses import dataclass
from hydra.utils import to_absolute_path
from functools import partial

@dataclass
class RocCurve:
    pr: np.array = None
    pr_err: np.array = None
    ratio: np.array = None
    thresholds: np.array = None
    auc_score: float = None

    def fill(self, cfg, create_ratio=False, ref_roc=None):
        fpr = np.array(cfg['false_positive_rate'])
        n_points = len(fpr)
        self.auc_score = cfg.get('auc_score')
        self.pr = np.empty((2, n_points))
        self.pr[0, :] = fpr
        self.pr[1, :] = cfg['true_positive_rate']

        if 'false_positive_rate_up' in cfg:
            self.pr_err = np.empty((2, 2, n_points))
            self.pr_err[0, 0, :] = cfg['false_positive_rate_up']
            self.pr_err[0, 1, :] = cfg['false_positive_rate_down']
            self.pr_err[1, 0, :] = cfg['true_positive_rate_up']
            self.pr_err[1, 1, :] = cfg['true_positive_rate_down']
        
        if 'thresholds' in cfg:
            self.thresholds = np.empty(n_points)
            self.thresholds[:] = cfg['thresholds']
        
        if 'auc_score' in cfg:
            self.auc_score = cfg['auc_score']

        if 'plot_cfg' in cfg:
            self.color = cfg['plot_cfg'].get('color', 'black')
            self.alpha = cfg['plot_cfg'].get('alpha', 1.)
            self.dots_only = cfg['plot_cfg'].get('dots_only', False)
            self.dashed = cfg['plot_cfg'].get('dashed', False)
            self.marker_size = cfg['plot_cfg'].get('marker_size', 5)
            self.with_errors = cfg['plot_cfg'].get('with_errors', True)

        if create_ratio:
            if ref_roc is None:
                ref_roc = self
            self.ratio = self.create_roc_ratio(self.pr[1], self.pr[0], ref_roc.pr[1], ref_roc.pr[0], True)
        else:
            self.ratio = None
            
    def prune(self, tpr_decimals=3):
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

    def draw(self, ax, ax_ratio = None):
        main_plot_adjusted = False
        if self.pr_err is not None and self.with_errors:
            x = self.pr[1]
            y = self.pr[0]
            entry = ax.errorbar(x, y, xerr=self.pr_err[1], yerr=self.pr_err[0], color=self.color, alpha=self.alpha,
                        fmt='o', markersize=self.marker_size, linewidth=1)
        else:
            if self.dots_only:
                entry = ax.errorbar(self.pr[1], self.pr[0], color=self.color, alpha=self.alpha, fmt='o', markersize=self.marker_size)
            else:
                fmt = '--' if self.dashed else ''
                x = self.pr[1]
                y = self.pr[0]
                if x[-1] - x[-2] > 0.01:
                    x = x[:-1]
                    y = y[:-1]
                    x_max_main = x[-1]
                    main_plot_adjusted = True
                entry = ax.errorbar(x, y, color=self.color, alpha=self.alpha, fmt=fmt)
        if self.ratio is not None and ax_ratio is not None:
            if self.pr_err is not None and self.with_errors:
                x = self.ratio[1]
                y = self.ratio[0]
                ax_ratio.errorbar(x, y, color=self.color, alpha=self.alpha, fmt='o', markersize='5', linewidth=1)
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
                ax_ratio.plot(x, y, color=self.color, alpha=self.alpha, linewidth=1, linestyle=linestyle)
        return entry

    @staticmethod
    def create_roc_ratio(x1, y1, x2, y2, wp): # wp -> if ratio curve (x2,y2) is a WP
        if not wp: # compute ratio for interpolation of both curves over their common and joint domain
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
        else: # compute ratio for interpolation of probed curve only over common domain points with ratio curve
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

### ----------------------------------------------------------------------------------------------------------------------  

@dataclass
class PlotSetup:
    
    # general
    tick_size: int = 14
    xlim: list = None 

    # y-axis params
    ylabel: str = "Mis-id probability"
    ylabel_size: int = 16
    yscale: str = 'log'
    ylim: list = None

    # ratio params
    ratio_xlabel_size: int = 16
    ratio_yscale: str = 'linear'
    ratio_ylim: list = None
    ratio_ylabel: str = 'Ratio'
    ratio_ylabel_size: int = 16
    ratio_ylabel_pad: int = 15
    ratio_tick_size: int = 12

    # legend params
    legend_loc: str = 'upper left'
    legend_fontsize: int = 14

    def apply(self, names, entries, ax, ax_ratio = None):
        if self.xlim is not None:
            ax.set_xlim(self.xlim)
        if self.ylim is not None:
            ax.set_ylim(self.ylim)

        ax.set_yscale(self.yscale)
        ax.set_ylabel(self.ylabel, fontsize=self.ylabel_size)
        ax.tick_params(labelsize=self.tick_size)
        ax.grid(True)
        lentries = []
        lnames = []
        for e,n in zip(entries, names):
          if n not in lnames:
            lentries.append(e)
            lnames.append(n)
        ax.legend(lentries, lnames, fontsize=self.legend_fontsize, loc=self.legend_loc)

        if ax_ratio is not None:
            if self.ratio_ylim is not None:
                ax_ratio.set_ylim(self.ratio_ylim)

            ax_ratio.set_yscale(self.ratio_yscale)
            ax_ratio.set_xlabel('Tau ID efficiency', fontsize=self.ratio_xlabel_size)
            ax_ratio.set_ylabel(self.ratio_ylabel, fontsize=self.ratio_ylabel_size, labelpad=self.ratio_ylabel_pad)
            ax_ratio.tick_params(labelsize=self.ratio_tick_size)
            ax_ratio.grid(True, which='both')
    
    @staticmethod
    def get_pt_text(pt_min, pt_max):
        if pt_max == 1000:
            pt_text = r'$p_T > {}$ GeV'.format(pt_min)
        elif pt_min == 20:
            pt_text = r'$p_T < {}$ GeV'.format(pt_max)
        else:
            pt_text = r'$p_T\in ({}, {})$ GeV'.format(pt_min, pt_max)
        
        return pt_text

    @staticmethod
    def get_eta_text(eta_min, eta_max):
        eta_text = r'${} < |\eta| < {}$'.format(eta_min, eta_max)
        return eta_text
    
    @staticmethod
    def get_dm_text(dm_bin):
        if len(dm_bin)==1:
            dm_text = r'DM$ = {}$'.format(dm_bin[0])
        else:
            dm_text = r'DM$ \in {}$'.format(dm_bin)
        return dm_text

    def add_text(self, ax, n_entries, pt_min, pt_max, eta_min, eta_max, dm_bin, period):
        header_y = 1.02
        ax.text(0.03, 0.89 - n_entries*0.07, self.get_pt_text(pt_min, pt_max), fontsize=14, transform=ax.transAxes)
        ax.text(0.03, 0.82 - n_entries*0.07, self.get_eta_text(eta_min, eta_max), fontsize=14, transform=ax.transAxes)
        ax.text(0.03, 0.75 - n_entries*0.07, self.get_dm_text(dm_bin), fontsize=14, transform=ax.transAxes)
        ax.text(0.01, header_y, 'CMS', fontsize=14, transform=ax.transAxes, fontweight='bold', fontfamily='sans-serif')
        ax.text(0.12, header_y, 'Simulation Preliminary', fontsize=14, transform=ax.transAxes, fontstyle='italic',
                fontfamily='sans-serif')
        ax.text(0.73, header_y, period, fontsize=13, transform=ax.transAxes, fontweight='bold',
                fontfamily='sans-serif')

### ----------------------------------------------------------------------------------------------------------------------  

@dataclass
class Discriminator:
    name: str
    pred_column: str
    raw: bool
    wp_from: str = None
    wp_column: str = None
    wp_name_to_index: dict = None
    wp_thresholds: dict = None 

    def __post_init__(self):
        if self.wp_from is None:
            self.wp_thresholds = {}
        else: # create list of WP names from either of provided WP dictionaries
            if self.wp_name_to_index is not None and self.wp_thresholds is not None:
                assert set(self.wp_name_to_index.keys()) == set(self.wp_thresholds.keys())
            if self.wp_name_to_index is not None:
                self.wp_names = list(self.wp_name_to_index.keys())
            elif self.wp_thresholds is not None:
                self.wp_names = list(self.wp_thresholds.keys())
            else:
                raise RuntimeError(f"For wp_from={self.wp_from} either wp_name_to_index or wp_thresholds should be specified, but both are None.")

    def count_passed(self, df, wp_name):
        if self.wp_from == 'wp_column':
            assert self.wp_column in df.columns
            wp = self.wp_name_to_index[wp_name]
            flag = 1 << wp
            passed = (np.bitwise_and(df[self.wp_column], flag) != 0).astype(int)
            return np.sum(passed * df.weight.values)
        elif self.wp_from == 'pred_column':
            if len(self.wp_thresholds) > 0:
                assert self.pred_column in df.columns
                wp_thr = self.wp_thresholds[wp_name]
                return np.sum(df[df[self.pred_column] > wp_thr].weight.values)
            else:
                raise RuntimeError('Working points thresholds are not specified for discriminator "{}"'.format(self.name))
        else:
            raise RuntimeError(f'count_passed() behaviour not defined for: wp_from={self.wp_from}')
        
    def create_roc_curve(self, df):
        roc, wp_roc = None, None
        if self.raw: # construct ROC curve
            fpr, tpr, thresholds = metrics.roc_curve(df['gen_tau'].values, df[self.pred_column].values, sample_weight=df.weight.values)
            if not (np.isnan(fpr).any() or np.isnan(tpr).any()):
                auc_score = metrics.roc_auc_score(df['gen_tau'].values, df[self.pred_column].values, sample_weight=df.weight.values)
                roc = RocCurve()
                roc_cfg = {
                    'false_positive_rate': fpr,
                    'true_positive_rate': tpr,
                    'thresholds': thresholds,
                    'auc_score': auc_score,
                }
                roc.fill(roc_cfg, create_ratio=False, ref_roc=None)
            else:
                print('[INFO] ROC curve is empty!')
                return None, None
        else:
            print('[INFO] raw=False, will skip creating ROC curve')        
        
        # construct WPs
        if self.wp_from in ['wp_column', 'pred_column']:  
            if (n_wp:=len(self.wp_names)) > 0:
                wp_roc = RocCurve()
                wp_roc_cfg = {
                    'false_positive_rate': np.empty(n_wp),
                    'true_positive_rate': np.empty(n_wp),
                    'false_positive_rate_up': np.empty(n_wp),
                    'true_positive_rate_up': np.empty(n_wp),
                    'false_positive_rate_down': np.empty(n_wp),
                    'true_positive_rate_down': np.empty(n_wp),
                }
                for wp_i, wp_name in enumerate(self.wp_names):
                    for kind, pr_name in zip([0, 1], ['false_positive_rate', 'true_positive_rate']):
                        # central values
                        df_x = df[df['gen_tau'] == kind]
                        n_passed = self.count_passed(df_x, wp_name)
                        n_total = np.sum(df_x.weight.values)
                        eff = float(n_passed) / n_total if n_total > 0 else 0.0
                        wp_roc_cfg[pr_name][n_wp - wp_i - 1] = eff
                        
                        # up/down variations
                        ci_low, ci_upp = proportion_confint(n_passed, n_total, alpha=1-0.68, method='beta')
                        wp_roc_cfg[f'{pr_name}_down'][n_wp - wp_i - 1] = ci_upp - eff
                        wp_roc_cfg[f'{pr_name}_up'][n_wp - wp_i - 1] = eff - ci_low
                        
                wp_roc.fill(wp_roc_cfg, create_ratio=False, ref_roc=None)
            else:
                raise RuntimeError('No working points specified')
        elif self.wp_from is None:
            print('[INFO] wp_from=None, will skip creating WP')
        else:
            raise RuntimeError(f'create_roc_curve() behaviour not defined for: wp_from={self.wp_from}')
        return roc, wp_roc

### ----------------------------------------------------------------------------------------------------------------------  

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
    def find_common_suffix(l):
        if not all([isinstance(s, str) for s in l]):
            raise TypeError("Iterable for finding common suffix doesn't contain all strings")
        l_inverse = [s[::-1] for s in l]
        suffix = os.path.commonprefix(l_inverse)[::-1]
        return suffix
    def path_splitter(path, common_suffix):
        basename = os.path.splitext(os.path.basename(path))[0]
        if basename.endswith(common_suffix):
            return basename[:-len(common_suffix)]
        else:
            return basename
        
    # prepare list of files with inputs
    # if path_to_input is not None:
    #     path_to_input = os.path.abspath(to_absolute_path(fill_placeholders(path_to_input, {"{sample_alias}": sample_alias})))
    #     input_common_suffix = find_common_suffix(glob(path_to_input))
    #     input_files = sorted(glob(path_to_input), key=partial(path_splitter, common_suffix=input_common_suffix))
    # else:
    #     input_files = []
    
    # prepare list of files with target labels
    if path_to_target is not None:
        path_to_target = os.path.abspath(to_absolute_path(fill_placeholders(path_to_target, {"{sample_alias}": sample_alias})))
        if f'artifacts/predictions/{sample_alias}' in path_to_target: # fetch corresponding input files from mlflow logs
            #json_filemap_name = f'{path_to_artifacts}/predictions/{sample_alias}/pred_input_filemap.json'
            json_filemap_name = path_to_target.replace(path_to_target.split("/")[-1], 'pred_input_filemap.json')
            if os.path.exists(json_filemap_name):
                with open(json_filemap_name, 'r') as json_file:
                    target_input_map = json.load(json_file)
                    # target_common_suffix = find_common_suffix(target_input_map.keys())
                    # target_files, input_files = zip(*sorted(target_input_map.items(), key=lambda item: partial(path_splitter, common_suffix=target_common_suffix)(item[0])))  # sort by values (target files)
                    target_files = glob(path_to_target)
                    input_files = [target_input_map[file] for file in target_files]
            else:
                raise FileNotFoundError(f'File {json_filemap_name} does not exist. Please make sure that input<->target file mapping is stored in mlflow run artifacts.')
        else: # use paths from cfg 
            raise FileNotFoundError(f'Target files are not in the mlflow run artifacts.')
            # target_common_suffix = find_common_suffix(glob(path_to_target))
            # target_files = sorted(glob(path_to_target), key=partial(path_splitter, common_suffix=target_common_suffix)) 
            # if len(target_files) != len(input_files):
            #     raise Exception(f'Number of input files ({len(input_files)}) not equal to number of target files with labels ({len(target_files)})')
    else: # will assume that target branches "gen_*" are present in input files
        raise FileNotFoundError(f'Target is not provided. With last modification target is required')
        # assert len(input_files)>0
        # target_files = [None]*len(input_files)

    # prepare list of files with inputs/predictions
    if path_to_pred is not None:
        path_to_pred = os.path.abspath(to_absolute_path(fill_placeholders(path_to_pred, {"{sample_alias}": sample_alias})))
        # pred_common_suffix = find_common_suffix(glob(path_to_pred))
        # pred_files = sorted(glob(path_to_pred), key=partial(path_splitter, common_suffix=pred_common_suffix))
        if f'artifacts/predictions/{sample_alias}' in path_to_pred and path_to_pred==path_to_target:
            pred_files = target_files
        else:
            raise FileNotFoundError('path to target and prediction should be the same')   
        if len(pred_files) != len(input_files):
            raise Exception(f'Number of input files ({len(input_files)}) not equal to number of prediction files with labels ({len(pred_files)})')
    else: # will assume that predictions are present in input files
        assert len(input_files)>0
        pred_files = [None]*len(input_files)
    
    return input_files, pred_files, target_files

def fill_placeholders(string, placeholder_to_value):
    for placeholder, value in placeholder_to_value.items():
        string = string.replace(placeholder, str(value))
    return string

### ----------------------------------------------------------------------------------------------------------------------  

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
