import os
import json 
from glob import glob 
from omegaconf import ListConfig
from hydra.utils import to_absolute_path

import numpy as np
import pandas as pd
import uproot

def tau_vs_other(prob_tau, prob_other):
    return np.where(prob_tau > 0, prob_tau / (prob_tau + prob_other), np.zeros(prob_tau.shape))

def create_df(path_to_preds, pred_samples, tau_type_to_select, selection, 
              pred_group_name, pred_column_prefix, target_group_name, target_column_prefix, 
              add_columns_from, add_columns, group_or_tree_name, 
              **kwargs):
    df = []
    path_to_preds = os.path.abspath(to_absolute_path(path_to_preds))

    # loop over input samples
    for sample_name, sample_cfg in pred_samples.items():

        # make list of files with predictions depending on the specified format
        filename_pattern = sample_cfg['filename_pattern']
        if isinstance(filename_pattern, ListConfig): 
            pred_files = [f'{path_to_preds}/{sample_name}/{filename}' for filename in filename_pattern]
        elif isinstance(filename_pattern, str):
            pred_files = glob(f'{path_to_preds}/{sample_name}/{filename_pattern}')
        else:
            raise Exception(f"unknown type of filename_pattern: {type(filename_pattern)}")
        
        df_sample = []
        for pred_file in pred_files:
            # read predictions and labels
            l_ = []
            for group in [pred_group_name, target_group_name]:
                df_ = pd.read_hdf(pred_file, group)
                df_ = df_.rename(columns={column: f'{group}_{column}' for column in df_.columns})
                l_.append(df_)
            assert l_[0].shape[0] == l_[1].shape[0], "Sizes of prediction and target dataframes don't match."
            df_pred = pd.concat(l_, axis=1)

            if add_columns_from == 'inputs':# read add_columns from the corresponding input file
                json_filemap_name = f'{path_to_preds}/{sample_name}/pred_input_filemap.json'
                with open(json_filemap_name, 'r') as json_file:
                    target_input_map = json.load(json_file) 

                with uproot.open(target_input_map[pred_file]) as f:
                    df_add = f[group_or_tree_name].arrays(add_columns, library='pd')
                
                # concatenate input branches and predictions/labels
                assert df_pred.shape[0] == df_add.shape[0], "Sizes of prediction and input dataframes don't match."
            elif add_columns_from == 'predictions':
                df_add = pd.read_hdf(pred_file, group_or_tree_name)
                df_add = df_add[add_columns]
            else:
                raise RuntimeError(f'add_columns_from should be either predictions or inputs, got {add_columns_from}')

            df_sample_ = pd.concat([df_pred, df_add], axis=1)
            assert not any(df_sample_.isna().any(axis=0)), 'found NaNs!'
            df_sample.append(df_sample_)
        
        # concat together files for a given sample and apply tau_type matching & selection
        df_sample = pd.concat(df_sample, axis=0)
        df_sample = df_sample.query(f'{target_group_name}_{target_column_prefix}{tau_type_to_select}==1')
        if selection is not None:
            df_sample = df_sample.query(selection)

        # compute weight variable
        df_sample['weight'] = 1.
        if sample_cfg['reweight_to_lumi'] is not None:
            df_sample['weight'] *= sample_cfg['sample_lumi'] / (sample_cfg['reweight_to_lumi'] * df_sample.shape[0])
        
        df.append(df_sample)

    # concat across samples
    df = pd.concat(df, axis=0)

    # compute vs_type discriminator scores
    vs_types = ['e', 'mu', 'jet']
    for vs_type in vs_types:
        df['score_vs_' + vs_type] = tau_vs_other(df[f'{pred_group_name}_{pred_column_prefix}tau'].values, df[f'{pred_group_name}_{pred_column_prefix}' + vs_type].values)
   
    print(f'\n-> Selected {df.shape[0]} {tau_type_to_select}s')
    return df
