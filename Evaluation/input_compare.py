import os
import sys
import json
import git
import yaml
import glob
from tqdm import tqdm
from shutil import rmtree

import uproot
import numpy as np
import pandas as pd

import mlflow
import hydra
from hydra.utils import to_absolute_path
from omegaconf import DictConfig, OmegaConf

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

sys.path.insert(0, "../Training/python")

epsilon = 0.00001
# epsilon = 0.000001

class FeatureDecoder:
    """A Class to compere the difference of an input tensor"""

    def __init__(self, cfg):
        path_to_artifacts = to_absolute_path(f'{cfg.path_to_mlflow}/{cfg.experiment_id}/{cfg.run_id}/artifacts/')
        train_cfg_path = f'{path_to_artifacts}/input_cfg/training_cfg.yaml'
        with open(train_cfg_path, "r") as stream:
            train_cfg = yaml.safe_load(stream)
        all_features = {}
        for key in train_cfg["Features_all"].keys():
            all_features[key] = []
            for var in train_cfg["Features_all"][key]:
                assert(len(list(var.keys()))==1)
                var_str = list(var.keys())[0]
                if not(var_str in train_cfg["Features_disable"][key]):
                    all_features[key].append(var_str)

        self.feature_map = {}
        for tensor in cfg.tensor_map.keys():
            self.feature_map[tensor] = []
            for block in cfg.tensor_map[tensor]:
                self.feature_map[tensor].extend(all_features[block])

        for elem in self.feature_map.keys():
            print("Tensor added to the map:",elem, len(self.feature_map[elem]))

    def get(self, tensor_name, index):
        return self.feature_map[tensor_name][index]

def grid(row, col, row_col_idx):
    """version with string concatenation"""
    string = '\n' + '+---'*col + '+\n'
    for r in range(1, row+1):
        for c in range(1, col+1):
            string += '| + ' if [r,c] in row_col_idx else '|   '
        string += '|'
        string += '\n' + '+---'*col + '+\n'
    return string

def compare_ids(cfg, sort=False, print_n=30, plot_deltas=True):
    path_to_artifacts = to_absolute_path(f'{cfg.path_to_mlflow}/{cfg.experiment_id}/{cfg.run_id}/artifacts/')

    prediction_path = f'{path_to_artifacts}/predictions/{cfg.sample_alias}/{cfg.input_filename}_pred.h5'
    df_default = pd.read_hdf(prediction_path, key='deeptauIDs')
    df_default [df_default < 0] = None
    df_current = pd.read_hdf(prediction_path, key='predictions')
    assert(len(df_default)==len(df_current))
    df_tar = pd.read_hdf(prediction_path, key='targets')
    df_dis = pd.concat([df_default, df_current], axis=1)

    dis_types = ["e", "mu", "jet"]
    for dis_type in dis_types:
        df_dis[f'VS{dis_type}'] = df_dis.node_tau / (df_dis[f"node_{dis_type}"] + df_dis.node_tau)
        df_dis[f'delta_VS{dis_type}'] = (df_dis[f'VS{dis_type}'] - df_dis[f'tau_byDeepTau2018v2p5VS{dis_type}raw'])
        df_dis[f'abs_delta_VS{dis_type}'] = (df_dis[f'delta_VS{dis_type}']).abs()

        # to compare tau_byDeepTau2017v2p1VS with tau_byDeepTau2017v2p5VS
        # df_dis[f'delta_VS{dis_type}'] = (df_dis[f'tau_byDeepTau2017v2p1VS{dis_type}raw'] - df_dis[f'tau_byDeepTau2017v2p1ReRunVS{dis_type}raw'])
        # df_dis[f'abs_delta_VS{dis_type}'] = (df_dis[f'delta_VS{dis_type}']).abs()

        # To make sure score is available for all taus:
        assert( np.all(np.isnan(df_dis[f'tau_byDeepTau2018v2p5VS{dis_type}raw']) == np.isnan(df_dis[f'delta_VS{dis_type}'])) )
        s = df_dis.shape[0]
        n = np.isnan(df_dis[f'tau_byDeepTau2018v2p5VS{dis_type}raw']).sum()
        print("Number of not nans:", s - n)
    
    
    df_dis["max_delta"] = df_dis[[f'abs_delta_VS{t}' for t in dis_types]].max(axis=1)

    # Display all events droping nans:
    # When Droping Nan's, new index -> should coorespond to the index of json file
    print("Top inconsistent DeepTauID scores are listed below:")
    df_dis_noNan = df_dis
    df_dis_noNan["old_indx"] = df_dis_noNan.index
    df_dis_noNan = df_dis_noNan.dropna().reset_index(drop=True)
    df_dis_noNan = df_dis_noNan.sort_values(['max_delta'], ascending=False)
    if print_n: df_dis_noNan = df_dis_noNan[:print_n]

    with pd.option_context('display.max_rows', None, 'display.max_columns', None, 'display.width', 500):  # more options can be specified also
        # print(df_dis_noNan[['old_indx','event', 'tau_idx']+[f'delta_VS{t}' for t in dis_types]])
        print(df_dis_noNan[['old_indx','event', 'tau_idx'] + [f'delta_VS{t}' for t in dis_types]
                                                           + [f'VS{t}' for t in dis_types]
                                                           + [f'tau_byDeepTau2018v2p5VS{t}raw' for t in dis_types]])

    if plot_deltas:
        img_path = 'deltaIDs'
        if not os.path.exists(img_path): os.makedirs(img_path)
        for dis_type in dis_types:
            plt.hist(df_dis[f'delta_VS{dis_type}'], density=True, bins=100)  # density=False would make counts
            plt.ylabel('arb. units')
            plt.xlabel(f'updatedVS{dis_type} - tau_byDeepTau2018v2p5VS{dis_type}raw')
            plt.savefig(f'{img_path}/delta_VS{dis_type}.pdf')
            plt.cla()
            plt.clf()
            with mlflow.start_run(experiment_id=cfg.experiment_id, run_id=cfg.run_id) as active_run:
                mlflow.log_artifact(img_path, f'predictions/{cfg.sample_alias}')

def compare_input(cfg, print_grid=False):
    assert(cfg.compare_input)
    path_to_artifacts = to_absolute_path(f'{cfg.path_to_mlflow}/{cfg.experiment_id}/{cfg.run_id}/artifacts/')
    file_cmssw_path = to_absolute_path(f'{cfg.path_to_input_dir}/{cfg.compare_input.input_cmssw}')
    files_cmssw_python = f'{path_to_artifacts}/predictions/{cfg.sample_alias}/{cfg.input_filename}_pred_input'

    with open(file_cmssw_path) as file:
        json_input = json.load(file)
        data_cmssw = {}
        for tensor_name in json_input.keys():
            data_cmssw[tensor_name] = np.array(json_input[tensor_name])

    event = cfg.compare_input.input_python.event
    index = cfg.compare_input.input_python.tau_index
    data_python = np.load(f'{files_cmssw_python}/tensor_{event}_{index}.npy',allow_pickle=True)[()]

    for key in data_cmssw.keys():
        print("Shape consistency check:",data_cmssw[key].shape, data_python[key].shape)
        assert(data_cmssw[key].shape == data_python[key].shape)

    map_f = FeatureDecoder(cfg)
    for key in list(json_input.keys()):
        print("\n--------->",key,"<---------")
        delta = np.abs(data_cmssw[key] - data_python[key])
        if key == list(json_input.keys())[0]:
            print("cmssw tau:", data_cmssw[key])
            print("python tau:", data_python[key])
            f_idx = np.where(delta > epsilon)
            print(f"Inconsistent features:\n")
            for f in np.unique(f_idx):
                print(map_f.get(key,f))
        else:
            row_idx, col_idx, f_idx = np.where(delta > epsilon)
            if row_idx.size != 0:
                print("cmssw grid:", data_cmssw[key][row_idx[0]][col_idx[0]][f_idx])
                print("python grid:", data_python[key][row_idx[0]][col_idx[0]][f_idx])
            print(f"Inconsistent features:\n")
            for f in np.unique(f_idx):
                print(map_f.get(key,f))


        if print_grid and key != list(json_input.keys())[0]: # if print & not first tensor (plane features)
            grid_idx = np.stack([row_idx, col_idx],axis=1).tolist()
            print("\nInconsistent cells:")
            print(
                grid(data_cmssw[key].shape[0],
                     data_cmssw[key].shape[1],
                     grid_idx
                    )
                )

@hydra.main(config_path='configs', config_name='input_compare')
def main(cfg: DictConfig) -> None:
    mlflow.set_tracking_uri(f"file://{to_absolute_path(cfg.path_to_mlflow)}")
    
    if 'compare_ids' in cfg.mode: compare_ids(cfg)
    if 'compare_input' in cfg.mode: compare_input(cfg)

if __name__ == '__main__':
    main()