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

sys.path.insert(0, "../Training/python")

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


def compare_ids(cfg, sort=False, print_n=10):
    path_to_artifacts = to_absolute_path(f'{cfg.path_to_mlflow}/{cfg.experiment_id}/{cfg.run_id}/artifacts/')

    prediction_path = f'{path_to_artifacts}/predictions/{cfg.sample_alias}/{cfg.input_filename}_pred.h5'
    df_default = pd.read_hdf(prediction_path, key='deeptauIDs')
    df_default [df_default < 0] = None
    df_current = pd.read_hdf(prediction_path, key='predictions')
    assert(len(df_default)==len(df_current))
    df_dis = pd.concat([df_default, df_current], axis=1)

    dis_types = ["e", "mu", "jet"]
    for dis_type in dis_types:
        df_dis[f'VS{dis_type}'] = df_dis.node_tau / (df_dis[f"node_{dis_type}"] + df_dis.node_tau)
        df_dis[f'delta_VS{dis_type}'] = (df_dis[f'VS{dis_type}'] - df_dis[f'tau_byDeepTau2017v2p5VS{dis_type}raw']).abs()
    df_dis["max_delta"] = df_dis[[f'delta_VS{t}' for t in dis_types]].max(axis=1)

    # check that for all taus there are non-nan output

    sort_df = df_dis.sort_values(['max_delta'], ascending=False)
    if print_n: sort_df = sort_df[:print_n]

    print(sort_df[['event', 'tau_idx']+[f'delta_VS{t}' for t in dis_types]])

def compare_input(cfg):
    assert(cfg.compare_input)
    path_to_artifacts = to_absolute_path(f'{cfg.path_to_mlflow}/{cfg.experiment_id}/{cfg.run_id}/artifacts/')
    file_cmssw_path = to_absolute_path(f'{cfg.path_to_input_dir}/{cfg.compare_input.input_cmssw}')
    files_cmssw_python = f'{path_to_artifacts}/predictions/{cfg.sample_alias}/{cfg.input_filename}_pred_input'

    # file_python_path = f'{output_filename}_input/tensor_{evnt}_{idx}.npy'
    # file_python_path = f'{path_to_artifacts}/prediction/{cfg.sample_alias}/{cfg.}_pred_input{/_input/tensor_{evnt}_{idx}.npy'

    with open(file_cmssw_path) as file:
        json_input = json.load(file)
        data_cmssw = {}
        for tensor_name in json_input.keys():
            data_cmssw[tensor_name] = np.array(json_input[tensor_name])

    event = cfg.compare_input.input_python.event
    index = cfg.compare_input.input_python.tau_index
    data_python = np.load(f'{files_cmssw_python}/tensor_{event}_{index}.npy',allow_pickle=True)[()]

    for key in data_cmssw.keys():
        print(data_cmssw[key].shape, data_python[key].shape) 

    print(data_cmssw["input_tau"], data_python["input_tau"])
    # print(data_python)
    
    map_f = FeatureDecoder(cfg)
    

@hydra.main(config_path='.', config_name='input_compare')
def main(cfg: DictConfig) -> None:
    print(cfg)
    mlflow.set_tracking_uri(f"file://{to_absolute_path(cfg.path_to_mlflow)}")
    
    # file = to_absolute_path(f'{cfg.path_to_input_dir}/{cfg.input_filename}.root')
    # prediction_output = f'{path_to_artifacts}/{cfg.sample_alias}/{cfg.input_filename}_pred.h5'
    # input_grids = f'{path_to_artifacts}/{cfg.sample_alias}/{cfg.input_filename}_pred.h5'

    if 'compare_ids' in cfg.mode:
        compare_ids(cfg)

    if 'compare_input' in cfg.mode:
        compare_input(cfg)

if __name__ == '__main__':
    main()