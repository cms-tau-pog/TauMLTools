import os
import sys
import json
import git
from tqdm import tqdm

import uproot
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model

import mlflow
import hydra
from hydra.utils import to_absolute_path
from omegaconf import DictConfig, OmegaConf
import glob

sys.path.insert(0, "../Training/python")
from common import setup_gpu
from commonReco import *

@hydra.main(config_path='.', config_name='apply_training')
def main(cfg: DictConfig) -> None:
    # set up paths & gpu
    mlflow.set_tracking_uri(f"file://{to_absolute_path(cfg.path_to_mlflow)}")
    path_to_artifacts = to_absolute_path(f'{cfg.path_to_mlflow}/{cfg.experiment_id}/{cfg.run_id}/artifacts/')

    path_to_files = to_absolute_path(f'{cfg.path_to_mlflow}/{cfg.experiment_id}/{cfg.run_id}/artifacts/predictions/{cfg.sample_alias}')

    predictions = []
    targets = []
    for filename in glob.glob(f'{path_to_files}/*.h5'):
        predictions.append(pd.read_hdf(filename,key='predictions'))
        targets.append(pd.read_hdf(filename,key='targets'))
    df_predictions = pd.concat(predictions)
    df_targets = pd.concat(targets)

    d_pt = ((df_predictions['pt'] - df_targets['pt']) / df_targets['pt']) [df_targets['pt'] > -600]
    pt_tar = df_targets['pt'][df_targets['pt'] > -600]
    pt_pred = df_predictions['pt'][df_targets['pt'] > -600]

    d_m = (np.sqrt(df_predictions['m2']) - np.sqrt(df_targets['m2'])) [df_targets['pt'] > -600]
    m_tar = np.sqrt(df_targets['m2'][df_targets['pt'] > -600])
    m_pred = np.sqrt(df_predictions['m2'][df_targets['pt'] > -600])

    import matplotlib.pyplot as plt

    # relative difference in pt
    fig, axs = plt.subplots(nrows=2, ncols=2, figsize = (20,10))
    fig.suptitle('relative pt difference')

    axs[0][0].hist(pt_tar, bins=100, range=(0, 1000), label="All pt bins")
    axs[0][0].set_xlabel('pt')
    axs[0][0].set_ylabel('arb. units')
    axs[0][1].hist(pt_pred, bins=100, range=(0, 1000), label="All pt bins")
    axs[0][1].set_xlabel('pt_pred')
    axs[0][1].set_ylabel('arb. units')
    axs[1][0].hist(d_pt, bins=100, range=(-2.5, 2.5), label="All pt bins")
    axs[1][0].set_xlabel('(pt_pred - pt) / pt ')
    axs[1][0].set_ylabel('arb. units')

    plt.savefig('pt_delta.png')

    fig, axs = plt.subplots(nrows=2, ncols=2, figsize = (20,10))
    fig.suptitle('mass difference')

    axs[0][0].hist(m_tar, bins=200, range=(0, 5), label="All pt bins")
    axs[0][0].set_xlabel('mass')
    axs[0][0].set_ylabel('arb. units')
    axs[0][1].hist(m_pred, bins=200, range=(0, 5), label="All pt bins")
    axs[0][1].set_xlabel('mass_pred')
    axs[0][1].set_ylabel('arb. units')
    axs[1][0].hist(d_m, bins=200, range=(-5, 5), label="All pt bins")
    axs[1][0].set_xlabel('mass difference')
    axs[1][0].set_ylabel('arb. units')

    plt.savefig('mass_delta.png')

if __name__ == '__main__':
    main()
