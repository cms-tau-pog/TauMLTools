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

sys.path.insert(0, "../Training/python")
from common import setup_gpu
from commonReco import *

@hydra.main(config_path='.', config_name='apply_training')
def main(cfg: DictConfig) -> None:
    # set up paths & gpu
    mlflow.set_tracking_uri(f"file://{to_absolute_path(cfg.path_to_mlflow)}")
    path_to_artifacts = to_absolute_path(f'{cfg.path_to_mlflow}/{cfg.experiment_id}/{cfg.run_id}/artifacts/')
    if cfg.gpu_cfg is not None:
        setup_gpu(cfg.gpu_cfg)
    else:
        os.environ["CUDA_VISIBLE_DEVICES"]="-1"

    # load the model
    with open(to_absolute_path(f'{path_to_artifacts}/input_cfg/metric_names.json')) as f:
        metric_names = json.load(f)
    path_to_model = f'{path_to_artifacts}/model'
    # model = load_model(path_to_model, {name: lambda _: None for name in metric_names.keys()}) # workaround to load the model without loading metric functions
    # 
    CustomMSE.mode = "p4"
    custom_objects = {
        "CustomMSE": CustomMSE,
        "pt_res" : pt_res,
        "m2_res" : m2_res,
        "my_mse_pt"   : my_mse_pt,
        "my_mse_mass" : my_mse_mass,
        "pt_res_rel" : pt_res_rel,
    }
    model = load_model(path_to_model, custom_objects=custom_objects, compile=True)


    # load baseline training cfg and update it with parsed arguments
    training_cfg = OmegaConf.load(to_absolute_path(cfg.path_to_training_cfg))
    if cfg.training_cfg_upd is not None:
        training_cfg = OmegaConf.merge(training_cfg, cfg.training_cfg_upd)
    training_cfg = OmegaConf.to_object(training_cfg)

    if cfg.checkout_train_repo: # fetch historic git commit used to run training
        with mlflow.start_run(experiment_id=cfg.experiment_id, run_id=cfg.run_id) as active_run:
            train_git_commit = active_run.data.params.get('git_commit')

        # stash local changes and checkout 
        if train_git_commit is not None:
            repo = git.Repo(to_absolute_path('.'), search_parent_directories=True)
            if cfg.verbose: print(f'\n--> Stashing local changes and checking out training commit: {train_git_commit}\n')
            repo.git.stash('save', 'stored_stash')
            repo.git.checkout(train_git_commit)
        else:
            if cfg.verbose: print('\n--> Didn\'t find git commit hash in run artifacts, continuing with current repo state\n')

    # instantiate DataLoader and get generator
    import DataLoaderReco
    scaling_cfg  = to_absolute_path(cfg.scaling_cfg)
    dataloader = DataLoaderReco.DataLoader(training_cfg, scaling_cfg)
    gen_predict = dataloader.get_predict_generator()
    # tau_types_names = training_cfg['Setup']['tau_types_names']
       
    # open input file
    input_file_name = to_absolute_path(cfg.path_to_file)
    output_file_name = os.path.splitext(os.path.basename(input_file_name))[0] + '_pred'
    with uproot.open(input_file_name) as f:
        t = f['taus']
        n_taus = len(t['evt'].array())

    # run predictions
    predictions = []
    targets = []
    if cfg.verbose: print(f'\n\n--> Processing file {input_file_name}, number of taus: {n_taus}\n')
    for X,y in tqdm(gen_predict(input_file_name), total=n_taus/training_cfg['Setup']['n_tau']):
        # print(model.predict(X)[0], y[0])
        predictions.append(model.predict(X))
        targets.append(y)
    
    # concat and check for validity
    predictions = np.concatenate(predictions, axis=0)
    targets = np.concatenate(targets, axis=0)
    print(predictions, targets)

    if np.any(np.isnan(predictions)):
        raise RuntimeError("NaN in predictions. Total count = {} out of {}".format(
                            np.count_nonzero(np.isnan(predictions)), predictions.shape))
    # # if np.any(predictions < 0) or np.any(predictions > 1):
    # #     raise RuntimeError("Predictions outside [0, 1] range.")

    # # store into intermediate hdf5 file
    # predictions = pd.DataFrame({f'node_{tau_type}': predictions[:, int(idx)] for idx, tau_type in tau_types_names.items()})
    # targets = pd.DataFrame({f'node_{tau_type}': targets[:, int(idx)] for idx, tau_type in tau_types_names.items()}, dtype=np.int64)

    predictions = pd.DataFrame({f'{name}': predictions[:, i] for name, i in [("pt",0),("m2",1)] })
    targets = pd.DataFrame({f'{name}': targets[:, i] for name, i in [("pt",0),("m2",1)] })

    predictions.to_hdf(f'{output_file_name}.h5', key='predictions', mode='w', format='fixed', complevel=1, complib='zlib')
    targets.to_hdf(f'{output_file_name}.h5', key='targets', mode='r+', format='fixed', complevel=1, complib='zlib')
    
    # log to mlflow and delete intermediate file
    with mlflow.start_run(experiment_id=cfg.experiment_id, run_id=cfg.run_id) as active_run:
        mlflow.log_artifact(f'{output_file_name}.h5', f'predictions/{cfg.sample_alias}')
    os.remove(f'{output_file_name}.h5')

    # log mapping between prediction file and corresponding input file 
    json_filemap_name = f'{path_to_artifacts}/predictions/{cfg.sample_alias}/pred_input_filemap.json'
    json_filemap_exists = os.path.exists(json_filemap_name)
    json_open_mode = 'r+' if json_filemap_exists else 'w'
    with open(json_filemap_name, json_open_mode) as json_file:
        if json_filemap_exists: # read performance data to append additional info 
            filemap_data = json.load(json_file)
        else: # create dictionary to fill with data
            filemap_data = {}
        filemap_data[os.path.abspath(f'{path_to_artifacts}/predictions/{cfg.sample_alias}/{output_file_name}.h5')] = input_file_name
        json_file.seek(0) 
        json_file.write(json.dumps(filemap_data, indent=4))
        json_file.truncate()

if __name__ == '__main__':

    main()

    # repo = git.Repo(to_absolute_path('.'), search_parent_directories=True)
    # current_git_branch = repo.active_branch.name
    # try:
    #     main()  
    # except Exception as e:
    #     print(e)
    # finally:
    #     if 'stored_stash' in repo.git.stash('list'):
    #         print(f'\n--> Checking out back branch: {current_git_branch}')
    #         repo.git.checkout(current_git_branch)
    #         print(f'--> Popping stashed changes\n')
    #         repo.git.stash('pop')