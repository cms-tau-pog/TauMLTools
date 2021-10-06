import os
import sys
import json
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
import DataLoader

@hydra.main(config_path='.', config_name='apply_training')
def main(cfg: DictConfig) -> None:
    # set up paths
    mlflow.set_tracking_uri(f"file://{to_absolute_path(cfg.path_to_mlflow)}")
    with mlflow.start_run(run_id=cfg.run_id) as active_run:
        path_to_artifacts = to_absolute_path(f'{cfg.path_to_mlflow}/{active_run.info.experiment_id}/{cfg.run_id}/artifacts/')

    # load the model
    with open(to_absolute_path(f'{path_to_artifacts}/input_cfg/metric_names.json')) as f:
        metric_names = json.load(f)
    path_to_model = f'{path_to_artifacts}/model'
    model = load_model(path_to_model, {name: lambda _: None for name in metric_names.keys()})

    # instantiate DataLoader and get generator
    training_cfg   = OmegaConf.to_object(cfg.training_cfg) # convert to a classical dictionary
    scaling_cfg  = f'{path_to_artifacts}/input_cfg/{cfg.scaling_cfg}'
    dataloader = DataLoader.DataLoader(training_cfg, scaling_cfg)
    gen_predict = dataloader.get_predict_generator()
    tau_types_names = training_cfg['Setup']['tau_types_names']

    # time_checkpoints = [time.time()]
    for input_file in tqdm(cfg.files_to_predict):
        assert len(input_file)==1
        input_file_key = list(input_file.keys())[0]
        input_file_name = to_absolute_path(input_file_key)
        input_file_cfg = input_file[input_file_key]
        output_file_name = f"{input_file_cfg['vs_type']}_{input_file_cfg['alias']}"
        with uproot.open(input_file_name) as f:
            t = f['taus']
            n_taus = len(t['evt'].array())
        
        predictions = []
        targets = []
        print(f'\n\n--> Processing file {input_file_name}, number of taus: {n_taus}')
        for X,y in tqdm(gen_predict(input_file_name), total=n_taus/training_cfg['Setup']['n_tau']):
            predictions.append(model.predict(X))
            targets.append(y)
            # time_checkpoints.append(time.time())
            # print(i, " ", time_checkpoints[-1]-time_checkpoints[-2], "s.")
        
        # concat and check for validity
        predictions = np.concatenate(predictions, axis=0)
        targets = np.concatenate(targets, axis=0)

        if np.any(np.isnan(predictions)):
            raise RuntimeError("NaN in predictions. Total count = {} out of {}".format(
                                np.count_nonzero(np.isnan(predictions)), predictions.shape))
        if np.any(predictions < 0) or np.any(predictions > 1):
            raise RuntimeError("Predictions outside [0, 1] range.")

        # store into intermediate hdf5 file
        predictions = pd.DataFrame({f'node_{tau_type}': predictions[:, int(idx)] for idx, tau_type in tau_types_names.items()})
        targets = pd.DataFrame({f'target_{tau_type}': targets[:, int(idx)] for idx, tau_type in tau_types_names.items()}, dtype=np.int64)
        predictions.to_hdf(f'{output_file_name}.h5', key='predictions', mode='w', format='fixed', complevel=1, complib='zlib')
        targets.to_hdf(f'{output_file_name}.h5', key='targets', mode='r+', format='fixed', complevel=1, complib='zlib')
        
        # log to mlflow and delete intermediate file
        with mlflow.start_run(run_id=cfg.run_id) as active_run:
            mlflow.log_artifact(f'{output_file_name}.h5', 'predictions')
        os.remove(f'{output_file_name}.h5')

if __name__ == '__main__':
    main()  