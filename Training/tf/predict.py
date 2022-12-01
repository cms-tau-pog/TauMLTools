import os
import yaml
from glob import glob
import hydra
from hydra.utils import to_absolute_path
from omegaconf import DictConfig

import mlflow
import pandas as pd
import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow.keras.models import load_model

physical_devices = tf.config.list_physical_devices('GPU') 
tf.config.experimental.set_memory_growth(physical_devices[0], True)
tf.config.experimental.set_virtual_device_configuration(physical_devices[0],
    [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=10*1024)])

@hydra.main(config_path='configs', config_name='predict')
def main(cfg: DictConfig) -> None:
    mlflow.set_tracking_uri(f'file://{to_absolute_path(cfg["path_to_mlflow"])}')

    # setup gpu
    physical_devices = tf.config.list_physical_devices('GPU') 
    # tf.config.experimental.set_memory_growth(physical_devices[cfg["gpu_id"]], True)
    tf.config.set_logical_device_configuration(
            physical_devices[cfg["gpu_id"]],
            [tf.config.LogicalDeviceConfiguration(memory_limit=cfg["memory_limit"]*1024)])
    logical_gpus = tf.config.list_logical_devices('GPU')
    print(len(physical_devices), "Physical GPUs,", len(logical_gpus), "Logical GPUs")

    print('\n-> Loading model\n')
    if cfg["checkpoint"] is not None:
        path_to_model = to_absolute_path(f'{cfg["path_to_mlflow"]}/{cfg["experiment_id"]}/{cfg["run_id"]}/artifacts/checkpoints/{cfg["checkpoint"]}')
    else:
        path_to_model = to_absolute_path(f'{cfg["path_to_mlflow"]}/{cfg["experiment_id"]}/{cfg["run_id"]}/artifacts/model/')
    
    with mlflow.start_run(experiment_id=cfg["experiment_id"], run_id=cfg["run_id"]) as active_run:
        trained_with_custom_schedule = 'CustomSchedule' in active_run.data.params["opt_learning_rate"]
    if trained_with_custom_schedule: # have to pass schedule signature as custom_objects
        model = load_model(path_to_model, {'CustomSchedule': lambda **unused_kwargs: None})
    else:
        model = load_model(path_to_model)

    if cfg["n_files"] == -1: # take all the files
        paths = glob(to_absolute_path(f'{cfg["path_to_dataset"]}/{cfg["dataset_name"]}/{cfg["dataset_type"]}/*/{cfg["tau_type"]}'))
    else: # take only first n_files
        paths = glob(to_absolute_path(f'{cfg["path_to_dataset"]}/{cfg["dataset_name"]}/{cfg["dataset_type"]}/*/{cfg["tau_type"]}'))[:cfg["n_files"]]
    for p in paths:
        file_name = p.split('/')[-2]
        dataset = tf.data.experimental.load(p)
        dataset = dataset.batch(cfg["batch_size"])
        dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)

        # load cfg used to produce dataset and retrieve column names
        with open(to_absolute_path(f'{p}/cfg.yaml'), 'r') as f:
            dataset_cfg = yaml.safe_load(f)
        label_column_names = dataset_cfg["label_columns"]
        add_column_names = dataset_cfg['data_cfg']['input_data'][cfg["dataset_type"]]["add_columns"]

        print(f'\n-> Predicting {file_name}')
        predictions, labels, add_columns = [], [], []
        for (*X, y, add_data) in dataset:
            predictions.append(model.predict(X))
            labels.append(y)
            add_columns.append(add_data)

        predictions = tf.concat(predictions, axis=0).numpy()
        labels = tf.concat(labels, axis=0).numpy()
        add_columns = tf.concat(add_columns, axis=0).numpy()
        
        # log to mlflow and delete intermediate file
        with mlflow.start_run(experiment_id=cfg["experiment_id"], run_id=cfg["run_id"]) as active_run:

            # extract mapping between model nodes and corresponding label names
            model_node_to_name = {k: v for k,v in active_run.data.params.items() if k.startswith('model_node_')}
            model_node_to_name = {int(k.split("model_node_")[-1]): v for k,v in model_node_to_name.items()}
            model_node_names = [model_node_to_name[k] for k in sorted(model_node_to_name)]

            predictions = pd.DataFrame(data=predictions, columns=[f'pred_{tau_type}' for tau_type in model_node_names])
            labels = pd.DataFrame(data=labels, columns=label_column_names, dtype=np.int64)
            add_columns = pd.DataFrame(data=add_columns, columns=add_column_names)
            
            print(f'   Saving to hdf5\n')
            predictions.to_hdf(f'{cfg["output_filename"]}.h5', key='predictions', mode='w', format='fixed', complevel=1, complib='zlib')
            labels.to_hdf(f'{cfg["output_filename"]}.h5', key='labels', mode='r+', format='fixed', complevel=1, complib='zlib')
            add_columns.to_hdf(f'{cfg["output_filename"]}.h5', key='add_columns', mode='r+', format='fixed', complevel=1, complib='zlib')
        

            mlflow.log_artifact(f'{cfg["output_filename"]}.h5', f'predictions/{cfg["dataset_name"]}/{file_name}/{cfg["tau_type"]}')
        os.remove(f'{cfg["output_filename"]}.h5')

if __name__ == '__main__':
    main()