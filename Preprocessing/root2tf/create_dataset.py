import os
import time
import shutil
import gc
from glob import glob
from collections import defaultdict

import hydra
from hydra.utils import to_absolute_path
from omegaconf import DictConfig, OmegaConf, open_dict

from utils.data_preprocessing import load_from_file, preprocess_array, awkward_to_tf, compute_labels, _get_xrootd_filenames

import tensorflow as tf
import awkward as ak
import numpy as np

def fetch_file_list(_files, cfg):
    files = []
    for _entry in _files:
        if _entry.startswith('root://'): # stream with xrootd, assume _entry is a directory to read *all* ROOT files from
            files += _get_xrootd_filenames(_entry, verbose=cfg['verbose'])
        else: # complete the pattern with glob and append file names to the final list
            files += glob(to_absolute_path(_entry))
    return set(files)

def process_files(files, cfg, dataset_type, dataset_cfg):
    print(f'\n-> Processing input files ({dataset_type})')

    tau_type_map  = cfg['gen_cfg']['tau_type_map']
    tree_name     = cfg['tree_name']
    step_size     = cfg['step_size']
    feature_names = cfg['feature_names']

    n_samples = defaultdict(int)
    for file_name in files:
        time_0 = time.time()

        # open ROOT file, read awkward array
        a = load_from_file(file_name, tree_name, step_size)
        time_1 = time.time()
        if cfg['verbose']:
            print(f'\n        Loading: took {(time_1-time_0):.1f} s.')

        # preprocess awkward array
        a_preprocessed, label_data, gen_data, add_columns = preprocess_array(a, feature_names, dataset_cfg['add_columns'], cfg['verbose'])
        del a; gc.collect()

        # preprocess labels
        if dataset_cfg['recompute_tau_type']:
            _labels = compute_labels(cfg['gen_cfg'], gen_data, label_data)
        else:
            _labels = label_data['tauType']

        time_2 = time.time()
        if cfg['verbose']:
            print(f'\n        Preprocessing: took {(time_2-time_1):.1f} s.\n')

        # final tuple with elements to be stored into TF dataset
        data = []

        # add awkward arrays converted to TF ragged arrays
        for feature_type, feature_list in feature_names.items(): # do this separately for each particle collection
            is_ragged = feature_type != 'global'
            X = awkward_to_tf(a_preprocessed[feature_type], feature_list, is_ragged) # will keep only feats from feature_list
            data.append(X)
            del a_preprocessed[feature_type], X; gc.collect()

        # add one-hot encoded labels
        label_columns = []
        labels = []
        for tau_type, tau_type_value in tau_type_map.items():
            _l = ak.values_astype(_labels == tau_type_value, np.int32)
            labels.append(_l)
            n_samples[tau_type] = ak.sum(_l)
            label_columns.append(f'label_{tau_type}')
        labels = tf.stack(labels, axis=-1)
        data.append(labels)
        del labels, label_data; gc.collect()

        # save label names to the yaml cfg
        with open_dict(cfg):
            cfg["label_columns"] = label_columns

        # add additional columns if needed
        if add_columns is not None:
            add_columns = awkward_to_tf(add_columns, dataset_cfg['add_columns'], False)
            data.append(add_columns)
            del add_columns; gc.collect()

        # create TF dataset
        dataset = tf.data.Dataset.from_tensor_slices(tuple(data))
        time_3 = time.time()
        if cfg['verbose']:
            print(f'\n        Preparing TF datasets: took {(time_3-time_2):.1f} s.')

        # remove existing datasets
        path_to_dataset = to_absolute_path(f'{cfg["path_to_dataset"]}/{cfg["dataset_name"]}/{dataset_type}/{os.path.splitext(os.path.basename(file_name))[0]}')
        if os.path.exists(path_to_dataset):
            shutil.rmtree(path_to_dataset)
        else:
            os.makedirs(path_to_dataset, exist_ok=True)

        # save TF dataset
        dataset.save(path_to_dataset, compression='GZIP')
        OmegaConf.save(config=cfg, f=f'{path_to_dataset}/cfg.yaml')
        time_4 = time.time()
        if cfg['verbose']:
            print(f'        Saving TF datasets: took {(time_4-time_3):.1f} s.\n')
        del dataset, data; gc.collect()
    return True

@hydra.main(config_path='configs', config_name='create_dataset')
def main(cfg: DictConfig) -> None:
    time_start = time.time()

    # read from cfg
    input_data = OmegaConf.to_object(cfg['input_data'])

    for dataset_type in input_data.keys(): # train/val/test
        # create list of file names to open
        dataset_cfg = input_data[dataset_type]
        _files = dataset_cfg.pop('files')
        files = fetch_file_list(_files)

        process_files(files=files, cfg=cfg, dataset_type=dataset_type, dataset_cfg=dataset_cfg)

        if cfg['verbose']:
            print(f'\n-> Dataset ({dataset_type}) contains:')
            for k, v in n_samples.items():
                print(f'    {k}: {v} samples')
    
    if cfg['verbose']:
        print(f'\nTotal time: {(time_4-time_start):.1f} s.\n')

if __name__ == '__main__':
    main()