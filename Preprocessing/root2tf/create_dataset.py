import os
import time
import shutil
from glob import glob
from collections import defaultdict
import hydra
from hydra.utils import to_absolute_path
from omegaconf import DictConfig, OmegaConf, open_dict
from LawWorkflows.mass_copy import remote_glob
from utils.tfio import write_tfrecord, serialize_spec
#import tensorflow as tf
import awkward as ak
import numpy as np

# from hydra import compose, initialize
# from omegaconf import OmegaConf
# with initialize(version_base=None, config_path="configs"): cfg = compose(config_name="create_dataset")

def process_files(files, cfg, dataset_cfg):
    import tensorflow as tf
    from utils.data_preprocessing import load_from_file, preprocess_array, awkward_to_tf, compute_labels
    # print(f'\n-> Processing input files ({dataset_type})')
    tau_type_map  = cfg['gen_cfg']['tau_type_map']
    tree_name     = cfg['tree_name']
    step_size     = cfg['step_size']
    feature_names = cfg['feature_names']
    n_samples = defaultdict(int)
    for file_name in files:
        time_0 = time.time()
        # open ROOT file, read awkward array
        # print("HALO",file_name, tree_name, step_size)
        a = load_from_file(file_name, tree_name, step_size)
        time_1 = time.time()
        if cfg['verbose']:
            print(f'\n        Loading: took {(time_1-time_0):.1f} s.')
        # preprocess awkward array
        a_preprocessed, scaling_data, label_data, gen_data, add_columns = preprocess_array(a, feature_names, dataset_cfg.get('add_columns'), cfg.get('verbose'))
        # preprocess labels
        if dataset_cfg.get('recompute_tau_type'):
            _labels = compute_labels(cfg['gen_cfg'], gen_data, label_data) # TODO
        else:
            _labels = label_data['tauType']
        time_2 = time.time()
        if cfg['verbose']:
            print(f'\n        Preprocessing: took {(time_2-time_1):.1f} s.\n')
        # final tuple with elements to be stored into TF dataset
        data = []
        # add awkward arrays converted to TF ragged arrays
        for feature_type, feature_list in feature_names.items():  # Loop over particle collections
            is_ragged = feature_type != 'global'
            # Only compute row lengths once per feature_type if ragged
            type_lengths = None
            if is_ragged:
                type_lengths = ak.count(a_preprocessed[feature_type][feature_list[0]], axis=1)
            # Convert features to TensorFlow tensors
            X = awkward_to_tf(a_preprocessed[feature_type], feature_list, is_ragged, type_lengths)
            data.append(X)
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
        # add additional columns if needed
        if add_columns is not None:
            add_columns = awkward_to_tf(add_columns, dataset_cfg['add_columns'], False, None)
            data.append(add_columns)

        # save label names to the yaml cfg
        with open_dict(cfg):
            cfg["label_columns"] = label_columns
            cfg["scaling_data"] = scaling_data
            cfg["add_columns"] = dataset_cfg['add_columns']
        # create TF dataset
        dataset = tf.data.Dataset.from_tensor_slices(tuple(data))
        time_3 = time.time()
        if cfg['verbose']:
            print(f'\n        Preparing TF datasets: took {(time_3-time_2):.1f} s.')
        # remove existing datasets
        path_to_dataset = to_absolute_path(f'{cfg["path_to_dataset"]}/{os.path.splitext(os.path.basename(file_name))[0]}')
        if os.path.exists(path_to_dataset):
            shutil.rmtree(path_to_dataset)
        else:
            os.makedirs(path_to_dataset, exist_ok=True)
        # save TF dataset
        if cfg.get("file_format") == "tfsave":
            # Saving as tf save files
            dataset.save(path_to_dataset, compression='GZIP')
        elif cfg.get("file_format") == "tfrecord":
            # Saving as tf Record
            with open_dict(cfg):
                cfg["element_spec"] = serialize_spec(dataset.element_spec)
            print(f"Element spec: {cfg['element_spec']}")
            write_tfrecord(dataset, f'{path_to_dataset}/data.tfrecord', add_columns)
        else:
            raise ValueError(f"Unsupported file format: {cfg.get('file_format')}")

        OmegaConf.save(config=cfg, f=f'{path_to_dataset}/cfg.yaml')
        time_4 = time.time()
        if cfg['verbose']:
            print(f'        Saving TF datasets: took {(time_4-time_3):.1f} s.\n')
    return n_samples

@hydra.main(config_path='configs', config_name='create_dataset')
def main(cfg: DictConfig) -> None:
    time_start = time.time()
    # read from cfg
    input_data = OmegaConf.to_object(cfg['input_data'])
    for dataset_type in input_data.keys(): # train/val/test
        # create list of file names to open
        dataset_cfg = input_data[dataset_type]
        _files = dataset_cfg.pop('files')
        files = []
        for file_path in _files:
            files += remote_glob(file_path)
        n_samples = process_files(files=files, cfg=cfg, dataset_cfg=dataset_cfg)
        if cfg.get('verbose'):
            print(f'\n-> Dataset ({dataset_type}) contains:')
            for k, v in n_samples.items():
                print(f'    {k}: {v} samples')
    time_5 = time.time()
    # save the config (to be fetched during the training) after removing file specific data
    for key in ['path_to_dataset', 'scaling_data', 'input_data']:
        cfg.pop(key)
    OmegaConf.save(config=cfg, f=to_absolute_path(f'{cfg["path_to_dataset"]}/cfg.yaml'))
    if cfg['verbose']:
        print(f'\nTotal time: {(time_5-time_start):.1f} s.\n')

if __name__ == '__main__':
    main()
