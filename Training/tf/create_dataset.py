import os
import time
import shutil
import gc
from glob import glob
from collections import defaultdict
import hydra
from hydra.utils import to_absolute_path
from omegaconf import DictConfig, OmegaConf, open_dict
from utils.data_preprocessing import load_from_file, preprocess_array, awkward_to_tf
from utils.gen_preprocessing import compute_genmatch_dR, recompute_tau_type, dict_to_numba

import tensorflow as tf
import awkward as ak
import numpy as np
from numba.core import types

@hydra.main(config_path='configs', config_name='create_dataset')
def main(cfg: DictConfig) -> None:
    time_start = time.time()

    # read from cfg
    tau_type_map = cfg['gen_cfg']['tau_type_map']
    tree_name = cfg['data_cfg']['tree_name']
    step_size = cfg['data_cfg']['step_size']
    feature_names = cfg['feature_names']
    input_data = OmegaConf.to_object(cfg['data_cfg']['input_data'])

    for dataset_type in input_data.keys():
        dataset_cfg = input_data[dataset_type]
        files = dataset_cfg.pop('files')
        if len(files)==1 and "*" in (file_name_regex:=list(files.keys())[0]):
            files = {f: files[file_name_regex] for f in glob(to_absolute_path(file_name_regex))}

        print(f'\n-> Processing input files ({dataset_type})')
        n_samples = defaultdict(int)
        for file_name, tau_types in files.items():
            time_0 = time.time()

            # open ROOT file, read awkward array
            a = load_from_file(file_name, tree_name, step_size)
            time_1 = time.time()
            if cfg['verbose']:
                print(f'        Loading: took {(time_1-time_0):.1f} s.')

            # preprocess awkward array
            a_preprocessed, label_data, gen_data, add_columns = preprocess_array(a, feature_names, dataset_cfg['add_columns'], cfg['verbose'])
            del a; gc.collect()

            # preprocess labels
            if dataset_cfg['recompute_tau_type']:
                # lazy compute dict with gen data
                gen_data = {_k: _v.compute() for _k, _v in gen_data.items()}
                
                # convert dictionaries to numba dict
                genLepton_match_map = dict_to_numba(cfg['gen_cfg']['genLepton_match_map'], key_type=types.unicode_type, value_type=types.int32)
                genLepton_kind_map = dict_to_numba(cfg['gen_cfg']['genLepton_kind_map'], key_type=types.unicode_type, value_type=types.int32)
                sample_type_map = dict_to_numba(cfg['gen_cfg']['sample_type_map'], key_type=types.unicode_type, value_type=types.int32)
                tau_type_map = dict_to_numba(tau_type_map, key_type=types.unicode_type, value_type=types.int32)
                
                # bool mask with dR gen matching
                genmatch_dR = compute_genmatch_dR(gen_data)
                is_dR_matched = genmatch_dR < cfg['gen_cfg']['genmatch_dR']

                # recompute labels
                tau_type_column = 'tauType_recomputed'
                recomputed_labels = recompute_tau_type(genLepton_match_map, genLepton_kind_map, sample_type_map, tau_type_map,
                                                            label_data['sampleType'], is_dR_matched,
                                                            gen_data['genLepton_index'], gen_data['genJet_index'], gen_data['genLepton_kind'], gen_data['genLepton_vis_pt'])
                label_data[tau_type_column] = ak.Array(recomputed_labels)

                # check the fraction of recomputed labels comparing to the original
                if sum_:=np.sum(label_data[tau_type_column]!=label_data["tauType"]):
                    print(f'\n        [WARNING] non-zero fraction of recomputed tau types: {sum_/len(label_data["tauType"])*100:.1f}%\n')
            else:
                tau_type_column = 'tauType'
  
            time_2 = time.time()
            if cfg['verbose']:
                print(f'        Preprocessing: took {(time_2-time_1):.1f} s.')

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
                labels.append(ak.values_astype(label_data[tau_type_column] == tau_type_value, np.int32))
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
                print(f'        Preparing TF datasets: took {(time_3-time_2):.1f} s.')

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

        if cfg['verbose']:
            print(f'\n-> Dataset ({dataset_type}) contains:')
            for k, v in n_samples.items():
                print(f'    {k}: {v} samples')
    
    if cfg['verbose']:
        print(f'\nTotal time: {(time_4-time_start):.1f} s.\n') 

if __name__ == '__main__':
    main()