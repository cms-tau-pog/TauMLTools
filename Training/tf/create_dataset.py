import os
import time
import shutil
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
    input_branches = cfg['data_cfg']['input_branches']
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
            a = load_from_file(file_name, tree_name, input_branches)
            time_1 = time.time()
            print(f'        Loading: took {(time_1-time_0):.1f} s.')

            # preprocess awkward array
            a = preprocess_array(a)

            # preprocess labels
            if dataset_cfg['recompute_tau_type']:
                genLepton_match_map = dict_to_numba(cfg['gen_cfg']['genLepton_match_map'], key_type=types.unicode_type, value_type=types.int32)
                genLepton_kind_map = dict_to_numba(cfg['gen_cfg']['genLepton_kind_map'], key_type=types.unicode_type, value_type=types.int32)
                sample_type_map = dict_to_numba(cfg['gen_cfg']['sample_type_map'], key_type=types.unicode_type, value_type=types.int32)
                tau_type_map = dict_to_numba(tau_type_map, key_type=types.unicode_type, value_type=types.int32)
                genmatch_dR = compute_genmatch_dR(a)
                is_dR_matched = genmatch_dR < cfg['gen_cfg']['genmatch_dR']

                tau_type_column = 'tauType_recomputed'
                a[tau_type_column] = recompute_tau_type(genLepton_match_map, genLepton_kind_map, sample_type_map, tau_type_map,
                                                            a['sampleType'], is_dR_matched,
                                                            a['genLepton_index'], a['genJet_index'], a['genLepton_kind'], a['genLepton_vis_pt']) # first execution might be slower due to compilation
                if sum_:=np.sum(a[tau_type_column]!=a["tauType"]):
                    print(f'\n        [WARNING] non-zero fraction of recomputed tau types: {sum_/len(a["tauType"])*100:.1f}%\n')
            else:
                tau_type_column = 'tauType'

            # create one-hot encoded labels
            label_columns = []
            for tau_type, tau_type_value in tau_type_map.items():
                a[f'label_{tau_type}'] = ak.values_astype(a[tau_type_column] == tau_type_value, np.int32)
                label_columns.append(f'label_{tau_type}')
            with open_dict(cfg):
                cfg["label_columns"] = label_columns
  
            time_2 = time.time()
            print(f'        Preprocessing: took {(time_2-time_1):.1f} s.')

            for tau_type in tau_types:
                time_2 = time.time()
                # select only given tau_type
                a_selected = a[a[f'label_{tau_type}'] == 1]
                n_selected = len(a_selected)
                n_samples[tau_type] += n_selected
                print(f'        Selected: {tau_type}: {n_selected} samples')

                # final tuple with elements to be stored into TF dataset
                data = ()

                # add awkward arrays converted to TF ragged arrays
                for particle_type, feature_names in cfg['feature_names'].items(): # do this separately for each particle collection
                    X = awkward_to_tf(a_selected, particle_type, feature_names) # will keep only feats from feature_names
                    data += (X,)
                
                # all labels to final dataset
                y = ak.to_pandas(a_selected[cfg["label_columns"]]).values
                data += (y,)

                # add additional columns if needed
                if dataset_cfg['add_columns'] is not None:
                    add_columns = ak.to_pandas(a_selected[dataset_cfg['add_columns']])
                    add_columns = np.squeeze(add_columns.values)
                    data += (add_columns,)
                
                # create TF dataset 
                dataset = tf.data.Dataset.from_tensor_slices(data)
                time_3 = time.time()
                print(f'        Preparing TF datasets: took {(time_3-time_2):.1f} s.')

                # remove existing datasets
                path_to_dataset = to_absolute_path(f'{cfg["path_to_dataset"]}/{cfg["dataset_name"]}/{dataset_type}/{os.path.splitext(os.path.basename(file_name))[0]}/{tau_type}')
                if os.path.exists(path_to_dataset):
                    shutil.rmtree(path_to_dataset)
                else:
                    os.makedirs(path_to_dataset, exist_ok=True)

                # save
                tf.data.experimental.save(dataset, path_to_dataset)
                OmegaConf.save(config=cfg, f=f'{path_to_dataset}/cfg.yaml')
                time_4 = time.time()
                print(f'        Saving TF datasets: took {(time_4-time_3):.1f} s.\n')

        print(f'\n-> Dataset ({dataset_type}) contains:')
        for k, v in n_samples.items():
            print(f'    {k}: {v} samples')

    print(f'\nTotal time: {(time_4-time_start):.1f} s.\n') 

if __name__ == '__main__':
    main()