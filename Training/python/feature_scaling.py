import uproot
import numpy as np

import os
import time
import gc
import argparse
import yaml
from glob import glob
# from tqdm import tqdm
from collections import defaultdict

from scaling_utils import dR, dR_signal_cone, mask_inf, mask_nan, fill_aggregators, get_quantiles
from scaling_utils import init_dictionaries, dump_to_json

def run_scaling(cfg, var_types, file_list=None, output_folder=None):
    with open(cfg) as f:
        scaling_dict = yaml.load(f, Loader=(yaml.FullLoader))
    
    assert type(var_types) == list, "--var-types should be a list"
    
    setup_dict                  = scaling_dict['Scaling_setup']
    features_dict               = scaling_dict['Features_all']
    var_types                   = features_dict.keys() if (var_types[0] == '-1' and len(var_types) == 1) else var_types
    file_path                   = setup_dict['file_path']
    output_json_folder          = setup_dict['output_json_folder'] if output_folder is None else output_folder
    file_range                  = setup_dict['file_range']
    tree_name                   = setup_dict['tree_name']
    log_step                    = setup_dict['log_step']
    version                     = setup_dict['version']
    selection_dict              = setup_dict['selection']
    scaling_params_json_prefix  = f"{output_json_folder}/scaling_params_v{version}"
    quantile_params_json_prefix = f"{output_json_folder}/quantile_params_v{version}"
    cone_definition_dict        = setup_dict['cone_definition']
    cone_selection_dict         = setup_dict['cone_selection']

    if not os.path.exists(output_json_folder):
        os.makedirs(output_json_folder)

    assert log_step > 0 and type(log_step) == int
    
    if file_list is None:
        if file_range==-1:
            file_names = sorted(glob(file_path))
            n_files = len(file_names)
        elif type(file_range)==list and len(file_range)==2 and file_range[0]<=file_range[1]:
            file_names = sorted(glob(file_path))[file_range[0]:file_range[1]]
            n_files = file_range[1]-file_range[0]
        else:
            raise ValueError('Specified file_range is not valid: should be either -1 (run on all files in file_path) or range [a, b] with a<=b')
    else:
        assert type(file_list) == list
        file_names = file_list
        n_files = len(file_names)
    file_names_ix = [fname.split('_')[-1].split('.root')[0] for fname in file_names] # id as taken from the name: used as file identifier (key field) in the output json files with quantiles
    # initialise dictionaries to be filled
    sums, sums2, counts, scaling_params, quantile_params = init_dictionaries(features_dict, cone_selection_dict, n_files)
    #
    print(f'\n[INFO] will process {n_files} input files from {file_path}')
    print(f'[INFO] will dump scaling parameters to {scaling_params_json_prefix}_*.json after every {log_step} files')
    print(f'[INFO] will dump quantile parameters for every file into {quantile_params_json_prefix}.json')
    print('[INFO] starting to accumulate sums & counts:\n')
    #
    skip_counter = 0 # counter of files which were skipped during processing
    inf_counter = defaultdict(list) # counter of features with inf values and their fraction
    nan_counter = defaultdict(list) # counter of features with nan values and their fraction
    processed_last_file = time.time()
    
    # loop over input files
    for file_i, (file_name_i, file_name) in enumerate(zip(file_names_ix, file_names)): # file_i used internally to count number of processed files
        print("Processing file:",file_i,",",file_name)
        log_scaling_params = not (file_i%log_step) or (file_i == n_files-1)
        with uproot.open(file_name, array_cache='5 GB') as f:
            if len(f.keys()) == 0: # some input ROOT files can be corrupted and uproot can't recover for it. These files are skipped in computations
                print(f'[WARNING] couldn\'t find any object in {file_name}: skipping the file')
                skip_counter += 1
            else:
                tree = f[tree_name]
                # loop over variable type
                for var_type in var_types:
                    # loop over variables of the given type
                    for var_dict in features_dict[var_type]:
                        begin_var = time.time()
                        (var, (selection_cut, aliases, scaling_type, *lim_params)), = var_dict.items()
                        if scaling_type == 'linear':
                            # dict with scaling params already fully filled after init_dictionaries() call, here compute only variable's quantiles
                            if len(lim_params) == 2 and lim_params[0] <= lim_params[1]:
                                var_array = tree.arrays(var, cut=selection_cut, aliases=aliases)[var]
                                var_array = mask_inf(var_array, var, inf_counter, raise_exception=False)
                                var_array = mask_nan(var_array, var, nan_counter, raise_exception=False)
                                quantile_params[var_type][var]['global'][file_name_i] = get_quantiles(var_array)
                                # del(var_array)
                            elif len(lim_params) == 1 and type(lim_params[0]) == dict:
                                tau_pt_name, tau_eta_name, tau_phi_name = cone_selection_dict['TauFlat']['var_names']['pt'], cone_selection_dict['TauFlat']['var_names']['eta'], cone_selection_dict['TauFlat']['var_names']['phi']
                                # NB: selection cut is not applied on tau branches
                                tau_pt_array, tau_eta_array, tau_phi_array = tree.arrays([tau_pt_name, tau_eta_name, tau_phi_name], cut=None, aliases=None, how=tuple)
                                constituent_eta_name, constituent_phi_name = cone_selection_dict[var_type]['var_names']['eta'], cone_selection_dict[var_type]['var_names']['phi']
                                var_array, constituent_eta_array, constituent_phi_array = tree.arrays([var, constituent_eta_name, constituent_phi_name], cut=selection_cut, aliases=aliases, how=tuple)
                                var_array = mask_inf(var_array, var, inf_counter, raise_exception=False)
                                var_array = mask_nan(var_array, var, nan_counter, raise_exception=False)
                                dR_tau_signal_cone = dR_signal_cone(tau_pt_array,
                                                                    cone_definition_dict['inner']['min_pt'],
                                                                    cone_definition_dict['inner']['min_radius'],
                                                                    cone_definition_dict['inner']['opening_coef'])
                                for cone_type in cone_selection_dict[var_type]['cone_types']:
                                    assert cone_type in lim_params[0].keys() # constrain only to those cone_types in cfg
                                    constituent_dR = dR(tau_eta_array - constituent_eta_array, tau_phi_array - constituent_phi_array)
                                    if cone_type == 'inner':
                                        cone_mask = constituent_dR <= dR_tau_signal_cone
                                    elif cone_type == 'outer':
                                        cone_mask = (constituent_dR > dR_tau_signal_cone) & (constituent_dR < cone_definition_dict['outer']['dR'])
                                    else:
                                        raise ValueError(f'For {var} cone_type should be either inner or outer, got {cone_type}.')
                                    quantile_params[var_type][var][cone_type][file_name_i] = get_quantiles(var_array[cone_mask])
                                # del(constituent_eta_array, constituent_phi_array, var_array)
                            else:
                                raise ValueError(f'Unrecognised lim_params for {var} in quantile computation')
                        elif scaling_type == 'normal':
                            for cone_type in cone_selection_dict[var_type]['cone_types']:
                                fill_aggregators(tree, var, var_type, file_i, file_name_i, cone_type, cone_definition_dict, cone_selection_dict, inf_counter, nan_counter,
                                                 selection_cut, aliases, sums, sums2, counts, fill_scaling_params=log_scaling_params,
                                                 scaling_params=scaling_params, quantile_params=quantile_params
                                                 )
                            # del(constituent_eta_array, constituent_phi_array, var_array)
                        end_var = time.time()
                        # print(f'---> processed {var} in {end_var - begin_var:.2f} s\n')
                # del(tau_pt_array, tau_eta_array, tau_phi_array)
        gc.collect()
        # snapshot scaling params into json if log_step is reached
        if log_scaling_params:
            if file_i == n_files-1:
                scaling_params_json_name = scaling_params_json_prefix
            else:
                if log_step == 1:
                    scaling_params_json_name = f'{scaling_params_json_prefix}_log_{(file_i)//log_step}'
                else:
                    scaling_params_json_name = f'{scaling_params_json_prefix}_log_{(file_i+1)//log_step}'
            dump_to_json({scaling_params_json_name: scaling_params})
        processed_current_file = time.time()
        # print(f'---> processed {file_name} in {processed_current_file - processed_last_file:.2f} s')
        processed_last_file = processed_current_file
    dump_to_json({f'{quantile_params_json_prefix}': quantile_params})
    print()
    if skip_counter > 0:
        print(f'[WARNING] during the processing {skip_counter} files with no objects were skipped\n')
    for inf_feature, inf_frac_counts in inf_counter.items():
        print(f"[WARNING] in {inf_feature} encountered inf values with average per file fraction: {format(np.mean(inf_frac_counts), '.2g')}")
    for nan_feature, nan_frac_counts in nan_counter.items():
        print(f"[WARNING] in {nan_feature} encountered nan values with average per file fraction: {format(np.mean(nan_frac_counts), '.2g')}")
    print('\nDone!')
    if skip_counter < n_files:
        return True
    else:
        return False

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, help='Path to yaml configuration file', default='Training/configs/trainingReco_v1.yaml')
    parser.add_argument('--var_types', nargs='+', help="Variable types from field 'Features_all' of the cfg file for which to derive scaling parameters. Defaults to -1 for running on all those specified in the cfg", default=['-1'])
    args = parser.parse_args()
    run_scaling(cfg=(args.cfg), var_types=(args.var_types))
