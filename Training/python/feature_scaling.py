import uproot
import awkward as ak
import numpy as np

import time
import gc
import argparse
import yaml
import json
import collections
from glob import glob
from tqdm import tqdm
from collections import defaultdict

from scaling_utils import Phi_mpi_pi, dR, dR_signal_cone, compute_mean, compute_std, fill_aggregators
from scaling_utils import nested_dict, init_dictionaries, dump_to_json

if __name__ == '__main__':
    # parse command line parameters
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", type=str, help="Path to yaml configuration file")
    parser.add_argument('--var_types', nargs='+', help="Variable types from field 'Features_all' of the cfg file for which to derive scaling parameters. Defaults to -1 for running on all those specified in the cfg", default=-1)
    args = parser.parse_args()
    with open(args.cfg) as f:
        scaling_dict = yaml.load(f, Loader=yaml.FullLoader)

    # read cfg parameters
    setup_dict = scaling_dict['Scaling_setup']
    features_dict = scaling_dict['Features_all']
    #
    assert type(args.var_types) == list
    if args.var_types[0] == "-1" and len(args.var_types) == 1:
        var_types = features_dict.keys()
    else:
        var_types = args.var_types
    file_path = setup_dict['file_path']
    output_json_folder = setup_dict['output_json_folder']
    file_range = setup_dict['file_range']
    tree_name = setup_dict['tree_name']
    log_step = setup_dict['log_step']
    version = setup_dict['version']
    scaling_params_json_prefix = f'{output_json_folder}/scaling_params_v{version}'
    #
    selection_dict = setup_dict['selection']
    cone_definition_dict = setup_dict['cone_definition']
    cone_selection_dict = setup_dict['cone_selection']
    #
    assert log_step > 0 and type(log_step) == int
    assert len(file_range)==2 and file_range[0]<=file_range[1]
    file_names = sorted(glob(file_path))[file_range[0]:file_range[1]]
    #
    dR_tau_outer_cone = cone_definition_dict['outer']['dR']
    tau_pt_name, tau_eta_name, tau_phi_name = cone_selection_dict['TauFlat']['var_names']['pt'], cone_selection_dict['TauFlat']['var_names']['eta'], cone_selection_dict['TauFlat']['var_names']['phi']
    inner_cone_min_pt = cone_definition_dict['inner']['min_pt']
    inner_cone_min_radius = cone_definition_dict['inner']['min_radius']
    inner_cone_opening_coef = cone_definition_dict['inner']['opening_coef']
    #
    # initialise dictionaries to be filled
    sums, sums2, counts, scaling_params = init_dictionaries(features_dict, cone_selection_dict, len(file_names))
    #
    print(f'\n[INFO] will process {len(file_names)} input files from {file_path}')
    print(f'[INFO] will dump scaling parameters to {scaling_params_json_prefix}_*.json after every {log_step} files')
    print('[INFO] starting to accumulate sums & counts:\n')
    #
    skip_counter = 0 # counter of files which were skipped during processing
    inf_counter = defaultdict(list) # counter of features with inf values and their fraction
    processed_last_file = time.time()

    # loop over input files
    for file_i, file_name in enumerate(tqdm(file_names)):
        log_scaling_params = not (file_i%log_step) or (file_i == len(file_names)-1)
        with uproot.open(file_name, array_cache='5 GB') as f:
            if len(f.keys()) == 0: # some input ROOT files can be corrupted and uproot can't recover for it. These files are skipped in computations
                print(f'[WARNING] couldn\'t find any object in {file_name}: skipping the file')
                skip_counter += 1
            else:
                tree = f[tree_name]
                # NB: selection cut is not applied on tau branches
                tau_pt_array, tau_eta_array, tau_phi_array = tree.arrays([tau_pt_name, tau_eta_name, tau_phi_name], cut=None, aliases=None, how=tuple)
                # loop over variable type
                for var_type in var_types:
                    # loop over variables of the given type
                    for var_dict in features_dict[var_type]:
                        begin_var = time.time()
                        (var, (selection_cut, aliases, scaling_type, *lim_params)), = var_dict.items()
                        if scaling_type != 'normal': continue # other scaling_type are already initialised with mean, std and lim_min/lim_max
                        constituent_eta_name, constituent_phi_name = cone_selection_dict[var_type]['var_names']['eta'], cone_selection_dict[var_type]['var_names']['phi']
                        # NB: selection cut is applied, broadcasting with tau array (w/o cut) correctly handles the difference
                        var_array, constituent_eta_array, constituent_phi_array = tree.arrays([var, constituent_eta_name, constituent_phi_name], cut=selection_cut, aliases=aliases, how=tuple)
                        if np.sum(np.isinf(var_array)) > 0:
                            is_inf_mask = np.isinf(var_array)
                            inf_counter[var].append(np.sum(is_inf_mask) / ak.count(var_array))
                            var_array = ak.mask(var_array, is_inf_mask, valid_when=False) # mask inf values with None
                        dR_tau_signal_cone = dR_signal_cone(tau_pt_array, inner_cone_min_pt, inner_cone_min_radius, inner_cone_opening_coef)
                        # loop over cone types specified for a given var_type in the cfg file
                        for cone_type in cone_selection_dict[var_type]['cone_types']:
                            fill_aggregators(var_array, tau_eta_array, tau_phi_array, constituent_eta_array, constituent_phi_array,
                                             var, var_type, file_i, cone_type, dR_tau_signal_cone, dR_tau_outer_cone,
                                             sums, sums2, counts, fill_scaling_params=log_scaling_params, scaling_params=scaling_params
                                             )
                        del(constituent_eta_array, constituent_phi_array, var_array)
                        end_var = time.time()
                        # print(f'---> processed {var} in {end_var - begin_var:.2f} s\n')
                del(tau_pt_array, tau_eta_array, tau_phi_array)
        gc.collect()
        # snapshot scaling params into json if log_step is reached
        if log_scaling_params:
            if file_i == len(file_names)-1:
                scaling_params_json_name = scaling_params_json_prefix
            else:
                scaling_params_json_name = f'{scaling_params_json_prefix}_log_{(file_i+1)//log_step}'
            dump_to_json({scaling_params_json_name: scaling_params})
        processed_current_file = time.time()
        # print(f'---> processed {file_name} in {processed_current_file - processed_last_file:.2f} s')
        processed_last_file = processed_current_file
    print()
    if skip_counter > 0:
        print(f'[WARNING] during the processing {skip_counter} files with no objects were skipped\n')
    for inf_feature, inf_frac_counts in inf_counter.items():
        print(f'[WARNING] in {inf_feature} encountered inf values with average count fraction: {np.mean(inf_frac_counts)}')
    print('\nDone!')