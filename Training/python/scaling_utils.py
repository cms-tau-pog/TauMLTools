import awkward as ak
import numpy as np

import gc
import json
import collections

def Phi_mpi_pi(array_phi):
    array_phi = ak.where(array_phi <= np.pi, array_phi, array_phi - 2*np.pi)
    array_phi = ak.where(array_phi >= -np.pi, array_phi, array_phi + 2*np.pi)
    assert ak.sum(array_phi > np.pi) + ak.sum(array_phi < -np.pi) == 0
    return array_phi

def dR(deta, dphi):
    ### TODO: use https://github.com/scikit-hep/vector instead of this custom implementation
    return np.sqrt(deta**2 + Phi_mpi_pi(dphi)**2)

def dR_signal_cone(pt_tau, min_pt, min_radius, opening_coef):
    return np.maximum(opening_coef/np.maximum(pt_tau, min_pt), min_radius)

def nested_dict():
    return collections.defaultdict(nested_dict)

def init_dictionaries(features_dict, cone_selection_dict, n_files):
    sums, sums2, counts, scaling_params = nested_dict(), nested_dict(), nested_dict(), nested_dict()
    for var_type in features_dict.keys():
        for var_dict in features_dict[var_type]:
            assert len(var_dict) == 1
            (var, (_, _, scaling_type, *lim_params)), = var_dict.items()
            if scaling_type == 'no_scaling':
                scaling_params[var_type][var] = {"mean": 0, "std": 1, "lim_min": "-inf", "lim_max": "inf"}
            elif scaling_type == 'linear':
                # NB: initialisation below assumes shift by mean, scaling by std and then clamping on lim_min, lim_max = [-1, 1] range downstream in DataLoader
                if len(lim_params) == 2:
                    assert lim_params[0] <= lim_params[1]
                    scaling_params[var_type][var] = {"mean": (lim_params[0]+lim_params[1])/2.,
                                                     "std": (lim_params[1]-lim_params[0])/2., "lim_min": -1., "lim_max": 1.}
                elif len(lim_params) == 1:
                    cone_dict = lim_params[0]
                    assert type(cone_dict) == dict
                    for cone_type, cone_lim_params in cone_dict.items():
                        assert cone_type in cone_selection_dict[var_type]['cone_types']
                        assert len(cone_lim_params)==2 and cone_lim_params[0]<=cone_lim_params[1]
                        scaling_params[var_type][var][cone_type] = {"mean": (cone_lim_params[0]+cone_lim_params[1])/2.,
                                                                    "std": (cone_lim_params[1]-cone_lim_params[0])/2., "lim_min": -1., "lim_max": 1.}
                else:
                    raise ValueError(f"In variable {var}: lim_params should be either pair numbers (min & max), or dictionary (min & max as values, cone types as keys)")
            elif scaling_type == 'normal':
                for cone_type in cone_selection_dict[var_type]['cone_types']:
                    if cone_type is not None:
                        if len(lim_params) == 2:
                            assert lim_params[0] <= lim_params[1]
                            scaling_params[var_type][var][cone_type] = {'mean': None, 'std': None, "lim_min": lim_params[0], "lim_max": lim_params[1]}
                        elif len(lim_params) == 1:
                            cone_dict = lim_params[0]
                            assert type(cone_dict) == dict
                            assert cone_type in cone_dict.keys()
                            cone_lim_params = cone_dict[cone_type]
                            assert len(cone_lim_params)==2 and cone_lim_params[0]<=cone_lim_params[1]
                            scaling_params[var_type][var][cone_type] = {'mean': None, 'std': None, "lim_min": cone_lim_params[0], "lim_max": cone_lim_params[1]}
                        elif len(lim_params) == 0:
                            scaling_params[var_type][var][cone_type] = {'mean': None, 'std': None, "lim_min": "-inf", "lim_max": "inf"}
                        else:
                            raise ValueError(f'In variable {var}: too many lim_params specified, expect either None, or 1 (dictionary with min/max values for various cone types), or 2 (min/max values)')
                        sums[var_type][var][cone_type] = np.zeros(n_files, dtype='float64')
                        sums2[var_type][var][cone_type] = np.zeros(n_files, dtype='float64')
                        counts[var_type][var][cone_type] = np.zeros(n_files, dtype='int64')
                    else:
                        if len(lim_params) == 2:
                            assert lim_params[0] <= lim_params[1]
                            scaling_params[var_type][var] = {'mean': None, 'std': None, "lim_min": lim_params[0], "lim_max": lim_params[1]}
                        elif len(lim_params) == 0:
                            scaling_params[var_type][var] = {'mean': None, 'std': None, "lim_min": "-inf", "lim_max": "inf"}
                        else:
                            raise ValueError(f'In variable {var}: too many lim_params specified, expect either None, or 2 (min/max values)')
                        sums[var_type][var] = np.zeros(n_files, dtype='float64')
                        sums2[var_type][var] = np.zeros(n_files, dtype='float64')
                        counts[var_type][var] = np.zeros(n_files, dtype='int64')
            else:
                raise ValueError(f"In variable {var}: scaling_type should be either no_scaling, or linear, or normal")
    return sums, sums2, counts, scaling_params

def compute_mean(sums, counts, aggregate=True, *file_range):
    if aggregate:
        if file_range:
            assert len(file_range) == 2 and file_range[0] <= file_range[1]
            return sums[file_range[0]:file_range[1]].sum()/counts[file_range[0]:file_range[1]].sum()
        else:
            return sums.sum()/counts.sum()
    else:
        return sums/counts

def compute_std(sums, sums2, counts, aggregate=True, *file_range):
    if aggregate:
        if file_range:
            assert len(file_range) == 2 and file_range[0] <= file_range[1]
            average2 = sums2[file_range[0]:file_range[1]].sum()/counts[file_range[0]:file_range[1]].sum()
            average = sums[file_range[0]:file_range[1]].sum()/counts[file_range[0]:file_range[1]].sum()
            return np.sqrt(average2 - average**2)
        else:
            return np.sqrt(sums2.sum()/counts.sum() - (sums.sum()/counts.sum())**2)
    else:
        return np.sqrt(sums2/counts - (sums/counts)**2)

def fill_aggregators(var_array, tau_eta_array, tau_phi_array, constituent_eta_array, constituent_phi_array,
                     var, var_type, file_i, cone_type, dR_tau_signal_cone, dR_tau_outer_cone,
                     sums, sums2, counts, fill_scaling_params=False, scaling_params=None):
    if cone_type == None:
        sums[var_type][var][file_i] += ak.sum(var_array)
        sums2[var_type][var][file_i] += ak.sum(var_array**2)
        counts[var_type][var][file_i] += ak.count(var_array)
        if fill_scaling_params:
            scaling_params[var_type][var]['mean'] = compute_mean(sums[var_type][var], counts[var_type][var], aggregate=True)
            scaling_params[var_type][var]['std'] = compute_std(sums[var_type][var], sums2[var_type][var], counts[var_type][var], aggregate=True)
    elif cone_type == 'inner' or cone_type == 'outer':
        constituent_dR = dR(tau_eta_array - constituent_eta_array, tau_phi_array - constituent_phi_array)
        if cone_type == 'inner':
            cone_mask = constituent_dR <= dR_tau_signal_cone
        elif cone_type == 'outer':
            cone_mask = (constituent_dR > dR_tau_signal_cone) & (constituent_dR < dR_tau_outer_cone)
        sums[var_type][var][cone_type][file_i] += ak.sum(var_array[cone_mask])
        sums2[var_type][var][cone_type][file_i] += ak.sum(var_array[cone_mask]**2)
        counts[var_type][var][cone_type][file_i] += ak.count(var_array[cone_mask])
        if fill_scaling_params:
            scaling_params[var_type][var][cone_type]['mean'] = compute_mean(sums[var_type][var][cone_type], counts[var_type][var][cone_type], aggregate=True)
            scaling_params[var_type][var][cone_type]['std'] = compute_std(sums[var_type][var][cone_type], sums2[var_type][var][cone_type], counts[var_type][var][cone_type], aggregate=True)
    else:
        raise ValueError(f'cone_type for {var_type} should be either inner, or outer')

def dump_to_json(dict_map):
    for fout_name, dict in dict_map.items():
        with open(f'{fout_name}.json', 'w') as fout:
            json.dump(dict, fout)
