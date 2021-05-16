import awkward as ak
import numpy as np

import gc
import json
import collections

def Phi_mpi_pi(array_phi):
    """
    Periodically (T=2*pi, shift step done only once) bring values of the given array into [-pi, pi] range.

    Arguments:
        array_phi: awkward array, values assumed to be radian measure of phi angle

    Returns:
        Awkward array, values of input array brought to [-pi, pi] range
    """
    array_phi = ak.where(array_phi <= np.pi, array_phi, array_phi - 2*np.pi)
    array_phi = ak.where(array_phi >= -np.pi, array_phi, array_phi + 2*np.pi)
    assert ak.sum(array_phi > np.pi) + ak.sum(array_phi < -np.pi) == 0
    return array_phi

def dR(deta, dphi):
    """
    Calculate dR=np.sqrt(deta**2 + dphi_shifted**2) between two vectors given differences of their eta and phi coordinates.
    Internally call Phi_mpi_pi() to bring delta eta values into [-pi, pi] range.

    Arguments:
        deta: awkward array, differences in eta coordinate of two arrays (i.e. eta_1-eta_2)
        dphi: awkward array, differences in phi coordinate of two arrays (i.e. phi_1-phi_2)

    Returns:
        Awkward array, dR values
    """
    # TODO: use https://github.com/scikit-hep/vector instead of the custom implementation
    return np.sqrt(deta**2 + Phi_mpi_pi(dphi)**2)

def dR_signal_cone(pt_tau, min_pt, min_radius, opening_coef):
    """
    Given a vector of taus' pt values calculate dR values of the corresponding signal cones according to the formula:
        dR_signal_cone = max(opening_coef/max(pt_tau, min_pt), min_radius)

    Arguments:
        pt_tau: awkward array, vector of taus' pt
        min_pt: float, defines minimal tau pt to be considered
        min_radius: float, defines minimal cone dR size to be considered
        opening_coef: float, cone parameter

    Returns:
        Awkward array, signal cone dR for each tau candidate
    """
    return np.maximum(opening_coef/np.maximum(pt_tau, min_pt), min_radius)

def nested_dict():
    """
    Construct a recursively instantiated dictionary for convenient arbitrary initialisation.

    Returns:
        collections.defaultdict, recursively nested
    """
    return collections.defaultdict(nested_dict)

def init_dictionaries(features_dict, cone_selection_dict, n_files):
    """
    Initialise and return necessary dictionaries for scaling parameters computation.
    This is done by firstly going in the input `features_dict` through variable types and then variables themselves.
    Depending on the specified type of scaling for a given variable the following cases are implemented:

        - no scaling: initialise only scaling params
            -> mean=0, std=1, lim_min=-inf, lim_max=inf
        - linear: initialise only scaling params with an option of inclusive or separate (inner vs outer cone) initialisation
            NB: this assumes the clamping range downstream to be [-1, 1]
            -> mean=(lim_params[0]+lim_params[1])/2., std=(lim_params[1]-lim_params[0])/2., lim_min=-1., lim_max=1.
        - normal: initialise sums, sums2, counts, scaling params with an option of inclusive or separate (inner vs outer cone) initialisation
            -> sums, sums2 and counts are initialised with 0
            -> if lim_params specified:
                    mean=None, std=None, lim_min=lim_params[0], lim_max=lim_params[1]
               else:
                    mean=None, std=None, lim_min=-inf, lim_max=inf

    For the description of the parameters and corresponding input configuration format please refer to the comments in the main dataloading .yaml config file.

    Arguments:
        - features_dict: dict, scaling configuration per particle type and feature as it is read from the main data loading .yaml config file ("Features_all" field)
        - cone_selection_dict: dict, cone configuration per particle type as it is read from the main data loading .yaml config file ("cone_selection" field)
        - n_files: int, number of input files to be used for mean/std computation

    Returns:
        - sums: dict, container for accumulating sums of features' values
        - sums2: dict, container for accumulating squared sums of features' values
        - counts: dict, container for accumulating counts of features' values
        - scaling_params: dict, container for storing features' scaling parameters (mean, std, lim_min, lim_max)
    """
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
    """
    Assuming input arrays correspond to per file sums and counts for a given feature's values, derive means either on the file-by-file basis, or via aggregating & averaging values over all/specified range of input files.

    Arguments:
        - sums: np.array, summed values for a given feature per processed files
        - counts: np.array, counts of a given feature per processed files
        - aggregate (optional, default=True): bool, whether to aggregate sums and counts for the mean computation. If no `file_range` specified, do that for all the input array, otherwise over a specified range in `file_range`.
        - file_range (optional): if passed, assume to be a list with the range of file ids to run aggregation and mean computation on.

    Returns:
        float (np.array) with mean (means per file)
    """
    if aggregate:
        if file_range:
            assert len(file_range) == 2 and file_range[0] <= file_range[1]
            return sums[file_range[0]:file_range[1]].sum()/counts[file_range[0]:file_range[1]].sum()
        else:
            return sums.sum()/counts.sum()
    else:
        return sums/counts

def compute_std(sums, sums2, counts, aggregate=True, *file_range):
    """
    Assuming input arrays correspond to per file (squared) sums and counts for a given feature's values, derive standard deviation either on the file-by-file basis, or via aggregating & averaging values over all/specified range of input files.

    Arguments:
        - sums: np.array, summed values for a given feature per processed files
        - sums2: np.array, squared summed values for a given feature per processed files
        - counts: np.array, counts of a given feature per processed files
        - aggregate (optional, default=True): bool, whether to aggregate sums, sums2 and counts for the std computation. If no `file_range` specified, do that for all the input array, otherwise over a specified range in `file_range`.
        - file_range (optional): if passed, assume to be a list with the range of file ids to run aggregation and std computation on.

    Returns:
        float (np.array) with std (stds per file)
    """
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
    """
    Update `sums`, `sums2` and `counts` dictionaries with the values from `var_array` either inclusively or exclusively (based on `cone_type` argument) for inner/outer cones.
    In the latter case, derive `constituent_dR` with respect to the tau direction of flight and define cones as:
        - inner: `constituent_dR` <= `dR_tau_signal_cone`
        - outer: constituent_dR` > `dR_tau_signal_cone` and `constituent_dR` < `dR_tau_outer_cone`
    Then mask consitutents which appear in the `cone_type` and update sums/sums2/counts only using those constituents which enter the given cone.

    If `fill_scaling_params` is set to `True`, also update `scaling_params` dictionary (i.e. make a "snapshot" of scaling parameters based on the current state of sums/sums2/counts)

    Arguments:
        - var_array: awkward array, values of a given feature for a given set of taus
        - tau_eta_array: awkward array, eta values of taus
        - tau_phi_array: awkward array, phi values of taus
        - constituent_eta_array: awkward array, eta values of tau constituents
        - constituent_phi_array: awkward array, phi values of tau constituents
        - var: string, variable name
        - var_type: string, variable type
        - file_i: int, index of the file being processed
        - cone_type: string, type of cone being processed, should be either inner or outer
        - dR_tau_signal_cone: awkward array, per tau dR values defining the signal cone
        - dR_tau_outer_cone: float, dR value defining the tau outer cone
        - sums: dict, container for accumulating sums of features' values and to be filled based on the input `var_array`
        - sums2: dict, container for accumulating square sums of features' values and to be filled based on the input `var_array`
        - counts: dict, container for accumulating counts of features' values and to be filled based on the input `var_array`
        - fill_scaling_params (optional, default=False): bool, whether to update the `scaling_params` dictionary with the values from the current state of sums/sums2/counts
        - scaling_params(optional, default=None): dict, main dictionary storing scaling parameters per variable type/variable name/cone type. Used only if `fill_scaling_params` is set to `True`

    Returns:
        None
    """
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
    """
    For each entry in the input `dict_map` write the corresponding dictionary into a json file with the specified path.

    Arguments:
        - dict_map: dict, mapping of the string containg path with filename to write to (without .json extension) and corresponding dictionary

    Returns: None
    """
    for fout_name, dict in dict_map.items():
        with open(f'{fout_name}.json', 'w') as fout:
            json.dump(dict, fout)
