import uproot
import awkward as ak
import tensorflow as tf
import numpy as np
import dask
from dask.delayed import delayed
from utils.gen_preprocessing import compute_genmatch_dR, recompute_tau_type, dict_to_numba
from numba.core import types

def get_type(col):
    # Get the primitive type of the given column
    if col.ndim == 1:
        return col.type.content.primitive
    elif col.ndim == 2:
        return col.type.content.content.primitive
    else:
        raise ValueError(f"Unsupported dimension: {col.ndim}")

def is_finite(col):
    # Get the min and max values for the given column
    primitive_type_col = get_type(col)
    if np.issubdtype(primitive_type_col, np.integer):
        min_value = np.iinfo(primitive_type_col).min
        max_value = np.iinfo(primitive_type_col).max
    elif np.issubdtype(primitive_type_col, np.floating):
        min_value = np.finfo(primitive_type_col).min
        max_value = np.finfo(primitive_type_col).max
    else:
        raise TypeError(f"Unsupported DataType: {primitive_type_col}")
    # Check if the values are infinite 
    return ~np.isnan(col) & np.isfinite(col) & (col!=min_value) & (col!=max_value)


def process_feature(col):
    collumn = col
    # Flatten the column if it is 2D
    if collumn.ndim == 2:
        collumn = ak.flatten(collumn)
    # Get the mean, std and non-NaN count of the column
    mean = ak.nanmean(collumn)
    # "particle_type" will have only one kind of entry for non-pfCand collections
    # This is expected, but can lead to nan std values
    std = ak.nanstd(collumn)
    if np.isnan(std):
        std = 0
    min_ = ak.nanmin(collumn)
    max_ = ak.nanmax(collumn)
    count = ak.sum(is_finite(collumn))
    # Replace NaN values with 0
    no_nan_feature = ak.nan_to_num(col, nan=0)
    scaler_dict = {
        "mean": float(mean),
        "std": float(std),
        "min": float(min_),
        "max": float(max_),
        "count": int(count),
    }
    return no_nan_feature, scaler_dict

def load_from_file(file_name, tree_name, step_size):
    print(f'      - {file_name}')
    a = uproot.dask(f'{file_name}:{tree_name}', step_size=step_size, library='ak', timeout=3000, retries= 5)
    return a

def awkward_to_tf(a, feature_names, is_ragged, type_lengths=None):    
    tf_array = []
    for feature_name in feature_names:
        _a = a[feature_name]
        if not np.all(np.isfinite(_a)):
            raise ValueError(f"Feature '{feature_name}' contains NaN or non-finite values.")
        if is_ragged:
            _a = ak.flatten(_a)
            _a = ak.values_astype(_a, np.float32)
            _a = tf.RaggedTensor.from_row_lengths(_a, type_lengths)
        else:
            _a = ak.values_astype(_a, np.float32)
        tf_array.append(_a)
    tf_array = tf.stack(tf_array, axis=-1)
    return tf_array

def preprocess_array(a, feature_names, add_feature_names, verbose=False):

    # remove taus with abnormal phi
    a = a[is_finite(a["tau_pt"])]

    # dictionary to store preprocessed features (per feature type)
    a_preprocessed = {feature_type: {} for feature_type in feature_names.keys()}

    # fill lazily original features which don't require preprocessing  
    for feature_type, feature_list in feature_names.items():
        for feature_name in feature_list:
            try:
                f = f'{feature_type}_{feature_name}' if feature_type != 'global' else feature_name
                a_preprocessed[feature_type][feature_name] = a[f]
            except:
                if verbose:
                    print(f'        {f} not found in input ROOT file, will skip it') 

    # ------- Global features ------- #

    a_preprocessed['global']['tau_E_over_pt'] = np.sqrt((a['tau_pt']*np.cosh(a['tau_eta']))*(a['tau_pt']*np.cosh(a['tau_eta'])) + a['tau_mass']*a['tau_mass'])/a['tau_pt']
    a_preprocessed['global']['tau_n_charged_prongs'] = a['tau_decayMode']//5 + 1
    a_preprocessed['global']['tau_n_neutral_prongs'] = a['tau_decayMode']%5
    a_preprocessed['global']['tau_chargedIsoPtSumdR03_over_dR05'] = ak.where((a['tau_chargedIsoPtSum']!=0), (a['tau_chargedIsoPtSumdR03']/a['tau_chargedIsoPtSum']), np.nan)
    a_preprocessed['global']['tau_neutralIsoPtSumWeight_over_neutralIsoPtSum'] = ak.where((a['tau_neutralIsoPtSum']!=0), (a['tau_neutralIsoPtSumWeight']/a['tau_neutralIsoPtSum']), np.nan)
    a_preprocessed['global']['tau_neutralIsoPtSumWeightdR03_over_neutralIsoPtSum'] = ak.where((a['tau_neutralIsoPtSum']!=0), (a['tau_neutralIsoPtSumWeightdR03']/a['tau_neutralIsoPtSum']), np.nan)
    a_preprocessed['global']['tau_neutralIsoPtSumdR03_over_dR05'] = ak.where((a['tau_neutralIsoPtSum']!=0), (a['tau_neutralIsoPtSumdR03']/a['tau_neutralIsoPtSum']), np.nan)

    tau_dxy_valid = (is_finite(a['tau_dxy']) & (a['tau_dxy'] > -10) & is_finite(a['tau_dxy_error']) & (a['tau_dxy_error'] > 0)) #Used
    a_preprocessed['global']['tau_dxy_valid'] = ak.values_astype(tau_dxy_valid, np.float32)
    a_preprocessed['global']['tau_dxy'] = ak.where(tau_dxy_valid, (a['tau_dxy']), np.nan)
    a_preprocessed['global']['tau_dxy_sig'] = ak.where(tau_dxy_valid, (np.abs(a['tau_dxy'])/a['tau_dxy_error']), np.nan)

    a_preprocessed['global']['tau_ip3d_sig'] = np.abs(a['tau_ip3d'])/a['tau_ip3d_error']

    tau_dz_sig_valid = (is_finite(a['tau_dz']) & is_finite(a['tau_dz_error']) & (a['tau_dz_error'] > 0)) #Used
    a_preprocessed['global']['tau_dz_sig_valid'] = ak.values_astype(tau_dz_sig_valid, np.float32)
    # a_preprocessed['global']['tau_dz'] = ak.where(tau_dz_sig_valid, (a['tau_dz']), np.nan)
    a_preprocessed['global']['tau_dz_sig'] = ak.where(tau_dz_sig_valid, (np.abs(a['tau_dz'])/a['tau_dz_error']), np.nan)

    tau_gj_angle_diff_valid = (a['tau_gj_angle_diff'] >= 0) # Used
    a_preprocessed['global']['tau_gj_angle_diff_valid'] = ak.values_astype(tau_gj_angle_diff_valid, np.float32)
    a_preprocessed['global']['tau_gj_angle_diff'] = ak.where(tau_gj_angle_diff_valid, (a['tau_gj_angle_diff']), np.nan)

    a_preprocessed['global']['tau_leadChargedCand_etaAtEcalEntrance_minus_tau_eta'] = a['tau_leadChargedCand_etaAtEcalEntrance'] - a['tau_eta']
    a_preprocessed['global']['particle_type'] = 9*ak.ones_like(a['tau_pt']) # assign unique particle type to a "global" token 

    # ------- PF candidates ------- #

    # shift delta phi into [-pi, pi] range 
    pf_dphi = (a['pfCand_phi'] - a['tau_phi'])
    pf_dphi = ak.where(pf_dphi <= np.pi, pf_dphi, pf_dphi - 2*np.pi)
    pf_dphi = ak.where(pf_dphi >= -np.pi, pf_dphi, pf_dphi + 2*np.pi)
    pf_deta = (a['pfCand_eta'] - a['tau_eta'])

    a_preprocessed['pfCand']['dphi'] = pf_dphi
    a_preprocessed['pfCand']['deta'] = pf_deta
    a_preprocessed['pfCand']['rel_pt'] = a['pfCand_pt'] / a['tau_pt']
    a_preprocessed['pfCand']['r'] = np.sqrt(np.square(pf_deta) + np.square(pf_dphi))
    a_preprocessed['pfCand']['theta'] = np.arctan2(pf_dphi, pf_deta) # dphi -> y, deta -> x
    a_preprocessed['pfCand']['particle_type'] = a['pfCand_particleType'] - 1

    vertex_z_valid = is_finite(a['pfCand_vertex_z']) #Used missing flag
    a_preprocessed['pfCand']['vertex_dx'] = a['pfCand_vertex_x'] - a['pv_x']
    a_preprocessed['pfCand']['vertex_dy'] = a['pfCand_vertex_y'] - a['pv_y']
    a_preprocessed['pfCand']['vertex_z_valid'] = ak.values_astype(vertex_z_valid, np.float32)
    a_preprocessed['pfCand']['vertex_dz'] = ak.where(vertex_z_valid, (a['pfCand_vertex_z'] - a['pv_z']), np.nan)
    a_preprocessed['pfCand']['vertex_dx_tauFL'] = a['pfCand_vertex_x'] - a['pv_x'] - a['tau_flightLength_x']
    a_preprocessed['pfCand']['vertex_dy_tauFL'] = a['pfCand_vertex_y'] - a['pv_y'] - a['tau_flightLength_y']
    a_preprocessed['pfCand']['vertex_dz_tauFL'] = ak.where(vertex_z_valid, (a['pfCand_vertex_z'] - a['pv_z'] - a['tau_flightLength_z']), np.nan)

    has_track_details = (a['pfCand_hasTrackDetails'] == 1)
    has_track_details_track_ndof = has_track_details * (a['pfCand_track_ndof'] > 0) # Used
    has_track_details_dxy_sig_finite = has_track_details * (is_finite(a['pfCand_dxy']) & is_finite(a['pfCand_dxy_error'])) #Used
    has_track_details_dz_sig_finite = has_track_details * (is_finite(a['pfCand_dz']) & is_finite(a['pfCand_dz_error'])) #Used
    a_preprocessed['pfCand']['dxy_sig_valid'] = ak.values_astype(has_track_details_dxy_sig_finite, np.float32)
    a_preprocessed['pfCand']['dxy_sig'] = ak.where(has_track_details_dxy_sig_finite, (np.abs(a['pfCand_dxy'])/a['pfCand_dxy_error']), np.nan)
    a_preprocessed['pfCand']['dz_sig_valid'] = ak.values_astype(has_track_details_dz_sig_finite, np.float32)
    a_preprocessed['pfCand']['dz_sig'] = ak.where(has_track_details_dz_sig_finite, (np.abs(a['pfCand_dz'])/a['pfCand_dz_error']), np.nan)
    a_preprocessed['pfCand']['dz_valid'] = ak.values_astype(has_track_details_dz_sig_finite, np.float32)
    a_preprocessed['pfCand']['dz'] = ak.where(is_finite(a['pfCand_dz']), a['pfCand_dz'], np.nan)
    a_preprocessed['pfCand']['track_ndof_valid'] = ak.values_astype(has_track_details_track_ndof, np.float32)
    a_preprocessed['pfCand']['track_ndof'] = ak.where(has_track_details_track_ndof, a['pfCand_track_ndof'], np.nan)
    a_preprocessed['pfCand']['chi2_ndof'] = ak.where(has_track_details_track_ndof, (a['pfCand_track_chi2']/a['pfCand_track_ndof']), np.nan)

    # ------- Electrons ------- #

    # shift delta phi into [-pi, pi] range 
    ele_dphi = (a['ele_phi'] - a['tau_phi'])
    ele_dphi = ak.where(ele_dphi <= np.pi, ele_dphi, ele_dphi - 2*np.pi)
    ele_dphi = ak.where(ele_dphi >= -np.pi, ele_dphi, ele_dphi + 2*np.pi)
    ele_deta = (a['ele_eta'] - a['tau_eta'])

    a_preprocessed['ele']['dphi'] = ele_dphi
    a_preprocessed['ele']['deta'] = ele_deta
    a_preprocessed['ele']['rel_pt'] = a['ele_pt'] / a['tau_pt']
    a_preprocessed['ele']['r'] = np.sqrt(np.square(ele_deta) + np.square(ele_dphi))
    a_preprocessed['ele']['theta'] = np.arctan2(ele_dphi, ele_deta) # dphi -> y, deta -> x
    a_preprocessed['ele']['particle_type'] = 7*ak.ones_like(a['ele_pt']) # assuming PF candidate types are [0..6]

    ele_cc_valid = (a['ele_cc_ele_energy'] >= 0) #Used
    a_preprocessed['ele']['cc_valid'] = ak.values_astype(ele_cc_valid, np.float32)
    a_preprocessed['ele']['cc_ele_rel_energy'] = ak.where(ele_cc_valid, (a['ele_cc_ele_energy']/a['ele_pt']), np.nan)
    a_preprocessed['ele']['cc_gamma_rel_energy'] = ak.where(ele_cc_valid, (a['ele_cc_gamma_energy']/a['ele_cc_ele_energy']), np.nan)
    a_preprocessed['ele']['cc_n_gamma'] = ak.where(ele_cc_valid, a['ele_cc_n_gamma'], np.nan)
    a_preprocessed['ele']['rel_trackMomentumAtVtx'] = a['ele_trackMomentumAtVtx']/a['ele_pt']
    a_preprocessed['ele']['rel_trackMomentumAtCalo'] = a['ele_trackMomentumAtCalo']/a['ele_pt']
    a_preprocessed['ele']['rel_trackMomentumOut'] = a['ele_trackMomentumOut']/a['ele_pt']
    a_preprocessed['ele']['rel_trackMomentumAtEleClus'] = a['ele_trackMomentumAtEleClus']/a['ele_pt']
    a_preprocessed['ele']['rel_trackMomentumAtVtxWithConstraint'] = a['ele_trackMomentumAtVtxWithConstraint']/a['ele_pt']
    a_preprocessed['ele']['rel_ecalEnergy'] = a['ele_ecalEnergy']/a['ele_pt']
    a_preprocessed['ele']['ecalEnergy_sig'] = a['ele_ecalEnergy']/a['ele_ecalEnergy_error']
    a_preprocessed['ele']['rel_gsfTrack_pt'] = a['ele_gsfTrack_pt']/a['ele_pt']
    a_preprocessed['ele']['gsfTrack_pt_sig'] = a['ele_gsfTrack_pt']/a['ele_gsfTrack_pt_error']

    ele_has_closestCtfTrack = (a['ele_closestCtfTrack_normalizedChi2'] >= 0) #Used
    a_preprocessed['ele']['has_closestCtfTrack'] = ak.values_astype(ele_has_closestCtfTrack, np.float32)
    a_preprocessed['ele']['closestCtfTrack_normalizedChi2'] = ak.where(ele_has_closestCtfTrack, a['ele_closestCtfTrack_normalizedChi2'], np.nan)
    a_preprocessed['ele']['closestCtfTrack_numberOfValidHits'] = ak.where(ele_has_closestCtfTrack, a['ele_closestCtfTrack_numberOfValidHits'], np.nan)

    ele_mva_valid = is_finite(a['ele_e5x5']) #Used
    ele_features = ['sigmaEtaEta', 'sigmaIetaIeta', 'sigmaIphiIphi', 'sigmaIetaIphi', 'e1x5', 'e2x5Max', 'e5x5', 'r9', 'hcalDepth1OverEcal', 'hcalDepth2OverEcal', 'hcalDepth1OverEcalBc', 'hcalDepth2OverEcalBc','eLeft', 'eRight', 'eBottom', 'eTop','full5x5_sigmaEtaEta', 'full5x5_sigmaIetaIeta', 'full5x5_sigmaIphiIphi', 'full5x5_sigmaIetaIphi','full5x5_e1x5', 'full5x5_e2x5Max', 'full5x5_e5x5', 'full5x5_r9','full5x5_hcalDepth1OverEcal', 'full5x5_hcalDepth2OverEcal', 'full5x5_hcalDepth1OverEcalBc', 'full5x5_hcalDepth2OverEcalBc','full5x5_eLeft', 'full5x5_eRight', 'full5x5_eBottom', 'full5x5_eTop','full5x5_e2x5Left', 'full5x5_e2x5Right', 'full5x5_e2x5Bottom', 'full5x5_e2x5Top']
    a_preprocessed['ele']['mva_valid'] = ak.values_astype(ele_mva_valid, np.float32)
    for ele_feature in ele_features:
        a_preprocessed['ele'][ele_feature] = ak.where(ele_mva_valid, a[f'ele_{ele_feature}'], np.nan)

    # ------- Muons ------- #

    # shift delta phi into [-pi, pi] range 
    muon_dphi = (a['muon_phi'] - a['tau_phi'])
    muon_dphi = ak.where(muon_dphi <= np.pi, muon_dphi, muon_dphi - 2*np.pi)
    muon_dphi = ak.where(muon_dphi >= -np.pi, muon_dphi, muon_dphi + 2*np.pi)
    muon_deta = (a['muon_eta'] - a['tau_eta'])

    a_preprocessed['muon']['dphi'] = muon_dphi
    a_preprocessed['muon']['deta'] = muon_deta
    a_preprocessed['muon']['rel_pt'] = a['muon_pt'] / a['tau_pt']
    a_preprocessed['muon']['r'] = np.sqrt(np.square(muon_deta) + np.square(muon_dphi))
    a_preprocessed['muon']['theta'] = np.arctan2(muon_dphi, muon_deta) # dphi -> y, deta -> x
    a_preprocessed['muon']['particle_type'] = 8*ak.ones_like(a['muon_pt']) # assuming PF candidate types are [0..6]

    a_preprocessed['muon']['dxy_sig'] = np.abs(a['muon_dxy'])/a['muon_dxy_error']

    muon_normalizedChi2_valid = ((a['muon_normalizedChi2'] > 0) * is_finite(a['muon_normalizedChi2'])) #Used
    a_preprocessed['muon']['normalizedChi2_valid'] = ak.values_astype(muon_normalizedChi2_valid, np.float32)
    a_preprocessed['muon']['normalizedChi2'] = ak.where(muon_normalizedChi2_valid, a['muon_normalizedChi2'], np.nan)
    a_preprocessed['muon']['numberOfValidHits'] = ak.where(muon_normalizedChi2_valid, a['muon_numberOfValidHits'], np.nan)

    a_preprocessed['muon']['rel_pfEcalEnergy'] = a['muon_pfEcalEnergy']/a['muon_pt']

    a_done = dask.compute(a_preprocessed)[0]

    # Initialize scaling data dictionary
    scaling_data = {feature_type: {} for feature_type in feature_names.keys()}

    # Create delayed computations
    delayed_computations = []
    for feature_type in feature_names.keys():
        for feature_name in feature_names[feature_type]:
            delayed_result = delayed(process_feature)(a_done[feature_type][feature_name])
            delayed_computations.append((feature_type, feature_name, delayed_result))

    # Compute all operations in parallel
    results = dask.compute(*[dc[2] for dc in delayed_computations])

    # Update scaling_data with results
    for (feature_type, feature_name, _), result in zip(delayed_computations, results):
        a_done[feature_type][feature_name], scaling_data[feature_type][feature_name] = result

    # data for labels
    label_data = dask.compute({_f: a[_f] for _f in ['sampleType', 'tauType']})[0]

    # data for gen leve matching
    gen_data = dask.compute({_f: a[_f] for _f in ['genLepton_index', 'genJet_index', 'genLepton_kind', 
                                        'tau_pt', 'tau_eta', 'tau_phi',
                                        'genLepton_vis_pt', 'genLepton_vis_eta', 'genLepton_vis_phi']})[0]

    # additional features (not used in the training)
    add_columns = dask.compute({_f: a[_f] for _f in add_feature_names})[0] if add_feature_names is not None else None

    return a_done, scaling_data, label_data, gen_data, add_columns

def compute_labels(gen_cfg, gen_data, label_data):
    # lazy compute dict with gen data
    # gen_data = {_k: _v.compute() for _k, _v in gen_data.items()}
    # convert dictionaries to numba dict
    genLepton_match_map = dict_to_numba(gen_cfg['genLepton_match_map'], key_type=types.unicode_type, value_type=types.int32)
    genLepton_kind_map = dict_to_numba(gen_cfg['genLepton_kind_map'], key_type=types.unicode_type, value_type=types.int32)
    sample_type_map = dict_to_numba(gen_cfg['sample_type_map'], key_type=types.unicode_type, value_type=types.int32)
    tau_type_map = dict_to_numba(gen_cfg['tau_type_map'], key_type=types.unicode_type, value_type=types.int32)
    # bool mask with dR gen matching
    genmatch_dR = compute_genmatch_dR(gen_data)
    is_dR_matched = genmatch_dR < gen_cfg['genmatch_dR']
    # recompute labels
    recomputed_labels = recompute_tau_type(genLepton_match_map, genLepton_kind_map, sample_type_map, tau_type_map, label_data['sampleType'], is_dR_matched, gen_data['genLepton_index'], gen_data['genJet_index'], gen_data['genLepton_kind'], gen_data['genLepton_vis_pt'])
    recomputed_labels = ak.Array(recomputed_labels)
    # check the fraction of recomputed labels comparing to the original
    if sum_:=np.sum(recomputed_labels != label_data["tauType"]):
        print(f'\n        [WARNING] non-zero fraction of recomputed tau types: {sum_/len(label_data["tauType"])*100:.1f}%\n')
    return recomputed_labels

