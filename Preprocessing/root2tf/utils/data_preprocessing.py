import uproot
import awkward as ak
import tensorflow as tf
import numpy as np

import os
import gc
import re
from XRootD import client
from XRootD.client.flags import DirListFlags, StatInfoFlags

from utils.gen_preprocessing import compute_genmatch_dR, recompute_tau_type, dict_to_numba
from numba.core import types

def _get_xrootd_filenames(prompt, verbose=False):
    client_path = re.findall("^(root://.*/)/.*$", prompt)[0]
    xrootd_client = client.FileSystem(client_path)
    if verbose:
        print(f"\nStream input files with client {client_path}\n")
    data_path = re.findall("^root://.*/(/.*$)", prompt)[0]
    if verbose:
        print(f"\nStream input from path {data_path}\n")
    if data_path.endswith(".root"):
        status, stats = xrootd_client.stat(data_path)
        if status.status != 0:
            print(f"\nStatus of {data_path} failed.\n")
            return None
        else:
            if not stats.flags & StatInfoFlags.IS_DIR:
                return [prompt]
    status, listing = xrootd_client.dirlist(data_path, DirListFlags.STAT)
    if status.status != 0:
        print(f"\nDirList of {data_path} failed.\n")
    data_files_base = [entry.name for entry in listing if re.search(".*\.root$", entry.name)]
    return [os.path.dirname(prompt) + '/' + root_file for root_file in data_files_base]

def load_from_file(file_name, tree_name, step_size):
    print(f'      - {file_name}')
    a = uproot.dask(f'{file_name}:{tree_name}', step_size=step_size, library='ak', timeout=300)
    return a

def awkward_to_tf(a, feature_names, is_ragged):
    if is_ragged:
        type_lengths = ak.count(a[feature_names[0]], axis=1)
    
    tf_array = []
    for feature_name in feature_names:
        try:
            _a = a[feature_name].compute()
        except AttributeError:
            _a = a[feature_name]
        finally:
            assert not np.any(np.isnan(_a)), f'Found NaN in {feature_name}'
            assert np.all(np.isfinite(_a)), f'Found not finite value in {feature_name}'
            if is_ragged:
                _a = ak.flatten(_a)
                _a = ak.values_astype(_a, np.float32)
                _a = tf.RaggedTensor.from_row_lengths(_a, type_lengths)
            tf_array.append(_a)
            del _a, a[feature_name]; gc.collect()
    tf_array = tf.stack(tf_array, axis=-1)
    return tf_array

def preprocess_array(a, feature_names, add_feature_names, verbose=False):
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
    a_preprocessed['global']['tau_chargedIsoPtSumdR03_over_dR05'] = ak.where((a['tau_chargedIsoPtSum']!=0).compute(), 
                                                      (a['tau_chargedIsoPtSumdR03']/a['tau_chargedIsoPtSum']).compute(), 
                                                      0)
    a_preprocessed['global']['tau_neutralIsoPtSumWeight_over_neutralIsoPtSum'] = ak.where((a['tau_neutralIsoPtSum']!=0).compute(), 
                                                                   (a['tau_neutralIsoPtSumWeight']/a['tau_neutralIsoPtSum']).compute(), 
                                                                   0)
    a_preprocessed['global']['tau_neutralIsoPtSumWeightdR03_over_neutralIsoPtSum'] = ak.where((a['tau_neutralIsoPtSum']!=0).compute(), 
                                                                       (a['tau_neutralIsoPtSumWeightdR03']/a['tau_neutralIsoPtSum']).compute(), 
                                                                       0)
    a_preprocessed['global']['tau_neutralIsoPtSumdR03_over_dR05'] = ak.where((a['tau_neutralIsoPtSum']!=0).compute(), 
                                                      (a['tau_neutralIsoPtSumdR03']/a['tau_neutralIsoPtSum']).compute(), 
                                                      0)

    a_preprocessed['global']['tau_sv_minus_pv_x'] = ak.where((a['tau_hasSecondaryVertex']).compute(), (a['tau_sv_x']-a['pv_x']).compute(), 0)
    a_preprocessed['global']['tau_sv_minus_pv_y'] = ak.where((a['tau_hasSecondaryVertex']).compute(), (a['tau_sv_y']-a['pv_y']).compute(), 0)
    a_preprocessed['global']['tau_sv_minus_pv_z'] = ak.where((a['tau_hasSecondaryVertex']).compute(), (a['tau_sv_z']-a['pv_z']).compute(), 0)

    tau_dxy_valid = ((a['tau_dxy'] > -10) & (a['tau_dxy_error'] > 0)).compute()
    a_preprocessed['global']['tau_dxy_valid'] = ak.values_astype(tau_dxy_valid, np.float32)
    a_preprocessed['global']['tau_dxy'] = ak.where(tau_dxy_valid, (a['tau_dxy']).compute(), 0)
    a_preprocessed['global']['tau_dxy_sig'] = ak.where(tau_dxy_valid, (np.abs(a['tau_dxy'])/a['tau_dxy_error']).compute(), 0)

    tau_ip3d_valid = ((a['tau_ip3d'] > -10) & (a['tau_ip3d_error'] > 0)).compute()
    a_preprocessed['global']['tau_ip3d_valid'] = ak.values_astype(tau_ip3d_valid, np.float32)
    a_preprocessed['global']['tau_ip3d'] = ak.where(tau_ip3d_valid, (a['tau_ip3d']).compute(), 0)
    a_preprocessed['global']['tau_ip3d_sig'] = ak.where(tau_ip3d_valid, (np.abs(a['tau_ip3d'])/a['tau_ip3d_error']).compute(), 0)
    
    tau_dz_sig_valid = (a['tau_dz_error'] > 0).compute()
    a_preprocessed['global']['tau_dz_sig_valid'] = ak.values_astype(tau_dz_sig_valid, np.float32)
    # a_preprocessed['global']['tau_dz'] = ak.where(tau_dz_sig_valid, (a['tau_dz']).compute(), 0)
    a_preprocessed['global']['tau_dz_sig'] = ak.where(tau_dz_sig_valid, (np.abs(a['tau_dz'])/a['tau_dz_error']).compute(), 0)
    
    tau_e_ratio_valid = (a['tau_e_ratio'] > 0).compute()
    a_preprocessed['global']['tau_e_ratio_valid'] = ak.values_astype(tau_e_ratio_valid, np.float32)
    a_preprocessed['global']['tau_e_ratio'] = ak.where(tau_e_ratio_valid, (a['tau_e_ratio']).compute(), 0)
    
    tau_gj_angle_diff_valid = (a['tau_gj_angle_diff'] >= 0).compute()
    a_preprocessed['global']['tau_gj_angle_diff_valid'] = ak.values_astype(tau_gj_angle_diff_valid, np.float32)
    a_preprocessed['global']['tau_gj_angle_diff'] = ak.where(tau_gj_angle_diff_valid, (a['tau_gj_angle_diff']).compute(), -1)

    a_preprocessed['global']['tau_leadChargedCand_etaAtEcalEntrance_minus_tau_eta'] = a['tau_leadChargedCand_etaAtEcalEntrance'] - a['tau_eta']
    a_preprocessed['global']['particle_type'] = 9*ak.ones_like(a['tau_pt'].compute()) # assign unique particle type to a "global" token 

    # ------- PF candidates ------- #

    # shift delta phi into [-pi, pi] range 
    pf_dphi = (a['pfCand_phi'] - a['tau_phi']).compute()
    pf_dphi = np.where(pf_dphi <= np.pi, pf_dphi, pf_dphi - 2*np.pi)
    pf_dphi = np.where(pf_dphi >= -np.pi, pf_dphi, pf_dphi + 2*np.pi)
    pf_deta = (a['pfCand_eta'] - a['tau_eta']).compute()

    a_preprocessed['pfCand']['dphi'] = pf_dphi
    a_preprocessed['pfCand']['deta'] = pf_deta
    a_preprocessed['pfCand']['rel_pt'] = a['pfCand_pt'] / a['tau_pt']
    a_preprocessed['pfCand']['r'] = np.sqrt(np.square(pf_deta) + np.square(pf_dphi))
    a_preprocessed['pfCand']['theta'] = np.arctan2(pf_dphi, pf_deta) # dphi -> y, deta -> x
    a_preprocessed['pfCand']['particle_type'] = a['pfCand_particleType'] - 1

    vertex_z_valid = np.isfinite( a['pfCand_vertex_z']).compute()
    a_preprocessed['pfCand']['vertex_dx'] = a['pfCand_vertex_x'] - a['pv_x']
    a_preprocessed['pfCand']['vertex_dy'] = a['pfCand_vertex_y'] - a['pv_y']
    a_preprocessed['pfCand']['vertex_dz'] = ak.where(vertex_z_valid, (a['pfCand_vertex_z'] - a['pv_z']).compute(), -10)
    a_preprocessed['pfCand']['vertex_dx_tauFL'] = a['pfCand_vertex_x'] - a['pv_x'] - a['tau_flightLength_x']
    a_preprocessed['pfCand']['vertex_dy_tauFL'] = a['pfCand_vertex_y'] - a['pv_y'] - a['tau_flightLength_y']
    a_preprocessed['pfCand']['vertex_dz_tauFL'] = ak.where(vertex_z_valid, (a['pfCand_vertex_z'] - a['pv_z'] - a['tau_flightLength_z']).compute(), -10)

    has_track_details = (a['pfCand_hasTrackDetails'] == 1).compute()
    has_track_details_track_ndof = has_track_details * (a['pfCand_track_ndof'] > 0).compute()
    has_track_details_dxy_finite = has_track_details * np.isfinite(a['pfCand_dxy']).compute()
    has_track_details_dxy_sig_finite = has_track_details * np.isfinite(np.abs(a['pfCand_dxy'])/a['pfCand_dxy_error']).compute()
    has_track_details_dz_finite = has_track_details * np.isfinite(a['pfCand_dz']).compute()
    has_track_details_dz_sig_finite = has_track_details * np.isfinite(np.abs(a['pfCand_dz'])/a['pfCand_dz_error']).compute()
    a_preprocessed['pfCand']['dxy'] = ak.where(has_track_details_dxy_finite, a['pfCand_dxy'].compute(), 0)
    a_preprocessed['pfCand']['dxy_sig'] = ak.where(has_track_details_dxy_sig_finite, (np.abs(a['pfCand_dxy'])/a['pfCand_dxy_error']).compute(), 0)
    a_preprocessed['pfCand']['dz'] = ak.where(has_track_details_dz_finite, a['pfCand_dz'].compute(), 0)
    a_preprocessed['pfCand']['dz_sig'] = ak.where(has_track_details_dz_sig_finite, (np.abs(a['pfCand_dz'])/a['pfCand_dz_error']).compute(), 0)
    a_preprocessed['pfCand']['track_ndof'] = ak.where(has_track_details_track_ndof, a['pfCand_track_ndof'].compute(), 0)
    a_preprocessed['pfCand']['chi2_ndof'] = ak.where(has_track_details_track_ndof, (a['pfCand_track_chi2']/a['pfCand_track_ndof']).compute(), 0)

    # ------- Electrons ------- #
    
    # shift delta phi into [-pi, pi] range 
    ele_dphi = (a['ele_phi'] - a['tau_phi']).compute()
    ele_dphi = np.where(ele_dphi <= np.pi, ele_dphi, ele_dphi - 2*np.pi)
    ele_dphi = np.where(ele_dphi >= -np.pi, ele_dphi, ele_dphi + 2*np.pi)
    ele_deta = (a['ele_eta'] - a['tau_eta']).compute()

    a_preprocessed['ele']['dphi'] = ele_dphi
    a_preprocessed['ele']['deta'] = ele_deta
    a_preprocessed['ele']['rel_pt'] = a['ele_pt'] / a['tau_pt']
    a_preprocessed['ele']['r'] = np.sqrt(np.square(ele_deta) + np.square(ele_dphi))
    a_preprocessed['ele']['theta'] = np.arctan2(ele_dphi, ele_deta) # dphi -> y, deta -> x
    a_preprocessed['ele']['particle_type'] = 7*ak.ones_like(a['ele_pt'].compute()) # assuming PF candidate types are [0..6]

    ele_cc_valid = (a['ele_cc_ele_energy'] >= 0).compute()
    a_preprocessed['ele']['cc_valid'] = ak.values_astype(ele_cc_valid, np.float32)
    a_preprocessed['ele']['cc_ele_rel_energy'] = ak.where(ele_cc_valid, (a['ele_cc_ele_energy']/a['ele_pt']).compute(), 0)
    a_preprocessed['ele']['cc_gamma_rel_energy'] = ak.where(ele_cc_valid, (a['ele_cc_gamma_energy']/a['ele_cc_ele_energy']).compute(), 0)
    a_preprocessed['ele']['cc_n_gamma'] = ak.where(ele_cc_valid, a['ele_cc_n_gamma'].compute(), -1)
    a_preprocessed['ele']['rel_trackMomentumAtVtx'] = a['ele_trackMomentumAtVtx']/a['ele_pt']
    a_preprocessed['ele']['rel_trackMomentumAtCalo'] = a['ele_trackMomentumAtCalo']/a['ele_pt']
    a_preprocessed['ele']['rel_trackMomentumOut'] = a['ele_trackMomentumOut']/a['ele_pt']
    a_preprocessed['ele']['rel_trackMomentumAtEleClus'] = a['ele_trackMomentumAtEleClus']/a['ele_pt']
    a_preprocessed['ele']['rel_trackMomentumAtVtxWithConstraint'] = a['ele_trackMomentumAtVtxWithConstraint']/a['ele_pt']
    a_preprocessed['ele']['rel_ecalEnergy'] = a['ele_ecalEnergy']/a['ele_pt']
    a_preprocessed['ele']['ecalEnergy_sig'] = a['ele_ecalEnergy']/a['ele_ecalEnergy_error']
    a_preprocessed['ele']['rel_gsfTrack_pt'] = a['ele_gsfTrack_pt']/a['ele_pt']
    a_preprocessed['ele']['gsfTrack_pt_sig'] = a['ele_gsfTrack_pt']/a['ele_gsfTrack_pt_error']

    ele_has_closestCtfTrack = (a['ele_closestCtfTrack_normalizedChi2'] >= 0).compute()
    a_preprocessed['ele']['has_closestCtfTrack'] = ak.values_astype(ele_has_closestCtfTrack, np.float32)
    a_preprocessed['ele']['closestCtfTrack_normalizedChi2'] = ak.where(ele_has_closestCtfTrack, a['ele_closestCtfTrack_normalizedChi2'].compute(), 0)
    a_preprocessed['ele']['closestCtfTrack_numberOfValidHits'] = ak.where(ele_has_closestCtfTrack, a['ele_closestCtfTrack_numberOfValidHits'].compute(), 0)

    ele_features = ['sigmaEtaEta', 'sigmaIetaIeta', 'sigmaIphiIphi', 'sigmaIetaIphi',
                    'e1x5', 'e2x5Max', 'e5x5', 'r9', 
                    'hcalDepth1OverEcal', 'hcalDepth2OverEcal', 'hcalDepth1OverEcalBc', 'hcalDepth2OverEcalBc',
                    'eLeft', 'eRight', 'eBottom', 'eTop',
                    'full5x5_sigmaEtaEta', 'full5x5_sigmaIetaIeta', 'full5x5_sigmaIphiIphi', 'full5x5_sigmaIetaIphi',
                    'full5x5_e1x5', 'full5x5_e2x5Max', 'full5x5_e5x5', 'full5x5_r9',
                    'full5x5_hcalDepth1OverEcal', 'full5x5_hcalDepth2OverEcal', 'full5x5_hcalDepth1OverEcalBc', 'full5x5_hcalDepth2OverEcalBc',
                    'full5x5_eLeft', 'full5x5_eRight', 'full5x5_eBottom', 'full5x5_eTop',
                    'full5x5_e2x5Left', 'full5x5_e2x5Right', 'full5x5_e2x5Bottom', 'full5x5_e2x5Top']
    for ele_feature in ele_features:
        _a = a[f'ele_{ele_feature}'].compute()
        a_preprocessed['ele'][ele_feature] = ak.where(_a > -1, _a, -1)

    # ------- Muons ------- #

    # shift delta phi into [-pi, pi] range 
    muon_dphi = (a['muon_phi'] - a['tau_phi']).compute()
    muon_dphi = np.where(muon_dphi <= np.pi, muon_dphi, muon_dphi - 2*np.pi)
    muon_dphi = np.where(muon_dphi >= -np.pi, muon_dphi, muon_dphi + 2*np.pi)
    muon_deta = (a['muon_eta'] - a['tau_eta']).compute()

    a_preprocessed['muon']['dphi'] = muon_dphi
    a_preprocessed['muon']['deta'] = muon_deta
    a_preprocessed['muon']['rel_pt'] = a['muon_pt'] / a['tau_pt']
    a_preprocessed['muon']['r'] = np.sqrt(np.square(muon_deta) + np.square(muon_dphi))
    a_preprocessed['muon']['theta'] = np.arctan2(muon_dphi, muon_deta) # dphi -> y, deta -> x
    a_preprocessed['muon']['particle_type'] = 8*ak.ones_like(a['muon_pt'].compute()) # assuming PF candidate types are [0..6]

    muon_dxy_sig_finite = np.isfinite(np.abs(a['muon_dxy'])/a['muon_dxy_error']).compute()
    a_preprocessed['muon']['dxy_sig'] = ak.where(muon_dxy_sig_finite, (np.abs(a['muon_dxy'])/a['muon_dxy_error']).compute(), 0)

    muon_normalizedChi2_valid = ((a['muon_normalizedChi2'] > 0) * np.isfinite(a['muon_normalizedChi2'])).compute()
    a_preprocessed['muon']['normalizedChi2_valid'] = ak.values_astype(muon_normalizedChi2_valid, np.float32)
    a_preprocessed['muon']['normalizedChi2'] = ak.where(muon_normalizedChi2_valid, a['muon_normalizedChi2'].compute(), 0)
    a_preprocessed['muon']['numberOfValidHits'] = ak.where(muon_normalizedChi2_valid, a['muon_numberOfValidHits'].compute(), 0)
    
    muon_pfEcalEnergy_valid = (a['muon_pfEcalEnergy'] >= 0).compute()
    a_preprocessed['muon']['pfEcalEnergy_valid'] = ak.values_astype(muon_pfEcalEnergy_valid, np.float32)
    a_preprocessed['muon']['rel_pfEcalEnergy'] = ak.where(muon_pfEcalEnergy_valid, (a['muon_pfEcalEnergy']/a['muon_pt']).compute(), 0)

    # data for labels
    label_data = {_f: a[_f].compute() for _f in ['sampleType', 'tauType']}

    # data for gen leve matching
    gen_data = {_f: a[_f] for _f in ['genLepton_index', 'genJet_index', 'genLepton_kind', 
                                     'tau_pt', 'tau_eta', 'tau_phi',
                                     'genLepton_vis_pt', 'genLepton_vis_eta', 'genLepton_vis_phi']} # will be computed on demand later

    # additional features (not used in the training)
    add_columns = {_f: a[_f] for _f in add_feature_names} if add_feature_names is not None else None

    return a_preprocessed, label_data, gen_data, add_columns

def compute_labels(gen_cfg, gen_data, label_data):
    # lazy compute dict with gen data
    gen_data = {_k: _v.compute() for _k, _v in gen_data.items()}
    
    # convert dictionaries to numba dict
    genLepton_match_map = dict_to_numba(gen_cfg['genLepton_match_map'], key_type=types.unicode_type, value_type=types.int32)
    genLepton_kind_map = dict_to_numba(gen_cfg['genLepton_kind_map'], key_type=types.unicode_type, value_type=types.int32)
    sample_type_map = dict_to_numba(gen_cfg['sample_type_map'], key_type=types.unicode_type, value_type=types.int32)
    tau_type_map = dict_to_numba(gen_cfg['tau_type_map'], key_type=types.unicode_type, value_type=types.int32)
    
    # bool mask with dR gen matching
    genmatch_dR = compute_genmatch_dR(gen_data)
    is_dR_matched = genmatch_dR < gen_cfg['genmatch_dR']

    # recompute labels
    recomputed_labels = recompute_tau_type(genLepton_match_map, genLepton_kind_map, sample_type_map, tau_type_map,
                                                label_data['sampleType'], is_dR_matched,
                                                gen_data['genLepton_index'], gen_data['genJet_index'], gen_data['genLepton_kind'], gen_data['genLepton_vis_pt'])
    recomputed_labels = ak.Array(recomputed_labels)

    # check the fraction of recomputed labels comparing to the original
    if sum_:=np.sum(recomputed_labels != label_data["tauType"]):
        print(f'\n        [WARNING] non-zero fraction of recomputed tau types: {sum_/len(label_data["tauType"])*100:.1f}%\n')
    
    return recomputed_labels

