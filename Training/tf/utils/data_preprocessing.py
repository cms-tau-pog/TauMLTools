import uproot
import awkward as ak
import tensorflow as tf
import numpy as np
from hydra.utils import to_absolute_path

def load_from_file(file_name, tree_name, input_branches):
    print(f'      - {file_name}')
    
    # open ROOT file and retireve awkward arrays
    with uproot.open(to_absolute_path(file_name)) as f:
        a = f[tree_name].arrays(input_branches, how='zip')

    return a

def awkward_to_tf(a, particle_type, feature_names):
    if particle_type == 'global':
        # tf_array = np.squeeze(ak.to_numpy(a[feature_names]))
        # tf_array = tf.constant(tf_array) # will return pd.DataFrame, convertion to TF happens at `tf.data.Dataset.from_tensor_slices(data)`` step
        tf_array = np.squeeze(ak.to_pandas(a[feature_names]).values)
        tf_array = tf.constant(tf_array) 
    else:
        pf_lengths = ak.count(a[particle_type, feature_names[0]], axis=1)
        ragged_pf_features = []
        for feature in feature_names:
            pf_a = ak.flatten(a[particle_type, feature])
            pf_a = ak.values_astype(pf_a, np.float32)
            ragged_pf_features.append(tf.RaggedTensor.from_row_lengths(pf_a, pf_lengths))
        tf_array = tf.stack(ragged_pf_features, axis=-1)
    return tf_array

def preprocess_array(a):
    # remove taus with abnormal phi
    a = a[np.abs(a['tau_phi'])<2.*np.pi] 

    # ------- Global features ------- #

    a['tau_E_over_pt'] = np.sqrt((a['tau_pt']*np.cosh(a['tau_eta']))*(a['tau_pt']*np.cosh(a['tau_eta'])) + a['tau_mass']*a['tau_mass'])/a['tau_pt']
    a['tau_n_charged_prongs'] = a['tau_decayMode']//5 + 1
    a['tau_n_neutral_prongs'] = a['tau_decayMode']%5
    a['tau_chargedIsoPtSumdR03_over_dR05'] = ak.where(a['tau_chargedIsoPtSum']!=0, 
                                                      a['tau_chargedIsoPtSumdR03']/a['tau_chargedIsoPtSum'], 
                                                      0)
    a['tau_neutralIsoPtSumWeight_over_neutralIsoPtSum'] = ak.where(a['tau_neutralIsoPtSum']!=0, 
                                                                   a['tau_neutralIsoPtSumWeight']/a['tau_neutralIsoPtSum'], 
                                                                   0)
    a['tau_neutralIsoPtSumWeightdR03_over_neutralIsoPtSum'] = ak.where(a['tau_neutralIsoPtSum']!=0, 
                                                                       a['tau_neutralIsoPtSumWeightdR03']/a['tau_neutralIsoPtSum'], 
                                                                       0)
    a['tau_neutralIsoPtSumdR03_over_dR05'] = ak.where(a['tau_neutralIsoPtSum']!=0, 
                                                      a['tau_neutralIsoPtSumdR03']/a['tau_neutralIsoPtSum'], 
                                                      0)

    a['tau_sv_minus_pv_x'] = ak.where(a['tau_hasSecondaryVertex'], a['tau_sv_x']-a['pv_x'], 0)
    a['tau_sv_minus_pv_y'] = ak.where(a['tau_hasSecondaryVertex'], a['tau_sv_y']-a['pv_y'], 0)
    a['tau_sv_minus_pv_z'] = ak.where(a['tau_hasSecondaryVertex'], a['tau_sv_z']-a['pv_z'], 0)

    tau_dxy_valid = (a['tau_dxy'] > -10) & (a['tau_dxy_error'] > 0)
    a['tau_dxy_valid'] = ak.values_astype(tau_dxy_valid, np.float32)
    a['tau_dxy'] = ak.where(tau_dxy_valid, a['tau_dxy'], 0)
    a['tau_dxy_sig'] = ak.where(tau_dxy_valid, np.abs(a['tau_dxy'])/a['tau_dxy_error'], 0)

    tau_ip3d_valid = (a['tau_ip3d'] > -10) & (a['tau_ip3d_error'] > 0)
    a['tau_ip3d_valid'] = ak.values_astype(tau_ip3d_valid, np.float32)
    a['tau_ip3d'] = ak.where(tau_ip3d_valid, a['tau_ip3d'], 0)
    a['tau_ip3d_sig'] = ak.where(tau_ip3d_valid, np.abs(a['tau_ip3d'])/a['tau_ip3d_error'], 0)
    
    tau_dz_sig_valid = a['tau_dz_error'] > 0
    a['tau_dz_sig_valid'] = ak.values_astype(tau_dz_sig_valid, np.float32)
    # a['tau_dz'] = ak.where(tau_dz_sig_valid, a['tau_dz'], 0)
    a['tau_dz_sig'] = ak.where(tau_dz_sig_valid, np.abs(a['tau_dz'])/a['tau_dz_error'], 0)
    
    tau_e_ratio_valid = a['tau_e_ratio'] > 0
    a['tau_e_ratio_valid'] = ak.values_astype(tau_e_ratio_valid, np.float32)
    a['tau_e_ratio'] = ak.where(tau_e_ratio_valid, a['tau_e_ratio'], 0)
    
    tau_gj_angle_diff_valid = a['tau_gj_angle_diff'] >= 0
    a['tau_gj_angle_diff_valid'] = ak.values_astype(tau_gj_angle_diff_valid, np.float32)
    a['tau_gj_angle_diff'] = ak.where(tau_gj_angle_diff_valid, a['tau_gj_angle_diff'], -1)

    a['tau_leadChargedCand_etaAtEcalEntrance_minus_tau_eta'] = a['tau_leadChargedCand_etaAtEcalEntrance'] - a['tau_eta']
    a['particle_type'] = 9 # assign unique particle type to a "global" token 

    # ------- PF candidates ------- #

    # shift delta phi into [-pi, pi] range 
    dphi_array = (a['pfCand', 'phi'] - a['tau_phi'])
    dphi_array = np.where(dphi_array <= np.pi, dphi_array, dphi_array - 2*np.pi)
    dphi_array = np.where(dphi_array >= -np.pi, dphi_array, dphi_array + 2*np.pi)

    # add features
    a['pfCand', 'dphi'] = dphi_array
    a['pfCand', 'deta'] = a['pfCand', 'eta'] - a['tau_eta']
    a['pfCand', 'rel_pt'] = a['pfCand', 'pt'] / a['tau_pt']
    a['pfCand', 'r'] = np.sqrt(np.square(a['pfCand', 'deta']) + np.square(a['pfCand', 'dphi']))
    a['pfCand', 'theta'] = np.arctan2(a['pfCand', 'dphi'], a['pfCand', 'deta']) # dphi -> y, deta -> x
    a['pfCand', 'particle_type'] = a['pfCand', 'particleType'] - 1

    # vertices
    a['pfCand', 'vertex_dx'] = a['pfCand', 'vertex_x'] - a['pv_x']
    a['pfCand', 'vertex_dy'] = a['pfCand', 'vertex_y'] - a['pv_y']
    a['pfCand', 'vertex_dz'] = a['pfCand', 'vertex_z'] - a['pv_z']
    a['pfCand', 'vertex_dx_tauFL'] = a['pfCand', 'vertex_x'] - a['pv_x'] - a['tau_flightLength_x']
    a['pfCand', 'vertex_dy_tauFL'] = a['pfCand', 'vertex_y'] - a['pv_y'] - a['tau_flightLength_y']
    a['pfCand', 'vertex_dz_tauFL'] = a['pfCand', 'vertex_z'] - a['pv_z'] - a['tau_flightLength_z']

    # IP, track info
    has_track_details = a['pfCand', 'hasTrackDetails'] == 1
    has_track_details_track_ndof = has_track_details * (a['pfCand', 'track_ndof'] > 0)
    a['pfCand', 'dxy'] = ak.where(has_track_details, a['pfCand', 'dxy'], 0)
    a['pfCand', 'dxy_sig'] = ak.where(has_track_details, np.abs(a['pfCand', 'dxy'])/a['pfCand', 'dxy_error'], 0)
    a['pfCand', 'dz'] = ak.where(has_track_details, a['pfCand', 'dz'], 0)
    a['pfCand', 'dz_sig'] = ak.where(has_track_details, np.abs(a['pfCand', 'dz'])/a['pfCand', 'dz_error'], 0)
    a['pfCand', 'track_ndof'] = ak.where(has_track_details_track_ndof, a['pfCand', 'track_ndof'], 0)
    a['pfCand', 'chi2_ndof'] = ak.where(has_track_details_track_ndof, a['pfCand', 'track_chi2']/a['pfCand', 'track_ndof'], 0)

    # ------- Electrons ------- #
    
    # shift delta phi into [-pi, pi] range 
    dphi_array = (a['ele', 'phi'] - a['tau_phi'])
    dphi_array = np.where(dphi_array <= np.pi, dphi_array, dphi_array - 2*np.pi)
    dphi_array = np.where(dphi_array >= -np.pi, dphi_array, dphi_array + 2*np.pi)

    a['ele', 'dphi'] = dphi_array
    a['ele', 'deta'] = a['ele', 'eta'] - a['tau_eta']
    a['ele', 'rel_pt'] = a['ele', 'pt'] / a['tau_pt']
    a['ele', 'r'] = np.sqrt(np.square(a['ele', 'deta']) + np.square(a['ele', 'dphi']))
    a['ele', 'theta'] = np.arctan2(a['ele', 'dphi'], a['ele', 'deta']) # dphi -> y, deta -> x
    a['ele', 'particle_type'] = 7 # assuming PF candidate types are [0..6]

    ele_cc_valid = a['ele', 'cc_ele_energy'] >= 0
    a['ele', 'cc_valid'] = ak.values_astype(ele_cc_valid, np.float32)
    a['ele', 'cc_ele_rel_energy'] = ak.where(ele_cc_valid, a['ele', 'cc_ele_energy']/a['ele', 'pt'], 0)
    a['ele', 'cc_gamma_rel_energy'] = ak.where(ele_cc_valid, a['ele', 'cc_gamma_energy']/a['ele', 'cc_ele_energy'], 0)
    a['ele', 'cc_n_gamma'] = ak.where(ele_cc_valid, a['ele', 'cc_n_gamma'], -1)
    a['ele', 'rel_trackMomentumAtVtx'] = a['ele', 'trackMomentumAtVtx']/a['ele', 'pt']
    a['ele', 'rel_trackMomentumAtCalo'] = a['ele', 'trackMomentumAtCalo']/a['ele', 'pt']
    a['ele', 'rel_trackMomentumOut'] = a['ele', 'trackMomentumOut']/a['ele', 'pt']
    a['ele', 'rel_trackMomentumAtEleClus'] = a['ele', 'trackMomentumAtEleClus']/a['ele', 'pt']
    a['ele', 'rel_trackMomentumAtVtxWithConstraint'] = a['ele', 'trackMomentumAtVtxWithConstraint']/a['ele', 'pt']
    a['ele', 'rel_ecalEnergy'] = a['ele', 'ecalEnergy']/a['ele', 'pt']
    a['ele', 'ecalEnergy_sig'] = a['ele', 'ecalEnergy']/a['ele', 'ecalEnergy_error']
    a['ele', 'rel_gsfTrack_pt'] = a['ele', 'gsfTrack_pt']/a['ele', 'pt']
    a['ele', 'gsfTrack_pt_sig'] = a['ele', 'gsfTrack_pt']/a['ele', 'gsfTrack_pt_error']

    ele_has_closestCtfTrack = a['ele', 'closestCtfTrack_normalizedChi2'] >= 0
    a['ele', 'has_closestCtfTrack'] = ak.values_astype(ele_has_closestCtfTrack, np.float32)
    a['ele', 'closestCtfTrack_normalizedChi2'] = ak.where(ele_has_closestCtfTrack, a['ele', 'closestCtfTrack_normalizedChi2'], 0)
    a['ele', 'closestCtfTrack_numberOfValidHits'] = ak.where(ele_has_closestCtfTrack, a['ele', 'closestCtfTrack_numberOfValidHits'], 0)

    a['ele', 'sigmaEtaEta'] = ak.where(a['ele', 'sigmaEtaEta']>-1, a['ele', 'sigmaEtaEta'], -1)
    a['ele', 'sigmaIetaIeta'] = ak.where(a['ele', 'sigmaIetaIeta']>-1, a['ele', 'sigmaIetaIeta'], -1)
    a['ele', 'sigmaIphiIphi'] = ak.where(a['ele', 'sigmaIphiIphi']>-1, a['ele', 'sigmaIphiIphi'], -1)
    a['ele', 'sigmaIetaIphi'] = ak.where(a['ele', 'sigmaIetaIphi']>-1, a['ele', 'sigmaIetaIphi'], -1)

    a['ele', 'e1x5'] = ak.where(a['ele', 'e1x5']>-1, a['ele', 'e1x5'], -1)
    a['ele', 'e2x5Max'] = ak.where(a['ele', 'e2x5Max']>-1, a['ele', 'e2x5Max'], -1)
    a['ele', 'e5x5'] = ak.where(a['ele', 'e5x5']>-1, a['ele', 'e5x5'], -1)
    a['ele', 'r9'] = ak.where(a['ele', 'r9']>-1, a['ele', 'r9'], -1)

    a['ele', 'hcalDepth1OverEcal'] = ak.where(a['ele', 'hcalDepth1OverEcal']>-1, a['ele', 'hcalDepth1OverEcal'], -1)
    a['ele', 'hcalDepth2OverEcal'] = ak.where(a['ele', 'hcalDepth2OverEcal']>-1, a['ele', 'hcalDepth2OverEcal'], -1)
    a['ele', 'hcalDepth1OverEcalBc'] = ak.where(a['ele', 'hcalDepth1OverEcalBc']>-1, a['ele', 'hcalDepth1OverEcalBc'], -1)
    a['ele', 'hcalDepth2OverEcalBc'] = ak.where(a['ele', 'hcalDepth2OverEcalBc']>-1, a['ele', 'hcalDepth2OverEcalBc'], -1)
    
    a['ele', 'eLeft'] = ak.where(a['ele', 'eLeft']>-1, a['ele', 'eLeft'], -1)
    a['ele', 'eRight'] = ak.where(a['ele', 'eRight']>-1, a['ele', 'eRight'], -1)
    a['ele', 'eBottom'] = ak.where(a['ele', 'eBottom']>-1, a['ele', 'eBottom'], -1)
    a['ele', 'eTop'] = ak.where(a['ele', 'eTop']>-1, a['ele', 'eTop'], -1)
    
    a['ele', 'full5x5_sigmaEtaEta'] = ak.where(a['ele', 'full5x5_sigmaEtaEta']>-1, a['ele', 'full5x5_sigmaEtaEta'], -1)
    a['ele', 'full5x5_sigmaIetaIeta'] = ak.where(a['ele', 'full5x5_sigmaIetaIeta']>-1, a['ele', 'full5x5_sigmaIetaIeta'], -1)
    a['ele', 'full5x5_sigmaIphiIphi'] = ak.where(a['ele', 'full5x5_sigmaIphiIphi']>-1, a['ele', 'full5x5_sigmaIphiIphi'], -1)
    a['ele', 'full5x5_sigmaIetaIphi'] = ak.where(a['ele', 'full5x5_sigmaIetaIphi']>-1, a['ele', 'full5x5_sigmaIetaIphi'], -1)
    
    a['ele', 'full5x5_e1x5'] = ak.where(a['ele', 'full5x5_e1x5']>-1, a['ele', 'full5x5_e1x5'], -1)
    a['ele', 'full5x5_e2x5Max'] = ak.where(a['ele', 'full5x5_e2x5Max']>-1, a['ele', 'full5x5_e2x5Max'], -1)
    a['ele', 'full5x5_e5x5'] = ak.where(a['ele', 'full5x5_e5x5']>-1, a['ele', 'full5x5_e5x5'], -1)
    a['ele', 'full5x5_r9'] = ak.where(a['ele', 'full5x5_r9']>-1, a['ele', 'full5x5_r9'], -1)
    
    a['ele', 'full5x5_hcalDepth1OverEcal'] = ak.where(a['ele', 'full5x5_hcalDepth1OverEcal']>-1, a['ele', 'full5x5_hcalDepth1OverEcal'], -1)
    a['ele', 'full5x5_hcalDepth2OverEcal'] = ak.where(a['ele', 'full5x5_hcalDepth2OverEcal']>-1, a['ele', 'full5x5_hcalDepth2OverEcal'], -1)
    a['ele', 'full5x5_hcalDepth1OverEcalBc'] = ak.where(a['ele', 'full5x5_hcalDepth1OverEcalBc']>-1, a['ele', 'full5x5_hcalDepth1OverEcalBc'], -1)
    a['ele', 'full5x5_hcalDepth2OverEcalBc'] = ak.where(a['ele', 'full5x5_hcalDepth2OverEcalBc']>-1, a['ele', 'full5x5_hcalDepth2OverEcalBc'], -1)
    
    a['ele', 'full5x5_eLeft'] = ak.where(a['ele', 'full5x5_eLeft']>-1, a['ele', 'full5x5_eLeft'], -1)
    a['ele', 'full5x5_eRight'] = ak.where(a['ele', 'full5x5_eRight']>-1, a['ele', 'full5x5_eRight'], -1)
    a['ele', 'full5x5_eBottom'] = ak.where(a['ele', 'full5x5_eBottom']>-1, a['ele', 'full5x5_eBottom'], -1)
    a['ele', 'full5x5_eTop'] = ak.where(a['ele', 'full5x5_eTop']>-1, a['ele', 'full5x5_eTop'], -1)
    
    a['ele', 'full5x5_e2x5Left'] = ak.where(a['ele', 'full5x5_e2x5Left']>-1, a['ele', 'full5x5_e2x5Left'], -1)
    a['ele', 'full5x5_e2x5Right'] = ak.where(a['ele', 'full5x5_e2x5Right']>-1, a['ele', 'full5x5_e2x5Right'], -1)
    a['ele', 'full5x5_e2x5Bottom'] = ak.where(a['ele', 'full5x5_e2x5Bottom']>-1, a['ele', 'full5x5_e2x5Bottom'], -1)
    a['ele', 'full5x5_e2x5Top'] = ak.where(a['ele', 'full5x5_e2x5Top']>-1, a['ele', 'full5x5_e2x5Top'], -1)

    # ------- Muons ------- #

    # shift delta phi into [-pi, pi] range 
    dphi_array = (a['muon', 'phi'] - a['tau_phi'])
    dphi_array = np.where(dphi_array <= np.pi, dphi_array, dphi_array - 2*np.pi)
    dphi_array = np.where(dphi_array >= -np.pi, dphi_array, dphi_array + 2*np.pi)

    a['muon', 'dphi'] = dphi_array
    a['muon', 'deta'] = a['muon', 'eta'] - a['tau_eta']
    a['muon', 'rel_pt'] = a['muon', 'pt'] / a['tau_pt']
    a['muon', 'r'] = np.sqrt(np.square(a['muon', 'deta']) + np.square(a['muon', 'dphi']))
    a['muon', 'theta'] = np.arctan2(a['muon', 'dphi'], a['muon', 'deta']) # dphi -> y, deta -> x
    a['muon', 'particle_type'] = 8 # assuming PF candidate types are [0..6]

    a['muon', 'dxy_sig'] =  np.abs(a['muon', 'dxy'])/a['muon', 'dxy_error'] 

    muon_normalizedChi2_valid = a['muon', 'normalizedChi2'] >= 0
    a['muon', 'normalizedChi2_valid'] = ak.values_astype(muon_normalizedChi2_valid, np.float32)
    a['muon', 'normalizedChi2'] = ak.where(muon_normalizedChi2_valid, a['muon', 'normalizedChi2'], 0)
    a['muon', 'numberOfValidHits'] = ak.where(muon_normalizedChi2_valid, a['muon', 'numberOfValidHits'], 0)
    
    muon_pfEcalEnergy_valid = a['muon', 'pfEcalEnergy'] >= 0
    a['muon', 'pfEcalEnergy_valid'] = ak.values_astype(muon_pfEcalEnergy_valid, np.float32)
    a['muon', 'rel_pfEcalEnergy'] = ak.where(muon_pfEcalEnergy_valid, a['muon', 'pfEcalEnergy']/a['muon', 'pt'], 0)

    # preprocess NaNs
    a = ak.nan_to_num(a, nan=0., posinf=0., neginf=0.)

    return a 