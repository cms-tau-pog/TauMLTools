import uproot
import pandas
import numpy as np
import tensorflow as tf

truth_branches = [ 'gen_e', 'gen_mu', 'gen_tau', 'gen_jet' ]
weight_branches = [ 'trainingWeight' ]
navigation_branches = [ 'innerCells_begin', 'innerCells_end', 'outerCells_begin', 'outerCells_end']
tau_id_branches = [ 'againstElectronMVA6', 'againstElectronMVA6raw', 'againstElectronMVA62018',
                    'againstElectronMVA62018raw', 'againstMuon3', 'againstMuon3raw',
                    'byCombinedIsolationDeltaBetaCorr3Hits', 'byCombinedIsolationDeltaBetaCorr3Hitsraw',
                    'byIsolationMVArun2v1DBoldDMwLT2016', 'byIsolationMVArun2v1DBoldDMwLT2016raw',
                    'byIsolationMVArun2v1DBnewDMwLT2016', 'byIsolationMVArun2v1DBnewDMwLT2016raw',
                    'byIsolationMVArun2017v2DBoldDMwLT2017', 'byIsolationMVArun2017v2DBoldDMwLT2017raw',
                    'byIsolationMVArun2017v2DBoldDMdR0p3wLT2017', 'byIsolationMVArun2017v2DBoldDMdR0p3wLT2017raw',
                    'byIsolationMVArun2017v2DBnewDMwLT2017', 'byIsolationMVArun2017v2DBnewDMwLT2017raw',
                    'byDeepTau2017v1VSe', 'byDeepTau2017v1VSeraw', 'byDeepTau2017v1VSmu', 'byDeepTau2017v1VSmuraw',
                    'byDeepTau2017v1VSjet', 'byDeepTau2017v1VSjetraw', 'byDpfTau2016v0VSall', 'byDpfTau2016v0VSallraw' ]
global_event_branches = [ 'rho', 'pv_x', 'pv_y', 'pv_z', 'pv_chi2', 'pv_ndof' ]

input_tau_signal_branches = [ 'tau_pt', 'tau_eta', 'tau_mass', 'tau_charge', 'tau_decayMode',
                              'tau_dxy_pca_x', 'tau_dxy_pca_y', 'tau_dxy_pca_z', 'tau_dxy', 'tau_dxy_error',
                              'tau_ip3d', 'tau_ip3d_error', 'tau_dz', 'tau_dz_error',
                              'tau_hasSecondaryVertex', 'tau_sv_x', 'tau_sv_y', 'tau_sv_z',
                              'tau_flightLength_x', 'tau_flightLength_y', 'tau_flightLength_z', 'tau_flightLength_sig',
                              'tau_pt_weighted_deta_strip', 'tau_pt_weighted_dphi_strip', 'tau_pt_weighted_dr_signal',
                              'tau_leadingTrackNormChi2', 'tau_e_ratio', 'tau_gj_angle_diff', 'tau_n_photons',
                              'tau_emFraction', 'tau_inside_ecal_crack', 'leadChargedCand_etaAtEcalEntrance' ]
input_tau_iso_branches = [ 'chargedIsoPtSum', 'chargedIsoPtSumdR03', 'footprintCorrection', 'footprintCorrectiondR03',
                           'neutralIsoPtSum', 'neutralIsoPtSumWeight', 'neutralIsoPtSumWeightdR03',
                           'neutralIsoPtSumdR03', 'photonPtSumOutsideSignalCone', 'photonPtSumOutsideSignalConedR03',
                           'puCorrPtSum', 'tau_pt_weighted_dr_iso' ]
# input_tau_branches = input_tau_signal_branches + input_tau_iso_branches

input_tau_branches = [ 'tau_pt', 'tau_eta', 'tau_mass', 'tau_charge', 'tau_decayMode', 'chargedIsoPtSum',
                       'chargedIsoPtSumdR03', 'footprintCorrection', 'footprintCorrectiondR03', 'neutralIsoPtSum',
                       'neutralIsoPtSumWeight', 'neutralIsoPtSumWeightdR03', 'neutralIsoPtSumdR03',
                       'photonPtSumOutsideSignalCone', 'photonPtSumOutsideSignalConedR03', 'puCorrPtSum',
                       'tau_dxy_pca_x', 'tau_dxy_pca_y', 'tau_dxy_pca_z', 'tau_dxy', 'tau_dxy_error', 'tau_ip3d',
                       'tau_ip3d_error', 'tau_dz', 'tau_dz_error', 'tau_hasSecondaryVertex', 'tau_sv_x', 'tau_sv_y',
                       'tau_sv_z', 'tau_flightLength_x', 'tau_flightLength_y', 'tau_flightLength_z',
                       'tau_flightLength_sig', 'tau_pt_weighted_deta_strip', 'tau_pt_weighted_dphi_strip',
                       'tau_pt_weighted_dr_signal', 'tau_pt_weighted_dr_iso', 'tau_leadingTrackNormChi2',
                       'tau_e_ratio', 'tau_gj_angle_diff', 'tau_n_photons', 'tau_emFraction', 'tau_inside_ecal_crack',
                       'leadChargedCand_etaAtEcalEntrance' ]

input_cell_common_branches = [ 'eta_index', 'phi_index', 'tau_pt' ]
input_cell_pfCand_branches = [ 'pfCand_n_total', 'pfCand_n_ele', 'pfCand_n_muon', 'pfCand_n_gamma',
                               'pfCand_n_chargedHadrons', 'pfCand_n_neutralHadrons', 'pfCand_max_pt', 'pfCand_sum_pt',
                               'pfCand_sum_pt_scalar', 'pfCand_sum_E', 'pfCand_tauSignal',
                               'pfCand_leadChargedHadrCand', 'pfCand_tauIso', 'pfCand_pvAssociationQuality',
                               'pfCand_fromPV', 'pfCand_puppiWeight', 'pfCand_puppiWeightNoLep', 'pfCand_pdgId',
                               'pfCand_charge', 'pfCand_lostInnerHits', 'pfCand_numberOfPixelHits',
                               'pfCand_vertex_x', 'pfCand_vertex_y', 'pfCand_vertex_z',
                               'pfCand_hasTrackDetails', 'pfCand_dxy', 'pfCand_dxy_error', 'pfCand_dz',
                               'pfCand_dz_error', 'pfCand_track_chi2', 'pfCand_track_ndof', 'pfCand_hcalFraction',
                               'pfCand_rawCaloFraction' ]
input_cell_ele_branches = [ 'ele_n_total', 'ele_max_pt', 'ele_sum_pt', 'ele_sum_pt_scalar', 'ele_sum_E',
                            'ele_cc_ele_energy', 'ele_cc_gamma_energy', 'ele_cc_n_gamma', 'ele_trackMomentumAtVtx',
                            'ele_trackMomentumAtCalo', 'ele_trackMomentumOut', 'ele_trackMomentumAtEleClus',
                            'ele_trackMomentumAtVtxWithConstraint', 'ele_ecalEnergy', 'ele_ecalEnergy_error',
                            'ele_eSuperClusterOverP', 'ele_eSeedClusterOverP', 'ele_eSeedClusterOverPout',
                            'ele_eEleClusterOverPout', 'ele_deltaEtaSuperClusterTrackAtVtx',
                            'ele_deltaEtaSeedClusterTrackAtCalo', 'ele_deltaEtaEleClusterTrackAtCalo',
                            'ele_deltaPhiEleClusterTrackAtCalo', 'ele_deltaPhiSuperClusterTrackAtVtx',
                            'ele_deltaPhiSeedClusterTrackAtCalo', 'ele_mvaInput_earlyBrem', 'ele_mvaInput_lateBrem',
                            'ele_mvaInput_sigmaEtaEta', 'ele_mvaInput_hadEnergy', 'ele_mvaInput_deltaEta',
                            'ele_gsfTrack_normalizedChi2', 'ele_gsfTrack_numberOfValidHits', 'ele_gsfTrack_pt',
                            'ele_gsfTrack_pt_error', 'ele_closestCtfTrack_normalizedChi2',
                            'ele_closestCtfTrack_numberOfValidHits' ]
input_cell_muon_branches = [ 'muon_n_total', 'muon_max_pt', 'muon_sum_pt', 'muon_sum_pt_scalar', 'muon_sum_E',
                             'muon_dxy', 'muon_dxy_error', 'muon_normalizedChi2', 'muon_numberOfValidHits',
                             'muon_segmentCompatibility', 'muon_caloCompatibility', 'muon_pfEcalEnergy',
                             'muon_n_matches_DT_1', 'muon_n_matches_DT_2', 'muon_n_matches_DT_3',
                             'muon_n_matches_DT_4', 'muon_n_matches_CSC_1', 'muon_n_matches_CSC_2',
                             'muon_n_matches_CSC_3', 'muon_n_matches_CSC_4', 'muon_n_matches_RPC_1',
                             'muon_n_matches_RPC_2', 'muon_n_matches_RPC_3', 'muon_n_matches_RPC_4',
                             'muon_n_hits_DT_1', 'muon_n_hits_DT_2', 'muon_n_hits_DT_3', 'muon_n_hits_DT_4',
                             'muon_n_hits_CSC_1', 'muon_n_hits_CSC_2', 'muon_n_hits_CSC_3', 'muon_n_hits_CSC_4',
                             'muon_n_hits_RPC_1', 'muon_n_hits_RPC_2', 'muon_n_hits_RPC_3', 'muon_n_hits_RPC_4' ]

df_tau_branches = truth_branches + navigation_branches + weight_branches + input_tau_branches
df_cell_branches = input_cell_common_branches + input_cell_pfCand_branches + input_cell_ele_branches \
                 + input_cell_muon_branches

match_suffixes = [ 'e', 'mu', 'tau', 'jet' ]
e, mu, tau, jet = 0, 1, 2, 3
cell_locations = ['inner', 'outer']
n_cells_eta = { 'inner': 21, 'outer': 13 }
n_cells_phi = { 'inner': 21, 'outer': 13 }
n_cells = { 'inner': n_cells_eta['inner'] * n_cells_phi['inner'], 'outer': n_cells_eta['outer'] * n_cells_phi['outer'] }

component_branches = [ input_cell_common_branches + input_cell_pfCand_branches,
                       input_cell_common_branches + input_cell_ele_branches,
                       input_cell_common_branches + input_cell_muon_branches ]

component_names = [ 'pfCand', 'ele', 'muon' ]

n_outputs = len(truth_branches)

def ReadBranchesToDataFrame(file_name, tree_name, branches, entrystart=None, entrystop=None):
    with uproot.open(file_name) as file:
        tree = file[tree_name]
        df = tree.arrays(branches, entrystart=entrystart, entrystop=entrystop, outputtype=pandas.DataFrame)
    return df

def GetNumberOfEntries(file_name, tree_name):
    with uproot.open(file_name) as file:
        tree = file[tree_name]
        return tree.numentries

def load_graph(graph_filename):
    with tf.gfile.GFile(graph_filename, 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())

    with tf.Graph().as_default() as graph:
        tf.import_graph_def(graph_def, name="deepTau")
    return graph

def ExtractMuonDiscriminators(id_flags):
    mu_disc = np.zeros([id_flags.shape[0], 2], dtype=int)
    mu_disc[:, 0] = np.bitwise_and(np.right_shift(id_flags, 5), 1)
    mu_disc[:, 1] = np.bitwise_and(np.right_shift(id_flags, 6), 1)
    return mu_disc

class TauLosses:
    Le_sf = 1
    Lmu_sf = 1
    Ljet_sf = 1
    epsilon = 1e-6

    @staticmethod
    def SetSFs(sf_e, sf_mu, sf_jet):
        TauLosses.Le_sf = sf_e
        TauLosses.Lmu_sf = sf_mu
        TauLosses.Ljet_sf = sf_jet

    @staticmethod
    def Lbase(target, output, genuine_index, fake_index):
        epsilon = tf.convert_to_tensor(TauLosses.epsilon, output.dtype.base_dtype)
        genuine_vs_fake = output[:, genuine_index] / (output[:, genuine_index] + output[:, fake_index] + epsilon)
        genuine_vs_fake = tf.clip_by_value(genuine_vs_fake, epsilon, 1 - epsilon)
        loss = -target[:, genuine_index] * tf.log(genuine_vs_fake) - target[:, fake_index] * tf.log(1 - genuine_vs_fake)
        return loss

    @staticmethod
    def Le(target, output):
        return TauLosses.Lbase(target, output, tau, e)

    @staticmethod
    def Lmu(target, output):
        return TauLosses.Lbase(target, output, tau, mu)

    @staticmethod
    def Ljet(target, output):
        return TauLosses.Lbase(target, output, tau, jet)

    @staticmethod
    def sLe(target, output):
        sf = tf.convert_to_tensor(TauLosses.Le_sf, output.dtype.base_dtype)
        return sf * TauLosses.Le(target, output)

    @staticmethod
    def sLmu(target, output):
        sf = tf.convert_to_tensor(TauLosses.Lmu_sf, output.dtype.base_dtype)
        return sf * TauLosses.Lmu(target, output)

    @staticmethod
    def sLjet(target, output):
        sf = tf.convert_to_tensor(TauLosses.Ljet_sf, output.dtype.base_dtype)
        return sf * TauLosses.Ljet(target, output)

    @staticmethod
    def tau_crossentropy(target, output):
        return TauLosses.sLe(target, output) + TauLosses.sLmu(target, output) + TauLosses.sLjet(target, output)

    @staticmethod
    def tau_vs_other(prob_tau, prob_other):
        return prob_tau / (prob_tau + prob_other + TauLosses.epsilon)


def LoadModel(model_file, compile=True):
    from keras.models import load_model
    if compile:
        return load_model(model_file, custom_objects = {
            'tau_crossentropy': TauLosses.tau_crossentropy, 'Le': TauLosses.Le, 'Lmu': TauLosses.Lmu,
            'Ljet': TauLosses.Ljet, 'sLe': TauLosses.sLe, 'sLmu': TauLosses.sLmu, 'sLjet': TauLosses.sLjet
        })
    else:
        return load_model(model_file, compile = False)


def quantile_ex(data, quantiles, weights):
    quantiles = np.array(quantiles)
    indices = np.argsort(data)
    data_sorted = data[indices]
    weights_sorted = weights[indices]
    prob = np.cumsum(weights_sorted) - weights_sorted / 2
    prob = (prob[:] - prob[0]) / (prob[-1] - prob[0])
    return np.interp(quantiles, prob, data_sorted)
