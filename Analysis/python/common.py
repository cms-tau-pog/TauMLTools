import gc
import uproot
import pandas
import numpy as np
import tensorflow as tf
import functools
from keras.models import load_model

try:
    get_ipython
    from tqdm import tqdm_notebook as tqdm
except:
    from tqdm import tqdm

def MakeP4Branches(name, pt=True, ht=True, dEta=True, dPhi=True, energy=True, mass=True):
    branches = []
    if pt:
        branches.append(name + '_pt')
    if ht:
        branches.append(name + '_ht')
    if dEta:
        branches.append(name + '_dEta')
    if dPhi:
        branches.append(name + '_dPhi')
    if energy:
        branches.append(name + '_energy')
    if mass:
        branches.append(name + '_mass')
    return branches

def MakePFRegionBranches(cone_name, ChargedHadrCands=True, NeutrHadrCands=True, GammaCands=True):
    branches = []
    if ChargedHadrCands:
        branches.extend(MakeP4Branches(cone_name + '_ChargedHadrCands_sum'))
        branches.append(cone_name + '_ChargedHadrCands_nTotal')
    if NeutrHadrCands:
        branches.extend(MakeP4Branches(cone_name + '_NeutrHadrCands_sum'))
        branches.append(cone_name + '_NeutrHadrCands_nTotal')
    if GammaCands:
        branches.extend(MakeP4Branches(cone_name + '_GammaCands_sum'))
        branches.append(cone_name + '_GammaCands_nTotal')
    return branches


central_tau_id_branches = ['againstElectronMVA6Raw', 'byCombinedIsolationDeltaBetaCorrRaw3Hits',
    'byIsolationMVArun2v1DBoldDMwLTraw', 'byIsolationMVArun2v1DBdR03oldDMwLTraw',
    'byIsolationMVArun2v1DBoldDMwLTraw2016', 'byIsolationMVArun2017v2DBoldDMwLTraw2017',
    'byIsolationMVArun2017v2DBoldDMdR0p3wLTraw2017', 'id_flags']
truth_branches = ['gen_match']
input_branches = [ 'pt', 'eta', 'mass', 'decayMode', 'dxy', 'dxy_sig', 'dz', 'ip3d', 'ip3d_sig',
                   'hasSecondaryVertex', 'flightLength_r', 'flightLength_dEta', 'flightLength_dPhi', 'flightLength_sig',
                   'chargedIsoPtSum', 'chargedIsoPtSumdR03', 'footprintCorrection', 'footprintCorrectiondR03',
                   'neutralIsoPtSum', 'neutralIsoPtSumdR03', 'neutralIsoPtSumWeight', 'neutralIsoPtSumWeightdR03',
                   'photonPtSumOutsideSignalCone', 'photonPtSumOutsideSignalConedR03', 'puCorrPtSum',
                   'pt_weighted_deta_strip', 'pt_weighted_dphi_strip', 'pt_weighted_dr_signal', 'pt_weighted_dr_iso',
                   'leadingTrackNormChi2', 'e_ratio', 'gj_angle_diff', 'n_photons', 'emFraction', 'inside_ecal_crack',
                   'has_gsf_track', 'gsf_ele_matched', 'gsf_ele_pt', 'gsf_ele_dEta', 'gsf_ele_dPhi', 'gsf_ele_energy',
                   'gsf_ele_Ee', 'gsf_ele_Egamma', 'gsf_ele_Pin', 'gsf_ele_Pout', 'gsf_ele_Eecal',
                   'gsf_ele_dEta_SeedClusterTrackAtCalo', 'gsf_ele_dPhi_SeedClusterTrackAtCalo', 'gsf_ele_mvaIn_sigmaEtaEta',
                   'gsf_ele_mvaIn_hadEnergy', 'gsf_ele_mvaIn_deltaEta', 'gsf_ele_Chi2NormGSF', 'gsf_ele_GSFNumHits',
                   'gsf_ele_GSFTrackResol', 'gsf_ele_GSFTracklnPt', 'gsf_ele_Chi2NormKF', 'gsf_ele_KFNumHits',
                   'n_matched_muons', 'muon_pt', 'muon_dEta', 'muon_dPhi',
                   'muon_n_matches_DT_1', 'muon_n_matches_DT_2', 'muon_n_matches_DT_3', 'muon_n_matches_DT_4',
                   'muon_n_matches_CSC_1', 'muon_n_matches_CSC_2', 'muon_n_matches_CSC_3', 'muon_n_matches_CSC_4',
                   'muon_n_hits_DT_2', 'muon_n_hits_DT_3', 'muon_n_hits_DT_4',
                   'muon_n_hits_CSC_2', 'muon_n_hits_CSC_3', 'muon_n_hits_CSC_4',
                   'muon_n_hits_RPC_2', 'muon_n_hits_RPC_3', 'muon_n_hits_RPC_4',
                   'leadChargedCand_etaAtEcalEntrance' ]
input_branches.extend(MakeP4Branches('leadChargedHadrCand', ht=False, energy=False))
input_branches.extend(MakePFRegionBranches('innerSigCone'))
input_branches.extend(MakePFRegionBranches('outerSigCone', ChargedHadrCands=False))
input_branches.extend(MakePFRegionBranches('isoRing02'))
input_branches.extend(MakePFRegionBranches('isoRing03'))
input_branches.extend(MakePFRegionBranches('isoRing04', NeutrHadrCands=False))
input_branches.extend(MakePFRegionBranches('isoRing05', ChargedHadrCands=False, NeutrHadrCands=False))

all_branches = truth_branches + input_branches + central_tau_id_branches
match_suffixes = [ 'e', 'mu', 'tau', 'jet' ]
gen_match_ex_branches = [ 'gen_{}'.format(suff) for suff in match_suffixes ]
e, mu, tau, jet = 0, 1, 2, 3
input_shape = (len(input_branches), )
n_outputs = len(gen_match_ex_branches)

class GenMatch:
    Electron = 1
    Muon = 2
    TauElectron = 3
    TauMuon = 4
    Tau = 5
    NoMatch = 6


def ReadBrancesToDataFrame(file_name, tree_name, branches, entrystart=None, entrystop=None):
    with uproot.open(file_name) as file:
        tree = file[tree_name]
        df = tree.arrays(branches, entrystart=entrystart, entrystop=entrystop, outputtype=pandas.DataFrame)
        df.columns = [ c.decode('utf-8') for c in df.columns ]
    return df

def GetNumberOfEntries(file_name, tree_name):
    with uproot.open(file_name) as file:
        tree = file[tree_name]
        return tree.numentries

def ReadBranchesTo2DArray(file_name, tree_name, branches, dtype, chunk_size = int(5e6), entrystart=None,
                          entrystop=None):
    if entrystop is None:
        entrystop = GetNumberOfEntries(file_name, tree_name)
        gc.collect()
    if entrystart is None:
        entrystart = 0
    nentries = entrystop - entrystart
    data = np.empty([nentries, len(branches)], dtype=dtype)

    if chunk_size is None:
        chunk_size = nentries

    step = 0
    current_start = entrystart
    with tqdm(total=nentries) as pbar:
        while current_start < entrystop:
            current_stop = min(current_start + chunk_size, entrystop)
            n_current = current_stop - current_start
            df = ReadBrancesToDataFrame(file_name, tree_name, branches, current_start, current_stop)
            for br_index in range(len(branches)):
                data[chunk_size*step:chunk_size*(step+1), br_index] = df[branches[br_index]].astype(dtype)
            del df
            gc.collect()
            current_start += chunk_size
            step += 1
            pbar.update(n_current)
    return data

def VectorizeGenMatch(data, dtype):
    if data.shape[1] != 1:
        raise RuntimeError("Invalid input")
    v_data = np.zeros([data.shape[0], 4], dtype=dtype)
    v_data[:, e] = ((data[:, 0] == GenMatch.Electron) | (data[:, 0] == GenMatch.TauElectron)).astype(dtype)
    v_data[:, mu] = ((data[:, 0] == GenMatch.Muon) | (data[:, 0] == GenMatch.TauMuon)).astype(dtype)
    v_data[:, tau] = (data[:, 0] == GenMatch.Tau).astype(dtype)
    v_data[:, jet] = (data[:, 0] == GenMatch.NoMatch).astype(dtype)
    return v_data

def ReadXY(file_name, tree_name, chunk_size = int(5e6), entrystart=None, entrystop=None):
    X = ReadBranchesTo2DArray(file_name, tree_name, input_branches, np.float32, chunk_size=chunk_size,
                              entrystart=entrystart, entrystop=entrystop)
    Y_raw = ReadBranchesTo2DArray(file_name, tree_name, truth_branches, int, entrystart=entrystart, entrystop=entrystop)
    Y = VectorizeGenMatch(Y_raw, int)
    return X, Y


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
    def Lbase(target, output, weights, genuine_index, fake_index, weights_index):
        epsilon = tf.convert_to_tensor(TauLosses.epsilon, output.dtype.base_dtype)
        genuine_vs_fake = output[:, genuine_index] / (output[:, genuine_index] + output[:, fake_index] + epsilon)
        genuine_vs_fake = tf.clip_by_value(genuine_vs_fake, epsilon, 1 - epsilon)
        loss = -target[:, genuine_index] * tf.log(genuine_vs_fake) - target[:, fake_index] * tf.log(1 - genuine_vs_fake)
        w_sum = tf.reduce_sum((target[:, genuine_index] + target[:, fake_index]) * weights[:, weights_index])
        n_items = tf.to_float(tf.shape(weights))[0]
        return loss * weights[:, weights_index] / w_sum * n_items

    @staticmethod
    def Le(target, output, weights):
        return TauLosses.Lbase(target, output, weights, tau, e, 0)

    @staticmethod
    def Lmu(target, output, weights):
        return TauLosses.Lbase(target, output, weights, tau, mu, 1)

    @staticmethod
    def Ljet(target, output, weights):
        return TauLosses.Lbase(target, output, weights, tau, jet, 2)

    @staticmethod
    def sLe(target, output, weights):
        sf = tf.convert_to_tensor(TauLosses.Le_sf, output.dtype.base_dtype)
        return sf * TauLosses.Le(target, output, weights)

    @staticmethod
    def sLmu(target, output, weights):
        sf = tf.convert_to_tensor(TauLosses.Lmu_sf, output.dtype.base_dtype)
        return sf * TauLosses.Lmu(target, output, weights)

    @staticmethod
    def sLjet(target, output, weights):
        sf = tf.convert_to_tensor(TauLosses.Ljet_sf, output.dtype.base_dtype)
        return sf * TauLosses.Ljet(target, output, weights)

    @staticmethod
    def tau_crossentropy(target, output, weights):
        return TauLosses.sLe(target, output, weights) + TauLosses.sLmu(target, output, weights) + \
               TauLosses.sLjet(target, output, weights)

    @staticmethod
    def tau_vs_other(prob_tau, prob_other):
        return prob_tau / (prob_tau + prob_other + TauLosses.epsilon)


def LoadModel(model_file, compile=True):
    if compile:
        weight_input = tf.placeholder(tf.float32, shape=(None, 3))

        tau_crossentropy = functools.partial(TauLosses.tau_crossentropy, weights=weight_input)
        Le = functools.partial(TauLosses.Le, weights=weight_input)
        Lmu = functools.partial(TauLosses.Lmu, weights=weight_input)
        Ljet = functools.partial(TauLosses.Ljet, weights=weight_input)
        sLe = functools.partial(TauLosses.sLe, weights=weight_input)
        sLmu = functools.partial(TauLosses.sLmu, weights=weight_input)
        sLjet = functools.partial(TauLosses.sLjet, weights=weight_input)

        functools.update_wrapper(tau_crossentropy, TauLosses.tau_crossentropy)
        functools.update_wrapper(Le, TauLosses.Le)
        functools.update_wrapper(Lmu, TauLosses.Lmu)
        functools.update_wrapper(Ljet, TauLosses.Ljet)
        functools.update_wrapper(sLe, TauLosses.sLe)
        functools.update_wrapper(sLmu, TauLosses.sLmu)
        functools.update_wrapper(sLjet, TauLosses.sLjet)


        return load_model(model_file, custom_objects = {
            'tau_crossentropy': tau_crossentropy, 'Le': Le, 'Lmu': Lmu, 'Ljet': Ljet,
            'sLe': sLe, 'sLmu': sLmu, 'sLjet': sLjet
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
