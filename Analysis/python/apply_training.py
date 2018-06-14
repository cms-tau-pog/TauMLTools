#!/usr/bin/env python

import argparse
parser = argparse.ArgumentParser(description='Apply training and store results.')
parser.add_argument('--input', required=True, type=str, help="Input directory")
parser.add_argument('--filelist', required=False, type=str, default=None, help="Txt file with input tuple list")
parser.add_argument('--output', required=True, type=str, help="Output file")
parser.add_argument('--model', required=True, type=str, help="Txt file with input tuple list")
parser.add_argument('--tree', required=False, type=str, default="taus", help="Tree name")
parser.add_argument('--batch-size', required=False, type=int, default=20000, help="Batch size")
parser.add_argument('--max-n-files', required=False, type=int, default=None, help="Maximum number of files to process")
args = parser.parse_args()

import os
import gc
import uproot
import pandas
import tensorflow as tf
import numpy as np
from tqdm import tqdm

central_tau_id_branches = ['againstElectronMVA6Raw', 'byCombinedIsolationDeltaBetaCorrRaw3Hits',
    'byIsolationMVArun2v1DBoldDMwLTraw', 'byIsolationMVArun2v1DBdR03oldDMwLTraw',
    'byIsolationMVArun2v1DBoldDMwLTraw2016', 'byIsolationMVArun2017v2DBoldDMwLTraw2017',
    'byIsolationMVArun2017v2DBoldDMdR0p3wLTraw2017', 'id_flags']
truth_branches = ['gen_match']
input_branches = ['pt', 'eta', 'mass', 'decayMode', 'chargedIsoPtSum', 'neutralIsoPtSum', 'neutralIsoPtSumWeight',
                  'photonPtSumOutsideSignalCone', 'puCorrPtSum',
                  'dxy', 'dxy_sig', 'dz', 'ip3d', 'ip3d_sig',
                  'hasSecondaryVertex', 'flightLength_r', 'flightLength_dEta', 'flightLength_dPhi',
                  'flightLength_sig', 'leadChargedHadrCand_pt', 'leadChargedHadrCand_dEta',
                  'leadChargedHadrCand_dPhi', 'leadChargedHadrCand_mass', 'pt_weighted_deta_strip',
                  'pt_weighted_dphi_strip', 'pt_weighted_dr_signal', 'pt_weighted_dr_iso',
                  'leadingTrackNormChi2', 'e_ratio', 'gj_angle_diff', 'n_photons', 'emFraction',
                  'has_gsf_track', 'inside_ecal_crack',
                  'gsf_ele_matched', 'gsf_ele_pt', 'gsf_ele_dEta', 'gsf_ele_dPhi', 'gsf_ele_mass', 'gsf_ele_Ee',
                  'gsf_ele_Egamma', 'gsf_ele_Pin', 'gsf_ele_Pout', 'gsf_ele_EtotOverPin', 'gsf_ele_Eecal',
                  'gsf_ele_dEta_SeedClusterTrackAtCalo', 'gsf_ele_dPhi_SeedClusterTrackAtCalo', 'gsf_ele_mvaIn_sigmaEtaEta',
                  'gsf_ele_mvaIn_hadEnergy',
                  'gsf_ele_mvaIn_deltaEta', 'gsf_ele_Chi2NormGSF', 'gsf_ele_GSFNumHits', 'gsf_ele_GSFTrackResol',
                  'gsf_ele_GSFTracklnPt', 'gsf_ele_Chi2NormKF', 'gsf_ele_KFNumHits',
                  'leadChargedCand_etaAtEcalEntrance', 'leadChargedCand_pt', 'leadChargedHadrCand_HoP',
                  'leadChargedHadrCand_EoP', 'tau_visMass_innerSigCone', 'n_matched_muons', 'muon_pt', 'muon_dEta', 'muon_dPhi',
                  'muon_n_matches_DT_1', 'muon_n_matches_DT_2', 'muon_n_matches_DT_3', 'muon_n_matches_DT_4',
                  'muon_n_matches_CSC_1', 'muon_n_matches_CSC_2', 'muon_n_matches_CSC_3', 'muon_n_matches_CSC_4',
                  'muon_n_hits_DT_2', 'muon_n_hits_DT_3', 'muon_n_hits_DT_4',
                  'muon_n_hits_CSC_2', 'muon_n_hits_CSC_3', 'muon_n_hits_CSC_4',
                  'muon_n_hits_RPC_2', 'muon_n_hits_RPC_3', 'muon_n_hits_RPC_4',
                  'muon_n_stations_with_matches_03', 'muon_n_stations_with_hits_23',
                  'signalChargedHadrCands_sum_innerSigCone_pt', 'signalChargedHadrCands_sum_innerSigCone_dEta',
                  'signalChargedHadrCands_sum_innerSigCone_dPhi', 'signalChargedHadrCands_sum_innerSigCone_mass',
                  'signalChargedHadrCands_sum_outerSigCone_pt', 'signalChargedHadrCands_sum_outerSigCone_dEta',
                  'signalChargedHadrCands_sum_outerSigCone_dPhi', 'signalChargedHadrCands_sum_outerSigCone_mass',
                  'signalChargedHadrCands_nTotal_innerSigCone', 'signalChargedHadrCands_nTotal_outerSigCone',
                  'signalNeutrHadrCands_sum_innerSigCone_pt', 'signalNeutrHadrCands_sum_innerSigCone_dEta',
                  'signalNeutrHadrCands_sum_innerSigCone_dPhi', 'signalNeutrHadrCands_sum_innerSigCone_mass',
                  'signalNeutrHadrCands_sum_outerSigCone_pt', 'signalNeutrHadrCands_sum_outerSigCone_dEta',
                  'signalNeutrHadrCands_sum_outerSigCone_dPhi', 'signalNeutrHadrCands_sum_outerSigCone_mass',
                  'signalNeutrHadrCands_nTotal_innerSigCone', 'signalNeutrHadrCands_nTotal_outerSigCone',
                  'signalGammaCands_sum_innerSigCone_pt', 'signalGammaCands_sum_innerSigCone_dEta',
                  'signalGammaCands_sum_innerSigCone_dPhi', 'signalGammaCands_sum_innerSigCone_mass',
                  'signalGammaCands_sum_outerSigCone_pt', 'signalGammaCands_sum_outerSigCone_dEta',
                  'signalGammaCands_sum_outerSigCone_dPhi', 'signalGammaCands_sum_outerSigCone_mass',
                  'signalGammaCands_nTotal_innerSigCone', 'signalGammaCands_nTotal_outerSigCone',
                  'isolationChargedHadrCands_sum_pt', 'isolationChargedHadrCands_sum_dEta',
                  'isolationChargedHadrCands_sum_dPhi', 'isolationChargedHadrCands_sum_mass',
                  'isolationChargedHadrCands_nTotal',
                  'isolationNeutrHadrCands_sum_pt', 'isolationNeutrHadrCands_sum_dEta',
                  'isolationNeutrHadrCands_sum_dPhi', 'isolationNeutrHadrCands_sum_mass',
                  'isolationNeutrHadrCands_nTotal',
                  'isolationGammaCands_sum_pt', 'isolationGammaCands_sum_dEta',
                  'isolationGammaCands_sum_dPhi', 'isolationGammaCands_sum_mass',
                  'isolationGammaCands_nTotal',
                 ]
all_branches = truth_branches + input_branches + central_tau_id_branches
match_suffixes = [ 'e', 'mu', 'tau', 'jet' ]
gen_match_ex_branches = [ 'gen_match_{}'.format(suff) for suff in match_suffixes ]
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

def load_graph(graph_filename):
    with tf.gfile.GFile(graph_filename, 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())

    with tf.Graph().as_default() as graph:
        tf.import_graph_def(graph_def, name="deepTau")
    return graph

def ReadBrancesToDataFrame(file_name, tree_name, branches, nentries=None):
    with uproot.open(file_name) as file:
        tree = file[tree_name]
        df = tree.arrays(branches, entrystop=nentries, outputtype=pandas.DataFrame)
        df.columns = [ c.decode('utf-8') for c in df.columns ]
    return df

def ReadBranchesTo2DArray(file_name, tree_name, branches, dtype, chunk_size = 20, nentries=None):
    data = None
    n = 0
    branch_chunks = [ branches[pos:pos+chunk_size] for pos in range(0, len(branches), chunk_size) ]
    if nentries is None:
        with uproot.open(file_name) as file:
            tree = file[tree_name]
            nentries = tree.numentries
        gc.collect()
    data = np.empty([nentries, len(branches)], dtype=dtype)
    for chunk in branch_chunks:
        df = ReadBrancesToDataFrame(file_name, tree_name, chunk, nentries)
        for br in chunk:
            data[:, n] = df[br].astype(dtype)
            print("branch '{}' loaded. {}/{}".format(br, n + 1, len(branches)))
            n += 1
        del df
        gc.collect()
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

def ExtractMuonDiscriminators(id_flags):
    mu_disc = np.zeros([id_flags.shape[0], 2], dtype=int)
    mu_disc[:, 0] = np.bitwise_and(np.right_shift(id_flags, 5), 1)
    mu_disc[:, 1] = np.bitwise_and(np.right_shift(id_flags, 6), 1)
    return mu_disc

def ProcessFile(session, graph, file_name):
    full_name = args.input + '/' + file_name

    print("Loading inputs...")

    result_branches = [
        'run', 'lumi', 'evt', 'tau_index', 'pt', 'eta', 'phi', 'decayMode', 'againstElectronMVA6Raw', 'id_flags',
        'byIsolationMVArun2017v2DBoldDMwLTraw2017'
    ]

    df = ReadBrancesToDataFrame(full_name, args.tree, result_branches)
    refId_mu = ExtractMuonDiscriminators(df.id_flags)

    X = ReadBranchesTo2DArray(full_name, args.tree, input_branches, np.float32, chunk_size=200)
    Y_raw = ReadBranchesTo2DArray(full_name, args.tree, truth_branches, int)
    Y = VectorizeGenMatch(Y_raw, int)
    N = X.shape[0]

    print("Running predictions...")
    x_gr_name = graph.get_operations()[0].name
    y_gr_name = graph.get_operations()[-1].name
    x_gr = graph.get_tensor_by_name(x_gr_name + ':0')
    y_gr = graph.get_tensor_by_name(y_gr_name + ':0')
    deepId = np.zeros([N, 4])

    with tqdm(total=N, unit='taus') as pbar:
        for n in range(0, N, args.batch_size):
            pred = session.run(y_gr, feed_dict={x_gr: X[n:n+args.batch_size, :]})
            deepId[n:n+args.batch_size, :] = pred[:, :]
            dn = min(N, n + args.batch_size) - n
            pbar.update(dn)

    return pandas.DataFrame(data = {
        'run': df.run, 'lumi': df.lumi, 'evt': df.evt, 'tau_index': df.tau_index,
        'pt': df.pt, 'eta': df.eta, 'phi': df.phi, 'decayMode': df.decayMode,
        'gen_e': Y[:, e], 'gen_mu': Y[:, mu], 'gen_tau': Y[:, tau], 'gen_jet': Y[:, jet],
        'deepId_e': deepId[:, e], 'deepId_mu': deepId[:, mu], 'deepId_tau': deepId[:, tau],
        'deepId_jet': deepId[:, jet],
        'refId_e': df.againstElectronMVA6Raw, 'refId_mu_loose': refId_mu[:, 0], 'refId_mu_tight': refId_mu[:, 1],
        'refId_jet': df.byIsolationMVArun2017v2DBoldDMwLTraw2017
    })

if args.filelist is None:
    file_list = [ f for f in os.listdir(args.input) if f.endswith('.root') ]
else:
    with open(args.filelist, 'r') as f_list:
        file_list = [ f.strip() for f in f_list if len(f) != 0 ]

if args.max_n_files is not None and args.max_n_files > 0:
    file_list = file_list[0:args.max_n_files]

if os.path.isfile(args.output):
    os.remove(args.output)

graph = load_graph(args.model)
sess = tf.Session(graph=graph)

for file_name in file_list:
    print("Processing '{}'".format(file_name))
    df = ProcessFile(sess, graph, file_name)
    print("Saving output into '{}'...".format(args.output))
    df.to_hdf(args.output, 'taus', append=True, complevel=9, complib='zlib')

print("All files processed.")
