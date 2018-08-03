#!/usr/bin/env python

import argparse
parser = argparse.ArgumentParser(description='Apply training and store results.')
parser.add_argument('--input', required=True, type=str, help="Input directory")
parser.add_argument('--filelist', required=False, type=str, default=None, help="Txt file with input tuple list")
parser.add_argument('--output', required=True, type=str, help="Output file")
parser.add_argument('--model', required=True, type=str, help="Model file")
parser.add_argument('--tree', required=False, type=str, default="taus", help="Tree name")
parser.add_argument('--chunk-size', required=False, type=int, default=5000000, help="Chunk size")
parser.add_argument('--batch-size', required=False, type=int, default=10000, help="Batch size")
parser.add_argument('--max-n-files', required=False, type=int, default=None, help="Maximum number of files to process")
parser.add_argument('--max-n-entries-per-file', required=False, type=int, default=None,
                    help="Maximum number of entries per file")
args = parser.parse_args()

import os
import pandas
import tensorflow as tf
import numpy as np
from tqdm import tqdm

from common import *

def ProcessFile(session, graph, file_name, entrystart, entrystop):
    print("Loading inputs...")

    result_branches = [
        'run', 'lumi', 'evt', 'tau_index', 'pt', 'eta', 'phi', 'decayMode', 'againstElectronMVA6Raw', 'id_flags',
        'byIsolationMVArun2017v2DBoldDMwLTraw2017'
    ]

    df = ReadBrancesToDataFrame(file_name, args.tree, result_branches, entrystart=entrystart, entrystop=entrystop)
    refId_mu = ExtractMuonDiscriminators(df.id_flags)

    X, Y = ReadXY(file_name, args.tree, entrystart=entrystart, entrystop=entrystop)
    N = X.shape[0]

    print("Running predictions...")

    # for op in graph.get_operations():
    #     print(op.name)
    # raise RuntimeError("stop")
    x_gr_name = "deepTau/main_input" #graph.get_operations()[0].name
    y_gr_name = "deepTau/main_output/Softmax"#graph.get_operations()[-1].name
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
    if os.path.isdir(args.input):
        file_list = [ f for f in os.listdir(args.input) if f.endswith('.root') ]
        prefix = args.input + '/'
    else:
        file_list = [ args.input ]
        prefix = ''
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
    full_name = prefix + file_name
    n_entries = GetNumberOfEntries(full_name, args.tree)
    if args.max_n_entries_per_file is not None:
        n_entries = min(n_entries, args.max_n_entries_per_file)
    current_start = 0
    while current_start < n_entries:
        current_stop = min(current_start + args.chunk_size, n_entries)
        print("Loading entries [{}, {}) out of {}...".format(current_start, current_stop, n_entries))
        df = ProcessFile(sess, graph, full_name, current_start, current_stop)
        print("Saving output into '{}'...".format(args.output))
        df.to_hdf(args.output, args.tree, append=True, complevel=1, complib='zlib')
        del df
        gc.collect()
        current_start += args.chunk_size

print("All files processed.")
