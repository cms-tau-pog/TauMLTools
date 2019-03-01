#!/usr/bin/env python

import argparse
parser = argparse.ArgumentParser(description='Apply training and store results.')
parser.add_argument('--input', required=True, type=str, help="Input directory")
parser.add_argument('--filelist', required=False, type=str, default=None, help="Txt file with input tuple list")
parser.add_argument('--output', required=True, type=str, help="Output file")
parser.add_argument('--model', required=True, type=str, help="Model file")
parser.add_argument('--tree', required=False, type=str, default="taus", help="Tree name")
parser.add_argument('--chunk-size', required=False, type=int, default=1000, help="Chunk size")
parser.add_argument('--batch-size', required=False, type=int, default=250, help="Batch size")
parser.add_argument('--max-n-files', required=False, type=int, default=None, help="Maximum number of files to process")
parser.add_argument('--max-n-entries-per-file', required=False, type=int, default=None,
                    help="Maximum number of entries per file")
args = parser.parse_args()

import os
import gc
import pandas
import tensorflow as tf
import numpy as np
from tqdm import tqdm

from common import *
from DataLoader import DataLoader

def Predict(session, graph, X_taus, X_inner_pfCand, X_inner_ele, X_inner_muon, X_outer_pfCand, X_outer_ele,
            X_outer_muon):
#    for op in graph.get_operations():
#        print(op.name)
#    raise RuntimeError("stop")

    gr_name_prefix = "deepTau/input_"
    tau_gr = graph.get_tensor_by_name(gr_name_prefix + "tau:0")
    inner_pfCand_gr = graph.get_tensor_by_name(gr_name_prefix + "inner_pfCand:0")
    inner_ele_gr = graph.get_tensor_by_name(gr_name_prefix + "inner_ele:0")
    inner_muon_gr = graph.get_tensor_by_name(gr_name_prefix + "inner_muon:0")
    outer_pfCand_gr = graph.get_tensor_by_name(gr_name_prefix + "outer_pfCand:0")
    outer_ele_gr = graph.get_tensor_by_name(gr_name_prefix + "outer_ele:0")
    outer_muon_gr = graph.get_tensor_by_name(gr_name_prefix + "outer_muon:0")

    y_gr = graph.get_tensor_by_name("deepTau/main_output/Softmax:0")
    N = X_taus.shape[0]
    # if np.any(np.isnan(X_taus)) or np.any(np.isnan(X_cells_pfCand)) or np.any(np.isnan(X_cells_ele)) \
    #     or np.any(np.isnan(X_cells_muon)):
    #     raise RuntimeErrror("Nan in inputs")
    pred = session.run(y_gr, feed_dict={
        tau_gr: X_taus, inner_pfCand_gr: X_inner_pfCand, inner_ele_gr: X_inner_ele, inner_muon_gr: X_inner_muon,
        outer_pfCand_gr: X_outer_pfCand, outer_ele_gr: X_outer_ele, outer_muon_gr: X_outer_muon,
    })
    if np.any(np.isnan(pred)):
        raise RuntimeError("NaN in predictions. Total count = {} out of {}".format(np.count_nonzero(np.isnan(pred)), pred.shape))
    if np.any(pred < 0) or np.any(pred > 1):
        raise RuntimeError("Predictions outside [0, 1] range.")
    return pandas.DataFrame(data = {
        'deepId_e': pred[:, e], 'deepId_mu': pred[:, mu], 'deepId_tau': pred[:, tau],
        'deepId_jet': pred[:, jet]
    })

if args.filelist is None:
    if os.path.isdir(args.input):
        file_list = [ f for f in os.listdir(args.input) if f.endswith('.root') or f.endswith('.h5') ]
        prefix = args.input + '/'
    else:
        file_list = [ args.input ]
        prefix = ''
else:
    with open(args.filelist, 'r') as f_list:
        file_list = [ f.strip() for f in f_list if len(f) != 0 ]

if len(file_list) == 0:
    raise RuntimeError("Empty input list")
#if args.max_n_files is not None and args.max_n_files > 0:
#    file_list = file_list[0:args.max_n_files]


graph = load_graph(args.model)
sess = tf.Session(graph=graph)

file_index = 0
for file_name in file_list:
    if args.max_n_files is not None and file_index >= args.max_n_files: break
    print("Processing '{}'".format(file_name))
    full_name = prefix + file_name

    pred_output = args.output + '/' + file_name + '_pred.hdf5'
    if os.path.isfile(pred_output):
        os.remove(pred_output)

#    n_entries = GetNumberOfEntries(full_name, args.tree)
#    if args.max_n_entries_per_file is not None:
#        n_entries = min(n_entries, args.max_n_entries_per_file)
#    current_start = 0

    loader = DataLoader(full_name, args.batch_size, args.chunk_size, max_data_size=args.max_n_entries_per_file,
                        n_passes = 1)

    with tqdm(total=loader.data_size, unit='taus') as pbar:
        for inputs in loader.generator(return_truth = False, return_weights = False):
            df = Predict(sess, graph, *inputs)
            df.to_hdf(args.output + '/' + file_name + '_pred.hdf5', args.tree, append=True, complevel=1, complib='zlib')
            pbar.update(df.shape[0])
            del df
            gc.collect()
    file_index += 1

print("All files processed.")
