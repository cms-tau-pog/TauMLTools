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
parser.add_argument('--max-queue-size', required=False, type=int, default=8, help="Maximal queue size")
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
from DataLoader import DataLoader, read_hdf_lock

class Predictor:
    def __init__(self, graph, net_config):
        gr_name_prefix = "deepTau/input_"
        self.x_graphs = []
        if len(net_config.tau_branches):
            self.x_graphs.append(graph.get_tensor_by_name(gr_name_prefix + "tau:0"))
        for loc in net_config.cell_locations:
            for comp_name in net_config.comp_names:
                gr_name = '{}{}_{}:0'.format(gr_name_prefix, loc, comp_name)
                self.x_graphs.append(graph.get_tensor_by_name(gr_name))
        self.y_graph = graph.get_tensor_by_name("deepTau/main_output/Softmax:0")

    def Predict(self, session, X):
        if len(self.x_graphs) != len(X):
            raise RuntimeError("Inconsistent size of the inputs.")

        feed_dict = {}
        for n in range(len(self.x_graphs)):
            feed_dict[self.x_graphs[n]] = X[n]
        pred = session.run(self.y_graph, feed_dict=feed_dict)
        if np.any(np.isnan(pred)):
            raise RuntimeError("NaN in predictions. Total count = {} out of {}".format(
                               np.count_nonzero(np.isnan(pred)), pred.shape))
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
net_conf = netConf_full
predictor = Predictor(graph, net_conf)

file_index = 0
for file_name in file_list:
    if args.max_n_files is not None and file_index >= args.max_n_files: break
    full_name = prefix + file_name

    pred_output = args.output + '/' + os.path.splitext(file_name)[0] + '_pred.h5'
    if os.path.isfile(pred_output):
        print('"{}" already present in the output directory.'.format(pred_output))
        continue
        #os.remove(pred_output)
    print("Processing '{}' -> '{}'".format(file_name, os.path.basename(pred_output)))

    loader = DataLoader(full_name, net_conf, args.batch_size, args.chunk_size,
                        max_data_size = args.max_n_entries_per_file, max_queue_size = args.max_queue_size,
                        n_passes = 1, return_grid = True)

    with tqdm(total=loader.data_size, unit='taus') as pbar:
        for inputs in loader.generator(return_truth = False, return_weights = False):
            df = predictor.Predict(sess, inputs)
            read_hdf_lock.acquire()
            df.to_hdf(pred_output, args.tree, append=True, complevel=1, complib='zlib')
            read_hdf_lock.release()
            pbar.update(df.shape[0])
            del df
            gc.collect()
    file_index += 1

print("All files processed.")
