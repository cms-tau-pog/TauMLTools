#!/usr/bin/env python

import argparse
parser = argparse.ArgumentParser(description='Apply training and store results.')
parser.add_argument('--input', required=True, type=str, help="Input directory")
parser.add_argument('--filelist', required=False, type=str, default=None, help="Txt file with input tuple list")
parser.add_argument('--output', required=True, type=str, help="Output file")
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
import math

from TauMLTools.Training.common import *
from TauMLTools.Training.DataLoader import DataLoader, read_hdf_lock

class Predictor:
    def Predict(self, X):
        if len(X[0][0]) != 2:
            raise RuntimeError("Inconsistent size of the inputs.")
       
        pred = []
        numRows = len(X[0])
        for idxRow in range(numRows):
            tau_pt = X[0][idxRow][0]
            chargedIsoPtSum = X[0][idxRow][1]
            score = 0.5*(1.0 + math.tanh(-0.01*chargedIsoPtSum/max(1.e-9, tau_pt)))
            ##print("row #%i: pT = %1.3f, I_ch = %1.3f --> score = %1.3f" % (idxRow, tau_pt, chargedIsoPtSum, score))
            pred.append(score)
        pred = np.array(pred)
        if np.any(np.isnan(pred)):
            raise RuntimeError("NaN in predictions. Total count = {} out of {}".format(
                               np.count_nonzero(np.isnan(pred)), pred.shape))
        if np.any(pred < 0) or np.any(pred > 1):
            raise RuntimeError("Predictions outside [0, 1] range.")
        return pandas.DataFrame(data = {
            'deepId_e': 0., 'deepId_mu': 0., 'deepId_tau': pred,
            'deepId_jet': [ (1.0 - pred[i]) for i in range(len(pred)) ]
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

predictor = Predictor()
net_conf = NetConf("dummyTau", False, [ "tau_pt", "chargedIsoPtSum" ], [], [], [])

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
            df = predictor.Predict(inputs)
            read_hdf_lock.acquire()
            df.to_hdf(pred_output, args.tree, append=True, complevel=1, complib='zlib')
            read_hdf_lock.release()
            pbar.update(df.shape[0])
            del df
            gc.collect()
    file_index += 1

print("All files processed.")
