#!/usr/bin/env python

import argparse
parser = argparse.ArgumentParser(description='Convert hdf5 container to ROOT TTree.')
parser.add_argument('--input', required=True, type=str, help="Input HDF5 file")
parser.add_argument('--output', required=True, type=str, help="Output ROOT file")
parser.add_argument('--tree', required=False, type=str, default="taus", help="Tree name")
parser.add_argument('--chunk-size', required=False, type=int, default=100000, help="Number of entries per iteration")
parser.add_argument('--boolean-columns', required=False, type=str, default="", help="List of boolean columns")
args = parser.parse_args()

import os
import h5py
import pandas
import root_pandas
from tqdm import tqdm

if os.path.isfile(args.output):
    os.remove(args.output)

with h5py.File(args.input, 'r') as file:
    n_total = file[args.tree]['table'].shape[0]

boolean_columns = [ c.strip() for c in args.boolean_columns.split(',') if len(c.strip()) != 0 ]

with tqdm(total=n_total, unit='entries') as pbar:
    for df in pandas.read_hdf(args.input, args.tree, chunksize=args.chunk_size):
        for c in boolean_columns:
            df[c] = pandas.Series(df[c].astype(bool), index=df.index)
        df.to_root(args.output, mode='a', key=args.tree)
        pbar.update(df.shape[0])

print("All entries has been processed.")
