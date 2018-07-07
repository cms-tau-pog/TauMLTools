#!/usr/bin/env python

import argparse
parser = argparse.ArgumentParser(description='Convert ROOT TTree to hdf5 container.')
parser.add_argument('--input', required=True, type=str, help="Input ROOT file")
parser.add_argument('--output', required=True, type=str, help="Output HDF5 file")
parser.add_argument('--tree', required=False, type=str, default="taus", help="Tree name")
parser.add_argument('--chunk-size', required=False, type=int, default=100000, help="Number of entries per iteration")
args = parser.parse_args()

import os
import uproot
import pandas
from root_pandas import read_root
from tqdm import tqdm

if os.path.isfile(args.output):
    os.remove(args.output)

with uproot.open(args.input) as file:
    tree = file[args.tree]
    n_total = tree.numentries

first_pass = True
boolean_columns = ""
with tqdm(total=n_total, unit='entries') as pbar:
    for df in read_root(args.input, args.tree, chunksize=args.chunk_size):
        for n in range(df.shape[1]):
            if df.dtypes[n] == bool:
                df[df.columns[n]] = pandas.Series(df[df.columns[n]].astype(int), index=df.index)
                if first_pass:
                    boolean_columns += df.columns[n] + ","
        df.to_hdf(args.output, args.tree, append=True)
        pbar.update(df.shape[0])
        first_pass = False

print("Boolean columns: {}".format(boolean_columns))
print("All entries has been processed.")
