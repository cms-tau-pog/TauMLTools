#!/usr/bin/env python

import argparse
parser = argparse.ArgumentParser(description='Shuffle hdf5 container.')
parser.add_argument('--input', required=True, type=str, help="Input ROOT file")
parser.add_argument('--tree', required=False, type=str, default="taus", help="Tree name")
args = parser.parse_args()

import random
import h5py
from tqdm import tqdm

# based on: https://svn.python.org/projects/python/trunk/Lib/random.py
def shuffle(x):
    with tqdm(total=len(x) - 1, unit='entries') as pbar:
        n_proc = 0
        for i in reversed(range(1, len(x))):
            j = int(random.random() * (i+1))
            x[i], x[j] = x[j], x[i]
            n_proc += 1
            if (n_proc == 100000) or (i == 1):
                pbar.update(n_proc)
                n_proc = 0


with h5py.File(args.input, 'r+') as file:
    print("Number of entries = {}.".format(file[args.tree]["table"].shape[0]))
    shuffle(file[args.tree]["table"])
