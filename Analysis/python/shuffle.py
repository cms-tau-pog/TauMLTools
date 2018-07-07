#!/usr/bin/env python

import argparse
parser = argparse.ArgumentParser(description='Shuffle hdf5 container.')
parser.add_argument('--input', required=True, type=str, help="Input ROOT file")
parser.add_argument('--tree', required=False, type=str, default="taus", help="Tree name")
args = parser.parse_args()

import random
import h5py

with h5py.File(args.input, 'r+') as file:
    print("Number of entries = {}.".format(file[args.tree]["table"].shape[0]))
    random.shuffle(file[args.tree]["table"])
