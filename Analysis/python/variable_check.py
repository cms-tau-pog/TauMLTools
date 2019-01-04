#!/usr/bin/env python

import argparse
parser = argparse.ArgumentParser(description='Sanity check for all variables.')
parser.add_argument('--input', required=True, type=str, help="Input root file")
parser.add_argument('--tree', required=True, type=str, help="Tree name")
args = parser.parse_args()

import uproot
import pandas
import numpy as np

with uproot.open(args.input) as file:
    tree = file[args.tree]
    df = tree.arrays('*', outputtype=pandas.DataFrame)

def get_flat(values):
    if type(df[column].values[0]) != np.ndarray:
        return values
    result = np.array([])
    for n in range(values.shape[0]):
        result = np.append(result, values[n])
    return result


std_thr = 1e-7

print("dataset size: {} features, {} entries".format(df.shape[1], df.shape[0]))
print("{:<50}{:>15}{:>15}{:>15}{:>15}{:>10}".format("feature", "min", "max", "average", "std", "is_const"))
for column in df.columns:
    values = get_flat(df[column])
    if values.shape[0] == 0:
        raise RuntimeError("Feature '' is empty.".format(column))
    avg = np.average(values)
    std = np.std(values)
    amin = np.amin(values)
    amax = np.amax(values)
    is_const = std < std_thr
    print("{:<50}{:>15.4E}{:>15.4E}{:>15.4E}{:>15.4E}{:>10}".format(column, amin, amax, avg, std, str(is_const)))

