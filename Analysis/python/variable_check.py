#!/usr/bin/env python

import argparse
parser = argparse.ArgumentParser(description='Sanity check for all variables.')
parser.add_argument('--input', required=True, type=str, help="Input root file")
parser.add_argument('--tree', required=True, type=str, help="Tree name")
parser.add_argument('--max-entries', required=False, type=int, default=None,
                    help="Maximal number of entries to process")
args = parser.parse_args()

import uproot
import pandas
import numpy as np

with uproot.open(args.input) as file:
    tree = file[args.tree]
    df = tree.arrays('*', outputtype=pandas.DataFrame, entrystop=args.max_entries)

def get_flat(values):
    if type(df[column].values[0]) != np.ndarray:
        return values
    result = np.array([])
    for n in range(values.shape[0]):
        result = np.append(result, values[n])
    return result


std_thr = 1e-7

print("dataset size: {} features, {} entries".format(df.shape[1], df.shape[0]))
print("{:<50}{:>15}{:>15}{:>15}{:>15}{:>16}{:>10}".format("feature", "min", "max", "average", "std", "abs_max/abs_avg",
                                                          "is_const"))
for column in df.columns:
    values = get_flat(df[column])
    if values.shape[0] == 0:
        raise RuntimeError("Feature '' is empty.".format(column))
    avg = np.average(values)
    std = np.std(values)
    amin = np.amin(values)
    amax = np.amax(values)
    abs_values = np.abs(values)
    abs_max = np.amax(abs_values)
    abs_avg = np.average(abs_values)
    is_const = std < std_thr
    print("{:<50}{:>15.4E}{:>15.4E}{:>15.4E}{:>15.4E}{:>16.4E}{:>10}".format(column, amin, amax, avg, std,
          abs_max / abs_avg, str(is_const)))
