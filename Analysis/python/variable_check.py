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
    #df = tree.arrays('*', outputtype=pandas.DataFrame, entrystop=args.max_entries)
    df = tree.arrays('*', entrystop=args.max_entries)

std_thr = 1e-7
valid_thr = -1e9

#print("dataset size: {} features, {} entries".format(df.shape[1], df.shape[0]))
print(",".join(["feature", "min", "max", "average", "std", "abs_max", "abs_avg", "is_const"]))
# print("{:<50}{:>15}{:>15}{:>15}{:>15}{:>15}{:>15}{:>10}".format("feature", "min", "max", "average", "std", "abs_max",
#                                                                 "abs_avg", "is_const"))
for column in sorted(df):
    #values = get_flat(df[column])
    values = df[column].flatten()
    values = values[values > valid_thr]
    # if values.shape[0] == 0:
    #     raise RuntimeError("Feature '' is empty.".format(column))
    # if column == "isoTrack_dz_error":
    #     print(column)
    #     print(values)
    if values.shape[0] > 0:
        avg = np.average(values)
        std = np.std(values)
        amin = np.amin(values)
        amax = np.amax(values)
        abs_values = np.abs(values)
        abs_avg = np.average(abs_values)
        abs_max = np.amax(abs_values)
    else:
        avg = std = abs_avg = abs_max = amax = amin = np.NAN

    is_const = not (std > std_thr)

    print("{},{:.4E},{:.4E},{:.4E},{:.4E},{:.4E},{:.4E},{}".format(column, amin, amax, avg, std, abs_max, abs_avg,
                                                                  str(is_const)))
    # print("{:<50}{:>15.4E}{:>15.4E}{:>15.4E}{:>15.4E}{:>15.4E}{:>15.4E}{:>10}".format(column, amin, amax, avg, std,
    #       abs_max, abs_avg, str(is_const)))
