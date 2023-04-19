#!/usr/bin/env python

import argparse
parser = argparse.ArgumentParser(description='Sanity check for all variables.')
parser.add_argument('--input', required=True, type=str, help="Input root file")
parser.add_argument('--tree', required=True, type=str, help="Tree name")
parser.add_argument('--max-entries', required=False, type=int, default=None,
                    help="Maximal number of entries to process")
parser.add_argument('--std-thr', required=False, type=float, default=1e-7,
                    help="If std(var) < std_thr, var is considered constant.")
parser.add_argument('--valid-thr', required=False, type=float, default=-1e9,
                    help="Only values > valid_thr will be used.")
parser.add_argument('--separator', required=False, type=str, default=',', help="Separator for the CSV output.")


args = parser.parse_args()

import uproot
import awkward as ak
import numpy as np

with uproot.open(args.input) as file:
    tree = file[args.tree]
    df = tree.arrays(entry_stop=args.max_entries)

print(args.separator.join(["feature", "min", "max", "average", "std", "abs_max", "abs_avg", "is_const"]))
for column in sorted(tree.keys()):
    values = df[column]
    values = values[values > args.valid_thr]
    if len(values) > 0:
        if values.ndim > 1:
            values = ak.flatten(values)
        avg = np.average(values)
        std = np.std(values)
        amin = np.amin(values)
        amax = np.amax(values)
        abs_values = np.abs(values)
        abs_avg = np.average(abs_values)
        abs_max = np.amax(abs_values)
    else:
        avg = std = abs_avg = abs_max = amax = amin = np.NAN

    is_const = not (std > args.std_thr)
    row = [ column, f'{amin:.4E}', f'{amax:.4E}', f'{avg:.4E}', f'{std:.4E}', f'{abs_max:.4E}', f'{abs_avg:.4E}',
            str(is_const) ]
    print(args.separator.join(row))
