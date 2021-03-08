#!/usr/bin/env python

import os
import sys
import argparse
import fnmatch
import uproot

parser = argparse.ArgumentParser(description='Create size list.')
parser.add_argument('--input', required=True, type=str, help="Input directory")
parser.add_argument('--prev-output', required=False, type=str, default=None, help="Previous output")
args = parser.parse_args()

if not os.path.isdir(args.input):
    raise RuntimeError("Input directory '{}' not found".format(args.input))

prev_results = {}
if args.prev_output is not None:
    with open(args.prev_output, 'r') as prev_output:
        for line in prev_output.readlines():
            split = line.split(' ')
            if len(split) != 2:
                raise RuntimeError("invalid previous ouptput")
            file_name = split[0].strip()
            n_events = int(split[1])
            prev_results[file_name] = n_events

all_file_names = []

for root_dir, dir_names, file_names in os.walk(args.input):
    for file_name in fnmatch.filter(file_names, '*.root'):
        full_file_name = os.path.join(root_dir, file_name)
        all_file_names.append(full_file_name)

all_file_names = sorted(all_file_names)

for full_file_name in all_file_names:
    rel_file_name = os.path.relpath(full_file_name, args.input)
    if rel_file_name in prev_results:
        n_events = prev_results[rel_file_name]
    else:
        with uproot.open(full_file_name) as file:
            keys = [ key[:key.rindex(b';')].decode('utf-8') for key in file.keys() ]
            if 'taus' in keys:
                tree = file['taus']
                n_events = tree.numentries
            else:
                raise RuntimeError('TTree with name "taus" is not found in "{}". Available keys: {}' \
                      .format(full_file_name, file.keys()))
    print("{} {}".format(rel_file_name, n_events))
