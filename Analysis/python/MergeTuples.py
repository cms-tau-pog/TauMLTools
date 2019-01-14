#!/usr/bin/env python

import os
import argparse
import subprocess

parser = argparse.ArgumentParser(description='Deploy keras model.')
parser.add_argument('--input', required=True, type=str, help="input directory")
parser.add_argument('--output', required=True, type=str, help="output directory")
parser.add_argument('--n-threads', required=False, type=int, default=1, help="number of threads")
args = parser.parse_args()

if not os.path.isdir(args.input):
    raise RuntimeError("Input directory '{}' not found".format(args.input))

if not os.path.isdir(args.output):
	os.makedirs(args.output)

crab_prefix = "crab_"

for dir_name in os.listdir(args.input):
    dir_path = os.path.join(args.input, dir_name)
    if not os.path.isdir(dir_path): continue
    if dir_name[0:len(crab_prefix)] == crab_prefix:
        dataset_name = dir_name[len(crab_prefix):]
    else:
        dataset_name = dir_name
    file_name = os.path.join(args.output, dataset_name + ".root")
    if os.path.exists(file_name):
        print("{} was already processed.".format(dataset_name))
        continue

    print("Processing {}...".format(dataset_name))

    cmd = './run.sh MergeTuples --output "{}" --input-dir "{}" --n-threads {}' \
          .format(file_name, dir_path, args.n_threads)
    result = subprocess.call([cmd], shell=True)
    if result != 0:
        if os.path.exists(file_name):
            os.remove(file_name)
        raise RuntimeError("MergeTuples has failed.")
    print("{} has been successfully processed".format(dataset_name))
