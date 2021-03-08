#!/usr/bin/env python

import os
import re
import argparse
import subprocess
import shutil

parser = argparse.ArgumentParser(description='Deploy keras model.')
parser.add_argument('--input', required=True, type=str, help="input directory")
parser.add_argument('--output', required=True, type=str, help="output directory")
parser.add_argument('--n-threads', required=False, type=int, default=1, help="number of threads")
parser.add_argument('--filter', required=False, type=str, default='.*', help="regex filter for dataset names")
args = parser.parse_args()

if not os.path.isdir(args.input):
    raise RuntimeError("Input directory '{}' not found".format(args.input))

if not os.path.isdir(args.output):
	os.makedirs(args.output)

crab_prefix = "crab_"
pt_bins = "20, 25, 30, 35, 40, 45, 50, 55, 60, 70, 80, 90, 100, 120, 140, 160, 180, 200, 225, 250, 275, 300, 350, 400, 450, 500, 600, 700, 800, 900, 1000"
eta_bins = "0., 0.575, 1.15, 1.725, 2.3"

for dir_name in os.listdir(args.input):
    input_dir_path = os.path.join(args.input, dir_name)
    if not os.path.isdir(input_dir_path): continue
    if dir_name[0:len(crab_prefix)] == crab_prefix:
        dataset_name = dir_name[len(crab_prefix):]
    else:
        dataset_name = dir_name
    if re.match(args.filter, dataset_name) is None: continue
    
    output_dir_path = os.path.join(args.output, dataset_name)
    if os.path.exists(output_dir_path):
        print("{} was already processed.".format(dataset_name))
        continue

    print("Processing {}...".format(dataset_name))

    cmd = './run.sh CreateBinnedTuples --output "{}" --input-dir "{}" --n-threads {} --pt-bins "{}" --eta-bins "{}"' \
          .format(output_dir_path, input_dir_path, args.n_threads, pt_bins, eta_bins)
    result = subprocess.call([cmd], shell=True)
    if result != 0:
        if os.path.exists(output_dir_path):
            shutil.rmtree(output_dir_path)
        raise RuntimeError("MergeTuples has failed.")
    print("{} has been successfully processed".format(dataset_name))
