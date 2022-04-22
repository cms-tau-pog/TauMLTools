#!/usr/bin/env python

import os
import re
import argparse
import shutil
import random
from glob import glob
import subprocess

parser = argparse.ArgumentParser(description='Creating histograms for shuffle and merge.')
parser.add_argument('--input', required=True, type=str, help="input directory")
parser.add_argument('--output', required=True, type=str, help="output directory")
parser.add_argument('--n-threads', required=False, type=int, default=1, help="number of threads")
parser.add_argument('--filter', required=False, type=str, default='.*', help="regex filter for dataset names")
parser.add_argument('--rewrite',required=False, action='store_true', default=False, help="rewrite existing histograms")
args = parser.parse_args()

if not os.path.isdir(args.input):
    raise RuntimeError("Input directory '{}' not found".format(args.input))

if not os.path.isdir(args.output):
	os.makedirs(args.output)

pt_hist = "4990, 10, 5000"
eta_hist = "25, 0, 2.5"

print ("regex filter for dataset names: "+args.filter)

input_path = []
for dir_name in glob(args.input+"/*"):
    if (re.match(args.filter, dir_name) is None):
        continue
    else:
        input_path.append(dir_name)

print ("list of input files: ")
print (input_path)

for dir_path in input_path:

    split_path = dir_path.split("/")
    if not os.path.isdir(dir_path): continue

    output_root = args.output + "/" \
                + split_path[-1] + ".root"
    output_entries = args.output + "/" \
                + split_path[-1] + ".txt"
                

    if os.path.exists(output_root):
        print("{} was already processed.".format(dir_path))
        if args.rewrite:
            print("{} was removed".format(output_root))
            os.remove(output_root)
        else:
            continue
    if os.path.exists(output_entries):
        print("{} was already processed.".format(dir_path))
        if args.rewrite:
            print("{} was removed".format(output_entries))
            os.remove(output_entries)
        else:
            continue

    print("Added {}...".format(dir_path))

    cmd = 'CreateSpectralHists --outputfile "{}" --output_entries "{}" --input-dir "{}" --pt-hist "{}" --eta-hist "{}" --n-threads {}' \
        .format(output_root, output_entries, dir_path, pt_hist, eta_hist, args.n_threads)
    result = subprocess.call([cmd], shell=True)

    if result != 0:
        if os.path.exists(output_root):
            os.remove(output_root)
        raise RuntimeError("MergeTuples has failed.")
    print("{} has been successfully processed".format(dir_path))

