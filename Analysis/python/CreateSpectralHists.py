#!/usr/bin/env python

import os
import re
import argparse
import shutil
from multiprocessing import Pool
import random

parser = argparse.ArgumentParser(description='Creating histograms for shuffle and merge.')
parser.add_argument('--input', required=True, type=str, help="input directory")
parser.add_argument('--output', required=False, type=str, default="same", help="output directory or \
                                                            --output same to put at input directory")
parser.add_argument('--n-threads', required=False, type=int, default=1, help="number of threads")
parser.add_argument('--filter', required=False, type=str, default='.*', help="regex filter for dataset names")
args = parser.parse_args()

if not os.path.isdir(args.input):
    raise RuntimeError("Input directory '{}' not found".format(args.input))

if not os.path.isdir(args.output):
	os.makedirs(args.output)

def execute_cmd(cmd):
    return os.system(cmd)

pt_hist = "200, 0, 1000"
eta_hist = "4, 0, 2.3"

# shuffle is needed to distribute load among cores a bit
foldernames_all = os.listdir(args.input)
random.shuffle(foldernames_all)
print(foldernames_all)

foldernames=[]
for dir_name in foldernames_all:
    if not(re.match(args.filter, dir_name) is None):
        foldernames.append(dir_name)


commands = []
for dir_name in foldernames:

    input_dir_path = os.path.join(args.input, dir_name)
    if not os.path.isdir(input_dir_path): continue

    if args.output=="same":
        output_file = os.path.join(input_dir_path, dir_name+".root")
    else:
        output_file = os.path.join(args.output, dir_name+".root")

    if os.path.exists(output_file):
        print("{} was already processed.".format(input_dir_path))
        continue

    print("Added {}...".format(input_dir_path))

    cmd = 'CreateSpectralHists --output "{}" --input-dir "{}" --pt-hist "{}" --eta-hist "{}"' \
        .format(output_file, input_dir_path, pt_hist, eta_hist)
    commands.append(cmd)

# print commands
p = Pool(args.n_threads)
p.map(execute_cmd,commands)
