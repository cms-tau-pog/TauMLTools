#!/usr/bin/env python

import os
import re
import argparse
from glob import glob

from RunKit.sh_tools import sh_call, ShCallError

parser = argparse.ArgumentParser(description='Creating histograms for shuffle and merge.')
parser.add_argument('--input', required=True, type=str, help="input directory")
parser.add_argument('--output', required=True, type=str, help="output directory")
parser.add_argument('--n-threads', required=False, type=int, default=1, help="number of threads")
parser.add_argument('--filter', required=False, type=str, default='.*', help="regex filter for dataset names")
parser.add_argument('--rewrite',required=False, action='store_true', default=False, help="rewrite existing histograms")
parser.add_argument('--mode',required=False, type=str, default="tau",
    help="eta phi of the following object will be recorded. Currently available: 1)boostedTau 2)tau 3)jet")
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

    ana_path = os.environ['ANALYSIS_PATH']

    cmd = [ 'python', os.path.join(ana_path, 'Core', 'python', 'run_cxx.py'),
        os.path.join(ana_path, 'PreProcessing', 'CreateSpectralHists.cxx'),
        '--outputfile', output_root,
        '--output_entries', output_entries,
        '--input-dir', dir_path,
        '--pt-hist', pt_hist,
        '--eta-hist', eta_hist,
        '--n-threads', str(args.n_threads),
        '--mode', args.mode,
    ]
    try:
        sh_call(cmd)
    except ShCallError as e:
        if os.path.exists(output_root):
            os.remove(output_root)
        raise RuntimeError("CreateSpectralHists has failed.") from e
    print("{} has been successfully processed".format(dir_path))

