#!/usr/bin/env python
# Skim TTree.
# This file is part of https://github.com/hh-italian-group/AnalysisTools.

import argparse
import os

parser = argparse.ArgumentParser(description='Skim tree.')
parser.add_argument('--input', required=True, type=str, help="input root file or txt file with a list of files")
parser.add_argument('--output', required=True, type=str, help="output root file")
parser.add_argument('--tree', required=True, type=str, help="selection")
parser.add_argument('--other-trees', required=False, type=None, help="other trees to copy")
parser.add_argument('--sel', required=False, type=str, default=None, help="selection")
parser.add_argument('--include-columns', required=False, default=None, type=str, help="columns to be included")
parser.add_argument('--exclude-columns', required=False, default=None, type=str, help="columns to be excluded")
parser.add_argument('--input-prefix', required=False, type=str, default='',
                    help="prefix to be added to input each input file")
parser.add_argument('--processing-module', required=False, type=str, default=None,
                    help="Python module used to process DataFrame. Should be in form file:method")
parser.add_argument('--comp-algo', required=False, type=str, default='LZMA', help="compression algorithm")
parser.add_argument('--comp-level', required=False, type=int, default=9, help="compression level")
parser.add_argument('--n-threads', required=False, type=int, default=None, help="number of threads")
parser.add_argument('--input-range', required=False, type=str, default=None,
                    help="read only entries in range begin:end (before any selection)")
parser.add_argument('--output-range', required=False, type=str, default=None,
                    help="write only entries in range begin:end (after all selections)")
args = parser.parse_args()

import ROOT
ROOT.gROOT.SetBatch(True)

if args.n_threads is None:
    if args.input_range is not None or args.output_range is not None:
        args.n_threads = 1
    else:
        args.n_threads = 4

if args.n_threads > 1:
    ROOT.ROOT.EnableImplicitMT(args.n_threads)
columns_to_include = []
if args.include_columns is not None:
    columns_to_include = args.include_columns.split(',')
columns_to_exclude = []
if args.exclude_columns is not None:
    columns_to_exclude = args.exclude_columns.split(',')

inputs = ROOT.vector('string')()
if args.input.endswith('.root'):
    inputs.push_back(args.input_prefix + args.input)
    print("Adding input '{}'...".format(args.input))
elif args.input.endswith('.txt'):
    with open(args.input, 'r') as input_list:
        for name in input_list.readlines():
            name = name.strip()
            if len(name) > 0 and name[0] != '#':
                inputs.push_back(args.input_prefix + name)
                print("Adding input '{}'...".format(name))
elif os.path.isdir(args.input):
    for f in os.listdir(args.input):
        if not f.endswith('.root'): continue
        file_name = os.path.join(args.input, f)
        inputs.push_back(file_name)
        print("Adding input '{}'...".format(file_name))
else:
    raise RuntimeError("Input format is not supported.")

df = ROOT.RDataFrame(args.tree, inputs)

if args.input_range is not None:
    begin, end = [ int(x) for x in args.input_range.split(':') ]
    df = df.Range(begin, end)

if args.processing_module is not None:
    module_desc = args.processing_module.split(':')
    import imp
    module = imp.load_source('processing', module_desc[0])
    fn = getattr(module, module_desc[1])
    df = fn(df)

branches = ROOT.vector('string')()
for column in df.GetColumnNames():
    include_column = False
    if len(columns_to_include) == 0 or column in columns_to_include:
        include_column = True
    if column in columns_to_exclude:
        include_column = False
    if include_column:
        branches.push_back(column)
        print("Adding column '{}'...".format(column))

if args.sel is not None:
    df = df.Filter(args.sel)

if args.output_range is not None:
    begin, end = [ int(x) for x in args.output_range.split(':') ]
    df = df.Range(begin, end)

print("Creating a snapshot...")
opt = ROOT.RDF.RSnapshotOptions()
opt.fCompressionAlgorithm = getattr(ROOT.ROOT, 'k' + args.comp_algo)
opt.fCompressionLevel = args.comp_level
df.Snapshot(args.tree, args.output, branches, opt)

if args.other_trees is not None:
    other_trees = args.other_trees.split(',')
    for tree_name in other_trees:
        print("Copying {}...".format(tree_name))
        other_df = ROOT.RDataFrame(tree_name, inputs)
        branches = ROOT.vector('string')()
        for column in other_df.GetColumnNames():
            branches.push_back(column)
            print("\tAdding column '{}'...".format(column))
        opt.fMode = 'UPDATE'
        other_df.Snapshot(tree_name, args.output, branches, opt)

print("Skim successfully finished.")
