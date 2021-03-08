#!/usr/bin/env python

import ROOT
import argparse

parser = argparse.ArgumentParser(description='Copy selected events.')
parser.add_argument('--input', required=True, type=str, help="Input tuple")
parser.add_argument('--output', required=True, type=str, help="Output tuple")
parser.add_argument('--nevents', required=True, type=float, help="Number of events to keep")
parser.add_argument('--ncells', required=True, type=float, help="Number of cells per tau")
args = parser.parse_args()

ROOT.gROOT.ProcessLine('ROOT::EnableImplicitMT(12)')

print("Number of events to keep: {}".format(args.nevents))

f_in = ROOT.TFile(args.input, "READ")
taus = f_in.Get("taus")
cells = f_in.Get("cells")

f_out = ROOT.TFile(args.output, "RECREATE")
f_out.SetCompressionAlgorithm(4) # LZ4
f_out.SetCompressionLevel(5)

taus_out = taus.CloneTree(0)
taus_out.CopyEntries(taus, int(args.nevents), "SortBasketsByBranch")
f_out.WriteTObject(taus_out, "taus", "Overwrite")

cells_out = cells.CloneTree(0)
cells_out.CopyEntries(cells, int(args.nevents * args.ncells), "SortBasketsByBranch")
f_out.WriteTObject(cells_out, "cells", "Overwrite")

f_out.Close()
f_in.Close()
