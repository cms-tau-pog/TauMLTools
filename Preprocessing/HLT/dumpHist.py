import os
import sys
import math
import numpy as np
import yaml
import json

if __name__ == "__main__":
  file_dir = os.path.dirname(os.path.abspath(__file__))
  base_dir = os.path.dirname(file_dir)
  if base_dir not in sys.path:
    sys.path.append(base_dir)
  __package__ = os.path.split(file_dir)[-1]

from .AnalysisTools import *
from .MixStep import MixStep

import ROOT
ROOT.gROOT.SetBatch(True)
ROOT.EnableThreadSafety()


def GetBinContent(hist, pt, eta):
  x_axis = hist.GetXaxis()
  x_bin = x_axis.FindFixBin(pt)
  y_axis = hist.GetYaxis()
  y_bin = y_axis.FindFixBin(eta)
  return hist.GetBinContent(x_bin, y_bin)

def GetBinEdges(axis):
  edges = []
  for bin in range(1, axis.GetNbins() + 2):
    edges.append(axis.GetBinLowEdge(bin))
  return np.array(edges)

def dump_hist(hist_name, input_files):
  result = None
  for file_name in input_files:
    file = ROOT.TFile.Open(file_name, "READ")
    hist = file.Get(hist_name)
    if result:
      result.Add(hist)
    else:
      result = hist
  x_bins = GetBinEdges(result.GetXaxis())
  y_bins = GetBinEdges(result.GetYaxis())
  content = np.zeros((len(x_bins) - 1, len(y_bins) - 1))
  for x_bin in range(1, result.GetNbinsX() + 1):
    for y_bin in range(1, result.GetNbinsY() + 1):
      content[x_bin-1, y_bin-1] = result.GetBinContent(x_bin, y_bin)

  bin_dict = []
  for x_bin in range(len(x_bins) - 1):
    for y_bin in range(len(y_bins) - 1):
      bin_dict.append( {
        'pt': (x_bins[x_bin], x_bins[x_bin + 1]),
        'eta': (y_bins[y_bin], y_bins[y_bin + 1]),
        'count': content[x_bin, y_bin],
      })
  print(f'x_bins = {x_bins.tolist()}')
  print(f'y_bins = {y_bins.tolist()}')
  #print(json.dumps(bin_dict, indent=2))

if __name__ == "__main__":
  import argparse
  parser = argparse.ArgumentParser(description='Make mix.')
  parser.add_argument('--hist', required=True, type=str, help="name of the histogram")
  parser.add_argument('input', nargs='+', type=str, help="input files")
  args = parser.parse_args()

  dump_hist(args.hist, args.input)
