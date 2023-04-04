from enum import Enum
import os
import sys

if __name__ == "__main__":
  file_dir = os.path.dirname(os.path.abspath(__file__))
  base_dir = os.path.dirname(file_dir)
  if base_dir not in sys.path:
    sys.path.append(base_dir)
  __package__ = os.path.split(file_dir)[-1]

from .AnalysisTools import *

import ROOT

def make_hists(input_files, input_tree, output, x_bins, y_bins, tau_types):
  input_files_vec = ListToVector(input_files, "string")
  df = ROOT.RDataFrame("Events", input_files_vec)
  df = ApplyCommonDefinitions(df=df)
  x_bins_vec = ListToVector(x_bins, "double")
  y_bins_vec = ListToVector(y_bins, "double")
  model = ROOT.RDF.TH2DModel("spectrum", "spectrum", x_bins_vec.size() - 1, x_bins_vec.data(),
                             y_bins_vec.size() - 1, y_bins_vec.data())
  hists = {}
  for tau_type in tau_types:
    df_tau = df.Define("L1Tau_gen_pt_sel", f"L1Tau_gen_pt[L1Tau_type == {tau_type.value}]") \
               .Define("L1Tau_gen_abs_eta_sel", f"L1Tau_gen_abs_eta[L1Tau_type == {tau_type.value}]")
    hists[tau_type] = df_tau.Histo2D(model, "L1Tau_gen_pt_sel", "L1Tau_gen_abs_eta_sel")
  os.makedirs(os.path.dirname(output), exist_ok=True)
  output_file = ROOT.TFile(output, "RECREATE")
  for tau_type, hist in hists.items():
    output_file.WriteTObject(hist.GetValue(), f'pt_eta_{tau_type.name}')
  output_file.Close()

if __name__ == "__main__":
  import argparse
  parser = argparse.ArgumentParser(description='Make histogram.')
  parser.add_argument('--input', required=True, type=str, help="Input file(s)")
  parser.add_argument('--output', required=True, type=str, help="output directory")
  parser.add_argument('--x-bins', required=True, type=str, help="x bins")
  parser.add_argument('--y-bins', required=True, type=str, help="y bins")
  parser.add_argument('--input-tree', required=False, type=str, default='Events', help="Input tree")
  parser.add_argument('--tau-types', required=False, type=str, default='e,mu,tau,jet', help="tau types")
  args = parser.parse_args()

  PrepareRootEnv()
  x_bins = [float(x) for x in args.x_bins.split(',')]
  y_bins = [float(y) for y in args.y_bins.split(',')]
  tau_types = [TauType[x] for x in args.tau_types.split(',')]
  make_hists(MakeFileList(args.input), args.input_tree, args.output, x_bins, y_bins, tau_types)
