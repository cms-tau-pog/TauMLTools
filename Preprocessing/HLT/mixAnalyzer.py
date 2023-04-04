import os
import sys
import math
import numpy as np
import yaml

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

def analyse_mix(cfg_file):
  with open(cfg_file, 'r') as f:
    cfg = yaml.safe_load(f)
  mix_steps, batch_size = MixStep.Load(cfg)
  print(f'Number of mix steps: {len(mix_steps)}')
  print(f'Batch size: {batch_size}')
  step_stat = np.zeros(len(mix_steps))
  step_stat_split = { 'tau': np.zeros(len(mix_steps)), 'jet': np.zeros(len(mix_steps)), 'e': np.zeros(len(mix_steps)) }
  n_taus = { }
  n_taus_batch = { }
  for step_idx, step in enumerate(mix_steps):
    n_available = 0
    for input in step.inputs:
      input_path = os.path.join(cfg['spectrum_root'], f'{input}.root')
      file = ROOT.TFile.Open(input_path, "READ")
      hist_name = f'pt_eta_{step.tau_type}'
      hist = file.Get(hist_name)
      n_available += hist.GetBinContent(step.pt_bin + 1, step.eta_bin + 1)
    n_taus[step.tau_type] = n_available + n_taus.get(step.tau_type, 0)
    n_taus_batch[step.tau_type] = step.count + n_taus_batch.get(step.tau_type, 0)
    n_batches = math.floor(n_available / step.count)
    step_stat[step_idx] = n_batches
    step_stat_split[step.tau_type][step_idx] = n_batches
  step_idx = np.argmin(step_stat)
  print(f'Total number of samples = {sum(n_taus.values())}: {n_taus}')
  n_taus_active = { name: x * step_stat[step_idx] for name, x in n_taus_batch.items()}
  print(f'Total number of used samples = {sum(n_taus_active.values())}: {n_taus_active}')
  n_taus_frac = { name: n_taus_active[name] / x for name, x in n_taus.items()}
  print(f'Used fraction: {n_taus_frac}')
  print(f'Number of samples per batch: {n_taus_batch}')
  n_taus_rel = { name: x / n_taus_batch['tau'] for name, x in n_taus_batch.items()}
  print(f'Relative number of samples: {n_taus_rel}')
  print('Step with minimum number of batches:')
  print(f'n_batches: {step_stat[step_idx]}')
  mix_steps[step_idx].Print()
  for name, stat in step_stat_split.items():
    step_idx = np.argmax(stat)
    print(f'Step with maximum number of batches for {name}:')
    print(f'n_batches: {stat[step_idx]}')
    mix_steps[step_idx].Print()

if __name__ == "__main__":
  import argparse
  parser = argparse.ArgumentParser(description='Make mix.')
  parser.add_argument('--cfg', required=True, type=str, help="Mix config file")
  args = parser.parse_args()

  analyse_mix(args.cfg)
