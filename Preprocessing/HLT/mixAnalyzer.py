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

  step_stats = {}
  for tau_type in [ 'total', 'tau', 'displaced_tau', 'jet', 'e' ]:
    step_stats[tau_type] = np.ones((len(mix_steps), 2)) * -1
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
    n_batches = math.floor(n_available / step.count) if not step.allow_duplicates else float('inf')
    step.n_available = n_available
    for tau_type in [ 'total', step.tau_type ]:
      step_stats[tau_type][step_idx, 0] = n_batches
      step_stats[tau_type][step_idx, 1] = step_idx
  finite_steps = step_stats['total'][np.isfinite(step_stats['total'][:, 0]), :]
  step_idx = np.argmin(finite_steps[:, 0])
  print(f'Total number of samples = {sum(n_taus.values())}: {n_taus}')
  n_taus_active = { name: x * finite_steps[step_idx, 0] for name, x in n_taus_batch.items()}
  print(f'Total number of used samples = {sum(n_taus_active.values())}: {n_taus_active}')
  n_taus_frac = { name: n_taus_active[name] / x for name, x in n_taus.items()}
  print(f'Used fraction: {n_taus_frac}')
  print(f'Number of samples per batch: {n_taus_batch}')
  n_taus_rel = { name: x / n_taus_batch['tau'] for name, x in n_taus_batch.items()}
  print(f'Relative number of samples: {n_taus_rel}')
  print('Step with minimum number of batches:')
  print(f'n_batches: {finite_steps[step_idx, 0]}')
  mix_steps[int(finite_steps[step_idx, 1])].Print()
  for name, stat in step_stats.items():
    if name == 'total': continue
    finite_steps = stat[np.isfinite(stat[:, 0]) & (stat[:, 0] >= 0), :]
    if np.shape(finite_steps)[0] == 0: continue
    step_idx = np.argmax(finite_steps[:, 0])
    print(f'Step with maximum number of batches for {name}:')
    print(f'n_batches: {finite_steps[step_idx, 0]}')
    mix_steps[int(finite_steps[step_idx, 1])].Print()
  n_batches = cfg['n_batches']
  print(f'Target number of batches: {n_batches}')
  print('Steps with allowed duplicates:')
  for step_idx in range(len(mix_steps)):
    if mix_steps[step_idx].allow_duplicates:
      n_repetitions = math.ceil((n_batches * mix_steps[step_idx].count) / mix_steps[step_idx].n_available)
      print(f'step {step_idx} n_repetitions = {n_repetitions} selection: {mix_steps[step_idx].selection}')

if __name__ == "__main__":
  import argparse
  parser = argparse.ArgumentParser(description='Make mix.')
  parser.add_argument('--cfg', required=True, type=str, help="Mix config file")
  args = parser.parse_args()

  analyse_mix(args.cfg)
