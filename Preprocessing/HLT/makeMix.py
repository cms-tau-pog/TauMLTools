import math
import os
import shutil
import sys
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

def make_split_ranges(n_elem, n_splits):
  split_size = n_elem / n_splits
  splits = []
  for i in range(n_splits + 1):
    x = math.floor(i * split_size) if i < n_splits else n_elem
    if x in splits:
      raise RuntimeError(f'Unable to split {n_elem} into {n_splits} ranges')
    splits.append(x)
  return splits

def make_intput_list(inputs, input_root):
  input_lists = []
  all_files = []
  n_files_max = 0
  for input in inputs:
    input_path = os.path.join(input_root, input)
    file_list = MakeFileList(input_path)
    n_files_max = max(n_files_max, len(file_list))
    input_lists.append(file_list)
    all_files.extend(file_list)
  steps = [ len(input_lists[i]) / n_files_max for i in range(len(input_lists)) ]
  input_indices = []
  for i in range(n_files_max):
    for input_idx in range(len(inputs)):
      pos = int(i * steps[input_idx])
      entry = (input_idx, pos)
      if entry not in input_indices:
        input_indices.append(entry)
  input_files = []
  for input_idx, pos in input_indices:
    input_files.append(input_lists[input_idx][pos])
  for file in all_files:
    if file not in input_files:
      raise RuntimeError(f'make_intput_list logic error: {file} not in input_files')
  return input_files

def make_mix(cfg_file, output, n_jobs, job_id):
  with open(cfg_file, 'r') as f:
    cfg = yaml.safe_load(f)
  batch_ranges = make_split_ranges(cfg['n_batches'], n_jobs)
  n_batches = batch_ranges[job_id + 1] - batch_ranges[job_id]
  print(f'Job {job_id+1}/{n_jobs}: creating {n_batches} batches...')
  if os.path.exists(output):
    shutil.rmtree(output)
  os.makedirs(output, exist_ok=True)
  mix_steps, batch_size = MixStep.Load(cfg)
  print(f'Number of mix steps: {len(mix_steps)}')
  print(f'Batch size: {batch_size}')
  for step_idx, step in enumerate(mix_steps):
    print(f'{timestamp_str()}{step_idx+1}/{len(mix_steps)}: processing...')
    input_files = make_intput_list(step.inputs, cfg['input_root'])
    input_files_vec = ListToVector(input_files, "string")
    df_in = ROOT.RDataFrame(cfg['tree_name'], input_files_vec)
    if n_jobs != 1:
      event_ranges = make_split_ranges(df_in.Count().GetValue(), n_jobs)
      df_in = df_in.Range(event_ranges[job_id], event_ranges[job_id + 1])
      print(f'Total number of available input entries {event_ranges[-1]}')
      print(f'Considering events in range [{event_ranges[job_id]}, {event_ranges[job_id+1]})')
    if 'event_sel' in cfg:
      df_in = df_in.Filter(cfg['event_sel'])
    df_in = ApplyCommonDefinitions(df_in)
    df_in = df_in.Define('L1Tau_sel', step.selection)

    l1tau_columns = [
      'pt', 'eta', 'phi', 'hwEtSum', 'hwEta', 'hwIso', 'hwPhi', 'hwPt', 'isoEt', 'nTT', 'rawEt', 'towerIEta',
      'towerIPhi', 'type', 'gen_pt', 'gen_eta', 'gen_phi', 'gen_mass', 'gen_charge', 'gen_partonFlavour',
    ]

    df_in = df_in.Define(f'L1Tau_nPV', 'RVecI(L1Tau_pt.size(), nPFPrimaryVertex)')
    df_in = df_in.Define(f'L1Tau_event', 'RVecI(L1Tau_pt.size(), static_cast<int>(event))')
    df_in = df_in.Define(f'L1Tau_luminosityBlock', 'RVecI(L1Tau_pt.size(), static_cast<int>(luminosityBlock))')
    df_in = df_in.Define(f'L1Tau_run', 'RVecI(L1Tau_pt.size(), static_cast<int>(run))')
    other_columns = { 'nPV', 'event', 'luminosityBlock', 'run' }
    l1tau_columns.extend(other_columns)

    columns_in = [ 'L1Tau_sel' ]
    columns_out = []
    for col in l1tau_columns:
      columns_in.append(f'L1Tau_{col}')
      out_name = f'L1Tau_{col}' if col not in other_columns else col
      columns_out.append(out_name)

    columns_in_v = ListToVector(columns_in)
    column_types = [ str(df_in.GetColumnType(c)) for c in columns_in ]
    nTau_in = (step.stop_idx - step.start_idx) * n_batches
    print(f'nTaus = {nTau_in}')
    print(f'inputs: {step.input_setups}')
    print(f'selection: {step.selection}')

    if step_idx == 0:
      nTau_out = batch_size * n_batches
      df_out = ROOT.RDataFrame(nTau_out)
    else:
      output_prev_step = os.path.join(output, f'step_{step_idx}.root')
      df_out = ROOT.RDataFrame(cfg['tree_name'], output_prev_step)

    tuple_maker = ROOT.analysis.TupleMaker(*column_types)(100, nTau_in)
    df_out = tuple_maker.process(ROOT.RDF.AsRNode(df_in), ROOT.RDF.AsRNode(df_out), columns_in_v,
                                 step.start_idx, step.stop_idx, batch_size)

    define_fn_name = 'Define' if step_idx == 0 else 'Redefine'
    for column_idx in range(1, len(columns_in)):
      column_type = column_types[column_idx]
      {'ROOT::VecOps::RVec<int>', 'ROOT::VecOps::RVec<float>', 'ROOT::VecOps::RVec<ROOT::VecOps::RVec<int> >'}
      if column_type in ['ROOT::VecOps::RVec<int>', 'ROOT::VecOps::RVec<Int_t>' ]:
        default_value = '0'
        entry_col = 'int_values'
      elif column_type in [ 'ROOT::VecOps::RVec<float>', 'ROOT::VecOps::RVec<Float_t>']:
        default_value = '0.f'
        entry_col = 'float_values'
      elif column_type in [ 'ROOT::VecOps::RVec<ROOT::VecOps::RVec<int> >', 'ROOT::VecOps::RVec<vector<int> >']:
        default_value = 'RVecI()'
        entry_col = 'vint_values'
      elif column_type == 'ROOT::VecOps::RVec<ROOT::VecOps::RVec<float> >':
        default_value = 'RVecF()'
        entry_col = 'vfloat_values'
      else:
        raise Exception(f'Unknown column type {column_type}')

      if step_idx != 0:
        default_value = f'{columns_out[column_idx-1]}'
      define_str = f'_entry.valid ? _entry.{entry_col}.at({column_idx - 1}) : {default_value}'
      df_out = getattr(df_out, define_fn_name)(columns_out[column_idx - 1], define_str)

    if step_idx == 0:
      df_out = df_out.Define('is_valid', '_entry.valid')
      df_out = df_out.Define('step_idx', f'int({step_idx + 1})')
    else:
      df_out = df_out.Redefine('is_valid', '_entry.valid || is_valid')
      df_out = df_out.Redefine('step_idx', f'_entry.valid ? int({step_idx + 1}) : step_idx')
    columns_out.extend(['is_valid', 'step_idx'])

    opt = ROOT.RDF.RSnapshotOptions()
    opt.fCompressionAlgorithm = ROOT.ROOT.kLZMA
    opt.fCompressionLevel = 9
    opt.fMode = 'RECREATE'
    output_step = os.path.join(output, f'step_{step_idx+1}.root')
    columns_out_v = ListToVector(columns_out)
    df_out.Snapshot(cfg['tree_name'], output_step, columns_out_v, opt)
    tuple_maker.join()
    print(f'{timestamp_str()}{step_idx+1}/{len(mix_steps)}: done.')

if __name__ == "__main__":
  import argparse
  parser = argparse.ArgumentParser(description='Make mix.')
  parser.add_argument('--cfg', required=True, type=str, help="Mix config file")
  parser.add_argument('--output', required=True, type=str, help="output directory")
  parser.add_argument('--n-jobs', required=False, type=int, default=1, help="number of parallel jobs")
  parser.add_argument('--job-id', required=False, type=int, default=0, help="current job id")
  args = parser.parse_args()

  PrepareRootEnv()
  make_mix(args.cfg, args.output, args.n_jobs, args.job_id)
