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
from .MixBin import MixBin

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
  mix_steps, batch_size = MixBin.Load(cfg)
  print(f'Number of mix steps: {len(mix_steps)}')
  print(f'Batch size: {batch_size}')
  for step_idx, mix_bins in enumerate(mix_steps):
    print(f'{timestamp_str()}{step_idx+1}/{len(mix_steps)}: processing...')
    print(f'Number of bins in step: {len(mix_bins)}')
    input_files = make_intput_list(mix_bins[0].inputs, cfg['input_root'])
    print(f'Input files ({len(input_files)} total):')
    for file in input_files:
      print(f'  {file}')
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

    l1tau_columns = [
      'pt', 'eta', 'phi', 'hwEtSum', 'hwEta', 'hwIso', 'hwPhi', 'hwPt', 'isoEt', 'nTT', 'rawEt', 'towerIEta',
      'towerIPhi',
    ]

    df_in = df_in.Define(f'L1Tau_nPV', 'RVecI(L1Tau_pt.size(), nPFPrimaryVertex)')
    df_in = df_in.Define(f'L1Tau_event', 'RVecI(L1Tau_pt.size(), static_cast<int>(event))')
    df_in = df_in.Define(f'L1Tau_luminosityBlock', 'RVecI(L1Tau_pt.size(), static_cast<int>(luminosityBlock))')
    df_in = df_in.Define(f'L1Tau_run', 'RVecI(L1Tau_pt.size(), static_cast<int>(run))')
    other_columns = { 'nPV', 'event', 'luminosityBlock', 'run',
                     'Gen_type', 'Gen_pt', 'Gen_eta', 'Gen_phi', 'Gen_mass', 'Gen_charge', 'Gen_partonFlavour',
                     'Gen_flightLength_rho', 'Gen_flightLength_phi', 'Gen_flightLength_z' }

    tau_columns = [ 'Tau_pt', 'Tau_eta', 'Tau_phi', 'Tau_mass', 'Tau_charge',  'Tau_deepTauVSjet' ]
    for c in tau_columns:
      df_in = df_in.Define(f'L1Tau_{c}', f'GetVar(L1Tau_tauIdx, {c})')
      other_columns.add(c)

    jet_columns = [ 'Jet_pt', 'Jet_eta', 'Jet_phi', 'Jet_mass', 'Jet_PNet_probb', 'Jet_PNet_probc', 'Jet_PNet_probg',
                    'Jet_PNet_probtauhm', 'Jet_PNet_probtauhp', 'Jet_PNet_probuds', 'Jet_PNet_ptcorr', ]
    for c in jet_columns:
      df_in = df_in.Define(f'L1Tau_{c}', f'GetVar(L1Tau_jetIdx, {c})')
      other_columns.add(c)

    pfcand_columns = [ 'PFCand_EcalEnergy', 'PFCand_HcalEnergy', 'PFCand_charge', 'PFCand_eta', 'PFCand_longLived',
                       'PFCand_mass', 'PFCand_pdgId', 'PFCand_phi', 'PFCand_pt', 'PFCand_rawEcalEnergy',
                       'PFCand_rawHcalEnergy', 'PFCand_trackChi2', 'PFCand_trackDxy', 'PFCand_trackDxyError',
                       'PFCand_trackDz', 'PFCand_trackDzError', 'PFCand_trackEta', 'PFCand_trackEtaError',
                       'PFCand_trackHitsValidFraction', 'PFCand_trackIsValid', 'PFCand_trackNdof',
                       'PFCand_trackNumberOfLostHits', 'PFCand_trackNumberOfValidHits', 'PFCand_trackPhi',
                       'PFCand_trackPhiError', 'PFCand_trackPt', 'PFCand_trackPtError', 'PFCand_vx', 'PFCand_vy',
                       'PFCand_vz', ]
    df_in = df_in.Define('PFCand_p4', 'GetP4(PFCand_pt, PFCand_eta, PFCand_phi, PFCand_mass)')
    df_in = df_in.Define('L1Tau_PFCand_matched', 'FindMatchingSet(L1Tau_p4, PFCand_p4, 0.5)')
    for c in pfcand_columns:
      df_in = df_in.Define(f'L1Tau_{c}', f'GetVarVec(L1Tau_PFCand_matched, {c})')
      other_columns.add(c)

    pfpv_columns = [ 'PFPrimaryVertex_chi2', 'PFPrimaryVertex_ndof', 'PFPrimaryVertex_x', 'PFPrimaryVertex_y',
                     'PFPrimaryVertex_z', ]
    for c in pfpv_columns:
      df_in = df_in.Define(f'L1Tau_{c}', f'ROOT::VecOps::RVec<RVecF>(L1Tau_pt.size(), {c})')
      other_columns.add(c)

    pfsv_columns = [ 'PFSecondaryVertex_chi2', 'PFSecondaryVertex_eta', 'PFSecondaryVertex_mass',
                     'PFSecondaryVertex_ndof', 'PFSecondaryVertex_ntracks', 'PFSecondaryVertex_phi',
                     'PFSecondaryVertex_pt', 'PFSecondaryVertex_x', 'PFSecondaryVertex_y', 'PFSecondaryVertex_z', ]
    df_in = df_in.Define('PFSecondaryVertex_p4', '''GetP4(PFSecondaryVertex_pt, PFSecondaryVertex_eta,
                                                    PFSecondaryVertex_phi, PFSecondaryVertex_mass)''')
    df_in = df_in.Define('L1Tau_PFSecondaryVertex_matched', 'FindMatchingSet(L1Tau_p4, PFSecondaryVertex_p4, 0.5)')
    for c in pfsv_columns:
      df_in = df_in.Define(f'L1Tau_{c}', f'GetVarVec(L1Tau_PFSecondaryVertex_matched, {c})')
      other_columns.add(c)

    pixelTrack_columns = [ 'PixelTrack_charge', 'PixelTrack_chi2', 'PixelTrack_dxy', 'PixelTrack_dz', 'PixelTrack_eta',
                           'PixelTrack_nHits', 'PixelTrack_nLayers', 'PixelTrack_phi', 'PixelTrack_pt',
                           'PixelTrack_quality', 'PixelTrack_tip', 'PixelTrack_vtxIdx', 'PixelTrack_zip', ]
    df_in = df_in.Define('PixelTrack_mass', 'RVecF(PixelTrack_pt.size(), 0.f)')
    df_in = df_in.Define('PixelTrack_p4', 'GetP4(PixelTrack_pt, PixelTrack_eta, PixelTrack_phi, PixelTrack_mass)')
    df_in = df_in.Define('L1Tau_PixelTrack_matched', 'FindMatchingSet(L1Tau_p4, PixelTrack_p4, 0.5)')
    for c in pixelTrack_columns:
      df_in = df_in.Define(f'L1Tau_{c}', f'GetVarVec(L1Tau_PixelTrack_matched, {c})')
      other_columns.add(c)

    pixelVertex_columns = [ 'PixelVertex_chi2', 'PixelVertex_ndof', 'PixelVertex_ptv2', 'PixelVertex_weight',
                            'PixelVertex_z', ]
    for c in pixelVertex_columns:
      df_in = df_in.Define(f'L1Tau_{c}', f'ROOT::VecOps::RVec<RVecF>(L1Tau_pt.size(), {c})')
      other_columns.add(c)

    recHit_colums = {
      'EB': [ 'RecHitEB_chi2', 'RecHitEB_energy', 'RecHitEB_eta', 'RecHitEB_isTimeErrorValid',
                         'RecHitEB_isTimeValid', 'RecHitEB_phi', 'RecHitEB_rho', 'RecHitEB_time',
                         'RecHitEB_timeError', ],
      'EE': [ 'RecHitEE_chi2', 'RecHitEE_energy', 'RecHitEE_eta', 'RecHitEE_isTimeErrorValid',
                         'RecHitEE_isTimeValid', 'RecHitEE_phi', 'RecHitEE_rho', 'RecHitEE_time',
                         'RecHitEE_timeError', ],
      'HBHE': [ 'RecHitHBHE_chi2', 'RecHitHBHE_eaux', 'RecHitHBHE_energy', 'RecHitHBHE_eraw',
                           'RecHitHBHE_eta', 'RecHitHBHE_eta_front', 'RecHitHBHE_phi', 'RecHitHBHE_phi_front',
                           'RecHitHBHE_rho', 'RecHitHBHE_rho_front', 'RecHitHBHE_time', 'RecHitHBHE_timeFalling', ],
      'HO': [ 'RecHitHO_energy', 'RecHitHO_eta', 'RecHitHO_phi', 'RecHitHO_rho', 'RecHitHO_time', ],
    }
    for det, columns in recHit_colums.items():
      df_in = df_in.Define(f'RecHit{det}_mass', f'RVecF(RecHit{det}_rho.size(), 0.f)')
      df_in = df_in.Define(f'RecHit{det}_p4', f'''GetP4(RecHit{det}_rho, RecHit{det}_eta,
                                                        RecHit{det}_phi, RecHit{det}_mass)''')
      df_in = df_in.Define(f'L1Tau_RecHit{det}_matched', f'FindMatchingSet(L1Tau_p4, RecHit{det}_p4, 0.5)')
      for c in columns:
        df_in = df_in.Define(f'L1Tau_{c}', f'GetVarVec(L1Tau_RecHit{det}_matched, {c})')
        other_columns.add(c)

    l1tau_columns.extend(other_columns)

    columns_in = [ ]
    columns_out = []
    for col in l1tau_columns:
      columns_in.append(f'L1Tau_{col}')
      out_name = f'L1Tau_{col}' if col not in other_columns else col
      columns_out.append(out_name)

    column_types = [ str(df_in.GetColumnType(c)) for c in columns_in ]

    tuple_maker = ROOT.analysis.TupleMaker(*column_types)()

    print(f'inputs: {mix_bins[0].input_setups}')
    nTau_in = 0
    slot_sel_str = 'RVecI slot_sel(L1Tau_pt.size(), -1);'
    for bin in mix_bins:
      nTau_batch = bin.stop_idx - bin.start_idx
      nTau_bin = nTau_batch * n_batches
      max_queue_size = nTau_bin if bin.allow_duplicates else min(nTau_batch * 100, nTau_bin)
      slot_id = tuple_maker.AddBin(bin.start_idx, bin.stop_idx, max_queue_size, nTau_bin)
      nTau_in += nTau_bin
      slot_sel_str += f'''
        {{
          auto sel = {bin.selection};
          for(size_t n = 0; n < L1Tau_pt.size(); ++n) {{
            if(sel[n]) slot_sel[n] = {slot_id};
          }}
        }}
      '''
      print(f'bin: slot={slot_id} nTaus={nTau_bin} selection="{bin.selection}"')
    slot_sel_str += ' return slot_sel;'
    print(f'nTaus total = {nTau_in}')
    df_in = df_in.Define('slot_sel', slot_sel_str)
    df_in = df_in.Filter('for(auto slot : slot_sel) if(slot >= 0) return true; return false;')

    columns_in = [ 'slot_sel' ] + columns_in
    column_types = [ str(df_in.GetColumnType('slot_sel')) ] + column_types

    if step_idx == 0:
      nTau_out = batch_size * n_batches
      df_out = ROOT.RDataFrame(nTau_out)
    else:
      output_prev_step = os.path.join(output, f'step_{step_idx}.root')
      df_out = ROOT.RDataFrame(cfg['tree_name'], output_prev_step)

    columns_in_v = ListToVector(columns_in)
    df_out = tuple_maker.process(ROOT.RDF.AsRNode(df_in), ROOT.RDF.AsRNode(df_out), columns_in_v, batch_size, True)

    define_fn_name = 'Define' if step_idx == 0 else 'Redefine'
    for column_idx in range(1, len(columns_in)):
      column_type = column_types[column_idx]
      if column_type in ['ROOT::VecOps::RVec<int>', 'ROOT::VecOps::RVec<Int_t>' ]:
        default_value = '0'
        entry_type = 'int'
      elif column_type in [ 'ROOT::VecOps::RVec<float>', 'ROOT::VecOps::RVec<Float_t>']:
        default_value = '0.f'
        entry_type = 'float'
      elif column_type in [ 'ROOT::VecOps::RVec<ROOT::VecOps::RVec<int> >', 'ROOT::VecOps::RVec<vector<int> >']:
        default_value = 'RVecI()'
        entry_type = 'RVecI'
      elif column_type == 'ROOT::VecOps::RVec<ROOT::VecOps::RVec<float> >':
        default_value = 'RVecF()'
        entry_type = 'RVecF'
      else:
        raise Exception(f'Unknown column type {column_type}')

      if step_idx != 0:
        default_value = f'{columns_out[column_idx-1]}'
      define_str = f'_entry && _entry->valid ? std::get<{entry_type}>(_entry->values.at({column_idx - 1})) : {default_value}'
      df_out = getattr(df_out, define_fn_name)(columns_out[column_idx - 1], define_str)

    if step_idx == 0:
      df_out = df_out.Define('is_valid', '_entry && _entry->valid')
      df_out = df_out.Define('step_idx', f'int({step_idx + 1})')
    else:
      df_out = df_out.Redefine('is_valid', '(_entry && _entry->valid) || is_valid')
      df_out = df_out.Redefine('step_idx', f'(_entry && _entry->valid) ? int({step_idx + 1}) : step_idx')
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
