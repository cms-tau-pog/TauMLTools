#!/usr/bin/env python
import argparse
parser = argparse.ArgumentParser('''
The script runs a binned KS test between different chunks of the same RDataFrame. For simplicity, all chunks are compared to the first one.
If a KS test is below the threshold, a warning message is printed on screen.
NOTE: pvalue = 99 means that one of the two histograms is empty.
''')

import ROOT
import glob
import json
from collections import OrderedDict

parser.add_argument('--input'       , required = True, type = str, help = 'input file. Accepts glob patterns')
parser.add_argument('--output'      , required = True, type = str, help = 'output file name')
parser.add_argument('--pdf'         , default  = None, type = str, help = 'output pdf directory')
parser.add_argument('--json'        , required = True, type = str, help = 'output json file name')
parser.add_argument('--nsplit'      , default  = 5   , type = str, help = 'number of chunks per file')
parser.add_argument('--pvthreshold' , default  = .05 , type = str, help = 'threshold of KS test (above = ok)')

parser.add_argument('--visual', action = 'store_true', help = 'Won\'t run the script in batch mode')
parser.add_argument('--legend', action = 'store_true', help = 'Draw a TLegent on canvases')
args = parser.parse_args()

import os
if not os.path.exists(args.pdf):
  os.makedirs(args.pdf)

ROOT.gROOT.SetBatch(not args.visual)
ROOT.gStyle.SetOptStat(0)

JSON_DICT      = OrderedDict()
OUTPUT_ROOT    = ROOT.TFile.Open('{}'.format(args.output), 'RECREATE')
OUTPUT_JSON    = open(args.json, 'w')
N_SPLITS       = args.nsplit
PVAL_THRESHOLD = args.pvthreshold

## binning of tested variables (cannot use unbinned distributions with python before root 6.18)
BINS = {
  'tau_pt'    : (50, 0, 5000),
  'tau_eta'   : (5, -2.5, 2.5),
  'lepton_gen_match' : (20, -1, 19),
  'sampleType': (20, -1, 19),      
  'dataset_id': (20, -1, 19),
  'dataset_group_id': (20, -1, 19),
}

def groupby(dataframe, by):
  _ = dataframe.Histo1D(by)
  hist = _.GetValue()
  hist.ClearUnderflowAndOverflow()
  types = list(set([round(hist.GetBinCenter(jj)) for jj in range(hist.GetNbinsX()) if hist.GetBinContent(jj)]))

  return {tt: dataframe.Filter('{} == {}'.format(by, tt)) for tt in types}

def get_histos(dataframe, branch, norm = False):
  size = int(dataframe.Count())
  sub_size = 1 + size / N_SPLITS
  subframes = [dataframe.Range(ii*sub_size, (ii+1)*sub_size) for ii in range(N_SPLITS)]
  
  model = (branch, "") + BINS[branch]
  hptrs = [sf.Histo1D(model, branch) for sf in subframes]
  histos = [hh.GetValue().Clone() for hh in hptrs]
  
  for hh in histos:
    hh.SetTitle(branch)
    hh.Sumw2()
    hh.ClearUnderflowAndOverflow()
    if norm and hh.Integral():
      hh.Scale(1. / hh.Integral())

  return histos

def save_histos(histos, fdir, pvalues):
  OUTPUT_ROOT.cd()
  OUTPUT_ROOT.cd(fdir)
  
  can = ROOT.TCanvas()
  leg = ROOT.TLegend(0.9, 0.1, 1., 0.9, "p-values (KS with the first chunk)")

  histos[0].GetYaxis().SetRangeUser(0, 1.1*max(hh.GetMaximum() for hh in histos))
  histos[0].SetMarkerStyle(20)

  for ii, hh in enumerate(histos): 
    hh.SetLineColor(ii+1)
    hh.SetMarkerColor(ii+1)
    leg.AddEntry(hh, 'chunk %d - pval = %.3f' %(ii, pvalues[ii]), 'lep')
    hh.Draw("PE" + " SAME" * (ii != 0))
    hh.Write()
  if args.legend:
    leg.Draw("SAME")
  if args.pdf is not None:
    can.SaveAs('{}/{}.pdf'.format(args.pdf, fdir.replace('/', '_')), 'pdf')
  can.Write()
  OUTPUT_ROOT.cd()

def run_validation(dataframe, branches, pwd = ''):
  OUTPUT_ROOT.cd()
  OUTPUT_ROOT.mkdir(pwd)
  
  if not pwd in JSON_DICT.keys():
    JSON_DICT[pwd] = OrderedDict()

  for branch in branches:
    OUTPUT_ROOT.cd() 
    OUTPUT_ROOT.mkdir('/'.join([pwd, branch]))

    histos = get_histos(dataframe, branch = branch, norm = True)
  
    pvalues = [histos[0].KolmogorovTest(hh) if histos[0].Integral()*hh.Integral() else 99 for hh in histos]
    if not histos[0].Integral():
      print '[WARNING] control histogram is empty for step {} inside {}'.format(branch, pwd)
    
    if not all([pv >= PVAL_THRESHOLD for pv in pvalues]):
      print '[WARNING] KS test failed for step {} inside {}. p-values are:'.format(branch, pwd)
      print '\t', pvalues
    
    JSON_DICT[pwd][branch] = pvalues
    
    save_histos(histos, fdir = '/'.join([pwd, branch]), pvalues = pvalues)

if __name__ == '__main__':
  print '[INFO] reading files', args.input
  input_files = ROOT.std.vector('std::string')()
  for file in glob.glob(args.input):
    input_files.push_back(str(file))

  main_dir = 'KS_test'

  ## first, run on plain columns
  dataframe = ROOT.RDataFrame('taus', input_files)
  run_validation(dataframe = dataframe, pwd = main_dir, branches = ['lepton_gen_match', 'sampleType', 'dataset_group_id'])

  ## then, group by tau type
  tau_type_dataframes = groupby(dataframe = dataframe, by = 'lepton_gen_match')
  for ii, df in tau_type_dataframes.iteritems():
    run_validation(dataframe = df, pwd = '/'.join([main_dir, 'lepton_gen_match', str(ii)]), branches = ['tau_pt', 'tau_eta'])

  ## then, group by sample type
  sample_type_dataframes = groupby(dataframe = dataframe, by = 'sampleType')
  for ii, df in sample_type_dataframes.iteritems():
    run_validation(dataframe = df, pwd = '/'.join([main_dir, 'sampleType', str(ii)]), branches = ['tau_pt', 'tau_eta'])

  ## then, group by dataset group id
  group_id_dataframes = groupby(dataframe = dataframe, by = 'dataset_group_id')
  for ii, df in group_id_dataframes.iteritems():
    run_validation(dataframe = df, pwd = '/'.join([main_dir, 'dataset_group_id', str(ii)]), branches = ['tau_pt', 'tau_eta', 'dataset_id'])

  OUTPUT_ROOT.Close()
  json.dump(JSON_DICT, OUTPUT_JSON, indent = 4)
  print '[INFO] all done. Files', args.output, 'and', args.json, 'have been created'