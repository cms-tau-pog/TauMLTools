from __future__ import print_function
import sys; PYTHON_MAJOR = int(sys.version_info.major)
dict_iterator = 'items' if PYTHON_MAJOR == 3 else 'iteritems'

import ROOT
import glob
import json
from collections import OrderedDict

import argparse
parser = argparse.ArgumentParser('''
The script runs a binned KS test between different chunks of the same RDataFrame. For simplicity, all chunks are compared to the first one.
If a KS test is below the threshold, a warning message is printed on screen.
NOTE: the binning of each variable must be hard coded in the script (using the BINS dictionary)
NOTE: pvalue = 99 means that one of the two histograms is empty.
''')

parser.add_argument('--input'       , required = True, type = str, help = 'input file. Accepts glob patterns (use quotes)')
parser.add_argument('--output'      , required = True, type = str, help = 'output directory name')
parser.add_argument('--nsplit'      , default  = 100 , type = int, help = 'number of chunks per file')
parser.add_argument('--pvthreshold' , default  = .05 , type = str, help = 'threshold of KS test (above = ok)')
parser.add_argument('--n_threads'   , default  = 1   , type = int, help = 'enable ROOT implicit multithreading')

parser.add_argument('--visual', action = 'store_true', help = 'Won\'t run the script in batch mode')
parser.add_argument('--legend', action = 'store_true', help = 'Draw a TLegent on canvases')
args = parser.parse_args()

ROOT.gROOT.SetBatch(not args.visual)
ROOT.gStyle.SetOptStat(0)

import os
pdf_dir = '/'.join([args.output, 'pdf'])
if not os.path.exists(pdf_dir):
  os.makedirs(pdf_dir)

JSON_DICT      = OrderedDict()
OUTPUT_ROOT    = ROOT.TFile.Open('{}/histograms.root'.format(args.output), 'RECREATE')
OUTPUT_JSON    = open('{}/pvalues.json'.format(args.output), 'w')
N_SPLIT        = args.nsplit
PVAL_THRESHOLD = args.pvthreshold

## binning of tested variables (cannot use unbinned distributions with python before root 6.18)
BINS = {
  'tau_pt'    : (50, 0, 5000),
  'tau_eta'   : (5, -3.2, 3.2),
  'lepton_gen_match' : (20, -1, 19),
  'sampleType': (20, -1, 19),      
  'dataset_id': (20, -1, 19),
  'dataset_group_id': (20, -1, 19),
}

class Lazy_container:
  def __init__(self, ptr, hst = None):
    self.ptr = ptr
    self.hst = hst
  def load_histogram(self):
    self.hst = self.ptr.GetValue()

class Entry:  ## 2D entry (chunk_id x variable)
  def __init__(self, var, histo, tdir = None):
    self.var = var
    self.hst = histo
    self.tdir = tdir if not tdir is None else self.var

  def run_KS_test(self, norm = True):    
    self.chunks = [self.hst.ProjectionY('chunk_{}'.format(cc), cc+1, cc+1).Clone() for cc in range(N_SPLIT)]
    self.chunks[0].SetMarkerStyle(20)
    for jj, hh in enumerate(self.chunks):
      hh.SetTitle(self.tdir)
      hh.Sumw2()
      hh.SetLineColor(jj+1)
      if hh.Integral() and norm:
        hh.Scale(1. / hh.Integral())
    
    if not self.chunks[0].Integral():
      print ('[WARNING] control histogram is empty inside {}'.format(self.tdir))
    
    self.pvalues = [self.chunks[0].KolmogorovTest(hh) if self.chunks[0].Integral()*hh.Integral() else 99 for hh in self.chunks]
    if not all([pv >= PVAL_THRESHOLD for pv in self.pvalues]):
      print ('[WARNING] KS test failed for step {}. p-values are:'.format(self.tdir))
      print ('\t', self.pvalues)
  
  def save_data(self):
    OUTPUT_ROOT.cd()
    
    if not OUTPUT_ROOT.GetDirectory(self.tdir):
      OUTPUT_ROOT.mkdir(self.tdir)
    
    OUTPUT_ROOT.cd(self.tdir)
    
    can = ROOT.TCanvas()
    leg = ROOT.TLegend(0.9, 0.1, 1., 0.9, "p-values (KS with the first chunk)")
    for ii, hh in enumerate(self.chunks):
      hh.Write()
      hh.Draw('PE'+' SAME'*(ii != 0))
      leg.AddEntry(hh, 'chunk %d - pval = %.3f' %(ii, self.pvalues[ii]), 'lep')
    if args.legend:
      leg.Draw("SAME")
    
    can.SaveAs('{}/pdf/{}.pdf'.format(args.output, self.tdir.replace('/', '_')), 'pdf')
    can.Write()

    OUTPUT_ROOT.cd()

    json_here = JSON_DICT
    for here in self.tdir.split('/'):
      if not here in json_here.keys():
        json_here[here] = OrderedDict()
      json_here = json_here[here]
    json_here['pvalues'] = self.pvalues

def to_2D(histo, vbin):
  histo.GetZaxis().SetRange(vbin, vbin)
  return histo.Project3D('yx').Clone()

if __name__ == '__main__':
  print ('[INFO] reading files', args.input)
  
  if args.n_threads > 1:
    ROOT.ROOT.EnableImplicitMT(args.n_threads)

  input_files = ROOT.std.vector('std::string')()
  
  for file in glob.glob(args.input):
    input_files.push_back(str(file))

  model = lambda main, third = None: (main, '', N_SPLIT, 0, N_SPLIT)+BINS[main]+BINS[third] if not third is None else (main, '', N_SPLIT, 0, N_SPLIT)+BINS[main]
  
  dataframe = ROOT.RDataFrame('taus', input_files)
  dataframe = dataframe.Define('chunk_id', 'rdfentry_ % {}'.format(N_SPLIT))

  ## unbinned distributions
  ptr_lgm = Lazy_container(dataframe.Histo2D(model('lepton_gen_match'), 'chunk_id', 'lepton_gen_match'))
  ptr_st  = Lazy_container(dataframe.Histo2D(model('sampleType')      , 'chunk_id', 'sampleType'      ))
  ptr_dgi = Lazy_container(dataframe.Histo2D(model('dataset_group_id'), 'chunk_id', 'dataset_group_id'))
  ptr_di  = Lazy_container(dataframe.Histo2D(model('dataset_id')      , 'chunk_id', 'dataset_id'      ))

  ## binned distributions
  ptrs_tau_pt = {
    binned_variable: Lazy_container(dataframe.Histo3D(model('tau_pt', third = binned_variable), 'chunk_id', 'tau_pt', binned_variable))
      for binned_variable in ['lepton_gen_match', 'sampleType', 'dataset_group_id', 'dataset_id']
  }
  ptrs_tau_eta = {
    binned_variable: Lazy_container(dataframe.Histo3D(model('tau_eta', third = binned_variable), 'chunk_id', 'tau_eta', binned_variable))
      for binned_variable in ['lepton_gen_match', 'sampleType', 'dataset_group_id', 'dataset_id']
  }
  ptrs_dataset_id = {
    binned_variable: Lazy_container(dataframe.Histo3D(model('dataset_id', third = binned_variable), 'chunk_id', 'dataset_id', binned_variable))
      for binned_variable in ['dataset_group_id']
  }

  lazy_containers = [ptr_lgm, ptr_st, ptr_dgi, ptr_di ] +\
    [lc for lc in ptrs_tau_pt.values()]  +\
    [lc for lc in ptrs_tau_eta.values()] +\
    [lc for lc in ptrs_dataset_id.values()]
  
  for lc in lazy_containers:
    lc.load_histogram()
  
  ## run validation
  entry_lgm = Entry(var = 'lepton_gen_match', histo = ptr_lgm.hst)
  entry_st  = Entry(var = 'sampleType'      , histo = ptr_st .hst)
  entry_dgi = Entry(var = 'dataset_group_id', histo = ptr_dgi.hst)
  entry_di  = Entry(var = 'dataset_id'      , histo = ptr_di .hst)

  entries_tau_pt = [
    Entry(var = 'tau_pt', histo = to_2D(ptrs_tau_pt[binned_variable].hst, jj+1), tdir = '/'.join([binned_variable, str(bb), 'tau_pt']))
      for binned_variable in ['lepton_gen_match', 'sampleType', 'dataset_group_id', 'dataset_id']
      for jj, bb in enumerate(range(*BINS[binned_variable][1:]))
  ] ; entries_tau_pt = [ee for ee in entries_tau_pt if ee.hst.GetEntries()]

  entries_tau_eta = [
    Entry(var = 'tau_eta', histo = to_2D(ptrs_tau_eta[binned_variable].hst, jj+1), tdir = '/'.join([binned_variable, str(bb), 'tau_eta']))
      for binned_variable in ['lepton_gen_match', 'sampleType', 'dataset_group_id', 'dataset_id']
      for jj, bb in enumerate(range(*BINS[binned_variable][1:]))
  ] ; entries_tau_eta = [ee for ee in entries_tau_eta if ee.hst.GetEntries()]
  
  entries_dataset_id = [
    Entry(var = 'dataset_id', histo = to_2D(ptrs_dataset_id[binned_variable].hst, jj+1), tdir = '/'.join([binned_variable, str(bb), 'dataset_id']))
      for binned_variable in ['dataset_group_id']
      for jj, bb in enumerate(range(*BINS[binned_variable][1:]))
  ]; entries_dataset_id = [ee for ee in entries_dataset_id if ee.hst.GetEntries()]

  entries = [entry_lgm, entry_st, entry_dgi, entry_di] +\
    [ee for ee in entries_tau_pt]  +\
    [ee for ee in entries_tau_eta] +\
    [ee for ee in entries_dataset_id]

  for ee in entries:
    ee.run_KS_test()
    ee.save_data()

  OUTPUT_ROOT.Close()
  json.dump(JSON_DICT, OUTPUT_JSON, indent = 4)
  print ('[INFO] all done. Files saved in', args.output)