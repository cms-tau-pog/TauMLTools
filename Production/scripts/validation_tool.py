from __future__ import print_function

import ROOT
import json
from collections import OrderedDict
import os
import re

import argparse
parser = argparse.ArgumentParser('''
The script runs a binned KS test between different chunks of the same RDataFrame. For simplicity, all chunks are compared to the first one.
If a KS test is below the threshold, a warning message is printed on screen.
NOTE: the binning of each variable must be hard coded in the script (using the BINS dictionary)
NOTE: pvalue = 99 means that one of the two histograms is empty.
''')

parser.add_argument('--input'         , required = True, type = str, help = 'input directory. Will loop inside all subdirectories')
parser.add_argument('--regex'  , default  = '.*\.root$', type = str, help = 'regular expression to match to input files')
parser.add_argument('--output'        , required = True, type = str, help = 'output directory name')
parser.add_argument('--test'          , default ='Chi2', type = str, help = 'test to perform. Accepts Chi2 and KS')
parser.add_argument('--nsplit'        , default  = 100 , type = int, help = 'number of chunks per file')
parser.add_argument('--pvthreshold'   , default  = .05 , type=float, help = 'threshold of KS test (above = ok)')
parser.add_argument('--n_threads'     , default  = 1   , type = int, help = 'enable ROOT implicit multithreading')
parser.add_argument('--id_json'       , required = True, type = str, help = 'dataset id json file')
parser.add_argument('--group_id_json' , required = True, type = str, help = 'dataset group id json file')
parser.add_argument('--use_dataset_id', action = 'store_true', help = 'Add \'dataset_id\' column for comparison and binning')
parser.add_argument('--formats'       , default ='pdf', type = str, help = 'comma separated list of output file formats')

parser.add_argument('--visual', action = 'store_true', help = 'Won\'t run the script in batch mode')
parser.add_argument('--legend', action = 'store_true', help = 'Draw a TLegend on canvases')
args = parser.parse_args()

ROOT.gROOT.SetBatch(not args.visual)
ROOT.gStyle.SetOptStat(0)

for formats in args.formats.split(","):
  pdf_dir = os.path.join(args.output, formats)
  if not os.path.exists(pdf_dir):
    os.makedirs(pdf_dir)

JSON_DICT      = OrderedDict()
OUTPUT_ROOT    = ROOT.TFile.Open('{}/histograms.root'.format(args.output), 'RECREATE')
OUTPUT_JSON    = open('{}/pvalues.json'.format(args.output), 'w')
N_SPLIT        = args.nsplit
PVAL_THRESHOLD = args.pvthreshold

## binning of tested variables (dataset group id and dataset id are guessed from jsons)
BINS = {
  'tau_pt'    : (100, 0, 1000),
  'tau_eta'   : (10, -2.5, 2.5),
  'tauType' : (4, 0, 4),
  'sampleType': (5, 0, 5),
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

  def run_test(self, test, norm = True):
    self.chunks = [self.hst.ProjectionY('chunk_{}'.format(cc), cc+1, cc+1).Clone() for cc in range(N_SPLIT)]

    self.chunks[0].SetMarkerStyle(20)

    for jj, hh in enumerate(self.chunks):
      hh.SetTitle(self.tdir)
      hh.Sumw2()
      hh.SetLineColor(jj+1)
      if hh.Integral() and norm:
        hh.Scale(1. / hh.Integral())
    
    self.chunks[0].GetYaxis().SetRangeUser(0, 1.1*max(hh.GetMaximum() for hh in self.chunks))

    if not self.chunks[0].Integral():
      print ('[WARNING] control histogram is empty inside {}'.format(self.tdir))

    if test == 'Chi2':
      # see https://root.cern.ch/doc/master/classTH1.html#a6c281eebc0c0a848e7a0d620425090a5
      self.pvalues = [self.chunks[0].Chi2Test(hh, "NORM" if norm else "UU") if self.chunks[0].Integral()*hh.Integral() else 99 for hh in self.chunks]
    elif test == 'KS':
      # see https://root.cern.ch/doc/master/classTH1.html#aeadcf087afe6ba203bcde124cfabbee4
      self.pvalues = [self.chunks[0].KolmogorovTest(hh)   if self.chunks[0].Integral()*hh.Integral() else 99 for hh in self.chunks]
    else:
      raise ValueError("Test {} not valid".format(args.test))

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
    
    for formats in args.formats.split(","):
      can.SaveAs('{}/{}/{}.{}'.format(args.output, formats, self.tdir.replace('/', '_'), formats), formats)
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
  print ('[INFO] reading files from', args.input)
  
  if args.n_threads > 1:
    ROOT.ROOT.EnableImplicitMT(args.n_threads)

  input_files = ROOT.std.vector('std::string')()
  file_list = ['/'.join([root, ff]) for root, dirs, files in os.walk(args.input) for ff in files]
  file_list = sorted([ff for ff in file_list if not re.match(args.regex, ff) is None])

  for ff in file_list:
    input_files.push_back(ff)

  model = lambda main, third = None: (main, '', N_SPLIT, 0, N_SPLIT)+BINS[main]+BINS[third] if not third is None else (main, '', N_SPLIT, 0, N_SPLIT)+BINS[main]
  
  cpp_function = '''
std::vector<ULong64_t> hash_list = {%s};
int hash, line = 0;
for(std::vector<ULong64_t>::iterator it = hash_list.begin(); it != hash_list.end(); it++, line++){
if (*it == %s) return line;
} return line;'''

  id_json       = json.load(open(args.id_json, 'r'))
  group_id_json = json.load(open(args.group_id_json, 'r'))
  ids = [id_json[key] for key in sorted(id_json.keys())]
  group_ids = [group_id_json[key] for key in sorted(group_id_json.keys())]

  BINS['uh_dataset_group_id'] = (len(group_ids), 0, len(group_ids))
  BINS['uh_dataset_id']       = (len(ids), 0, len(ids))

  dataframe = ROOT.RDataFrame('taus', input_files)
  tot_entries = dataframe.Count().GetValue()

  dataframe = dataframe.Define('chunk_id', 'rdfentry_ * {} / {}'.format(N_SPLIT, tot_entries))
  dataframe = dataframe.Define('uh_dataset_id'      , cpp_function % (','.join(ids), 'dataset_id'))
  dataframe = dataframe.Define('uh_dataset_group_id', cpp_function % (','.join(group_ids), 'dataset_group_id'))

  ## unbinned distributions
  ptr_lgm = Lazy_container(dataframe.Histo2D(model('tauType'), 'chunk_id', 'tauType'))
  ptr_st  = Lazy_container(dataframe.Histo2D(model('sampleType')      , 'chunk_id', 'sampleType'      ))
  ptr_dgi = Lazy_container(dataframe.Histo2D(model('uh_dataset_group_id'), 'chunk_id', 'uh_dataset_group_id'))
  ptr_di  = Lazy_container(dataframe.Histo2D(model('uh_dataset_id')      , 'chunk_id', 'uh_dataset_id'      ))

  ## binned distributions
  BINNED_VARIABLES = ['tauType', 'sampleType', 'uh_dataset_group_id'] + ['uh_dataset_id']*args.use_dataset_id
  BINNED_VARIABLENAMES = {
    'tauType': {0:'e', 1:'mu', 2:'tau', 3:'jet', 4:'emb_e', 5:'emb_mu', 6:'emb_tau', 7:'emb_jet', 8:'data'},
    'sampleType': {1:'MC', 2:'Embedded', 3:'Data'},
    'uh_dataset_group_id': {ii:jj for ii,jj in enumerate(sorted(group_id_json.keys()))},
    'uh_dataset_id': {ii:jj for ii,jj in enumerate(sorted(id_json.keys()))}
  }
  ptrs_tau_pt = {
    binned_variable: Lazy_container(dataframe.Histo3D(model('tau_pt', third = binned_variable), 'chunk_id', 'tau_pt', binned_variable))
      for binned_variable in BINNED_VARIABLES
  }
  ptrs_tau_eta = {
    binned_variable: Lazy_container(dataframe.Histo3D(model('tau_eta', third = binned_variable), 'chunk_id', 'tau_eta', binned_variable))
      for binned_variable in BINNED_VARIABLES
  }
  ptrs_dataset_id = {
    binned_variable: Lazy_container(dataframe.Histo3D(model('uh_dataset_id', third = binned_variable), 'chunk_id', 'uh_dataset_id', binned_variable))
      for binned_variable in ['uh_dataset_group_id']
  }

  lazy_containers = [ptr_lgm, ptr_st, ptr_dgi] + [ptr_di]*args.use_dataset_id +\
    [lc for lc in ptrs_tau_pt.values()]  +\
    [lc for lc in ptrs_tau_eta.values()] +\
    [lc for lc in ptrs_dataset_id.values()]*args.use_dataset_id

  for lc in lazy_containers:
    lc.load_histogram()
  
  ## run validation
  entry_lgm = Entry(var = 'tauType', histo = ptr_lgm.hst)
  entry_st  = Entry(var = 'sampleType'      , histo = ptr_st .hst)
  entry_dgi = Entry(var = 'uh_dataset_group_id', histo = ptr_dgi.hst)
  entry_di  = Entry(var = 'uh_dataset_id'      , histo = ptr_di .hst)

  entries_tau_pt = [
    Entry(var = 'tau_pt', histo = to_2D(ptrs_tau_pt[binned_variable].hst, jj+1), tdir = '/'.join([binned_variable, str(bb) if bb not in BINNED_VARIABLENAMES[binned_variable] else BINNED_VARIABLENAMES[binned_variable][bb], 'tau_pt']))
      for binned_variable in BINNED_VARIABLES
      for jj, bb in enumerate(range(*BINS[binned_variable][1:]))
  ] ; entries_tau_pt = [ee for ee in entries_tau_pt if ee.hst.GetEntries()]

  entries_tau_eta = [
    Entry(var = 'tau_eta', histo = to_2D(ptrs_tau_eta[binned_variable].hst, jj+1), tdir = '/'.join([binned_variable, str(bb) if bb not in BINNED_VARIABLENAMES[binned_variable] else BINNED_VARIABLENAMES[binned_variable][bb], 'tau_eta']))
      for binned_variable in BINNED_VARIABLES
      for jj, bb in enumerate(range(*BINS[binned_variable][1:]))
  ] ; entries_tau_eta = [ee for ee in entries_tau_eta if ee.hst.GetEntries()]
  
  entries_dataset_id = [
    Entry(var = 'uh_dataset_id', histo = to_2D(ptrs_dataset_id[binned_variable].hst, jj+1), tdir = '/'.join([binned_variable, str(bb) if bb not in BINNED_VARIABLENAMES[binned_variable] else BINNED_VARIABLENAMES[binned_variable][bb], 'uh_dataset_id']))
      for binned_variable in ['uh_dataset_group_id']
      for jj, bb in enumerate(range(*BINS[binned_variable][1:]))
      if args.use_dataset_id
  ]; entries_dataset_id = [ee for ee in entries_dataset_id if ee.hst.GetEntries()]

  entries = [entry_lgm, entry_st, entry_dgi] + [entry_di]*args.use_dataset_id +\
    [ee for ee in entries_tau_pt]  +\
    [ee for ee in entries_tau_eta] +\
    [ee for ee in entries_dataset_id]

  for ee in entries:
    ee.run_test(test = args.test)
    ee.save_data()
  print ("ID names for uh_dataset_group_id:", sorted(BINNED_VARIABLENAMES['uh_dataset_group_id'].items()))
  if args.use_dataset_id: print ("ID names for uh_dataset_id:", sorted(BINNED_VARIABLENAMES['uh_dataset_id'].items()))

  OUTPUT_ROOT.Close()
  json.dump(JSON_DICT, OUTPUT_JSON, indent = 4)
  print ('[INFO] all done. Files saved in', args.output, '. N. of processed events:', tot_entries)
