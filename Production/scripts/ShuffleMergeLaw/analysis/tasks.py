## see https://github.com/riga/law/tree/master/examples/htcondor_at_cern

import six
import law
import subprocess
import os
import re
import sys

from analysis.framework import Task, HTCondorWorkflow
import luigi

class ShuffleMergeSpectral(Task, HTCondorWorkflow, law.LocalWorkflow):
  ## '_' will be converted to '-' for the shell command invocation
  cfg               = luigi.Parameter(description = 'configuration file with the list of input sources')
  input_path        = luigi.Parameter(description = 'input path with tuples for all the samples')
  output_path       = luigi.Parameter(description = 'output directory')
  pt_bins           = luigi.Parameter(description = 'pt bins')
  eta_bins          = luigi.Parameter(description = 'eta bins')
  mode              = luigi.Parameter(description = 'merging mode: MergeAll or MergePerEntry')
  input_spec        = luigi.Parameter(description = '')
  tau_ratio         = luigi.Parameter(description = '')
  n_jobs            = luigi.IntParameter(description = 'number of HTCondor jobs to run')
  ## optional arguments (default will be None: don't override ShuffleMergeSpectral.cxx default values)
  start_entry       = luigi.FloatParameter(description = 'starting point', default = 0)
  end_entry         = luigi.FloatParameter(description = 'ending point'  , default = 1)
  compression_algo  = luigi.OptionalParameter(default = '', description = 'ZLIB, LZMA, LZ4')
  compression_level = luigi.OptionalParameter(default = '', description = 'compression level of output file')
  disabled_branches = luigi.OptionalParameter(default = '', description = 'disabled-branches list of branches to disabled in the input tuples')
  parity            = luigi.OptionalParameter(default = '', description = 'take only even:0, take only odd:1, take all entries:3')
  max_entries       = luigi.OptionalParameter(default = '', description = 'maximal number of entries in output train+test tuples')
  n_threads         = luigi.OptionalParameter(default = '', description = 'number of threads')
  exp_disbalance    = luigi.OptionalParameter(default = '', description = 'maximal expected disbalance between low pt and high pt regions')
  seed              = luigi.OptionalParameter(default = '', description = 'random seed to initialize the generator used for sampling')
  
  def create_branch_map(self):
    step = 1. * (self.end_entry - self.start_entry) / self.n_jobs
    return {i: (round(self.start_entry + i*step, 4), round(self.start_entry + (i+1)*step, 4)) for i in range(self.n_jobs)}

  def output(self):
    return self.local_target("empty_file_{}.txt".format(self.branch))

  def run(self):
    file_name   = '_'.join(['ShuffleMergeSpectral', str(self.branch)]) + '.root'
    output_name = '/'.join([self.output_path, file_name]) if self.mode == 'MergeAll' else self.output_path

    quote = lambda x: str('\"{}\"'.format(str(x)))
    command = ' '.join(['ShuffleMergeSpectral',
      '--cfg'               , str(self.cfg)             ,
      '--input'             , str(self.input_path)      ,
      '--output'            , output_name               ,
      '--pt-bins'           , quote(str(self.pt_bins))  ,
      '--eta-bins'          , quote(str(self.eta_bins)) ,
      '--mode'              , str(self.mode)            ,
      '--input-spec'        , str(self.input_spec)      ,
      '--start-entry'       , str(self.branch_data[0])  ,
      '--end-entry'         , str(self.branch_data[1])  ,
      '--tau-ratio'         , quote(str(self.tau_ratio))] +\
      ## optional arguments
      ['--seed'             , str(self.seed)                    ] * (not self.seed              is None) +\
      ['--n-threads'        , str(self.n_threads)               ] * (not self.n_threads         is None) +\
      ['--disabled-branches', quote(str(self.disabled_branches))] * (not self.disabled_branches is None) +\
      ['--exp-disbalance'   , str(self.exp_disbalance)          ] * (not self.exp_disbalance    is None) +\
      ['--compression-algo' , str(self.compression_algo)        ] * (not self.compression_algo  is None) +\
      ['--compression_level', str(self.compression_level)       ] * (not self.compression_level is None) +\
      ['--parity'           , str(self.parity)                  ] * (not self.parity            is None) +\
      ['--max_entries'      , str(self.max_entries)             ] * (not self.max_entries       is None)  )

    print ('>> {}'.format(command))
    
    proc = subprocess.Popen(command, shell = True, stdout = subprocess.PIPE, stderr = subprocess.PIPE)
    stdout, stderr = proc.communicate()

    sys.stdout.write(stdout + '\n')
    sys.stderr.write(stderr + '\n')

    proc.check_returncode()
