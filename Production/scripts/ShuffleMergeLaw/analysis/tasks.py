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
  compression_algo  = luigi.Parameter(default = '', description = 'ZLIB, LZMA, LZ4')
  compression_level = luigi.Parameter(default = '', description = 'compression level of output file')
  disabled_branches = luigi.Parameter(default = '', description = 'disabled-branches list of branches to disabled in the input tuples')
  parity            = luigi.Parameter(default = '', description = 'take only even:0, take only odd:1, take all entries:3')
  max_entries       = luigi.Parameter(default = '', description = 'maximal number of entries in output train+test tuples')
  n_threads         = luigi.Parameter(default = '', description = 'number of threads')
  exp_disbalance    = luigi.Parameter(default = '', description = 'maximal expected disbalance between low pt and high pt regions')
  seed              = luigi.Parameter(default = '', description = 'random seed to initialize the generator used for sampling')

  def create_branch_map(self):
    step = 1. * (self.end_entry - self.start_entry) / self.n_jobs
    return {i: (round(self.start_entry + i*step, 6), round(self.start_entry + (i+1)*step, 6)) for i in range(self.n_jobs)}

  def output(self):
    return self.local_target("empty_file_{}.txt".format(self.branch))

  def run(self):
    file_name   = '_'.join(['ShuffleMergeSpectral', str(self.branch)]) + '.root'
    self.output_dir = getattr(self, 'output_dir', self.output_path)
    output_name = '/'.join([self.output_dir, file_name]) if self.mode == 'MergeAll' else self.output_path

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
      ['--seed'             , str(self.seed)                    ] * (not self.seed              is '') +\
      ['--n-threads'        , str(self.n_threads)               ] * (not self.n_threads         is '') +\
      ['--disabled-branches', quote(str(self.disabled_branches))] * (not self.disabled_branches is '') +\
      ['--exp-disbalance'   , str(self.exp_disbalance)          ] * (not self.exp_disbalance    is '') +\
      ['--compression-algo' , str(self.compression_algo)        ] * (not self.compression_algo  is '') +\
      ['--compression_level', str(self.compression_level)       ] * (not self.compression_level is '') +\
      ['--parity'           , str(self.parity)                  ] * (not self.parity            is '') +\
      ['--max-entries'      , str(self.max_entries)             ] * (not self.max_entries       is '')  )

    print ('>> {}'.format(command))
    proc = subprocess.Popen(command, shell = True, stdout = subprocess.PIPE, stderr = subprocess.PIPE)
    stdout, stderr = proc.communicate()

    sys.stdout.write(stdout + '\n')
    #sys.stderr.write(stderr + '\n')
    sys.stdout.write(stderr + '\n')

    retcode = proc.returncode
    if proc.returncode != 0:
      raise Exception('job {} return code is {}'.format(self.branch, retcode))

class HaddFiles(Task, HTCondorWorkflow, law.LocalWorkflow):
  class InputFile:
    def __init__(self, path, size):
      self.path = path
      self.size = size
  class FileBatch:
    def __init__(self, dataset):
      self.files = []
      self.dataset = dataset
    def size(self):
      return sum([ff.size for ff in self.files])

  ## '_' will be converted to '-' for the shell command invocation
  input_path  = luigi.Parameter(description = 'input path with tuples for all the samples')
  output_path = luigi.Parameter(description = 'output directory')
  output_size = luigi.FloatParameter(description = 'output file size in GB', default = 10.)

  def create_branch_map(self):
    batches = []
    datasets = [fol for fol in os.listdir(self.input_path) if os.path.isdir('/'.join([self.input_path, fol]))]
    for ds in datasets:
      if not os.path.exists('/'.join([self.output_path, ds])):
        os.mkdir('/'.join([self.output_path, ds]))

      batches.append(self.FileBatch(dataset = ds))

      dataset_path  = '/'.join([self.input_path, ds])
      dataset_files = ['/'.join([dataset_path, fil]) for fil in os.listdir(dataset_path) if os.path.isfile('/'.join([dataset_path, fil]))]

      for ff in dataset_files:
        if batches[-1].size() > self.output_size*1.e+9:
          batches.append(self.FileBatch(dataset = ds))
        batches[-1].files.append(self.InputFile(path = ff, size = os.path.getsize(ff)))

    return dict(enumerate(batches))

  def output(self):
    return self.local_target("empty_file_{}.txt".format(self.branch))

  def run(self):
    quote = lambda x: str('\"{}\"'.format(str(x)))
    command = 'hadd -O -ff -k {OUT} {IN}'.format(
      OUT = '/'.join([self.output_path, self.branch_data.dataset, 'HaddFile_{}.root'.format(self.branch)]),
      IN  = ' '.join([ff.path for ff in self.branch_data.files])
    )

    print ('>> {}'.format(command))
    proc = subprocess.Popen(command, shell = True, stdout = subprocess.PIPE, stderr = subprocess.PIPE)
    stdout, stderr = proc.communicate()

    sys.stdout.write(stdout + '\n')
    #sys.stderr.write(stderr + '\n')
    sys.stdout.write(stderr + '\n')

    retcode = proc.returncode
    if proc.returncode != 0:
      raise Exception('job {} return code is {}'.format(self.branch, retcode))

