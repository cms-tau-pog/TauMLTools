## see https://github.com/riga/law/tree/master/examples/htcondor_at_cern

import six
import law
import subprocess
import os
import re
import sys

from framework import Task, HTCondorWorkflow
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
    self.output_dir = '/'.join([self.output_path, 'tmp'])
    ## create the .root file output directory
    if not os.path.exists(self.output_dir):
      os.makedirs(self.output_dir)
    ## this directory will store the json has dictionaries
    if not os.path.exists('/'.join([self.output_dir, '..', 'hashes'])):
      os.makedirs('/'.join([self.output_dir, '..', 'hashes']))


    step = 1. * (self.end_entry - self.start_entry) / self.n_jobs
    return {i: (round(self.start_entry + i*step, 6), round(self.start_entry + (i+1)*step, 6)) for i in range(self.n_jobs)}

  def output(self):
    return self.local_target("empty_file_{}.txt".format(self.branch))

  def run(self):
    self.output_dir = '/'.join([self.output_path, 'tmp'])
    file_name   = '_'.join(['ShuffleMergeSpectral', str(self.branch)]) + '.root'
    output_name = '/'.join([self.output_dir, file_name]) if self.mode == 'MergeAll' else self.output_dir

    if not self.mode == 'MergeAll':
      raise Exception('Only --mode MergeAll is supported by the law tool for now')

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

    sys.stdout.write(str(stdout) + '\n')
    sys.stderr.write(str(stderr) + '\n')

    retcode = proc.returncode

    if retcode != 0:
      raise Exception('job {} return code is {}'.format(self.branch, retcode))
    elif retcode == 0 and self.mode == 'MergeAll':
      os.rename(output_name, '/'.join([self.output_dir, '..', file_name]))
      os.rename('./out', '/'.join([self.output_dir, '..', 'hashes', 'out_{}'.format(self.branch)]))
      taskout = self.output()
      taskout.dump('Task ended with code %s\n' %retcode)
    else:
      ## TODO: what happens if not MergeAll?
      pass
