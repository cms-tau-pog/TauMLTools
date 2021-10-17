## see https://github.com/riga/law/tree/master/examples/htcondor_at_cern

import six
import law
import subprocess
import os
import re
import sys
import shutil

from framework import Task, HTCondorWorkflow
import luigi

class ShuffleMergeFlat(Task, HTCondorWorkflow, law.LocalWorkflow):
  ## '_' will be converted to '-' for the shell command invocation
  cfg               = luigi.Parameter(description = 'configuration file with the list of input sources')
  input_path        = luigi.Parameter(description = 'Input file with the list of files to read.')
  output_path       = luigi.Parameter(description = 'output directory')
  n_jobs            = luigi.IntParameter(description = 'number of HTCondor jobs to run')
  ## optional arguments (default will be an empty string: don't override ShuffleMergeSpectral.cxx default values)
  compression_algo  = luigi.Parameter(default = '', description = 'ZLIB, LZMA, LZ4')
  compression_level = luigi.Parameter(default = '', description = 'compression level of output file')
  disabled_branches = luigi.Parameter(default = '', description = 'disabled-branches list of branches to disabled in the input tuples')
  file_entries      = luigi.Parameter(default = '', description = 'maximal number of entries per one file')
  max_entries       = luigi.Parameter(default = '', description = 'maximal number of entries in output train+test tuples')
  n_threads         = luigi.Parameter(default = '', description = 'number of threads')
  seed              = luigi.Parameter(default = '', description = 'random seed to initialize the generator used for sampling')
  # parity            = luigi.Parameter(default = '', description = 'take only even:0, take only odd:1, take all entries:3')

  def create_branch_map(self):
    self.output_dir = '/'.join([self.output_path, 'tmp'])
    ## create the .root file output directory
    if not os.path.exists(os.path.abspath(self.output_dir)):
      os.makedirs(os.path.abspath(self.output_dir))
    ## this directory will store the json has dictionaries
    if not os.path.exists(os.path.abspath('/'.join([self.output_dir, '..', 'hashes']))):
      os.makedirs(os.path.abspath('/'.join([self.output_dir, '..', 'hashes'])))

    return {i: i for i in range(self.n_jobs)}

  def output(self):
    return self.local_target("empty_file_{}.txt".format(self.branch))

  def move(self, src, dest):
    if os.path.exists(dest):
      if os.path.isdir(dest): shutil.rmtree(dest)
      else: os.remove(dest)
    shutil.move(src, dest)

  def run(self):
    self.output_dir = '/'.join([self.output_path, 'tmp'])
    file_name   = '_'.join(['ShuffleMergeFlat', str(self.branch)]) + '.root'
    output_name = '/'.join([self.output_dir, file_name])

    quote = lambda x: str('\"{}\"'.format(str(x)))
    command = ' '.join(['ShuffleMergeFlat',
      '--cfg'               , str(self.cfg)             ,
      '--input'             , str(self.input_path)      ,
      '--output'            , output_name               ,
      '--n-jobs'            , str(self.n_jobs)          ,
      '--job-idx'           , str(self.branch_data)     ] +\
      ## optional arguments
      ['--seed'             , str(self.seed)                    ] * (not self.seed               is '') +\
      ['--n-threads'        , str(self.n_threads)               ] * (not self.n_threads          is '') +\
      ['--disabled-branches', quote(str(self.disabled_branches))] * (not self.disabled_branches  is '') +\
      ['--compression-algo' , str(self.compression_algo)        ] * (not self.compression_algo   is '') +\
      ['--compression-level', str(self.compression_level)       ] * (not self.compression_level  is '') +\
      ['--max-entries'      , str(self.max_entries)             ] * (not self.max_entries        is '') +\
      # ['--parity'           , str(self.parity)                  ] * (not self.parity             is '') +\
      ['--file-entries'     , str(self.file_entries)            ] * (not self.file_entries       is ''))

    print ('>> {}'.format(command))
    proc = subprocess.Popen(command, shell = True, stdout = subprocess.PIPE, stderr = subprocess.PIPE)
    stdout, stderr = proc.communicate()

    sys.stdout.write(str(stdout) + '\n')
    sys.stderr.write(str(stderr) + '\n')

    retcode = proc.returncode

    if retcode != 0:
      raise Exception('job {} return code is {}'.format(self.branch, retcode))
    else:
      self.move(os.path.abspath(output_name), os.path.abspath('/'.join([self.output_dir, '..', file_name])))
      self.move(os.path.abspath('./out'    ), os.path.abspath('/'.join([self.output_dir, '..', 'hashes', 'out_{}'.format(self.branch)])))
      print('Output file and hash tables moved to {}\n'.format(os.path.abspath('/'.join([self.output_dir, '..']))))
      taskout = self.output()
      taskout.dump('Task ended with code %s\n' %retcode)
