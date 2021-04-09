## see https://github.com/riga/law/tree/master/examples/htcondor_at_cern

import six
import law
import subprocess
import os
import re
import sys

from framework import Task, HTCondorWorkflow
import luigi

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

