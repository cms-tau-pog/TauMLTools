## see https://github.com/riga/law/tree/master/examples/htcondor_at_cern

import six
import law
import subprocess
import os
import re

from analysis.framework import Task, HTCondorWorkflow
import luigi

class ShuffleMergeSpectral(Task, HTCondorWorkflow, law.LocalWorkflow):
  ## '_' will be converted to '-' for the shell command invocation
  cfg               = luigi.Parameter(description = 'configuration file with the list of input sources')
  input_path        = luigi.Parameter(description = 'input path with tuples for all the samples')
  output_path       = luigi.Parameter(description = 'output, depending on the merging mode: MergeAll - file MergePerEntry - directory.')
  pt_bins           = luigi.Parameter(description = 'pt bins')
  eta_bins          = luigi.Parameter(description = 'eta bins')
  mode              = luigi.Parameter(description = 'merging mode: MergeAll or MergePerEntry')
  input_spec        = luigi.Parameter(description = '')
  tau_ratio         = luigi.Parameter(description = '')
  disabled_branches = luigi.Parameter(description = 'disabled-branches list of branches to disabled in the input tuples',
                                      default = '')
  n_threads         = luigi.IntParameter(description = 'number of threads', default = 1)
  exp_disbalance    = luigi.IntParameter(description = '')
  n_jobs            = luigi.IntParameter(description = 'number of HTCondor jobs to run')
  start_entry       = luigi.FloatParameter(description = 'starting point')
  end_entry         = luigi.FloatParameter(description = 'ending point')
  
  def create_branch_map(self):
    step = 1. * (self.end_entry - self.start_entry) / self.n_jobs
    return {i: (round(self.start_entry + i*step, 4), round(self.start_entry + (i+1)*step, 4)) for i in range(self.n_jobs)}

  def output(self):
    return self.local_target("stdout_stderr_job_{}.txt".format(self.branch))

  def run(self):
    output = self.output()
    file_name   = '_'.join([self.output_path.split('.')[0], str(self.branch)])
    output_name = '.'.join([file_name, self.output_path.split('.')[1]])

    quote = lambda x: str('\"{}\"'.format(str(x)))
    command = ' '.join(['ShuffleMergeSpectral',
      '--cfg'                 , str(self.cfg)                       ,
      '--input'               , str(self.input_path)                ,
      '--output'              , output_name                         ,
      '--pt-bins'             , quote(str(self.pt_bins))            ,
      '--eta-bins'            , quote(str(self.eta_bins))           ,
      '--mode'                , str(self.mode)                      ,
      '--n-threads'           , str(self.n_threads)                 ,
      '--disabled-branches'   , quote(str(self.disabled_branches))  ,
      '--input-spec'          , str(self.input_spec)                ,
      '--tau-ratio'           , quote(str(self.tau_ratio))          ,
      '--exp-disbalance'      , str(self.exp_disbalance)            ,
      '--start-entry'         , str(self.branch_data[0])            ,
      '--end-entry'           , str(self.branch_data[1])            ,
    ])
    print ('>> {}'.format(command))
    
    proc = subprocess.Popen(command, shell = True, stdout = subprocess.PIPE, stderr = subprocess.PIPE)
    stdout, stderr = proc.communicate()

    print ('\n'.join([str(stdout), str(stderr)]))
    output.dump('\n'.join([str(stdout), str(stderr)]))
