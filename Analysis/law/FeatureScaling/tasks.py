## see https://github.com/riga/law/tree/master/examples/htcondor_at_cern
import six
import law
import subprocess
import os
import re
import sys
import glob
import shutil
import yaml

from framework import Task, HTCondorWorkflow
import luigi
sys.path.append('{}/../../../Training/python'.format(os.path.dirname(os.path.abspath(__file__))))
from feature_scaling import run_scaling as run_job

class FeatureScaling(Task, HTCondorWorkflow, law.LocalWorkflow):
  ## '_' will be converted to '-' for the shell command invocation
  cfg           = luigi.Parameter(description = 'location of the input yaml configuration file')
  var_types     = luigi.Parameter(default = "-1", description = 'variable types from field "Features_all" of the cfg file for which to derive scaling parameters. Defaults to -1 for running on all those specified in the cfg')
  files_per_job = luigi.IntParameter(default = 1, description = 'number of files to run a single job. This value defines the number of files used to log a single step')
  n_jobs        = luigi.IntParameter(default = 0, description = 'number of jobs to run. Together with --files-per-job determiines the total number of files processed. Default = 0: run on all files.')
  output_path   = luigi.Parameter(description = 'output directory')

  def __init__(self, *args, **kwargs):
    super(FeatureScaling, self).__init__(*args, **kwargs)
    # the task is re-init on the condor node, so os.path.abspath would refer to the condor node root directory
    # re-instantiating luigi parameters bypasses this and allows to pass local paths to the condor job
    self.cfg         = os.path.abspath(self.cfg)
    self.output_path = os.path.abspath(self.output_path)

    with open(self.cfg) as f:
      self.cfg_dict = yaml.load(f, Loader=(yaml.FullLoader))

  def create_branch_map(self):
    input_file_path  = self.cfg_dict['Scaling_setup']['file_path']
    files   = sorted(glob.glob(input_file_path))
    assert len(files), "Input file list is empty from path {}".format(input_file_path)

    batches = [files[j:j+self.files_per_job] for j in range(0, len(files), self.files_per_job)]
    if self.n_jobs:
      batches = batches[:self.n_jobs]

    return dict(enumerate(batches))

  def move(self, src, dest):
    if os.path.exists(dest):
      if src == dest:
        return
      if os.path.isdir(dest): shutil.rmtree(dest)
      else: os.remove(dest)
    shutil.move(src, dest)

  def output(self):
    return self.local_target("empty_file_{}.txt".format(self.branch))

  def run(self):
    destination_folder  = '/'.join([self.output_path, 'job{}'.format(self.branch)])
    if not os.path.exists(destination_folder):
      os.makedirs(destination_folder)
    temp_output_folder = os.path.abspath('./temp/'+'job{}'.format(self.branch))
    result = run_job( cfg = self.cfg                        , 
                      var_types = self.var_types.split(' ') , 
                      file_list = self.branch_data          , 
                      output_folder = temp_output_folder    )

    if not result:
      raise Exception('job {} failed'.format(self.branch))
    else:
      self.move(temp_output_folder, destination_folder)
      print('Output files moved to {}'.format(destination_folder))
      taskout = self.output()
      taskout.dump('Task ended succesfully')
