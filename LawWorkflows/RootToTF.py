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

from hydra import initialize, compose
from .framework import Task, HTCondorWorkflow
from omegaconf import DictConfig, OmegaConf, open_dict
sys.path.append(os.environ['ANALYSIS_PATH']+'/Preprocessing/root2tf/')
from create_dataset import fetch_file_list, process_files as run_job
import luigi

class RootToTF(Task, HTCondorWorkflow, law.LocalWorkflow):
  ## '_' will be converted to '-' for the shell command invocation
  cfg           = luigi.Parameter(description='location of the input yaml configuration file')
  files_per_job = luigi.IntParameter(default=1, description='number of files to run a single job.')
  n_jobs        = luigi.IntParameter(default=0, description='number of jobs to run. Together with --files-per-job determines the total number of files processed. Default=0 run on all files.')
  dataset_type  = luigi.Parameter(description="which samples to read (train/validation/test)")
  output_path   = luigi.Parameter(description="output path. Overrides 'path_to_dataset' in the cfg")

  def __init__(self, *args, **kwargs):
    ''' run the conversion of .root files to tensorflow datasets
    '''
    super(RootToTF, self).__init__(*args, **kwargs)
    # the task is re-init on the condor node, so os.path.abspath would refer to the condor node root directory
    # re-instantiating luigi parameters bypasses this and allows to pass local paths to the condor job
    self.cfg = os.path.relpath(self.cfg)

    with initialize(config_path=os.path.dirname(self.cfg)):
      self.cfg_dict = compose(config_name=os.path.basename(self.cfg))
    
    input_data  = OmegaConf.to_object(self.cfg_dict['input_data'])
    self.dataset_cfg = input_data[self.dataset_type]
#    self.output_path = os.path.abspath(self.cfg_dict['path_to_dataset'])
    self.output_path = os.path.abspath(self.output_path)
    if not os.path.exists(self.output_path):
      os.makedirs(self.output_path)
    self.cfg_dict['path_to_dataset'] = self.output_path

  def move(self, src, dest):
    #if os.path.exists(dest):
    #  if os.path.isdir(dest): shutil.rmtree(dest)
    #  else: os.remove(dest)
    shutil.move(src, dest)

  def create_branch_map(self):
    _files  = self.dataset_cfg.pop('files')
    files   = sorted([os.path.abspath(f) for f in _files])
    files   = list(fetch_file_list(files))
    assert len(files), "Input file list is empty: {}".format(_files)

    batches = [files[j:j+self.files_per_job] for j in range(0, len(files), self.files_per_job)]
    if self.n_jobs:
      batches = batches[:self.n_jobs]
    return dict(enumerate(batches))

  def output(self):
    return self.local_target("empty_file_{}.txt".format(self.branch))

  def run(self):
    temp_output_folder = os.path.abspath('./temp/'+'job{}'.format(self.branch))
    self.cfg_dict['path_to_dataset'] = temp_output_folder
    result = run_job( 
      cfg           = self.cfg_dict     ,
      dataset_type  = self.dataset_type ,
      files         = self.branch_data  ,
      dataset_cfg   = self.dataset_cfg  ,
    )
    if not result:
      raise Exception('job {} failed'.format(self.branch))
    else:
      self.move(temp_output_folder, self.output_path)
      print('Output files moved to {}'.format(self.output_path))
      taskout = self.output()
      taskout.dump('Task ended succesfully')
