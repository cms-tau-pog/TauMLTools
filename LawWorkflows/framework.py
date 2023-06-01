# coding: utf-8

import copy
import os
import math

import luigi
import law
law.contrib.load("htcondor")

def copy_param(ref_param, new_default):
  param = copy.deepcopy(ref_param)
  param._default = new_default
  return param

class Task(law.Task):
  """
  Base task that we use to force a version parameter on all inheriting tasks, and that provides
  some convenience methods to create local file and directory targets at the default data path.
  """

  version = luigi.Parameter()

  def store_parts(self):
    return (self.__class__.__name__, self.version)

  def local_path(self, *path):
    # ANALYSIS_DATA_PATH is defined in env.sh
    parts = (os.getenv("ANALYSIS_DATA_PATH"),) + self.store_parts() + path
    return os.path.join(*parts)

  def local_target(self, *path):
    return law.LocalFileTarget(self.local_path(*path))


class HTCondorWorkflow(law.htcondor.HTCondorWorkflow):
  """
  Batch systems are typically very heterogeneous by design, and so is HTCondor. Law does not aim
  to "magically" adapt to all possible HTCondor setups which would certainly end in a mess.
  Therefore we have to configure the base HTCondor workflow in law.contrib.htcondor to work with
  the CERN HTCondor environment. In most cases, like in this example, only a minimal amount of
  configuration is required.
  """
  max_runtime = law.DurationParameter(default=12.0, unit="h", significant=False, description="maximum runtime")
  max_memory  = luigi.Parameter(default = '2000', significant = False, description = 'maximum RAM usage')
  batch_name  = luigi.Parameter(default = 'TauML', description = 'HTCondor batch name')
  environment = luigi.ChoiceParameter(default = "", choices = ['', 'cmssw', 'conda', 'cmssw_conda'], var_type = str,
                                      description = "Environment used to run the job")
  requirements = luigi.Parameter(default='', significant=False, description='Requirements for HTCondor nodes')
  poll_interval = copy_param(law.htcondor.HTCondorWorkflow.poll_interval, 5) # set poll interval to 5 minutes

  def htcondor_output_directory(self):
    # the directory where submission meta data should be stored
    return law.LocalDirectoryTarget(self.local_path())

  def htcondor_bootstrap_file(self):
    # each job can define a bootstrap file that is executed prior to the actual job
    # in order to setup software and environment variables
    return law.util.rel_path(os.getenv("ANALYSIS_PATH"), "bootstrap.sh")

  def htcondor_job_config(self, config, job_num, branches):
    report_dir = str(self.htcondor_output_directory().path)
    for name in ['error', 'output', 'log']:
      log_dir = os.path.join(report_dir, f'{name}s')
      os.makedirs(log_dir, exist_ok=True)
      config.custom_content.append((name, os.path.join(log_dir, f'{name}.{job_num}.$(ClusterId).$(ProcId).txt')))

    # render_variables are rendered into all files sent with a job
    config.render_variables["analysis_path"] = os.getenv("ANALYSIS_PATH")
    config.render_variables["environment"] = self.environment
    if 'CONDA_EXE' in os.environ:
      config.render_variables["conda_path"]    = '/'.join(os.environ['CONDA_EXE'].split('/')[:-2])

    # maximum runtime
    config.custom_content.append(("+MaxRuntime", int(math.floor(self.max_runtime * 3600)) - 1))
    if len(self.requirements) > 0:
      config.custom_content.append(("requirements", self.requirements))

    config.custom_content.append(('request_memory', f'{self.max_memory}'))
    config.custom_content.append(('JobBatchName', self.batch_name))

    return config
