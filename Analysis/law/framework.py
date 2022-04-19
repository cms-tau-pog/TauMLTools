# coding: utf-8

"""
Law example tasks to demonstrate HTCondor workflows at CERN.

In this file, some really basic tasks are defined that can be inherited by
other tasks to receive the same features. This is usually called "framework"
and only needs to be defined once per user / group / etc.
"""


import os
import math

import luigi
import law


# the htcondor workflow implementation is part of a law contrib package
# so we need to explicitly load it
law.contrib.load("htcondor")


class Task(law.Task):
  """
  Base task that we use to force a version parameter on all inheriting tasks, and that provides
  some convenience methods to create local file and directory targets at the default data path.
  """

  version = luigi.Parameter()

  def store_parts(self):
    return (self.__class__.__name__, self.version)

  def local_path(self, *path):
    # ANALYSIS_DATA_PATH is defined in setup.sh
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
  max_runtime = law.DurationParameter(default=24.0, unit="h", significant=False,
    description="maximum runtime, default: 24h")
  max_memory  = luigi.Parameter(default = '2000', significant = False,
    description = 'maximum RAM usage')
  batch_name  = luigi.Parameter(default = 'TauML_law',
    description = 'HTCondor batch name')
  environment = luigi.ChoiceParameter(default = "CMSSW", choices = ['CMSSW', 'conda'], var_type = str,
    description = "Environment used to run the job")

  def htcondor_output_directory(self):
    # the directory where submission meta data should be stored
    return law.LocalDirectoryTarget(self.local_path())

  def htcondor_bootstrap_file(self):
    # each job can define a bootstrap file that is executed prior to the actual job
    # in order to setup software and environment variables
    return law.util.rel_path(__file__, "bootstrap.sh")

  def htcondor_job_config(self, config, job_num, branches):
    main_dir = os.getenv("ANALYSIS_PATH")
    report_dir = str(self.htcondor_output_directory().path)

    err_dir = '/'.join([report_dir, 'errors'])
    out_dir = '/'.join([report_dir, 'outputs'])
    log_dir = '/'.join([report_dir, 'logs'])

    if not os.path.exists(err_dir): os.makedirs(err_dir)
    if not os.path.exists(out_dir): os.makedirs(out_dir)
    if not os.path.exists(log_dir): os.makedirs(log_dir)

    good_envs = ['CMSSW', 'conda']
    assert self.environment in good_envs, \
    "--environment must be in the list {}. You wrote {}".format(good_envs, self.environment)

    # render_variables are rendered into all files sent with a job
    config.render_variables["analysis_path"] = main_dir
    config.render_variables["cmssw_base"]    = str(os.getenv('CMSSW_BASE'))
    config.render_variables["environment"]   = self.environment
    config.render_variables["conda_path"]    = '/'.join(os.environ['CONDA_EXE'].split('/')[:-2])
    config.render_variables["conda_env"]     = os.environ['CONDA_DEFAULT_ENV']
    config.render_variables["pythonpath"]    = os.environ['PYTHONPATH']
    config.render_variables["path"]          = os.environ['PATH']
    # force to run on CC7, http://batchdocs.web.cern.ch/batchdocs/local/submit.html#os-choice
    config.custom_content.append(("requirements", "(OpSysAndVer =?= \"CentOS7\")"))
    # maximum runtime
    config.custom_content.append(("+MaxRuntime", int(math.floor(self.max_runtime * 3600)) - 1))
    config.custom_content.append(("getenv", "true"))

    config.custom_content.append(('request_memory', '{}'.format(self.max_memory)))
    config.custom_content.append(('JobBatchName'  , self.batch_name))

    config.custom_content.append(("error" , '/'.join([err_dir, 'err_{}.txt'.format(job_num)])))
    config.custom_content.append(("output", '/'.join([out_dir, 'out_{}.txt'.format(job_num)])))
    config.custom_content.append(("log"   , '/'.join([log_dir, 'log_{}.txt'.format(job_num)])))

    config.custom_content.append(("x509userproxy", os.environ['X509_USER_PROXY']))

    return config
