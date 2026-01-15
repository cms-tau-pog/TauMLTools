# coding: utf-8

import copy
import os
import math

import luigi
import law
from datetime import datetime
from getpass import getuser
from tempfile import mkdtemp
law.contrib.load("htcondor")
law.contrib.load("wlcg")

if os.getenv("LOCAL_TIMESTAMP"):
    startup_time = os.getenv("LOCAL_TIMESTAMP")
else:
    startup_time = datetime.now().strftime("%Y_%m_%d_%H_%M_%S_%f")

def copy_param(ref_param, new_default):
    param = copy.deepcopy(ref_param)
    param._default = new_default
    return param

class Task(law.Task):
    """
    Base task that we use to force a version parameter on all inheriting tasks, and that provides
    some convenience methods to create local file and directory targets at the default data path.
    """

    version = luigi.Parameter(
      default="default/{}".format(startup_time),
      description="Versions of runs. Set to a timestamp as default."
    )
    wlcg_path = luigi.Parameter(description="Base-path to remote file location.")
    local_output_path = luigi.Parameter(
        description="Base-path to local file location.",
        default=os.getenv("ANALYSIS_DATA_PATH"),
    )
    is_local_output = luigi.BoolParameter(
        description="Whether to use local storage. False by default."
    )
    try:
        local_user = getuser()
    except:
        pass

    def store_parts(self):
      return (self.__class__.__name__, self.version)

    # Path of local targets.
    #   Composed from the analysis path set during the setup.sh
    #   or the local_output_path if is_local_output is set,
    #   the production_tag, the name of the task and an additional path if provided.
    def local_path(self, *path):
        return os.path.join(
            (
                self.local_output_path
                if self.is_local_output
                else os.getenv("ANALYSIS_DATA_PATH")
            ),
            *self.store_parts(),
            *path,
        )

    def temporary_local_path(self, *path):
        if os.environ.get("_CONDOR_JOB_IWD"):
            prefix = os.environ.get("_CONDOR_JOB_IWD") + "/tmp/"
        else:
            prefix = f"/tmp/{self.local_user}"
        temporary_dir = mkdtemp(dir=prefix)
        parts = (temporary_dir,) + (self.__class__.__name__,) + path
        return os.path.join(*parts)

    def local_target(self, path):
        if isinstance(path, (list, tuple)):
            return [law.LocalFileTarget(self.local_path(p)) for p in path]
        return law.LocalFileTarget(self.local_path(path))

    def local_directory_target(self, path):
        if isinstance(path, (list, tuple)):
            return [law.LocalDirectoryTarget(self.local_path(p)) for p in path]
        return law.LocalDirectoryTarget(self.local_path(path))

    def temporarylocal_target(self, *path):
        return law.LocalFileTarget(self.temporary_local_path(*path))

    # Path of remote targets. Composed from the production_tag,
    #   the name of the task and an additional path if provided.
    #   The wlcg_path will be prepended for WLCGFileTargets
    def remote_path(self, *path):
        parts = (self.version,) + (self.__class__.__name__,) + path
        #  os.path.join(*parts)
        tmp = os.path.join(*parts)
        print(tmp)
        return tmp

    def remote_target(self, path):
        if self.is_local_output:
            return self.local_target(path)

        if isinstance(path, (list, tuple)):
            return [law.wlcg.WLCGFileTarget(self.remote_path(p)) for p in path]

        return law.wlcg.WLCGFileTarget(self.remote_path(path))

    def remote_directory_target(self, path):
        if self.is_local_output:
            return self.local_directory_target(path)

        if isinstance(path, (list, tuple)):
            return [law.wlcg.WLCGDirectoryTarget(self.remote_path(p)) for p in path]

        # return law.wlcg.WLCGDirectoryTarget(self.remote_path(path))
        tmp = law.wlcg.WLCGDirectoryTarget(self.remote_path(path))
        print(tmp)
        return tmp


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
    max_disk  = luigi.Parameter(default = 'None', significant = False, description = 'maximum scratch space usage')
    num_CPUs   = luigi.Parameter(default = "None", significant = False, description = 'Number of requested CPU.')
    accounting_group   = luigi.Parameter(default = "1", significant = False, description = 'Accounting used for TOpAS.')
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
        config.render_variables["LOCAL_TIMESTAMP"] = startup_time
        if 'CONDA_EXE' in os.environ:
            config.render_variables["conda_path"]    = '/'.join(os.environ['CONDA_EXE'].split('/')[:-2])

        # maximum runtime
        config.custom_content.append(("+MaxRuntime", int(math.floor(self.max_runtime * 3600)) - 1))
        if len(self.requirements) > 0:
            config.custom_content.append(("requirements", self.requirements))

        config.custom_content.append(('request_memory', f'{self.max_memory}'))
        config.custom_content.append(('request_cpus', self.num_CPUs))
        config.custom_content.append(('JobBatchName', self.batch_name))
        config.custom_content.append(('RequestDisk', f'{self.max_disk}'))
        htcondor_user_proxy = law.wlcg.get_vomsproxy_file()
        config.custom_content.append(("x509userproxy", htcondor_user_proxy))
        config.custom_content.append(('accounting_group', self.accounting_group))

        return config

class HTCondorTOpASWorkflow(Task, HTCondorWorkflow, law.LocalWorkflow):
    # Remote workflow Task for TOpAS.
    # Special treatment for other clusters might require adjustments.
      # class RootToTF(Task, law.LocalWorkflow):
      ## '_' will be converted to '-' for the shell command invocation
      # cfg           = luigi.Parameter(description='location of the input yaml configuration file')
      # n_jobs        = luigi.IntParameter(default=0, description='number of jobs to run. Together with --files-per-job determines the total number of files processed. Default=0 run on all files.')
      # dataset_type  = luigi.Parameter(description="which samples to read (train/validation/test)")
    evictable  = luigi.Parameter(default = "False", description = 'Can job be evicted without breaking?')
    num_CPUs   = luigi.Parameter(default = "None", significant = False, description = 'Number of requested CPU.')
    num_GPUs   = luigi.Parameter(default = "None", significant = False, description = 'Number of requested GPU.')
    accounting_group   = luigi.Parameter(default = "None", significant = False, description = 'Accounting used for TOpAS.')
    cuda_memory  = luigi.Parameter(default = "None", significant = False, description = 'Amount of necessary device memory.')
    requirements = luigi.Parameter(default="None", significant = False, description = 'HTCondor requirements')
    max_disk  = luigi.Parameter(default = 'None', significant = False, description = 'maximum scratch space usage')
    max_runtime = law.DurationParameter(default=12.0, unit="h", significant=False, description="maximum runtime")
    max_memory  = luigi.Parameter(default = '2000', significant = False, description = 'maximum RAM usage')
    docker_image = luigi.Parameter(default='None', significant=False, description='Used docker image')

    comp_facility = luigi.Parameter(default = 'TOpAS',
                                    description = 'Computing facility for specific setups e.g: desy-naf, lxplus')

    # Redirect location of job files to <local_path>/"files"/...
    def htcondor_create_job_file_factory(self):
        jobdir = self.local_path("files")
        os.makedirs(jobdir, exist_ok=True)
        factory = super(HTCondorWorkflow, self).htcondor_create_job_file_factory(
            dir=jobdir,
            mkdtemp=False,
        )
        return factory

    def htcondor_job_config(self, config, job_num, branches):
        config.custom_content = []
        main_dir = os.getenv("ANALYSIS_PATH")
        report_dir = str(self.htcondor_output_directory().path)

        err_dir = '/'.join([report_dir, 'errors'])
        out_dir = '/'.join([report_dir, 'outputs'])
        log_dir = '/'.join([report_dir, 'logs'])

        if not os.path.exists(err_dir): os.makedirs(err_dir)
        if not os.path.exists(out_dir): os.makedirs(out_dir)
        if not os.path.exists(log_dir): os.makedirs(log_dir)

        # render_variables are rendered into all files sent with a job
        config.render_variables["analysis_path"] = main_dir
        # config.render_variables["copy_in"] = "False"

        full_req = str(self.requirements)
        if (self.num_GPUs != "None"):
            config.custom_content.append(('request_gpus', self.num_GPUs))
            if self.cuda_memory != "None":
                full_req = full_req + " && (GlobalMemoryMb > {})".format(str(self.cuda_memory))
        if full_req != "None":
            config.custom_content.append(("requirements", full_req))
        # if self.comp_facility=="desy-naf":
        #     config.custom_content.append(("+RequestRuntime", int(math.floor(self.max_runtime * 3600)) - 1))
        #     config.custom_content.append(('RequestMemory', '{}'.format(self.max_memory)))
        # elif self.comp_facility=="lxplus":
        #     config.custom_content.append(("+MaxRuntime", int(math.floor(self.max_runtime * 3600)) - 1))
        #     config.custom_content.append(('request_memory', '{}'.format(self.max_memory)))
        # elif self.comp_facility == "ETP":
            # Use proxy file located in $X509_USER_PROXY or /tmp/x509up_u$(id) if empty
        htcondor_user_proxy = law.wlcg.get_vomsproxy_file()
        config.render_variables["comp_facility"] = self.comp_facility
        config.custom_content.append(("x509userproxy", htcondor_user_proxy))
        config.custom_content.append(('+RemoteJob', 'True'))
        config.custom_content.append(("+RequestWalltime", int(math.floor(self.max_runtime * 3600)) - 1))
        for i in ["num_CPUs", "max_memory", "max_disk", "accounting_group", "docker_image"]:
            if getattr(self, i) == "None":
                raise Exception('TOpAS requires a value for {}.'.format(i))
        config.custom_content.append(('request_cpus', self.num_CPUs))
        config.custom_content.append(('RequestMemory', self.max_memory))
        # config.custom_content.append(('RequestMemory', f"{self.max_memory} + {self.max_memory} * (1/4 * NumJobStarts)"))
        # config.custom_content.append(('periodic_hold', "(HoldReasonCode == 34)"))
        # config.custom_content.append(('periodic_hold_reason', '"OOM, retrying"'))
        # config.custom_content.append(('periodic_release', "(HoldReasonCode == 34)"))
        # config.custom_content.append(('max_retries', "2")) # Max + 50% memory on retry

        if self.evictable:
            config.custom_content.append(('+evictable', self.evictable))
        config.custom_content.append(('RequestDisk', f'{self.max_disk}'))
        config.custom_content.append(('accounting_group', self.accounting_group))
        config.custom_content.append(("universe", "docker"))
        config.custom_content.append(("docker_image", self.docker_image))
        # tarball_dir = os.path.abspath(f"{main_dir}/tarballs/{self.version}")
        # tarball_local = law.LocalFileTarget(
        #     os.path.join(
        #         tarball_dir,
        #         self.__class__.__name__,
        #         "TauMLTools.tar.gz",
        #     )
        # )
        # if not tarball_local.exists():
        #     tarball_local.parent.touch()
        #     excludes = ["./.[^.]*", "./Analysis", "./Production", "./Evaluation", "./Core", "./Training", "./RunKit", "./soft", "./data", "./tarballs", "*/outputs", "*/mlruns", "__pycache__"]
        #     exclude_str = " ".join([f"--exclude={ex}" for ex in excludes])
        #     os.system(f'tar {exclude_str} -czf {tarball_local.path}  .')
        #     tarball_local.parent.touch()
        # config.input_files["Tau_tar"] = law.JobInputFile(tarball_local.path, render=False, copy=False)
        # else:
        #     raise Exception('no specific setups for {self.comp_facility} computing facility')

        # if self.comp_facility != "ETP":
        #     config.custom_content.append(("getenv", "true"))
        # config.render_variables["environment"] = self.environment
        config.render_variables["LOCAL_TIMESTAMP"] = startup_time
        config.custom_content.append(('JobBatchName'  , self.batch_name))
        config.custom_content.append(("error" , '/'.join([err_dir, 'err_{}.txt'.format(job_num)])))
        config.custom_content.append(("output", '/'.join([out_dir, 'out_{}.txt'.format(job_num)])))
        config.custom_content.append(("log"   , '/'.join([log_dir, 'log_{}.txt'.format(job_num)])))
        # config.custom_content.append(("stream_error", "True"))
        # config.custom_content.append(("stream_output", "True"))
        return config


    # def __init__(self, *args, **kwargs):
    #     ''' run the conversion of .root files to tensorflow datasets
    #     '''
    #     super(RootToTF, self).__init__(*args, **kwargs)
    #     # the task is re-init on the condor node, so os.path.abspath would refer to the condor node root directory
    #     # re-instantiating luigi parameters bypasses this and allows to pass local paths to the condor job
    #     rel_cfg = os.path.relpath(self.cfg, f"{os.getenv('ANALYSIS_PATH')}/LawWorkflows")
    #     with initialize(config_path=os.path.dirname(rel_cfg)):
    #         self.cfg_dict = compose(config_name=os.path.basename(rel_cfg))
    #     input_data  = OmegaConf.to_object(self.cfg_dict['input_data'])
    #     self.dataset_cfg = input_data[self.dataset_type]

    # def create_branch_map(self):
    #     from mass_copy import remote_glob
    #     _files = self.dataset_cfg.pop('files')
    #     files = []
    #     for file_path in _files:
    #         files += remote_glob(file_path)
    #     assert len(files), "Input file list is empty: {}".format(_files)
    #     branch_map = {i: j for i,j in enumerate(files)}
    #     print(branch_map)
    #     return branch_map


    # def output(self):
    #     file_path = self.branch_data
    #     file_name = os.path.splitext(os.path.basename(file_path))[0]
    #     output_target = self.remote_directory_target(file_name)
    #     output_target.parent.touch()
    #     return output_target

    # def run(self):
    #     from create_dataset import process_files as run_job
    #     file_path = self.branch_data
    #     file_name = os.path.splitext(os.path.basename(file_path))[0]
    #     temp_output_folder = os.path.abspath('./temp/{}'.format(file_name))
    #     self.cfg_dict['path_to_dataset'] = temp_output_folder
    #     print(f"file_path = {file_path}")
    #     result = run_job(
    #         cfg           = self.cfg_dict     ,
    #         files         = [file_path]  ,
    #         dataset_cfg   = self.dataset_cfg  ,
    #     )
    #     if not result:
    #         raise Exception('job {} failed'.format(self.branch))
    #     else:
    #         copy_files_to_remote(law.LocalDirectoryTarget(temp_output_folder), self.output().parent)
    #         print('Output files moved to {}'.format(self.output().path))

