# coding: utf-8

import copy
import os
import math
import subprocess
import select
import luigi
import law
from law.util import interruptable_popen
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

class TaskParameters(law.Task):
    """
    Base Parameter task for Task.
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

class Task(TaskParameters):
    """
    Base task that we use to force a version parameter on all inheriting tasks, and that provides
    some convenience methods to create local file and directory targets at the default data path.
    """
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

    def convert_env_to_dict(self, env):
        my_env = {}
        for line in env.splitlines():
            if line.find(" ") < 0:
                try:
                    key, value = line.split("=", 1)
                    my_env[key] = value
                except ValueError:
                    pass
        return my_env

    def set_environment(self, sourcescript, silent=False):
        if not silent:
            print("with source script: {}".format(sourcescript))
        if isinstance(sourcescript, str):
            sourcescript = [sourcescript]
        source_command = [
            "source {};".format(sourcescript) for sourcescript in sourcescript
        ] + ["env"]
        source_command_string = " ".join(source_command)
        code, out, error = interruptable_popen(
            source_command_string,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            # rich_console=console
        )
        if code != 0:
            print("source returned non-zero exit status {}".format(code))
            print("Error: {}".format(error))
            raise Exception("source failed")
        my_env = self.convert_env_to_dict(out)
        return my_env

    # Run a bash command
    #   Command can be composed of multiple parts (interpreted as seperated by a space).
    #   A sourcescript can be provided that is called by set_environment the resulting
    #       env is then used for the command
    #   The command is run as if it was called from run_location
    #   With "collect_out" the output of the run command is returned
    def run_command(
        self,
        command=[],
        sourcescript=[],
        run_location=None,
        collect_out=False,
        silent=False,
    ):
        if command:
            if isinstance(command, str):
                command = [command]
            logstring = "Running {}".format(command)
            if run_location:
                logstring += " from {}".format(run_location)
            if not silent:
                print(logstring)
            if sourcescript:
                run_env = self.set_environment(sourcescript, silent)
            else:
                run_env = None
            code, out, error = interruptable_popen(
                " ".join(command),
                shell=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                env=run_env,
                cwd=run_location,
            )
            if not silent:
                print("Output: {}".format(out))
            if not silent or code != 0:
                print("Error: {}".format(error))
            if code != 0:
                print("Error when running {}.".format(list(command)))
                print("Command returned non-zero exit status {}.".format(code))
                raise Exception("{} failed".format(list(command)))
            else:
                if not silent:
                    print("Command successful.")
            if collect_out:
                return out
        else:
            raise Exception("No command provided.")

    def run_command_readable(self, command=[], sourcescript=[], run_location=None):
        """
        This can be used, to run a command, where you want to read the output while the command is running.
        redirect both stdout and stderr to the same output.
        """
        if command:
            if isinstance(command, str):
                command = [command]
            if sourcescript:
                run_env = self.set_environment(sourcescript)
            else:
                run_env = None
            logstring = "Running {}".format(command)
            if run_location:
                logstring += " from {}".format(run_location)
            print("--------------------")
            print(logstring)
            try:
                p = subprocess.Popen(
                    " ".join(command),
                    shell=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    env=run_env,
                    cwd=run_location,
                    encoding="utf-8",
                )
                while True:
                    reads = [p.stdout.fileno(), p.stderr.fileno()]
                    ret = select.select(reads, [], [])

                    for fd in ret[0]:
                        if fd == p.stdout.fileno():
                            read = p.stdout.readline()
                            if read != "\n":
                                print(read.strip())
                        if fd == p.stderr.fileno():
                            read = p.stderr.readline()
                            if read != "\n":
                                print(read.strip())

                    if p.poll() != None:
                        break
                if p.returncode != 0:
                    raise Exception(f"Error when running {command}.")
            except Exception as e:
                raise Exception(f"Error when running {command}.")
        else:
            raise Exception("No command provided.")

class HTCondorTOpASWorkflowParameters(Task):
    # Special treatment for other clusters might require adjustments.
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

class HTCondorTOpASWorkflow(HTCondorTOpASWorkflowParameters, HTCondorWorkflow, law.LocalWorkflow):
    # Remote workflow Task for TOpAS.

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
        config.render_variables["environment"] = self.environment

        full_req = str(self.requirements)
        if (self.num_GPUs != "None"):
            config.custom_content.append(('request_gpus', self.num_GPUs))
            if self.cuda_memory != "None":
                full_req = full_req + " && (GlobalMemoryMb > {})".format(str(self.cuda_memory))
        if full_req != "None":
            config.custom_content.append(("requirements", full_req))
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

        if self.evictable:
            config.custom_content.append(('+evictable', self.evictable))
        config.custom_content.append(('RequestDisk', f'{self.max_disk}'))
        config.custom_content.append(('accounting_group', self.accounting_group))
        config.custom_content.append(("universe", "docker"))
        config.custom_content.append(("docker_image", self.docker_image))
        config.render_variables["LOCAL_TIMESTAMP"] = startup_time
        HTC_name = self.version + "_" + str(branches)
        config.custom_content.append(('JobBatchName'  , HTC_name))
        config.custom_content.append(("error" , '/'.join([err_dir, 'err_{}.txt'.format(job_num)])))
        config.custom_content.append(("output", '/'.join([out_dir, 'out_{}.txt'.format(job_num)])))
        config.custom_content.append(("log"   , '/'.join([log_dir, 'log_{}.txt'.format(job_num)])))
        # config.custom_content.append(("stream_error", "True"))
        # config.custom_content.append(("stream_output", "True"))
        tarball_dir = os.path.abspath(f"tarballs/{self.version}")
        tarball_local = law.LocalFileTarget(
            os.path.join(
                tarball_dir,
                self.__class__.__name__,
                "TauMLTools.tar.gz",
            )
        )
        if not tarball_local.exists():
            tarball_local.parent.touch()
            excludes = ["./.[^.]*", "./Analysis", "./Production", "./Core", "./Preprocessing", "./RunKit", "./soft", "./data", "./tarballs", "*/outputs", "*/mlruns", "__pycache__"]
            exclude_str = " ".join([f"--exclude={ex}" for ex in excludes])
            os.system(f'tar {exclude_str} -czf {tarball_local.path}  .')
        config.input_files["Tau_tar"] = law.JobInputFile(tarball_local.path, render=False, copy=False)
        return config
