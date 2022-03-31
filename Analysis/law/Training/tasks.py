## see https://github.com/riga/law/tree/master/examples/htcondor_at_cern

import six
import law
import subprocess
import os
import re
import sys
import shutil
import math

from framework import Task, HTCondorWorkflow
import luigi

class Training(Task, HTCondorWorkflow, law.LocalWorkflow):

    working_dir  = luigi.Parameter(description = 'Path to the working directory.')
    enable_gpu   = luigi.Parameter(default = True, significant = False, description = 'Required GPU on node.')
    cuda_memory  = luigi.Parameter(default = 10000, significant = False, description = 'Required CUDA Global Memory (in Mb).')
    input_cmds   = luigi.Parameter(description = 'Path to the txt file with input commands.')


    comp_facility = luigi.Parameter(default = 'desy-naf',
                        description = 'Computing facility for specific setups e.g: desy-naf, lxplus')

    def htcondor_job_config(self, config, job_num, branches):
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

        if bool(self.enable_gpu):
            config.custom_content.append('Request_GPUs = 1')
            config.custom_content.append(("requirements",
                    f"(OpSysAndVer =?= \"CentOS7\") && (CUDAGlobalMemoryMb > {str(self.cuda_memory)})"))
        else:
            config.custom_content.append(("requirements", "(OpSysAndVer =?= \"CentOS7\")"))

        if self.comp_facility=="desy-naf":
            config.custom_content.append(("+RequestRuntime", int(math.floor(self.max_runtime * 3600)) - 1))
            config.custom_content.append(('RequestMemory', '{}'.format(self.max_memory)))
        elif self.comp_facility=="lxplus":
            config.custom_content.append(("+MaxRuntime", int(math.floor(self.max_runtime * 3600)) - 1))
            config.custom_content.append(('request_memory', '{}'.format(self.max_memory)))
        else:
            raise Exception('no specific setups for {self.comp_facility} computing facility')

        config.custom_content.append(("getenv", "true"))
        config.custom_content.append(('JobBatchName'  , self.batch_name))

        config.custom_content.append(("error" , '/'.join([err_dir, 'err_{}.txt'.format(job_num)])))
        config.custom_content.append(("output", '/'.join([out_dir, 'out_{}.txt'.format(job_num)])))
        config.custom_content.append(("log"   , '/'.join([log_dir, 'log_{}.txt'.format(job_num)])))

        return config

    def create_branch_map(self):

        # Opening file
        print(f"Reading commands from file: {self.input_cmds}")
        file1 = open(self.input_cmds, 'r')
        self.cmds_list = []
        for line in file1:
            self.cmds_list.append(line)
        file1.close()
        return {i: cmd for i, cmd in enumerate(self.cmds_list)}

    def output(self):
        return self.local_target("empty_file_{}.txt".format(self.branch))

    def run(self):
        
        if not os.path.exists(os.path.abspath(self.working_dir)):
            raise Exception('Working folder {} does not exist'.format(job_folder))

        command = "cd " + self.working_dir + ";\n"\
                  + self.branch_data

        print ('>> {}'.format(command))
        proc = subprocess.Popen(command, shell = True, stdout = subprocess.PIPE, stderr = subprocess.PIPE)
        stdout, stderr = proc.communicate()

        sys.stdout.write(str(stdout) + '\n')
        sys.stderr.write(str(stderr) + '\n')

        retcode = proc.returncode

        if retcode != 0:
            raise Exception('job {} return code is {}'.format(self.branch, retcode))
        else:
            taskout = self.output()
            taskout.dump('Task ended with code %s\n' %retcode)