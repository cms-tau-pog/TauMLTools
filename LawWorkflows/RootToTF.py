import law
import os
import sys
import shutil

from hydra import initialize, compose
from .framework import Task, HTCondorWorkflow, startup_time, HTCondorTOpASWorkflow
from omegaconf import OmegaConf
sys.path.append(os.environ['ANALYSIS_PATH']+'/Preprocessing/root2tf/')
import luigi
import math
from getpass import getuser
from LawWorkflows.mass_copy import mass_copy

law.contrib.load("wlcg")

# if os.getenv("LAW_LOCAL_USER"):
#     LAW_LOCAL_USER = os.getenv("LAW_LOCAL_USER")
#     os.environ["USER"] = LAW_LOCAL_USER
# else:
#     LAW_LOCAL_USER = getuser()


# Collect all files in the local directory recursively
def collect_files(dir_path):
    file_paths = []
    for root, _, files in os.walk(dir_path):
        for file_name in files:
            file_path = os.path.join(root, file_name)
            file_paths.append(file_path)
    return file_paths

# Copy each file to the remote target
def copy_files_to_remote(local_dir, remote_dir):
    file_paths = collect_files(local_dir.path)

    for local_file_path in file_paths:
        # Get the relative path to create the same structure on the remote side
        rel_path = os.path.relpath(local_file_path, local_dir.path)
        remote_file_target = remote_dir.child(rel_path, type="f")

        # Make sure the remote directory exists
        remote_file_target.parent.touch()

        # Copy the file
        local_file_target = local_dir.child(rel_path, type="f")
        # print("COPY", local_file_target, remote_file_target)
        remote_file_target.copy_from_local(local_file_target)
        print(f"Copied {local_file_path} to {remote_file_target.uri()}")

class RootToTF(HTCondorTOpASWorkflow):
    ## '_' will be converted to '-' for the shell command invocation
    cfg           = luigi.Parameter(description='location of the input yaml configuration file')
    dataset_type  = luigi.Parameter(description="which samples to read (train/validation/test)")

    def __init__(self, *args, **kwargs):
        ''' run the conversion of .root files to tensorflow datasets
        '''
        super(RootToTF, self).__init__(*args, **kwargs)
        # the task is re-init on the condor node, so os.path.abspath would refer to the condor node root directory
        # re-instantiating luigi parameters bypasses this and allows to pass local paths to the condor job
        rel_cfg = os.path.relpath(self.cfg, f"{os.getenv('ANALYSIS_PATH')}/LawWorkflows")
        with initialize(config_path=os.path.dirname(rel_cfg)):
            self.cfg_dict = compose(config_name=os.path.basename(rel_cfg))
        input_data  = OmegaConf.to_object(self.cfg_dict['input_data'])
        self.dataset_cfg = input_data[self.dataset_type]

    def create_branch_map(self):
        from LawWorkflows.mass_copy import remote_glob
        datasets = self.dataset_cfg['files']
        files = []
        for dataset_name, _files in datasets.items():
            for file_path in _files:
                globbed_files = remote_glob(file_path)
                files += [(_f, dataset_name) for _f in globbed_files]
            assert len(files), "Input file list is empty: {}".format(_files)
        branch_map = {i: j for i,j in enumerate(files)}
        return branch_map


    def output(self):
        file_path, dataset_name = self.branch_data
        file_name = os.path.join(dataset_name, os.path.splitext(os.path.basename(file_path))[0])
        # print(file_name)
        output_target = self.remote_directory_target(file_name)
        # print(output_target.path)
        output_target.parent.touch()
        return output_target

    def run(self):
        from create_dataset import process_files as run_job
        file_path, dataset_name = self.branch_data
        file_name = os.path.splitext(os.path.basename(file_path))[0]
        temp_output_folder = os.path.abspath('./temp/{}'.format(file_name))
        self.cfg_dict['path_to_dataset'] = temp_output_folder
        print(f"file_path = {file_path}")
        result = run_job(
            cfg           = self.cfg_dict     ,
            files         = [file_path]  ,
            dataset_cfg   = self.dataset_cfg  ,
        )
        if not result:
            raise Exception('job {} failed'.format(self.branch))
        else:
            print('Moving files to {}'.format(self.output().parent.path))
            self.output().parent.copy_from_local(temp_output_folder)
