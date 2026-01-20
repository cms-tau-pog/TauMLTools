import law
import os
import sys
import tarfile
import mlflow
from hydra import initialize, compose
from .framework import HTCondorTOpASWorkflow, HTCondorTOpASWorkflowParameters
from omegaconf import OmegaConf
sys.path.append(os.environ['ANALYSIS_PATH']+'/Preprocessing/root2tf/')
import luigi

law.contrib.load("wlcg")

def is_valid_tarfile(filepath):
    try:
        with tarfile.open(filepath, 'r') as tar:
            tar.getmembers()  # Try to read the metadata of each member
        return True
    except (tarfile.TarError, EOFError):
        return False

class Predict(HTCondorTOpASWorkflow):
    cfg_list_file = luigi.Parameter(default = None, description='Path to a file containing a list of config files, one per line')

    def htcondor_job_config(self, config, job_num, branches):
        config = super().htcondor_job_config(config, job_num, branches)

        for branch in branches:
            tarball_dir = os.path.abspath(f"tarballs/{self.version}")
            tarball_predict = law.LocalFileTarget(
                os.path.join(
                    tarball_dir,
                    self.__class__.__name__,
                    f"Predict_{branch}.tar.gz",
                )
            )
            if not tarball_predict.exists():
                tarball_predict.parent.touch()
                excludes = ["./.[^.]*", "./Analysis", "./Production", "./Evaluation", "./Core", "./Preprocessing", "./RunKit", "./soft", "./data", "./tarballs", "*/outputs", "*/mlruns", "__pycache__"]
                exclude_str = " ".join([f"--exclude={ex}" for ex in excludes])
                os.system(f'tar {exclude_str} -czf {tarball_local.path}  .')
            config.input_files["Tau_tar"] = law.JobInputFile(tarball_local.path, render=False, copy=False)


            config.input_files["Tau_tar"] = law.JobInputFile(tarball_local.path, render=False, copy=False)
            tarball_local_eval = law.LocalFileTarget(
                os.path.join(
                    tarball_dir,
                    self.__class__.__name__,
                    f"Eval.tar.gz",
                )
            )

        if not tarball_local_eval.exists():
            # if not tarball_local_eval.exists():
            tarball_local_eval.parent.touch()
            # Use pack.py's pack_tar function directly
            sys.path.append(os.path.join(main_dir, "Evaluation"))
            from pack import pack_tar
            # from ..Evaluation.pack import pack_tar
            # Use self.cfg as the config path, tarball_local.path as the output tar file
            pack_tar(self.cfgs, tarball_local_eval.path)
        if not is_valid_tarfile(tarball_local.path):
            raise Exception(f"Tarball {tarball_local.path} is not a valid tar file.")
        config.input_files["Eval_tar"] = law.JobInputFile(tarball_local_eval.path, render=False, copy=False)
        config.output_files.append("prediction.tar.gz")
        return config

    # def __init__(self, *args, **kwargs):
    #     ''' run the conversion of .root files to tensorflow datasets
    #     '''
    #     super(Predict, self).__init__(*args, **kwargs)
    #     if self.cfg_list_file:
    #         # If cfg_list_file is provided, read the first config from it
    #         with open(self.cfg_list_file, "r") as f:
    #             cfgs = [line.strip() for line in f if line.strip()]
    #         if not cfgs:
    #             raise ValueError("cfg_list_file is empty or does not contain valid configs.")
    #         self.cfgs = cfgs
    #     else:
    #         raise ValueError("cfg_list_file must be provided.")
    #     # print(self.cfgs)

    def create_branch_map(self):
        # Read config paths from the file
        # self._cfgs = cfgs  # Store for later use
        # print({i: cfg for i, cfg in enumerate(self.cfgs)})
        branch_map = {i: cfg for i, cfg in enumerate(self.cfgs)}
        print(branch_map)
        # exit(0)
        return branch_map

    # @property
    # def cfg(self):
    #     # Return the config for this branch
    #     if not hasattr(self, "_cfgs"):
    #         # Fallback: read again if not set (shouldn't happen in law, but for safety)
    #         with open(self.cfg_list_file, "r") as f:
    #             self._cfgs = [line.strip() for line in f if line.strip()]
    #     return self._cfgs[self.branch]

    def output(self):
        output_target = self.local_target(f"files/prediction_{self.branch}To{int(self.branch) + 1}.tar.gz")
        return output_target

    def run(self):
        cfg = self.branch_data
        cfg_name = os.path.basename(cfg).split('.')[0]
        # Use self.cfg as the config for this branch
        self.run_command(
            f"tar -xzf ${{_CONDOR_SCRATCH_DIR}}/model_pack_branch_*.tar.gz",
        )
        self.run_command(
            f"python batch_package/{cfg_name}/predict_remote.py --config-name {cfg_name}",
        )
        self.run_command(
            f"tar -czf ${{LAW_JOB_INIT_DIR}}/prediction_{self.branch}To{int(self.branch) + 1}.tar.gz -C batch_package/{cfg_name} predictions",
        )
