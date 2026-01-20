import os
import re
import law
import luigi
import yaml
import tarfile
import mlflow
import tempfile
from hydra import compose, initialize
from .framework import HTCondorTOpASWorkflow
from .Training import MLflowLogTraining
from LawWorkflows.mass_copy import mass_copy
from hydra.utils import to_absolute_path

class Predict(HTCondorTOpASWorkflow):
    """
    HTCondor Workflow to run predictions.
    Matches training runs with prediction tasks using the original training commands,
    packs the required model/checkpoint and scaler, and runs the prediction on HTCondor.
    """

    predict_cfg = luigi.Parameter(
        description="Path to the yaml config defining files for prediction."
    )
    training_cmds = luigi.Parameter(
        description="Path to the txt file with original training commands to match models."
    )
    mlruns_dir = luigi.Parameter(
        default="mlruns",
        description="Path to the local central mlruns directory."
    )

    def create_branch_map(self):
        # Opening file
        print(f"Reading commands from file: {self.training_cmds}")
        self.cmds_list = {}

        required_keys = {
            "input_files": r"input_files=(\S+)(\s|$)",
            "experiment_name": r"experiment_name=(\S+)(\s|$)",
            "run_name": r"run_name=(\S+)(\s|$)",
        }

        with open(self.training_cmds, 'r') as file1:
            for i, line in enumerate(file1):
                self.cmds_list[i] = {}
                for key, pattern in required_keys.items():
                    match = re.search(pattern, line)
                    if not match:
                        raise ValueError(
                            f"Missing required '{key}' in training_cmds line {i}:\n{line}"
                        )
                    self.cmds_list[i][key] = match.group(1)
        return self.cmds_list

    def output(self):
        return self.local_target("files/predict_{}To{}.tar.gz".format(self.branch, int(self.branch) + 1))

    def htcondor_job_config(self, config, job_num, branches):
        # print(branches, job_num)
        config = super().htcondor_job_config(config, job_num, branches)
        # config.custom_content.append(("stream_error", "True"))
        # config.custom_content.append(("stream_output", "True"))
        config.output_files.append("prediction.tar.gz")
        # print(self.create_branch_map())
        branch_data = self.create_branch_map()[job_num-1]
        client = mlflow.tracking.MlflowClient(tracking_uri=f"file:{self.mlruns_dir}")

        with open(self.predict_cfg, "r") as f:
            p_cfg = yaml.safe_load(f)

        # Extract names to resolve path
        exp_name = branch_data["experiment_name"]
        run_name = branch_data["run_name"]
        input_files = branch_data["input_files"]

        exp = client.get_experiment_by_name(exp_name)
        if not exp:
            raise Exception(f"Experiment {exp_name} not found in {self.mlruns_dir}")

        runs = client.search_runs(
            experiment_ids=[exp.experiment_id],
            filter_string=f"tags.mlflow.runName = '{run_name}'"
        )
        if not runs:
            raise Exception(f"Run {run_name} not found.")

        run_id = runs[0].info.run_id
        artifact_path = os.path.join(self.mlruns_dir, exp.experiment_id, run_id, "artifacts")

        # Determine target model path
        checkpoint_name = p_cfg.get("checkpoint")
        if checkpoint_name:
            model_src = os.path.join(artifact_path, "checkpoints", checkpoint_name)
            arc_model_name = f"checkpoints/{checkpoint_name}*"
        else:
            model_src = os.path.join(artifact_path, "model")
            arc_model_name = "model"

        # Pack the artifacts for this specific job
        tarball_dir = os.path.abspath(f"tarballs/{self.version}")
        print("TARDIR",tarball_dir)
        model_tar_target = law.LocalFileTarget(
            os.path.join(
                tarball_dir,
                self.__class__.__name__,
                f"model_pack_branch_{job_num-1}.tar.gz",
            )
        )
        # Pack the artifacts for this specific job if the target doesn't exist
        if not model_tar_target.exists():
            target_path = model_tar_target.path
            target_dir = os.path.dirname(target_path)

            # Ensure directory exists, but do NOT touch the target file
            os.makedirs(target_dir, exist_ok=True)

            # Create temp file in same directory (required for atomic rename)
            fd, tmp_path = tempfile.mkstemp(
                prefix=".tmp_model_tar_",
                suffix=".tar.gz",
                dir=target_dir,
            )
            os.close(fd)  # tarfile will reopen it

            try:
                with tarfile.open(tmp_path, "w:gz") as tar:
                    # 1. Add the model
                    if os.path.exists(model_src):
                        tar.add(model_src, arcname=f"artifacts/{arc_model_name}")
                    else:
                        raise FileNotFoundError(f"Model source not found: {model_src}")

                    # 2. Add scaler if requested
                    if p_cfg.get("scaler"):
                        scaler_path = os.path.join(
                            artifact_path, f"scaler_{input_files}.pkl"
                        )
                        if not os.path.exists(scaler_path):
                            raise FileNotFoundError(f"Scaler not found: {scaler_path}")
                        tar.add(
                            scaler_path,
                            arcname=f"artifacts/scaler_{input_files}.pkl",
                        )

                    # 3. Add predict config
                    tar.add(self.predict_cfg, arcname="predict_cfg.yaml")

                # Atomic publish (only happens if everything above succeeded)
                os.replace(tmp_path, target_path)

            except Exception as e:
                # Cleanup temp file on failure
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)
                raise e

        config.input_files["Predict_tar"] = law.JobInputFile(model_tar_target.path, render=False, copy=False)
        return config


    def run(self):
        # Remote Execution on the worker node
        working_dir = "Evaluation"
        # exp_name = self.branch_data["experiment_name"]
        # run_name = self.branch_data["run_name"]

        # Unpack prediction package with config + model + scaler
        self.run_command(
            f"tar -xzf ${{_CONDOR_SCRATCH_DIR}}/model_pack_branch_*.tar.gz",
        )

        position_from_law_dir = "../"
        full_cfg = position_from_law_dir + "/predict_cfg.yaml"
        # Copy in training files
        with initialize(version_base=None, config_path=os.path.dirname(full_cfg)):
            cfg_data = compose(config_name=os.path.basename(full_cfg))

        # Use the first dataset_name for element_spec
        dataset_names = cfg_data["test_datasets"]
        for key, values in dataset_names.items():
            print(f"Copying {key} tfrecord files to {working_dir}/data/test/{key}")
            mass_copy(
                sources=values,
                destination= os.path.abspath(f"{working_dir}/data/test/{key}"),
                max_workers=2,
                verbose=True
            )

        command = "python predict_remote.py"
        self.run_command(command, run_location=working_dir)
        self.run_command(
            f"tar -czf ${{LAW_JOB_INIT_DIR}}/prediction_{self.branch}To{int(self.branch) + 1}.tar.gz predictions",
        )

        # # Run the prediction script
        # self.run_command_readable(cmd)

        # # Move output to the expected law target location
        # output_target = self.output()
        # output_target.parent.touch()
        # self.run_command(f"mv predictions.tar.gz {output_target.path}")
