## see https://github.com/riga/law/tree/master/examples/htcondor_at_cern

import law
# import subprocess
import os
import re
import tarfile
# import math
# import select
from hydra import compose, initialize
from .framework import HTCondorTOpASWorkflow, HTCondorTOpASWorkflowParameters
import luigi
import mlflow
from LawWorkflows.mass_copy import mass_copy
# from law.util import interruptable_popen
law.contrib.load("wlcg")

class Training(HTCondorTOpASWorkflow):

    input_cmds = luigi.Parameter(description = 'Path to the txt file with input commands.')

    def htcondor_job_config(self, config, job_num, branches):
        config = super().htcondor_job_config(config, job_num, branches)
        config.custom_content.append(("stream_error", "True"))
        config.custom_content.append(("stream_output", "True"))
        config.output_files.append("mlruns.tar.gz")
        return config

    def create_branch_map(self):
        # Opening file
        print(f"Reading commands from file: {self.input_cmds}")
        self.cmds_list = {}

        required_keys = {
            "input_files": r"input_files=(\S+)(\s|$)",
            "experiment_name": r"experiment_name=(\S+)(\s|$)",
            "run_name": r"run_name=(\S+)(\s|$)",
        }

        with open(self.input_cmds, 'r') as file1:
            for i, line in enumerate(file1):
                for key, pattern in required_keys.items():
                    if not re.search(pattern, line):
                        raise ValueError(
                            f"Missing required '{key}' in input_cmds line {i}:\n{line}"
                        )
                self.cmds_list[i] = {"command": line.strip()}
        return self.cmds_list

    def output(self):
        # If run on ETP, check if the result .tar is present
        return self.local_target("files/mlruns_{}To{}.tar.gz".format(self.branch, int(self.branch) + 1))

    def run(self):
        working_dir = "Training/python/tf/"
        cfg = "configs/train.yaml"
        if not os.path.exists(os.path.abspath(working_dir)):
            raise Exception('Working folder {} does not exist'.format(working_dir))

        command = self.branch_data["command"]

        match_inp = re.search(r'input_files=(\S+)(\s|$)', command)
        if not match_inp:
            raise ValueError("Input files not provided! Necessary in either the ml cfg file or as LAW parameter.")
        input_files_cfg = match_inp.group(1)

        position_from_law_dir = "../"
        full_cfg = position_from_law_dir + working_dir + cfg
        # Copy in training files
        with initialize(version_base=None, config_path=os.path.dirname(full_cfg)):
            cfg_data = compose(config_name=os.path.basename(full_cfg), overrides=[f"input_files={input_files_cfg}"])

        paths_cfg = cfg_data["input_files"]["cfg"]
        paths_train = cfg_data["input_files"]["train"]
        paths_val = cfg_data["input_files"]["val"]

        mass_copy(paths_cfg, os.path.abspath(f"{working_dir}/data/"))
        mass_copy(paths_train, os.path.abspath(f"{working_dir}/data/train"), max_workers=64)
        mass_copy(paths_val, os.path.abspath(f"{working_dir}/data/val"), max_workers=64)

        self.run_command_readable(command, run_location=working_dir)
        # self.run_command(command, run_location=working_dir)
        self.run_command(
            "tar -czf ${{LAW_JOB_INIT_DIR}}/mlruns_{}To{}.tar.gz mlruns".format(
                self.branch,
                int(self.branch) + 1
            ),
            run_location=working_dir
        )

class MLflowLogTraining(HTCondorTOpASWorkflowParameters):
    """
    Local Task to merge mlruns tarballs from the remote Training branches.
    Uses MLflow bindings to verify completion by checking for matching
    experiment and run names in the local storage.
    """

    input_cmds = luigi.Parameter(description='Path to the txt file with input commands.')
    mlruns_dir = luigi.Parameter(description='Path to the local central mlruns directory.')

    def requires(self):
        # Trigger/Require the Training workflow
        return Training.req(self)

    def output(self):
        """
        Uses MLflow API to check if the specific experiment and run name
        already exist in the local mlruns directory.
        """
        training_task = self.requires()
        branch_map = training_task.branch_map

        # Set the MLflow tracking URI to the local directory
        local_uri = f"file:{os.path.abspath(self.mlruns_dir)}"
        client = mlflow.tracking.MlflowClient(tracking_uri=local_uri)

        outputs = {}
        for branch, data in branch_map.items():
            cmd = data["command"]

            # Extract names from the command line strings
            exp_match = re.search(r"experiment_name=(\S+)", cmd)
            run_match = re.search(r"run_name=(\S+)", cmd)

            if not (exp_match and run_match):
                raise ValueError(f"{cmd} does not contain experiment and run names.")

            exp_name = exp_match.group(1)
            run_name = run_match.group(1)

            # Use MLflow bindings to search for existing run
            found_target = None
            try:
                experiment = client.get_experiment_by_name(exp_name)
                if experiment:
                    # Filter runs by the specific run_name tag/attribute
                    runs = client.search_runs(
                        experiment_ids=[experiment.experiment_id],
                        filter_string=f"tags.mlflow.runName = '{run_name}'"
                    )
                    if runs:
                        # If found, the output target is the actual directory on disk
                        run_id = runs[0].info.run_id
                        found_target = law.LocalFileTarget(os.path.join(self.mlruns_dir, experiment.experiment_id, run_id))
            except Exception:
                pass # Fallback to incomplete if API calls fail

            # If not found via API, provide a path that won't exist yet to force 'run()'
            outputs[branch] = found_target or law.LocalFileTarget(f"{self.mlruns_dir}/needed_{exp_name}_{run_name}")
        print("HERE", outputs)
        return law.TargetCollection(outputs)

    def run(self):
        # 1. Access the collection of tarballs from Training
        branch_targets = self.input().collection.targets.values()

        # 2. Define the central local directory
        extract_parent = os.path.dirname(self.mlruns_dir)

        if not os.path.exists(self.mlruns_dir):
            os.makedirs(self.mlruns_dir, exist_ok=True)

        self.publish_message(f"Merging {len(branch_targets)} tarballs into {self.mlruns_dir}")

        # Keep track of newly extracted IDs to only patch what we just updated
        new_experiment_ids = set()
        new_run_ids = set()

        # 3. Extract all tarballs
        success_count = 0
        for target in branch_targets:
            tar_path = target.path
            if not target.exists() or target.stat().st_size < 500:
                continue

            try:
                with tarfile.open(tar_path, "r:gz") as tar:
                    if not tar.getmembers():
                        continue

                    # Track IDs from tar content before extraction
                    for member in tar.getmembers():
                        parts = member.name.strip("/").split("/")
                        if len(parts) >= 2 and parts[0] == "mlruns":
                            # Track valid Experiment and Run IDs
                            new_experiment_ids.add(parts[1])
                            if len(parts) >= 3:
                                new_run_ids.add((parts[1], parts[2]))

                    tar.extractall(path=extract_parent)
                    success_count += 1
            except Exception as e:
                print(f"Failed to process {tar_path}: {e}")

        # 4. Use MLflow bindings/File system to fix paths in meta files
        self.publish_message("Fixing artifact URIs for new runs and experiments...")

        try:
            # First, fix Experiment meta files for the newly inserted experiments
            for exp_id in new_experiment_ids:
                exp_meta_path = os.path.join(self.mlruns_dir, exp_id, "meta.yaml")
                if os.path.exists(exp_meta_path):
                    with open(exp_meta_path, 'r') as f:
                        content = f.read()

                    # Fix experiment-level artifact_location
                    local_exp_uri = f"file:{self.mlruns_dir}/{exp_id}"
                    new_content = re.sub(
                        r'artifact_location:.*',
                        f'artifact_location: {local_exp_uri}',
                        content
                    )

                    if new_content != content:
                        with open(exp_meta_path, 'w') as f:
                            f.write(new_content)

            # Second, fix Run meta files for the newly inserted runs
            for exp_id, run_id in new_run_ids:
                run_meta_path = os.path.join(self.mlruns_dir, exp_id, run_id, "meta.yaml")
                if os.path.exists(run_meta_path):
                    with open(run_meta_path, 'r') as f:
                        content = f.read()

                    local_run_artifact_uri = f"file:{self.mlruns_dir}/{exp_id}/{run_id}/artifacts"
                    new_content = re.sub(
                        r'artifact_uri:.*',
                        f'artifact_uri: {local_run_artifact_uri}',
                        content
                    )

                    if new_content != content:
                        with open(run_meta_path, 'w') as f:
                            f.write(new_content)

        except Exception as e:
            print(f"Warning: Failed to fix some MLflow metadata paths: {e}")

        self.publish_message(f"Successfully merged and patched {success_count} runs.")
