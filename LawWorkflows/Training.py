## see https://github.com/riga/law/tree/master/examples/htcondor_at_cern

import law
# import subprocess
import os
import re
import tarfile
import shutil
# import math
# import select
import yaml
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
    Merges runs into existing experiments if names match, and fixes all metadata.
    """

    input_cmds = luigi.Parameter(description='Path to the txt file with input commands.')
    mlruns_dir = luigi.Parameter(description='Path to the local central mlruns directory.')

    def requires(self):
        return Training.req(self)

    def output(self):
        training_task = self.requires()
        branch_map = training_task.branch_map
        local_uri = f"file:{os.path.abspath(self.mlruns_dir)}"
        client = mlflow.tracking.MlflowClient(tracking_uri=local_uri)

        outputs = {}
        for branch, data in branch_map.items():
            cmd = data["command"]
            exp_match = re.search(r"experiment_name=(\S+)", cmd)
            run_match = re.search(r"run_name=(\S+)", cmd)

            if not (exp_match and run_match):
                raise ValueError(f"{cmd} does not contain experiment and run names.")

            exp_name, run_name = exp_match.group(1), run_match.group(1)
            found_target = None
            try:
                experiment = client.get_experiment_by_name(exp_name)
                if experiment:
                    runs = client.search_runs(
                        experiment_ids=[experiment.experiment_id],
                        filter_string=f"tags.mlflow.runName = '{run_name}'"
                    )
                    if runs:
                        run_id = runs[0].info.run_id
                        found_target = law.LocalFileTarget(os.path.join(self.mlruns_dir, experiment.experiment_id, run_id))
            except Exception:
                pass
            outputs[branch] = found_target or law.LocalFileTarget(f"{self.mlruns_dir}/needed_{exp_name}_{run_name}")

        return law.TargetCollection(outputs)

    def run(self):
        branch_targets = self.input().collection.targets.values()
        local_uri = f"file:{self.mlruns_dir}"
        client = mlflow.tracking.MlflowClient(tracking_uri=local_uri)

        if not os.path.exists(self.mlruns_dir):
            os.makedirs(self.mlruns_dir, exist_ok=True)

        self.publish_message(f"Starting merge into {self.mlruns_dir}")

        patched_runs = [] # List of (local_exp_id, run_id)

        for target in branch_targets:
            with tarfile.open(target.path, "r:gz") as tar:
                # 1. Map remote experiment IDs to local IDs by checking names
                id_map = {} # {remote_id: local_id}

                # Pre-scan tar for experiment meta files to establish mapping
                for member in tar.getmembers():
                    if member.name.endswith("meta.yaml") and member.name.count('/') == 2:
                        # This is an experiment meta file: mlruns/<id>/meta.yaml
                        f = tar.extractfile(member)
                        meta = yaml.safe_load(f)
                        remote_name = meta.get("name")
                        print("HERE", remote_name)
                        remote_id = member.name.split('/')[1]

                        if remote_id in ["0", ".trash"]: continue

                        # Check if this experiment exists locally
                        local_exp = client.get_experiment_by_name(remote_name)
                        if local_exp:
                            id_map[remote_id] = local_exp.experiment_id
                        else:
                            # If it doesn't exist, it will be extracted as-is
                            id_map[remote_id] = remote_id

                # 2. Extract and Redirect
                for member in tar.getmembers():
                    parts = member.name.strip("/").split("/")
                    if len(parts) < 2 or parts[0] != "mlruns" or parts[1] in ["0", ".trash"]:
                        continue

                    remote_exp_id = parts[1]
                    local_exp_id = id_map.get(remote_exp_id, remote_exp_id)

                    # Construct new local path
                    rel_parts = [local_exp_id] + parts[2:]
                    dest_path = os.path.join(self.mlruns_dir, *rel_parts)

                    if member.isdir():
                        os.makedirs(dest_path, exist_ok=True)
                    else:
                        os.makedirs(os.path.dirname(dest_path), exist_ok=True)
                        with tar.extractfile(member) as source, open(dest_path, "wb") as target_file:
                            shutil.copyfileobj(source, target_file)

                        # Track if this was a run directory for patching
                        if len(parts) >= 3:
                            patched_runs.append((local_exp_id, parts[2]))

        # 3. Patch Metadata (Experiment and Run levels)
        self.publish_message("Patching metadata for merged consistency...")
        unique_runs = list(set(patched_runs))
        unique_exps = list(set([r[0] for r in unique_runs]))

        for exp_id in unique_exps:
            meta_path = os.path.join(self.mlruns_dir, exp_id, "meta.yaml")
            if os.path.exists(meta_path):
                with open(meta_path, 'r') as f: meta = yaml.safe_load(f)
                meta["artifact_location"] = f"file:{self.mlruns_dir}/{exp_id}"
                meta["experiment_id"] = exp_id
                with open(meta_path, 'w') as f: yaml.safe_dump(meta, f)

        for exp_id, run_id in unique_runs:
            meta_path = os.path.join(self.mlruns_dir, exp_id, run_id, "meta.yaml")
            if os.path.exists(meta_path):
                with open(meta_path, 'r') as f: meta = yaml.safe_load(f)
                meta["artifact_uri"] = f"file:{self.mlruns_dir}/{exp_id}/{run_id}/artifacts"
                meta["experiment_id"] = exp_id
                with open(meta_path, 'w') as f: yaml.safe_dump(meta, f)

        self.publish_message("Merge complete.")
