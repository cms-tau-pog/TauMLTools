#!/usr/bin/env python
# Move crab outputs into the permanent location.

import argparse
import os
import shutil
import sys

parser = argparse.ArgumentParser(description='Move crab outputs into the permanent location.',
                  formatter_class = lambda prog: argparse.HelpFormatter(prog,width=90))
parser.add_argument('--input', required=True, type=str, help="crab outputs")
parser.add_argument('--output', required=True, type=str, help="new location")
parser.add_argument('--dry-run', action="store_true", required=False, help="Do not perform actual moving.")
args = parser.parse_args()

dry_run_msg = ''
if args.dry_run:
    dry_run_msg = ' (DRY RUN) '
print('Moving crab outputs from "{}" to "{}"{}'.format(args.input, args.output, dry_run_msg))

error_flag = False
def report_error(message):
    global error_flag
    print("ERROR: {}".format(message))
    error_flag = True

class Dataset:
    job_name_prefix = 'crab_'
    def __init__(self, input_path):
        self.base_path, self.job_name = os.path.split(input_path)
        if not self.job_name.startswith(Dataset.job_name_prefix):
            raise RuntimeError('Unexpected job name "{}" in "{}"'.format(self.job_name, self.base_path))
        self.name = self.job_name[len(Dataset.job_name_prefix):]
        job_id_dirs = os.listdir(input_path)
        if len(job_id_dirs) == 0:
            raise RuntimeError('No crab jobs were found for "{}"'.format(input_path))
        if len(job_id_dirs) > 1:
            raise RuntimeError('Multiple crab jobs were found for "{}"'.format(input_path))
        job_id_dir = job_id_dirs[0]
        job_id_dir_path = os.path.join(input_path, job_id_dir)
        self.root_files = {}
        for chunk_dir in os.listdir(job_id_dir_path):
            chunk_dir_path = os.path.join(job_id_dir_path, chunk_dir)
            if not os.path.isdir(chunk_dir_path):
                raise RuntimeError('Unexpected file "{}" in "{}"'.format(chunk_dir, job_id_dir_path))
            for root_file in os.listdir(chunk_dir_path):
                root_file_path = os.path.join(chunk_dir_path, root_file)
                if not os.path.isfile(root_file_path) or not root_file.endswith('.root'):
                    raise RuntimeError('Unexpected file "{}" in "{}"'.format(root_file, chunk_dir_path))
                if root_file in self.root_files:
                    raise RuntimeError('Duplicated file "{}" in "{}"'.format(root_file, input_path))
                self.root_files[root_file] = root_file_path

datasets = {}
for process_dir in sorted(os.listdir(args.input)):
    process_dir_path = os.path.join(args.input, process_dir)
    if not os.path.isdir(process_dir_path):
        report_error('Unexpected file "{}" in "{}"'.format(process_dir, args.input))
        continue
    for job_dir in sorted(os.listdir(process_dir_path)):
        job_dir_path = os.path.join(process_dir_path, job_dir)
        if not os.path.isdir(job_dir_path):
            report_error('Unexpected file "{}" in "{}"'.format(job_dir, process_dir_path))
            continue
        try:
            ds = Dataset(job_dir_path)
            if ds.name in datasets:
                report_error('Duplicated dataset name = "".'.format(ds.name))
            datasets[ds.name] = ds
        except RuntimeError as e:
            report_error(e.message)

if error_flag:
    print("Stopping due to errors.")
    sys.exit(1)

try:
    for ds_name, ds in datasets.items():
        print('Moving {}...'.format(ds_name)),
        ds_output = os.path.join(args.output, ds_name)
        if os.path.exists(ds_output):
            raise RuntimeError('Output path "{}" already exists'.format(ds_output))
        if not args.dry_run:
            os.makedirs(ds_output)
            for root_file, root_file_path in ds.root_files.items():
                root_file_output_path = os.path.join(ds_output, root_file)
                shutil.move(root_file_path, root_file_output_path)
        print('done')
except RuntimeError as e:
    report_error(e.message)
