#!/usr/bin/env python
# Submit jobs on CRAB.
# This file is part of https://github.com/cms-tau-pog/TauTriggerTools.
from __future__ import print_function
import argparse
import subprocess
import sys

parser = argparse.ArgumentParser(description='Submit jobs on CRAB.',
                  formatter_class = lambda prog: argparse.HelpFormatter(prog,width=90))
parser.add_argument('--workArea', required=True, type=str, help="Working area")
parser.add_argument('--cfg', required=True, type=str, help="CMSSW configuration file")
parser.add_argument('--scriptExe', required=False, default="", help="Custom script to execute")
parser.add_argument('--site', required=True, type=str, help="Site for stage out.")
parser.add_argument('--dryrun', action="store_true", required=False, help="Submission dryrun.")
parser.add_argument('--output', required=True, type=str, help="output path after /store/user/USERNAME")
parser.add_argument('--blacklist', required=False, type=str, default="",
					help="list of sites where the jobs shouldn't run")
parser.add_argument('--whitelist', required=False, type=str, default="",
					help="list of sites where the jobs can run")
parser.add_argument('--jobNames', required=False, type=str, default="",
					help="list of job names to submit (if not specified - submit all)")
parser.add_argument('--lumiMask', required=False, type=str, default="",
					help="json file with a lumi mask (default: apply lumi mask from the config file)")
parser.add_argument('--jobNameSuffix', required=False, type=str, default="",
					help="suffix that will be added to each job name")
parser.add_argument('--inputDBS', required=False, default="global", help="DBS instance")
parser.add_argument('--splitting', required=False, default="Automatic",
					help="suffix that will be added to each job name")
parser.add_argument('--unitsPerJob', required=False, type=int, default=1000, help="number of units per job")
parser.add_argument('--maxMemory', required=False, type=int, default=2000,
					help="maximum amount of memory (in MB) a job is allowed to use (default: 2000 MB )")
parser.add_argument('--numCores', required=False, type=int, default=1, help="number of cores per job (default: 1)")
parser.add_argument('--allowNonValid', action="store_true", help="Allow nonvalid dataset as an input.")
parser.add_argument('--vomsGroup',required=False, type=str, default="", help="custom VOMS group of used proxy")
parser.add_argument('--vomsRole', required=False, type=str, default="", help="custom VOMS role of used proxy")
parser.add_argument('job_file', type=str, nargs='+', help="text file with jobs descriptions")
args = parser.parse_args()

interpreter='python' if sys.version_info.major==2 else 'python3'
for job_file in args.job_file:
    cmd = '{} $(which crab_submit_file.py) --jobFile "{}"'.format(interpreter, job_file)
    for arg_name,arg_value in getattr(vars(args), 'iteritems', vars(args).items)():
        if arg_name != 'job_file' and type(arg_value) != bool and (type(arg_value) != str or len(arg_value)):
            cmd += ' --{} {} '.format(arg_name, arg_value)
        elif type(arg_value) == bool and arg_value:
            cmd += ' --{} '.format(arg_name)
    print ('> {}'.format(cmd))
    result = subprocess.call([cmd], shell=True)
    if result != 0:
        print('ERROR: failed to submit jobs from "{}"'.format(job_file))
        sys.exit(1)
