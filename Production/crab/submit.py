#!/usr/bin/env python
# Submit jobs on CRAB.
# This file is part of https://github.com/hh-italian-group/h-tautau.

import argparse
import sys
import re
from sets import Set

parser = argparse.ArgumentParser(description='Submit jobs on CRAB.',
                  formatter_class = lambda prog: argparse.HelpFormatter(prog,width=90))
parser.add_argument('--work-area', required=False, dest='workArea', type=str, default="work_area",
                    help="Work area")
parser.add_argument('--cfg', required=True, dest='cfg', type=str, help="CMSSW configuration file")
parser.add_argument('--site', required=True, dest='site', type=str, help="Site for stage out.")
parser.add_argument('--dryrun', action="store_true", help="Submission dryrun.")
parser.add_argument('--output', required=True, dest='output', type=str,
                    help="output path after /store/user/USERNAME")
parser.add_argument('--blacklist', required=False, dest='blacklist', type=str, default="",
					help="list of sites where the jobs shouldn't run")
parser.add_argument('--whitelist', required=False, dest='whitelist', type=str, default="",
					help="list of sites where the jobs can run")
parser.add_argument('--jobNames', required=False, dest='jobNames', type=str, default="",
					help="list of job names to submit (if not specified - submit all)")
parser.add_argument('--lumiMask', required=False, dest='lumiMask', type=str, default="",
					help="json file with a lumi mask (default: apply lumi mask from the config file)")
parser.add_argument('--jobNameSuffix', required=False, dest='jobNameSuffix', type=str, default="",
					help="suffix that will be added to each job name")
parser.add_argument('--unitsPerJob', required=False, dest='unitsPerJob', type=int, default=-1,
					help="number of units per job (default: use values from the config file)")
parser.add_argument('--maxMemory', required=False, dest='maxMemory', type=int, default=2000,
					help="maximum amount of memory (in MB) a job is allowed to use (default: 2000 MB )")
parser.add_argument('job_file', type=str, nargs='+', help="text file with jobs descriptions")
args = parser.parse_args()

from CRABClient.UserUtilities import config, ClientException, getUsernameFromSiteDB
from CRABAPI.RawCommand import crabCommand
from httplib import HTTPException

config = config()

config.General.workArea = args.workArea

config.JobType.pluginName = 'Analysis'
config.JobType.psetName = args.cfg
config.JobType.maxMemoryMB = args.maxMemory

config.Data.inputDBS = 'global'
config.General.transferOutputs = True
config.General.transferLogs = True
config.Data.publication = False

config.Site.storageSite = args.site
config.Data.outLFNDirBase = "/store/user/{}/{}".format(getUsernameFromSiteDB(), args.output)

if len(args.blacklist) != 0:
	config.Site.blacklist = re.split(',', args.blacklist)
if len(args.whitelist) != 0:
	config.Site.whitelist = re.split(',', args.whitelist)

job_names = Set(filter(lambda s: len(s) != 0, re.split(",", args.jobNames)))

from crab_tools import JobCollection
try:
    for job_file in args.job_file:
        job_collection = JobCollection(job_file, job_names, args.lumiMask, args.jobNameSuffix, args.unitsPerJob)
        print job_file
        print job_collection
        job_collection.submit(config,args.dryrun)
except RuntimeError as err:
    print >> sys.stderr, "ERROR:", str(err)
    sys.exit(1)
