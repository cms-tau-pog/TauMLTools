#!/usr/bin/env python
# Submit jobs on CRAB.
# This file is part of https://github.com/cms-tau-pog/TauTriggerTools.

import argparse
import sys
import re
from sets import Set

parser = argparse.ArgumentParser(description='Submit jobs on CRAB.',
                  formatter_class = lambda prog: argparse.HelpFormatter(prog,width=90))
parser.add_argument('--workArea', required=True, type=str, help="Working area")
parser.add_argument('--cfg', required=True, type=str, help="CMSSW configuration file")
parser.add_argument('--scriptExe', required=False, default="", help="Custom script to execute")
parser.add_argument('--site', required=True, type=str, help="Site for stage out.")
parser.add_argument('--dryrun', action="store_true", help="Submission dryrun.")
parser.add_argument('--useAbsPath', action="store_true", required=False,
                    help="use absolute path for the output folder e.g: /store/group/phys_tau/")
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
parser.add_argument('--jobFile', required=True, type=str, help="text file with jobs descriptions")
args = parser.parse_args()

from CRABClient.UserUtilities import config, ClientException
from CRABClient.UserUtilities import getUsernameFromCRIC as getUsername
from CRABAPI.RawCommand import crabCommand
from httplib import HTTPException

config = config()

config.General.workArea = args.workArea

config.JobType.pluginName = 'Analysis'
config.JobType.psetName = args.cfg
config.JobType.maxMemoryMB = args.maxMemory
config.JobType.numCores = args.numCores

config.Data.inputDBS = args.inputDBS
config.Data.allowNonValidInputDataset = args.allowNonValid
config.General.transferOutputs = True
config.General.transferLogs = False
config.Data.publication = False

config.Site.storageSite = args.site

if args.useAbsPath:
    config.Data.outLFNDirBase = args.output
else:
    config.Data.outLFNDirBase = "/store/user/{}/{}".format(getUsername(), args.output)

if len(args.blacklist) != 0:
	config.Site.blacklist = re.split(',', args.blacklist)
if len(args.whitelist) != 0:
	config.Site.whitelist = re.split(',', args.whitelist)

job_names = Set(filter(lambda s: len(s) != 0, re.split(",", args.jobNames)))

from TauMLTools.Production.crab_tools import JobCollection
try:
    job_collection = JobCollection(args.jobFile, job_names, args.lumiMask, args.jobNameSuffix)
    print args.jobFile
    print job_collection
    print "Splitting: {} with {} units per job".format(args.splitting, args.unitsPerJob)
    job_collection.submit(config, args.splitting, args.unitsPerJob, args.dryrun)
except RuntimeError as err:
    print >> sys.stderr, "ERROR:", str(err)
    sys.exit(1)
