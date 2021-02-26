#!/usr/bin/env bash

# Bootstrap file for batch jobs that is sent with all jobs and
# automatically called by the law remote job wrapper script to find the
# setup.sh file of this example which sets up software and some environment
# variables. The "{{analysis_path}}" variable is defined in the workflow
# base tasks in analysis/framework.py.

action() {
  pushd "{{cmssw_base}}/src"
  eval `scramv1 runtime -sh`
  popd
  source "{{analysis_path}}/setup.sh"
  source "/afs/cern.ch/user/m/mrieger/public/law_sw/setup.sh" ""
  source "$( law completion )" ""
}
action
