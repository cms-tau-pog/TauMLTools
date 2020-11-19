#!/usr/bin/env bash

action() {
    local this_file="$( [ ! -z "$ZSH_VERSION" ] && echo "${(%):-%x}" || echo "${BASH_SOURCE[0]}" )"
    local this_dir="$( cd "$( dirname "$this_file" )" && pwd )"

    export PYTHONPATH="$this_dir:$PYTHONPATH"
    export LAW_HOME="$this_dir/.law"
    export LAW_CONFIG_FILE="$this_dir/law.cfg"

    export ANALYSIS_PATH="$this_dir"
    export ANALYSIS_DATA_PATH="$ANALYSIS_PATH/data"
    export SML_INPUT_DATA="/eos/cms/store/group/phys_tau/TauML/prod_2018_v1/full_tuples/WJetsToLNu_HT-1200To2500/"

    source "/afs/cern.ch/user/m/mrieger/public/law_sw/setup.sh" ""
    source "$( law completion )" ""
}
action
