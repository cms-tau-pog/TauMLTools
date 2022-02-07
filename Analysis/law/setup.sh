#!/usr/bin/env bash

action() {
    local this_dir="$(dirname $(realpath $0))"

    export PYTHONPATH="$this_dir:$PYTHONPATH"
    export LAW_HOME="$this_dir/.law"
    export LAW_CONFIG_FILE="$this_dir/law.cfg"

    export ANALYSIS_PATH="$this_dir"
    export ANALYSIS_DATA_PATH="$ANALYSIS_PATH/data"
    python -m law > /dev/null ; RET=$?
    if [ $RET -ne 0 ]; then
        echo "\n[ERROR] law python libraires not found, \
please append the path to the law python libraires to the PYTHONPATH env. variable" 1>&2
        return $RET
    fi

    source "$( law completion )" ""
}
action
