#!/usr/bin/env bash

action() {
  if [[ "{{environment}}" == *"conda"* ]] && [[ ! "{{comp_facility}}" == *"TOpAS"* ]]; then
    echo "Will use conda inside {{conda_path}}"

    __conda_setup="$('{{conda_path}}/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
    if [ $? -eq 0 ]; then
      eval "$__conda_setup"
    else
      if [ -f "{{conda_path}}/etc/profile.d/conda.sh" ]; then
        . "{{conda_path}}/etc/profile.d/conda.sh"
      else
        export PATH="{{conda_path}}/bin:$PATH"
      fi
    fi
    unset __conda_setup
  fi

  if [[ "{{comp_facility}}" == *"TOpAS"* ]]; then
    export HOME=${_CONDOR_JOB_IWD}
    export ANALYSIS_PATH="${_CONDOR_JOB_IWD}/tmp/TauMLTools"
    mkdir tmp/TauMLTools
    tar -xzf TauMLTools*.tar.gz -C tmp/TauMLTools
    cd tmp/TauMLTools
    source env.sh -m docker
  else
    echo "try to source local env.sh"
    source "{{analysis_path}}/env.sh" "-m {{environment}}"
  fi
}
action
