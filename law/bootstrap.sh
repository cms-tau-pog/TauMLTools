#!/usr/bin/env bash

# Bootstrap file for batch jobs that is sent with all jobs and
# automatically called by the law remote job wrapper script to find the
# setup.sh file of this example which sets up software and some environment
# variables. The "{{analysis_path}}" variable is defined in the workflow
# base tasks in analysis/framework.py.

action() {
  luigid --port 8082 --background --logdir ./luigid_logdir

  which eosfusebind
  if [ $? -eq 0 ]; then
    eosfusebind -g
  fi

  export PYTHONPATH={{pythonpath}}:$PYTHONPATH
  export PATH={{path}}:$PATH

  if [ "{{environment}}" == "CMSSW" ]; then
    pushd "{{cmssw_base}}/src"
    eval `scramv1 runtime -sh`
    popd
    source "{{analysis_path}}/setup.sh"
    source "$( law completion )" ""
  elif [ "{{environment}}" == "conda" ]; then
    echo "Will use conda inside {{conda_path}}"
    echo "Will use conda environment {{conda_env}}"

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
    conda activate {{conda_env}}
    if [ $? -ne 0 ]; then
      echo "Unrecognized conda environment: {{conda_env}}"
    else
      echo "Conda setup ended well"
    fi
    
    source "{{analysis_path}}/setup.sh"
    source "$( law completion )" ""
  else
    echo "Unrecognized shell_env: {{environment}}"
  fi
}
action
