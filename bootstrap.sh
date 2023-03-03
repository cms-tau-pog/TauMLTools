#!/usr/bin/env bash

action() {
  if [[ "{{environment}}" == *"conda"* ]]; then
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
  source "{{analysis_path}}/env.sh" "{{environment}}"
}
action
