#!/bin/bash

# if (( $# < 1 )) ; then
#     cat << EOF
# Setup environment for TauMLTools
# Usage: source env.sh mode [mode_arg_1] [mode_arg_2] ...
# Supported modes: run2 run3 phase2_112X phase2_113X lcg conda hlt
# Mode-specific arguments:
# conda
#   --update [env.yaml]  updates environment from env.yaml (default: tau-ml-env.yaml)
# EOF
#     return 1
# fi

run_cmd() {
  "$@"
  local RESULT=$?
  if (( $RESULT != 0 )); then
    echo "Error while running '$@'"
    kill -INT $$
  fi
}

do_install_cmssw() {
  local this_file="$( [ ! -z "$ZSH_VERSION" ] && echo "${(%):-%x}" || echo "${BASH_SOURCE[0]}" )"
  local this_dir="$( cd "$( dirname "$this_file" )" && pwd )"

  export SCRAM_ARCH=$1
  local CMSSW_VER=$2
  if ! [ -f "$this_dir/soft/$CMSSW_VER/.installed" ]; then
    run_cmd mkdir -p "$this_dir/soft"
    run_cmd cd "$this_dir/soft"
    run_cmd source /cvmfs/cms.cern.ch/cmsset_default.sh
    if [ -d $CMSSW_VER ]; then
      echo "Removing incomplete $CMSSW_VER installation..."
      run_cmd rm -rf $CMSSW_VER
    fi
    echo "Creating $CMSSW_VER area in $PWD ..."
    run_cmd scramv1 project CMSSW $CMSSW_VER
    run_cmd cd $CMSSW_VER/src
    run_cmd eval `scramv1 runtime -sh`
    run_cmd mkdir TauMLTools
    run_cmd cd TauMLTools
    run_cmd ln -s "$this_dir/Analysis" Analysis
    run_cmd ln -s "$this_dir/Core" Core
    run_cmd ln -s "$this_dir/Production" Production
    run_cmd scram b -j8
    run_cmd cd "$this_dir"
    run_cmd touch "$this_dir/soft/$CMSSW_VER/.installed"
  fi
}

install_cmssw() {
  local this_file="$( [ ! -z "$ZSH_VERSION" ] && echo "${(%):-%x}" || echo "${BASH_SOURCE[0]}" )"
  local this_dir="$( cd "$( dirname "$this_file" )" && pwd )"
  local scram_arch=$1
  local cmssw_version=$2
  local node_os=$3
  local target_os=$4
  if [[ $node_os == $target_os ]]; then
    local env_cmd=""
    local env_cmd_args=""
  else
    local env_cmd="cmssw-$target_os"
    if ! command -v $env_cmd &> /dev/null; then
      echo "Unable to do a cross-platform installation for $cmssw_version SCRAM_ARCH=$scram_arch. $env_cmd is not available."
      return 1
    fi
    local env_cmd_args="--command-to-run"
  fi
  if ! [ -f "$this_dir/soft/$CMSSW_VER/.installed" ]; then
    run_cmd $env_cmd $env_cmd_args /usr/bin/env -i HOME=$HOME bash "$this_file" install_cmssw $scram_arch $cmssw_version $target_os_version
  fi
}

action() {
  local this_file="$( [ ! -z "$ZSH_VERSION" ] && echo "${(%):-%x}" || echo "${BASH_SOURCE[0]}" )"
  local this_dir="$( cd "$( dirname "$this_file" )" && pwd )"
  local MODE=$1

  export ANALYSIS_PATH="$this_dir"
  export ANALYSIS_DATA_PATH="$ANALYSIS_PATH/data"
  export X509_USER_PROXY="$ANALYSIS_DATA_PATH/voms.proxy"

  export PYTHONPATH="$this_dir:$PYTHONPATH"
  export LAW_HOME="$this_dir/.law"
  export LAW_CONFIG_FILE="$this_dir/LawWorkflows/law.cfg"

  run_cmd mkdir -p "$ANALYSIS_DATA_PATH"

  local os_version=$(cat /etc/os-release | grep VERSION_ID | sed -E 's/VERSION_ID="([0-9]+).*"/\1/')
  if [[ $os_version < 8 ]] ; then
    local os_prefix="cc"
  else
    local os_prefix="el"
  fi
  local node_os=$os_prefix$os_version

  local default_cmssw_ver=CMSSW_14_0_0
  local target_os_version=8
  local target_os_prefix="el"
  local target_os=$target_os_prefix$target_os_version
  export DEFAULT_CMSSW_BASE="$ANALYSIS_PATH/soft/$default_cmssw_ver"

  if [[ $MODE = *"cmssw"* ]]; then
    run_cmd install_cmssw el8_amd64_gcc11 $default_cmssw_ver $node_os $target_os

    if [[ $node_os == $target_os ]]; then
      export CMSSW_SINGULARITY=""
      local env_cmd=""
    else
      export CMSSW_SINGULARITY="/cvmfs/cms.cern.ch/common/cmssw-$target_os"
      local env_cmd="$CMSSW_SINGULARITY --command-to-run"
    fi

    alias cmsEnv="$env_cmd env -i HOME=$HOME ANALYSIS_PATH=$ANALYSIS_PATH ANALYSIS_DATA_PATH=$ANALYSIS_DATA_PATH X509_USER_PROXY=$X509_USER_PROXY DEFAULT_CMSSW_BASE=$DEFAULT_CMSSW_BASE KRB5CCNAME=$KRB5CCNAME $ANALYSIS_PATH/RunKit/cmsEnv.sh"
  fi

  if [[ $MODE == *"conda"* ]]; then
    local CONDA=$(which conda 2>/dev/null)
    if [[ $CONDA = "" || $CONDA = "/usr/bin/conda" ]]; then
      local PRIVATE_CONDA_INSTALL_DEFAULT="$ANALYSIS_PATH/soft/conda"
      local PRIVATE_CONDA_INSTALL="$PRIVATE_CONDA_INSTALL_DEFAULT"
      if [ -f "$PRIVATE_CONDA_INSTALL_DEFAULT.ref" ]; then
        local PRIVATE_CONDA_INSTALL=$(cat "$PRIVATE_CONDA_INSTALL.ref")
      fi
      if ! [ -f "$PRIVATE_CONDA_INSTALL/.installed" ]; then
        echo "Please select path where conda environment and packages will be installed."
        if [[ $HOST = lxplus*  || $HOSTNAME = lxplus* ]]; then
          echo "On lxplus it is recommended to use /afs/cern.ch/work/${USER:0:1}/$USER/conda or /eos/home-${USER:0:1}/$USER/conda."
        fi
        printf "new or existing conda installation path (default $PRIVATE_CONDA_INSTALL_DEFAULT): "
        read PRIVATE_CONDA_INSTALL
        if [[ "$PRIVATE_CONDA_INSTALL" = "" ]]; then
          PRIVATE_CONDA_INSTALL="$PRIVATE_CONDA_INSTALL_DEFAULT"
        fi
        if [[ "$PRIVATE_CONDA_INSTALL" != "$PRIVATE_CONDA_INSTALL_DEFAULT" ]]; then
          echo $PRIVATE_CONDA_INSTALL > "$PRIVATE_CONDA_INSTALL_DEFAULT.ref"
        else
          rm -f "$PRIVATE_CONDA_INSTALL_DEFAULT.ref"
        fi
        if ! [ -f "$PRIVATE_CONDA_INSTALL/.installed" ]; then
          if [ -d $PRIVATE_CONDA_INSTALL ]; then
            printf "Incomplete private conda installation in $PRIVATE_CONDA_INSTALL. Proceed? [y/N] "
            read X
            if [[ $X = "y" || $X = "Y" || $X = "yes" ]]; then
              run_cmd rm -rf $PRIVATE_CONDA_INSTALL
            else
              echo "Aborting..."
              kill -INT $$
            fi
          fi
          echo "Installing conda..."
          run_cmd mkdir -p soft
          run_cmd cd soft
          run_cmd curl https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -o Miniconda3-latest-Linux-x86_64.sh
          run_cmd bash Miniconda3-latest-Linux-x86_64.sh -b -p "$PRIVATE_CONDA_INSTALL"
          run_cmd touch "$PRIVATE_CONDA_INSTALL/.installed"
          run_cmd rm Miniconda3-latest-Linux-x86_64.sh
          run_cmd cd ..
        fi
      fi
      __conda_setup="$($PRIVATE_CONDA_INSTALL/bin/conda shell.${SHELL##*/} hook)"
      if (( $? == 0 )); then
        eval "$__conda_setup"
      else
        if [ -f "$PRIVATE_CONDA_INSTALL/etc/profile.d/conda.sh" ]; then
          . "$PRIVATE_CONDA_INSTALL/etc/profile.d/conda.sh"
        else
          export PATH="$PRIVATE_CONDA_INSTALL/bin:$PATH"
        fi
      fi
      unset __conda_setup
    fi
    tau_env_found=$(conda env list | grep -E '^tau-ml .*' | wc -l)
    if (( $tau_env_found != 1 )); then
      echo "Creating tau-ml environment..."
      run_cmd conda env create -f $ANALYSIS_PATH/tau-ml-env.yaml
    fi
    run_cmd conda activate tau-ml
    ARG="$2"
    if [[ $ARG = "--update" ]]; then
      ENV_YAML="$3"
      if [[ $ENV_YAML = "" ]]; then
        ENV_YAML="tau-ml-env.yaml"
      fi
      echo "Updating conda environment from '$ENV_YAML'..."
      run_cmd conda env update --file $ENV_YAML --prune
    fi
    local TAU_ML_LIB_DIR=$(cd $(dirname $(which python))/..; pwd)
    export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$TAU_ML_LIB_DIR/lib
  else
    source /cvmfs/sft.cern.ch/lcg/views/setupViews.sh LCG_105 x86_64-el9-gcc13-opt
    for law_location in /afs/cern.ch/user/m/mrieger/public/law_sw/setup.sh /afs/desy.de/user/r/riegerma/public/law_sw/setup.sh; do
      if [ -f $law_location ]; then
        source $law_location
        break
      fi
    done
    current_args=( "$@" )
    set --
    source /cvmfs/cms.cern.ch/rucio/setup-py3.sh &> /dev/null
    set -- "${current_args[@]}"
  fi

  if [ ! -z $ZSH_VERSION ]; then
    autoload bashcompinit
    bashcompinit
  fi
  source "$( law completion )"

  which eosfusebind &> /dev/null
  if [ $? -eq 0 ]; then
    eosfusebind -g
  fi

  alias run_cxx="python $ANALYSIS_PATH/Core/python/run_cxx.py"

  echo "TauMLTools environment is successfully loaded."
}

if [ "X$1" = "Xinstall_cmssw" ]; then
  do_install_cmssw "${@:2}"
else
  action "$@"
fi
