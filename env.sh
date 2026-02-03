#!/bin/bash

parse_arguments() {
  # Default values
  DEFAULT_SETUP_MODE="conda"
  DEFAULT_ENV_PATH=""
  SETUP_MODE=${DEFAULT_SETUP_MODE}
  ENV_PATH=${DEFAULT_ENV_PATH}

  # Parse arguments
  while [[ $# -gt 0 ]]; do
    case $1 in
      -m|--mode)
        # Validate argument existence and structure
        if [[ -z $2 || $2 == -* ]]; then
          echo "Error: $1 requires a value (got '$2')"
          return 1
        fi
        SETUP_MODE="$2"
        SETUP_WAS_SET="True"
        shift 2
        ;;
      -e|--env-path)
        # Validate argument existence and structure
        if [[ -z $2 || $2 == -* ]]; then
          echo "Error: $1 requires a value (got '$2')"
          return 1
        fi
        ENV_PATH="$2"
        shift 2
        ;;
      -h|--help)
        echo "Usage: source env.sh [options]"
        echo ""
        echo "Options:"
        echo "  -m, --mode SETUP_MODE    Specify the setup mode to use use"
        echo "                            [default: ${DEFAULT_SETUP_MODE}]"
        echo "  -e, --env-path PATH       Specify custom environment path"
        echo "                            [default: auto-detected]"
        echo "  -l, --list                List available workflows"
        echo "  -h, --help                Show this help message"
        echo ""
        echo "Environment path precedence:"
        echo "1. Command line argument (-e/--env-path)"
        echo "2. Saved location from environment.location file"
        echo "3. Current directory"
        return 1
        ;;
      -l|--list)
          echo "Available setup modes:"
          echo "-------------------"
          echo "cmssw - Uses a CMSSW installation"
          echo "conda - Uses a local miniforge"
          echo "docker - Uses a pre-installed env in a docker container"
          return 1
          ;;
      *)
        echo "Error: Unknown option $1"
        echo "Use --help to see available options"
        return 1
        ;;
    esac
  done

  # Export for use in main script
  export PARSED_SETUP_MODE="${SETUP_MODE}"
  export PARSED_ENV_PATH="${ENV_PATH}"
  return 0
}

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

  parse_arguments "$@"
  # return 1
  if [[ $? -ne 0 ]]; then
    return 1
  fi

  if [[ -z ${SETUP_WAS_SET} ]]; then
    echo "No setup mode selected, defaulting to ${DEFAULT_SETUP_MODE}"
  fi

  local this_file="$( [ ! -z "$ZSH_VERSION" ] && echo "${(%):-%x}" || echo "${BASH_SOURCE[0]}" )"
  local this_dir="$( cd "$( dirname "$this_file" )" && pwd )"
  local MODE=${SETUP_MODE}

  export ANALYSIS_PATH="$this_dir"
  export ANALYSIS_DATA_PATH="$ANALYSIS_PATH/data"
  if [[ -z ${_CONDOR_SCRATCH_DIR} ]]; then
    export X509_USER_PROXY="$ANALYSIS_DATA_PATH/voms.proxy"
  else
    export X509_USER_PROXY="$_CONDOR_SCRATCH_DIR/voms.proxy"
  fi

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

  local default_cmssw_ver=CMSSW_15_0_2
  local target_os_version=9
  local target_os_prefix="el"
  local target_os=$target_os_prefix$target_os_version
  export DEFAULT_CMSSW_BASE="$ANALYSIS_PATH/soft/$default_cmssw_ver"

  if [[ $MODE = *"cmssw"* ]]; then
    run_cmd install_cmssw el9_amd64_gcc12 $default_cmssw_ver $node_os $target_os

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

    ENV_NAME="tau-ml-TV"

    if [[ ! -z ${PARSED_ENV_PATH} ]]; then
      ENV_PATH="$(realpath ${PARSED_ENV_PATH})"
    elif [[ -f "${ANALYSIS_PATH}/environment.location" ]]; then
      ENV_PATH="$(tail -n 1 ${ANALYSIS_PATH}/environment.location)"
    else
      ENV_PATH="${ANALYSIS_PATH}/soft"
    fi
    echo "Using environments from ${ENV_PATH}/conda."
    # Save env location to file if provided
    if [[ ! -z ${PARSED_ENV_PATH} ]]; then
      echo saving environment path to file for future setups.
      echo "### This file contains the environment location that was provided when the setup was last run ###" > ${ANALYSIS_PATH}/environment.location
      echo "${ENV_PATH}" >> ${ANALYSIS_PATH}/environment.location
    fi
    if [ ! -f "${ENV_PATH}/conda/bin/activate" ]; then
      # Miniforge version used for all environments
      MAMBAFORGE_VERSION="24.3.0-0"
      MAMBAFORGE_INSTALLER="Mambaforge-${MAMBAFORGE_VERSION}-$(uname)-$(uname -m).sh"
      echo "Miniforge could not be found, installing miniforge version ${MAMBAFORGE_INSTALLER}"
      echo "More information can be found in"
      echo "https://github.com/conda-forge/miniforge"
      curl -L -O https://github.com/conda-forge/miniforge/releases/download/${MAMBAFORGE_VERSION}/${MAMBAFORGE_INSTALLER}
      bash ${MAMBAFORGE_INSTALLER} -b -s -p ${ENV_PATH}/conda
      rm -f ${MAMBAFORGE_INSTALLER}
    fi

    # Source base env of conda
    source ${ENV_PATH}/conda/bin/activate ''

    # Check if correct conda env is running
    if [ -d "${ENV_PATH}/conda/envs/${ENV_NAME}" ]; then
      echo  "${ENV_NAME} env found using conda."
    else
      # Create conda env from yaml file if necessary
      echo "Creating ${ENV_NAME} env from ${ENV_NAME}-env.yaml..."
      if [[ ! -f "${ANALYSIS_PATH}/${ENV_NAME}-env.yaml" ]]; then
        echo "${ANALYSIS_PATH}/${ENV_NAME}-env.yml not found. Unable to create environment."
        return 1
      fi
      conda env create -f ${ANALYSIS_PATH}/${ENV_NAME}-env.yml -n ${ENV_NAME}
      echo  "${ENV_NAME} env built using conda."
    fi
    echo "Activating env ${ENV_NAME} from conda."
    conda activate ${ENV_NAME}

    local TAU_ML_LIB_DIR=$(cd $(dirname $(which python))/..; pwd)
    export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$TAU_ML_LIB_DIR/lib
  elif [[ $MODE == *"docker"* ]]; then
    source /miniforge/bin/activate tau-ml
  else
    source /cvmfs/sft.cern.ch/lcg/views/setupViews.sh LCG_107 x86_64-el9-gcc14-opt
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
