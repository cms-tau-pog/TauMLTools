#!/bin/bash

if (( $# < 1 )) ; then
    cat << EOF
Setup environment for TauMLTools
Usage: source env.sh mode [mode_arg_1] [mode_arg_2] ...
Supported modes: run2 run3 phase2_112X phase2_113X lcg conda hlt
Mode-specific arguments:
conda
  --update [env.yaml]  updates environment from env.yaml (default: tau-ml-env.yaml)
EOF
    return 1
fi

run_cmd() {
  "$@"
  local RESULT=$?
  if (( $RESULT != 0 )); then
    echo "Error while running '$@'"
    kill -INT $$
  fi
}

load_env() {
  local MODE=$1
  local BASE_PATH=$PWD

  export TAU_ML_DIR="$(pwd)"
  if [[ $MODE = "phase2_112X" || $MODE = "phase2_113X" || $MODE = "run2" || $MODE = "run3" || $MODE = "hlt" ]]; then
    local os_version=$(cat /etc/os-release | grep VERSION_ID | sed -E 's/VERSION_ID="([0-9]+)"/\1/')
    if [[ $MODE = "phase2_112X" ]]; then
      local CMSSW_VER=CMSSW_11_2_5
      export SCRAM_ARCH=slc7_amd64_gcc900
    elif [[ $MODE = "phase2_113X" ]]; then
      local CMSSW_VER=CMSSW_11_3_0
      export SCRAM_ARCH=slc7_amd64_gcc900
    elif [[ $MODE = "run2" || $MODE = "run3" || $MODE = "hlt" ]] ; then
      local CMSSW_VER=CMSSW_12_4_10
      if [[ $os_version = "7" ]]; then
        export SCRAM_ARCH=slc7_amd64_gcc10
      else
        export SCRAM_ARCH=el8_amd64_gcc10
      fi
    fi

    local cmssw_inst_root="$TAU_ML_DIR/soft/CentOS${os_version}"
    local cmssw_inst="$cmssw_inst_root/$CMSSW_VER"
    if ! [ -f $cmssw_inst/.installed ]; then
      run_cmd mkdir -p "$cmssw_inst_root"
      run_cmd cd "$cmssw_inst_root"
      if [ -d $CMSSW_VER ]; then
        echo "Removing incomplete $CMSSW_VER installation..."
        run_cmd rm -rf $CMSSW_VER
      fi
      echo "Creating new $CMSSW_VER area..."
      run_cmd scramv1 project CMSSW $CMSSW_VER
      run_cmd cd "$CMSSW_VER/src"
      run_cmd eval `scramv1 runtime -sh`
      run_cmd mkdir TauMLTools
      run_cmd cd TauMLTools
      run_cmd ln -s "$TAU_ML_DIR/Analysis" Analysis
      run_cmd ln -s "$TAU_ML_DIR/Core" Core
      run_cmd ln -s "$TAU_ML_DIR/Production" Production
      run_cmd touch "$cmssw_inst/.installed"
      run_cmd scram b -j8
      run_cmd cd "$TAU_ML_DIR"
    else
      run_cmd cd "$cmssw_inst/src"
      run_cmd eval `scramv1 runtime -sh`
      run_cmd cd "$TAU_ML_DIR"
    fi
  elif [[ $MODE = "conda" ]]; then
    local CONDA=$(which conda 2>/dev/null)
    if [[ $CONDA = "" || $CONDA = "/usr/bin/conda" ]]; then
      local PRIVATE_CONDA_INSTALL_DEFAULT="$BASE_PATH/soft/conda"
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
          run_cmd curl https://repo.anaconda.com/miniconda/Miniconda2-latest-Linux-x86_64.sh -o Miniconda2-latest-Linux-x86_64.sh
          run_cmd bash Miniconda2-latest-Linux-x86_64.sh -b -p "$PRIVATE_CONDA_INSTALL"
          run_cmd touch "$PRIVATE_CONDA_INSTALL/.installed"
          run_cmd rm Miniconda2-latest-Linux-x86_64.sh
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
      run_cmd conda env create -f $BASE_PATH/tau-ml-env.yaml
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
  elif [[ $MODE = "lcg" ]]; then
    run_cmd source /cvmfs/sft.cern.ch/lcg/views/setupViews.sh LCG_101cuda x86_64-centos7-gcc10-opt
  else
    echo 'Mode "$MODE" is not supported.'
  fi

  echo "$MODE environment is successfully loaded."
}

load_env "$@"