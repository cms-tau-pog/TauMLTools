#!/bin/bash

if (( $# < 1 )) ; then
    cat << EOF
Setup environment for TauMLTools
Usage: source env.sh mode [mode_arg_1] [mode_arg_2] ...

Supported modes: prod2018 prod2018UL phase2 lcg conda

Mode-specific arguments:
conda
  --update [env.yaml]  updates environment from env.yaml (default: tau-ml-env.yaml)
EOF
    return 1
fi

MODE=$1
BASE_PATH=$PWD

function run_cmd {
    "$@"
    RESULT=$?
    if (( $RESULT != 0 )); then
        echo "Error while running '$@'"
        kill -INT $$
    fi
}

if [[ $MODE = "prod2018" || $MODE = "phase2" || $MODE = "phase2_113X" || $MODE = "prod2018UL" || $MODE = "run3" ]]; then
    if [ $MODE = "prod2018" ] ; then
        CMSSW_VER=CMSSW_10_6_29
        APPLY_BOOSTED_FIX=1
        export SCRAM_ARCH=slc7_amd64_gcc700
    elif [[ $MODE = "phase2" ]]; then
        CMSSW_VER=CMSSW_11_2_5
        APPLY_BOOSTED_FIX=0
        export SCRAM_ARCH=slc7_amd64_gcc900
    elif [[ $MODE = "phase2_113X" ]]; then
        CMSSW_VER=CMSSW_11_3_0
        APPLY_BOOSTED_FIX=0
        export SCRAM_ARCH=slc7_amd64_gcc900
    elif [ $MODE = "prod2018UL" ] ; then
        CMSSW_VER=CMSSW_10_6_29
        APPLY_BOOSTED_FIX=0
        export SCRAM_ARCH=slc7_amd64_gcc700
    elif [ $MODE = "run3" ] ; then
        CMSSW_VER=CMSSW_12_4_0
        APPLY_BOOSTED_FIX=0
        export SCRAM_ARCH=slc7_amd64_gcc10
    fi

    if ! [ -f soft/$CMSSW_VER/.installed ]; then
        run_cmd mkdir -p soft
        run_cmd cd soft
        if [ -d $CMSSW_VER ]; then
            echo "Removing incomplete $CMSSW_VER installation..."
            run_cmd rm -rf $CMSSW_VER
        fi
        echo "Creating new $CMSSW_VER area..."
        run_cmd scramv1 project CMSSW $CMSSW_VER
        run_cmd cd $CMSSW_VER/src
        run_cmd eval `scramv1 runtime -sh`
        if (( $APPLY_BOOSTED_FIX == 1 )); then
            run_cmd git cms-merge-topic -u cms-tau-pog:CMSSW_10_6_X_tau-pog_boostedTausMiniFix
        fi

        run_cmd mkdir TauMLTools
        run_cmd cd TauMLTools
        run_cmd ln -s ../../../../Analysis Analysis
        run_cmd ln -s ../../../../Core Core
        run_cmd ln -s ../../../../Production Production
        run_cmd touch ../../.installed
        run_cmd scram b -j8
        run_cmd cd ../../../..
    else
        run_cmd cd soft/$CMSSW_VER/src
        run_cmd eval `scramv1 runtime -sh`
        run_cmd cd ../../..
    fi
elif [[ $MODE = "conda" ]]; then
    CONDA=$(which conda 2>/dev/null)
    if [[ $CONDA = "" || $CONDA = "/usr/bin/conda" ]]; then
        PRIVATE_CONDA_INSTALL_DEFAULT="$BASE_PATH/soft/conda"
        PRIVATE_CONDA_INSTALL="$PRIVATE_CONDA_INSTALL_DEFAULT"
        if [ -f "$PRIVATE_CONDA_INSTALL_DEFAULT.ref" ]; then
            PRIVATE_CONDA_INSTALL=$(cat "$PRIVATE_CONDA_INSTALL.ref")
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
    TAU_ML_DIR=$(cd $(dirname $(which python))/..; pwd)
    export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$TAU_ML_DIR/lib
elif [[ $MODE = "lcg" ]]; then
    run_cmd source /cvmfs/sft.cern.ch/lcg/views/setupViews.sh LCG_101cuda x86_64-centos7-gcc10-opt
else
    echo 'Mode "$MODE" is not supported.'
fi

echo "$MODE environment is successfully loaded."
