#!/bin/bash

if [ $# -ne 1 ] ; then
    echo "Setup environment for TauMLTools"
    echo "Usage: source setup.sh mode"
    echo "Supported modes: prod2018 prod2018UL phase2 lcg conda "
    return 1
fi

MODE=$1
BASE_PATH=$PWD

function run_cmd {
    "$@"
    RESULT=$?
    if [ $RESULT -ne 0 ] ; then
        echo "Error while running '$@'"
        kill -INT $$
    fi
}

if [ $MODE = "prod2018" -o $MODE = "phase2" -o $MODE = "prod2018UL" ] ; then
    if [ $MODE = "prod2018" ] ; then
        CMSSW_VER=CMSSW_10_6_20
        APPLY_BOOSTED_FIX=1
        export SCRAM_ARCH=slc7_amd64_gcc700
    elif [ $MODE = "phase2" ] ; then
        CMSSW_VER=CMSSW_11_2_5
        APPLY_BOOSTED_FIX=0
        export SCRAM_ARCH=slc7_amd64_gcc900
    elif [ $MODE = "prod2018UL" ] ; then
        CMSSW_VER=CMSSW_10_6_27
        APPLY_BOOSTED_FIX=0
        export SCRAM_ARCH=slc7_amd64_gcc700
    fi

    if ! [ -f soft/$CMSSW_VER/.installed ] ; then
        run_cmd mkdir -p soft
        run_cmd cd soft
        if [ -d $CMSSW_VER ] ; then
            echo "Removing incomplete $CMSSW_VER installation..."
            run_cmd rm -rf $CMSSW_VER
        fi
        echo "Creating new $CMSSW_VER area..."
        run_cmd scramv1 project CMSSW $CMSSW_VER
        run_cmd cd $CMSSW_VER/src
        run_cmd eval `scramv1 runtime -sh`
        if [ $APPLY_BOOSTED_FIX -eq 1 ]; then
            run_cmd git cms-merge-topic -u cms-tau-pog:CMSSW_10_6_X_tau-pog_boostedTausMiniFix
        fi
        run_cmd mkdir TauMLTools
        run_cmd cd TauMLTools
        run_cmd ln -s ../../../../Analysis Analysis
        run_cmd ln -s ../../../../Core Core
        run_cmd ln -s ../../../../Production Production
        run_cmd scram b -j8
        run_cmd touch ../../.installed
        run_cmd cd ../../../..
    else
        run_cmd cd soft/$CMSSW_VER/src
        run_cmd eval `scramv1 runtime -sh`
        run_cmd cd ../../..
    fi
elif [ $MODE = "conda" ] ; then
    CONDA=$(which conda 2>/dev/null)
    if [ x$CONDA = "x" -o x$CONDA = "x/usr/bin/conda" ] ; then
        PRIVATE_CONDA_INSTALL=$BASE_PATH/soft/conda
        if ! [ -f "$PRIVATE_CONDA_INSTALL/.installed" ] ; then
            if [ -d $PRIVATE_CONDA_INSTALL ] ; then
                echo "Removing incomplete private conda installation..."
                run_cmd rm -rf $PRIVATE_CONDA_INSTALL
            fi
            echo "Installing conda..."
            run_cmd mkdir -p soft
            run_cmd cd soft
            run_cmd curl https://repo.anaconda.com/miniconda/Miniconda2-latest-Linux-x86_64.sh -o Miniconda2-latest-Linux-x86_64.sh
            run_cmd bash Miniconda2-latest-Linux-x86_64.sh -b -p $PRIVATE_CONDA_INSTALL
            run_cmd touch "$PRIVATE_CONDA_INSTALL/.installed"
        fi
        __conda_setup="$($PRIVATE_CONDA_INSTALL/bin/conda shell.${SHELL##*/} hook)"
        if [ $? -eq 0 ]; then
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
    if [ $tau_env_found -ne 1 ]; then
        echo "Creating tau-ml environment..."
        run_cmd conda env create -f $BASE_PATH/tau-ml-env.yaml
    fi
    run_cmd conda activate tau-ml
elif [ $MODE = "lcg" ] ; then
    run_cmd source /cvmfs/sft.cern.ch/lcg/views/setupViews.sh LCG_100 x86_64-centos7-gcc10-opt
else
    echo 'Mode "$MODE" is not supported.'
fi

echo "$MODE environment is successfully loaded."
