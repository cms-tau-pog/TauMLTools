#!/bin/bash

if [ $# -ne 2 ] ; then
    echo "Usage: network epoch"
    exit
fi

NETWORK=$1
EPOCH=$2

WORK_DIR="$PWD"
NET_DIR="output/networks/$NETWORK"
NET_FILE="DeepTau${NETWORK}_step1_e${EPOCH}"

if ! [ -d "$NET_DIR" ]; then
    echo "ERROR: directory with networks '$NET_DIR' not found."
    exit 1
fi

if ! [ -f "$NET_DIR/${NET_FILE}.pb" ] ; then
    cd "$NET_DIR"
    if ! [ -f "${NET_FILE}.h5" ] ; then
        if [ $NETWORK == "2017v2p6" ] ; then
            scp "kes@cmsphase2sim:workspace/tau-ml/CMSSW_10_4_0/src/TauML/Training/python/2017v2/${NET_FILE}.h5" .
        elif [ $NETWORK == "2017v2p7" ] ; then
            scp "androsov@gridui1:${NET_FILE}.h5" .
        else
            echo "ERROR: network not supported"
        fi
    fi

    if ! [ -f "${NET_FILE}.h5" ] ; then
        echo "ERROR: unable to get training file ${NET_FILE}.h5"
        exit 1
    fi

    python3 "$WORK_DIR/TauML/Training/python/deploy_model.py" --input "${NET_FILE}.h5"

    if ! [ -f "${NET_FILE}.pb" ] ; then
        echo "ERROR: unable to deploy training file ${NET_FILE}.h5"
        exit 1
    fi
    cd "$WORK_DIR"
fi

#TUPLES_DIR="/data/tau-ml/tuples-v2-training-v2-t1/testing-short"
TUPLES_DIR="/data/tau-ml/tuples-v2-training-v2-t1/testing"
PRED_DIR="output/predictions/$NETWORK/step1_e$EPOCH"

mkdir -p "$PRED_DIR"

python3 TauML/Training/python/apply_training.py --input "$TUPLES_DIR" --output "$PRED_DIR" \
    --model "$NET_DIR/${NET_FILE}.pb" --chunk-size 1000 --batch-size 100 --max-queue-size 20
