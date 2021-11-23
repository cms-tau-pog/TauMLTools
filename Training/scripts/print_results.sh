#!/bin/bash

PATH_TO_MLFLOW=$1

if ! [ -d "$PATH_TO_MLFLOW" ]; then
    echo "ERROR: directory with mlflow output '$PATH_TO_MLFLOW' not found."
    exit 1
fi

if ! [[ $PATH_TO_MLFLOW == *"mlruns/" || $PATH_TO_MLFLOW == *"mlruns" ]]; then
  echo "ERROR: '$PATH_TO_MLFLOW' is not mlruns output directory."
  exit 1
fi

DIRS=`readlink -f ${PATH_TO_MLFLOW}/*/*/meta.yaml`

for PATH_GLOB in ${DIRS[@]}; do
    RUN_ID_PATH=`echo $PATH_GLOB | sed 's|\(.*\)/.*|\1|'`
    RUN_ID=`echo $RUN_ID_PATH |  sed 's|.*/||'`
    NAME=`cat ${RUN_ID_PATH}/artifacts/model_summary.txt | grep 'Model:'`
    TP=`cat ${RUN_ID_PATH}/artifacts/model_summary.txt | grep 'Trainable params:'`
    LOSE=`cat ${RUN_ID_PATH}/metrics/weighted_tau_crossentropy_v2 | awk '{ print $2 }'`
    LOSE_VAL=`cat ${RUN_ID_PATH}/metrics/val_weighted_tau_crossentropy_v2 | awk '{ print $2 }'`
    echo RunID: $RUN_ID $NAME $TP loss: $LOSE val_loss: $LOSE_VAL
done