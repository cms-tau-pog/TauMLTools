# !/bin/bash

PATH_TO_MLFLOW=$1

if ! [ -d "$PATH_TO_MLFLOW" ]; then
    echo "ERROR: directory with mlflow output '$PATH_TO_MLFLOW' not found."
    exit 1
fi

if ! [[ $PATH_TO_MLFLOW == *"mlruns/" || $PATH_TO_MLFLOW == *"mlruns" ]]; then
  echo "ERROR: '$PATH_TO_MLFLOW' is not mlruns output directory."
  exit 1
fi

DIRS=`readlink -f ${PATH_TO_MLFLOW}/*/*/artifacts`

CMD=""
for PATH_GLOB in ${DIRS[@]}; do
    NAME=`cat ${PATH_GLOB}/model_summary.txt | awk '$1=="Model:"{print $2}'`
    echo $NAME " - added"
    CMD+="${NAME}_train:${PATH_GLOB}/tensorboard_logs/train,"
    CMD+="${NAME}_val:${PATH_GLOB}/tensorboard_logs/val,"
done

CMD=`echo $CMD | head -c -2`
CMD="tensorboard --port=9090 --logdir_spec $CMD"
echo "$CMD"
eval $CMD
