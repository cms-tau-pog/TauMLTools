#!/bin/bash

NETWORK=$1
EPOCH=$2

TUPLES_DIR="/data/tau-ml/tuples-v2-training-v2-t1/testing"
PRED_DIR="output/predictions/$NETWORK/step1_e$EPOCH"
EVAL_DIR="output/eval_plots/$NETWORK/step1_e$EPOCH"

if ! [ -d "$PRED_DIR" ]; then
    echo "ERROR: directory with predictions '$PRED_DIR' not found."
    exit 1
fi

mkdir -p "$EVAL_DIR"

TAU_SAMPLE=HTT
OBJ_TYPES=(e mu jet)
SAMPLES=(DY DY QCD)

for n in $(seq 0 $(( ${#OBJ_TYPES[@]} - 1 )) ); do
    echo "tau_${TAU_SAMPLE} vs ${OBJ_TYPES[n]}_${SAMPLES[n]}"
    python3 TauML/Training/python/evaluate_performance.py --input-taus "$TUPLES_DIR/tau_${TAU_SAMPLE}.h5" \
        --input-other "$TUPLES_DIR/${OBJ_TYPES[n]}_${SAMPLES[n]}.h5" --other-type ${OBJ_TYPES[n]} \
        --deep-results "$PRED_DIR" --output "$EVAL_DIR/tau_vs_${OBJ_TYPES[n]}.pdf"

done
