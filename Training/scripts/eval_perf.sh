#!/bin/bash

if [ $# -ne 3 ] ; then
    echo "Usage: network epoch ref_epoch"
    exit
fi

NETWORK=$1
EPOCH=$2
REF_EPOCH=$3

#TUPLES_DIR="/data/tau-ml/tuples-v2-training-v2-t1/testing"
TUPLES_DIR="output/tuples-v2-training-v2-t1/testing"
PRED_DIR="output/predictions/$NETWORK/step1_e$EPOCH"
PREV_PRED_DIR="output/predictions/2017v2p6/step1_e$REF_EPOCH"
EVAL_DIR="output/eval_plots/$NETWORK/step1_e$EPOCH"

if ! [ -d "$PRED_DIR" ]; then
    echo "ERROR: directory with predictions '$PRED_DIR' not found."
    exit 1
fi

mkdir -p "$EVAL_DIR"

TAU_SAMPLE=HTT
#OBJ_TYPES=(e mu jet)
#SAMPLES=(DY DY QCD)
#OBJ_TYPES=( jet jet jet jet )
#SAMPLES=( QCD TT DY W )
#OBJ_TYPES=( e e )
#SAMPLES=( DY W )
OBJ_TYPES=( e e mu mu jet jet jet jet )
SAMPLES=( DY W DY W QCD TT DY W )
#OBJ_TYPES=( mu mu )
#SAMPLES=( DY W )
#OBJ_TYPES=( jet )
#SAMPLES=( QCD )
#OBJ_TYPES=( mu )
#SAMPLES=( DY )
#OBJ_TYPES=( e )
#SAMPLES=( DY )


for n in $(seq 0 $(( ${#OBJ_TYPES[@]} - 1 )) ); do
    echo "tau_${TAU_SAMPLE} vs ${OBJ_TYPES[n]}_${SAMPLES[n]}"
    python3 -u TauML/Training/python/evaluate_performance.py --input-taus "$TUPLES_DIR/tau_${TAU_SAMPLE}.h5" \
        --input-other "$TUPLES_DIR/${OBJ_TYPES[n]}_${SAMPLES[n]}.h5" --other-type ${OBJ_TYPES[n]} \
        --deep-results "$PRED_DIR" --deep-results-label "e$EPOCH" \
        --prev-deep-results "$PREV_PRED_DIR" --prev-deep-results-label "e$REF_EPOCH" \
        --output "$EVAL_DIR/tau_vs_${OBJ_TYPES[n]}_${SAMPLES[n]}.pdf" 2>&1 \
        | tee "$EVAL_DIR/tau_vs_${OBJ_TYPES[n]}_${SAMPLES[n]}.txt"

done
