#!/bin/bash

TUPLES_DIR="/eos/cms/store/group/phys_tau/TauML/DeepTau_v2/tuples-v2-training-v2-t1/testing"
PRED_DIR="/eos/cms/store/group/phys_tau/TauML/DeepTau_v2/predictions/2017v2p6/step1_e6_noPCA"
EVAL_DIR="/eos/cms/store/group/phys_tau/TauML/DeepTau_v2/plots/2017v2p6/step1_e6_noPCA"
WEIGHTS_DIR="/eos/cms/store/group/phys_tau/TauML/DeepTau_v2/spectrum_weights"


if ! [ -d "$PRED_DIR" ]; then
    echo "ERROR: directory with predictions '$PRED_DIR' not found."
    exit 1
fi

mkdir -p "$EVAL_DIR"

TAU_SAMPLE=HTT
OBJ_TYPES=( e mu jet jet )
SAMPLES=( DY DY W TT )

for n in $(seq 0 $(( ${#OBJ_TYPES[@]} - 1 )) ); do
    echo "tau_${TAU_SAMPLE} vs ${OBJ_TYPES[n]}_${SAMPLES[n]}"
    python TauMLTools/Training/python/evaluate_performance.py --input-taus "$TUPLES_DIR/tau_${TAU_SAMPLE}.h5" \
           --input-other "$TUPLES_DIR/${OBJ_TYPES[n]}_${SAMPLES[n]}.h5" --other-type ${OBJ_TYPES[n]} \
           --deep-results "$PRED_DIR" --output "$EVAL_DIR/tau_vs_${OBJ_TYPES[n]}_${SAMPLES[n]}.pdf" \
           --setup TauMLTools/Training/python/plot_setups/run2.py --weights "$WEIGHTS_DIR" \
           --store-json --public-plots --draw-wp
    RESULT=$?
    if [ $RESULT -ne 0 ]; then
        echo "An error occured while running the last command. Interupting the execution."
        exit 1
    fi
done
