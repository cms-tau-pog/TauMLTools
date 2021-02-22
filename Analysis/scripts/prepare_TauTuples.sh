#!/usr/bin/env bash

N_THREADS=12
MAX_OCCUPANCY=500000
PREP_OUTPUT=output/training_preparation
PT_BINS="20, 25, 30, 35, 40, 45, 50, 55, 60, 70, 80, 90, 100, 120, 140, 160, 180, 200, 225, 250, 275, 300, 350, 400, 450, 500, 600, 700, 800, 900, 1000"
ETA_BINS="0., 0.575, 1.15, 1.725, 2.3"
DISABLED_BRANCHES="trainingWeight"

MAX_OCCUPANCY_2=100000000
PT_BINS_2="20, 1000"
ETA_BINS_2="0., 2.3"

MAX_OCCUPANCY_TESTING=20000

PRODUCE_TRAINING=1
PRODUCE_TESTING=1

if [ -d $PREP_OUTPUT ] ; then
    echo "ERROR: output directory '$PREP_OUTPUT' already exists."
    exit 1
fi

set -x

if [ $PRODUCE_TRAINING -eq 1 ] ; then
    mkdir -p $PREP_OUTPUT/all

    for OBJ in e mu tau jet; do
        OUTPUT=$PREP_OUTPUT/all/${OBJ}_pt_20_eta_0.000.root
        /usr/bin/time ./run.sh ShuffleMerge \
            --cfg TauML/Analysis/config/training_inputs_${OBJ}.cfg --input tuples-v2 --output $OUTPUT \
            --pt-bins "$PT_BINS" --eta-bins "$ETA_BINS" --mode MergeAll --calc-weights true --ensure-uniformity true \
            --max-bin-occupancy $MAX_OCCUPANCY --n-threads $N_THREADS  --disabled-branches "$DISABLED_BRANCHES"
    done

    python -u TauML/Analysis/python/CreateTupleSizeList.py --input $PREP_OUTPUT > $PREP_OUTPUT/size_list.txt

    /usr/bin/time ./run.sh ShuffleMerge --cfg TauML/Analysis/config/training_inputs_step2.cfg --input $PREP_OUTPUT \
        --output $PREP_OUTPUT/training_tauTuple.root --pt-bins "$PT_BINS_2" --eta-bins "$ETA_BINS_2" --mode MergeAll \
        --calc-weights false --ensure-uniformity true --max-bin-occupancy $MAX_OCCUPANCY_2 --n-threads $N_THREADS
fi

if [ $PRODUCE_TESTING -eq 1 ] ; then
    mkdir -p $PREP_OUTPUT/testing

    /usr/bin/time ./run.sh ShuffleMerge --cfg TauML/Analysis/config/testing_inputs.cfg --input tuples-v2 \
        --output $PREP_OUTPUT/testing --pt-bins "$PT_BINS" --eta-bins "$ETA_BINS" --mode MergePerEntry \
        --calc-weights false --ensure-uniformity false --max-bin-occupancy $MAX_OCCUPANCY_TESTING \
        --n-threads $N_THREADS --disabled-branches "$DISABLED_BRANCHES"
fi
