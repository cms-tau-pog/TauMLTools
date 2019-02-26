#!/usr/bin/env bash

N_THREADS=2

#set -x
#for i in {0..11} ; do
#    ./run.sh TrainingTupleProducer --input output/training_preparation/training_tauTuple.root \
#        --output output/tuples-v2-t1/training/part_$i.root --n-cells $N_CELLS --cell-size $CELL_SIZE \
#         --start-entry $((i*N_PER_PROCESS)) --end-entry $(( (i+1) * N_PER_PROCESS )) \
#        &> output/tuples-v2-t1/training/part_$i.log &
#done
#wait

./run.sh TrainingTupleProducer --input output/training_preparation/training_tauTuple.root \
                               --output output/tuples-v2-t2/training.root --n-threads $N_THREADS
