#!/usr/bin/env bash

N_CELLS=13
CELL_SIZE=0.05

set -x
for file in $(ls output/training_preparation/testing/*.root) ; do
   ./run.sh TrainingTupleProducer --input $file \
       --output output/tuples-v2-t1/testing/$(basename $file) --n-cells $N_CELLS --cell-size $CELL_SIZE \
       &> output/tuples-v2-t1/testing/$(basename $file).log &
done

wait
