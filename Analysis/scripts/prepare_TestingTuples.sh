#!/usr/bin/env bash

set -x
for file in $(ls output/training_preparation/testing/*.root) ; do
   ./run.sh TrainingTupleProducer --input $file \
            --output output/tuples-v2-t2/testing/$(basename $file) \
            &> output/tuples-v2-t2/testing/$(basename $file).log &
done

wait
