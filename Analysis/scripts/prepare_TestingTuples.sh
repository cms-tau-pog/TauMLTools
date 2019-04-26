#!/usr/bin/env bash

TRAINING_IN="/data/tau-ml/tuples-v2-training-v2/testing"
TRAINING_OUT_ROOT="output/tuples-v2-training-v2-t1-root/testing"
TRAINING_OUT_HDF="output/tuples-v2-training-v2-t1/testing"

set -x
mkdir -p $TRAINING_OUT_ROOT
for file in $(ls $TRAINING_IN/*.root) ; do
    f_name_ext=${file##*/}
    f_name=${f_name_ext%.*}
   ./run.sh TrainingTupleProducer --input $file --output $TRAINING_OUT_ROOT/$f_name_ext \
       &> $TRAINING_OUT_ROOT/${f_name}.log
done
#wait

mkdir -p $TRAINING_OUT_HDF
for file in $(ls $TRAINING_OUT_ROOT/*.root) ; do
    f_name_ext=${file##*/}
    f_name=${f_name_ext%.*}
    python ./TauML/Analysis/python/root_to_hdf.py --input $file \
            --output $TRAINING_OUT_HDF/$f_name.h5 --trees taus,inner_cells,outer_cells \
            &> $TRAINING_OUT_ROOT/${f_name}_hdf.log
done
#wait
