#!/usr/bin/env bash

N_THREADS=1
N_PER_PROCESS=13000000

TRAINING_IN="/data/tau-ml/tuples-v2-training-v2/training_tauTuple.root"
TRAINING_OUT_ROOT="output/tuples-v2-training-v2-t1-root/training"
TRAINING_OUT_HDF="output/tuples-v2-training-v2-t1/training"

set -x
mkdir -p $TRAINING_OUT_ROOT
for i in {0..11} ; do
   ./run.sh TrainingTupleProducer --input $TRAINING_IN --output $TRAINING_OUT_ROOT/part_$i.root \
       --start-entry $((i*N_PER_PROCESS)) --end-entry $(( (i+1) * N_PER_PROCESS )) \
       &> $TRAINING_OUT_ROOT/part_$i.log &
done
wait

mkdir -p $TRAINING_OUT_HDF
for file in $(ls $TRAINING_OUT_ROOT/*.root) ; do
    f_name_ext=${file##*/}
    f_name=${f_name_ext%.*}
    python ./TauML/Analysis/python/root_to_hdf.py --input $file \
            --output $TRAINING_OUT_HDF/$f_name.h5 --trees taus,inner_cells,outer_cells \
            &> $TRAINING_OUT_ROOT/${f_name}_hdf.log &
done
wait


#./run.sh TrainingTupleProducer --input output/training_preparation/training_tauTuple.root \
#                               --output output/tuples-v2-t2/training.root --n-threads $N_THREADS
