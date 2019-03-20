#!/usr/bin/env bash

N_THREADS=1
N_PER_PROCESS=6100000

# set -x
# for i in {0..11} ; do
#    ./run.sh TrainingTupleProducer --input output/training_preparation/training_tauTuple.root \
#        --output output/tuples-v2-t3-root/training/part_$i.root \
#         --start-entry $((i*N_PER_PROCESS)) --end-entry $(( (i+1) * N_PER_PROCESS )) \
#        &> output/tuples-v2-t3-root/training/part_$i.log &
# done
# wait

for file in $(ls output/tuples-v2-t3-root/training/*.root) ; do
    f_name_ext=${file##*/}
    f_name=${f_name_ext%.*}
    python ./TauML/Analysis/python/root_to_hdf.py --input $file \
            --output output/tuples-v2-t3/training/$f_name.h5 --trees taus,inner_cells,outer_cells \
            &> output/tuples-v2-t3-root/training/${f_name}_hdf.log &
done
wait


#./run.sh TrainingTupleProducer --input output/training_preparation/training_tauTuple.root \
#                               --output output/tuples-v2-t2/training.root --n-threads $N_THREADS
