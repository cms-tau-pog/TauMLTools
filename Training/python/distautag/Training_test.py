import os
import sys
import time
import yaml

sys.path.insert(0, "..")
from common import *
import DataLoaderReco

with open(os.path.abspath( "../../configs/trainingDisTauTag_v1.yaml")) as f:
    config = yaml.safe_load(f)
scaling  = os.path.abspath("../../configs/scaling_params_vDisTauTag_v1.json")
dataloader = DataLoaderReco.DataLoader(config, scaling)

gen_train = dataloader.get_generator(primary_set = True)
# gen_val = dataloader.get_generator(primary_set = False)

input_shape, input_types  = dataloader.get_shape()

data_train = tf.data.Dataset.from_generator(
    gen_train, output_types = input_types, output_shapes = input_shape
    ).prefetch(10)

start = time.time()
time_checkpoints = []

for i,(x,y) in enumerate(data_train):
    time_checkpoints.append(time.time()-start)
    print(i, " ", time_checkpoints[-1], "s.")
    start = time.time()

print("AVR.:",sum(time_checkpoints)/len(time_checkpoints))