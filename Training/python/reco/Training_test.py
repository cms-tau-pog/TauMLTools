import os
import sys
import time

sys.path.insert(0, "..")
from common import *
import DataLoader

config   = os.path.abspath( "../../configs/trainingReco_v1.yaml")
scaling  = os.path.abspath("../../configs/scaling_params_vReco_v1.json")
dataloader = DataLoader.DataLoader(config, scaling)

gen_train = dataloader.get_generator(primary_set = True)
# gen_val = dataloader.get_generator(primary_set = False)

map_features, input_shape, input_types  = dataloader.get_config()

data_train = tf.data.Dataset.from_generator(
    gen_train, output_types = input_types, output_shapes = input_shape
    ).prefetch(10)

start = time.time()
time_checkpoints = []

for epoch in range(3):
    print("Epoch ->", epoch)
    for i,_ in enumerate(data_train):
        time_checkpoints.append(time.time()-start)
        print(i, " ", time_checkpoints[-1], "s.")
        start = time.time()
    print("AVR.:",sum(time_checkpoints)/len(time_checkpoints))
# for i,_ in enumerate(gen_train()):
#     time_checkpoints.append(time.time())
#     print(i, " ", time_checkpoints[-1]-time_checkpoints[-2], "s.")



