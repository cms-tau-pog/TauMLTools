import os
import yaml
import sys
import time

sys.path.insert(0, "..")
from common import *
import DataLoader

with open(os.path.abspath( "../../configs/training_v1.yaml")) as f:
    config = yaml.safe_load(f)
scaling  = os.path.abspath("../../configs/ShuffleMergeSpectral_trainingSamples-2_files_0_50.json")
dataloader = DataLoader.DataLoader(config, scaling)

gen_train = dataloader.get_generator(primary_set = True)
# gen_val = dataloader.get_generator(primary_set = False)

netConf_full = dataloader.get_net_config()
input_shape, input_types = dataloader.get_input_config()

data_train = tf.data.Dataset.from_generator(
    gen_train, output_types = input_types, output_shapes = input_shape
    ).prefetch(10)

time_checkpoints = [time.time()]

for epoch in range(3):
    print("Epoch ->", epoch)
    for i,_ in enumerate(data_train):
        # if i % 10 == 0:
        #     time.sleep(10)
        time_checkpoints.append(time.time())
        print(i, " ", time_checkpoints[-1]-time_checkpoints[-2], "s.")
    
# for i,_ in enumerate(gen_train()):
#     time_checkpoints.append(time.time())
#     print(i, " ", time_checkpoints[-1]-time_checkpoints[-2], "s.")



