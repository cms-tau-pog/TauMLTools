import os
import sys
import time

sys.path.insert(0, "..")
from common import *
import DataLoader

config   = os.path.abspath( "../../configs/training_v1.yaml")
scaling  = os.path.abspath("../../configs/scaling_params_v1.json")
dataloader = DataLoader.DataLoader(config, scaling)

gen_train = dataloader.get_generator(primary_set = True)
gen_val = dataloader.get_generator(primary_set = False)

data_train = tf.data.Dataset.from_generator(gen_train, output_types = input_types, output_shapes = input_shape)
data_val = tf.data.Dataset.from_generator(gen_val, output_types = input_types, output_shapes = input_shape)

time_checkpoints = [time.time()]

for i,_ in enumerate(gen_train()):
    time_checkpoints.append(time.time())
    print(i, " ", time_checkpoints[-1]-time_checkpoints[-2], "s.")
    
for i,_ in enumerate(gen_train()):
    time_checkpoints.append(time.time())
    print(i, " ", time_checkpoints[-1]-time_checkpoints[-2], "s.")



