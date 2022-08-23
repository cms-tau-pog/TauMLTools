import os
import yaml
import gc
import sys
from glob import glob
import math
import numpy as np
import tensorflow as tf
os.environ["CUDA_VISIBLE_DEVICES"]="-1"
sys.path.insert(0, "..")
from common import *
import DataLoader
import argparse

parser = argparse.ArgumentParser(description='Generate adversarial dataset in tf format')
parser.add_argument('--n_batches', required=True, type=int, help="Number of batches")
parser.add_argument('--save_path', required=True, type=str, help="Save path")
parser.add_argument('--input_dir', required=True, type=str, help="Directory containing mixed adversarial ROOT file")
parser.add_argument('--training_cfg', required=True, type=str, help="Training config")
args = parser.parse_args()


save_path = args.save_path 
scaling_cfg = "../configs/ShuffleMergeSpectral_trainingSamples-2_files_0_498.json"
training_cfg_path = args.training_cfg # ../configs/training_v1.yaml

with open(training_cfg_path) as file:
    training_cfg = yaml.full_load(file)
    print("Training Config Loaded")

training_cfg["Setup"]["input_dir"] = args.input_dir
training_cfg["Setup"]["n_tau"] = 100 # 100 taus/batch
training_cfg["Setup"]["tau_types_names"] = { "0":"e", "1":"mu", "2":"tau", "3":"jet" , "8":"data"} # add data type for labels
training_cfg["Setup"]["recompute_tautype"] = False # tau type already computed earlier 
training_cfg["SetupNN"]["n_batches"] = args.n_batches
training_cfg["SetupNN"]["n_batches_val"] = 0 # only generate training data as train/val split done later in training
training_cfg["SetupNN"]["validation_split"] = 0
training_cfg["Setup"]["input_type"]="AdversarialGenerate" # DataLoader in AdversarialGenerate mode


dataloader = DataLoader.DataLoader(training_cfg, scaling_cfg)
print("DataLoader Created")
gen_train = dataloader.get_generator(primary_set = True, return_weights = dataloader.use_weights, show_progress=True) 
print("Generator Loaded")
input_shape, input_types = dataloader.get_input_config()
print("Input shapes and Types acquired")
data_train = tf.data.Dataset.from_generator(gen_train, output_types = input_types, output_shapes = input_shape).prefetch(tf.data.AUTOTUNE)
print("Dataset extracted from DataLoader")
tf.data.experimental.save(data_train, save_path, compression = "GZIP")
print("Conversion Complete")