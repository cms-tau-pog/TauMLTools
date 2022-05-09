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
import threading

parser = argparse.ArgumentParser(description='Convert ROOT files to TF dataset')
parser.add_argument('--n_batches', required=True, type=int, help="Number of batches")
parser.add_argument('--scaling_cfg', required=True, type=str, help="Scaling config")
parser.add_argument('--save_path', required=True, type=str, help="Save path")
parser.add_argument('--training_cfg', required=True, type=str, help="Training config")
parser.add_argument('--file_start', required=True, type=int, help="Index of first file")
parser.add_argument('--file_end', required=True, type=int, help="Index of last file")
args = parser.parse_args()



save_path = args.save_path # "/home/russell/tfdata/testing"
scaling_cfg = args.scaling_cfg #"../../configs/ShuffleMergeSpectral_trainingSamples-2_files_0_50.json"
training_cfg_path = args.training_cfg #../../configs/training_v1.yaml

with open(training_cfg_path) as file:
    training_cfg = yaml.full_load(file)
    print("Training Config Loaded")

training_cfg["SetupNN"]["n_batches"]= args.n_batches
training_cfg["SetupNN"]["n_batches_val"]= 0 #args.n_batches # only generate training data as train/val split done later in training
training_cfg["SetupNN"]["validation_split"]=0.3 # same val split
training_cfg["Setup"]["input_type"]="ROOT" # make ROOT so generator loads



dataloader = DataLoader.DataLoader(training_cfg, scaling_cfg)
print("DataLoader Created")
dataloader.train_files=dataloader.train_files[args.file_start:args.file_end]
train_files = dataloader.train_files
print(f"Number of training files for conversion: {len(train_files)}")
print(f"Starting index: {args.file_start}, Ending index: {args.file_end}")
print(f"Converting {dataloader.n_batches} Batches")



gen_train = dataloader.get_generator(primary_set = True, return_weights = dataloader.use_weights, show_progress=True)  # SET PRIMARY FALSE IF VAL 
print("Generator Loaded")
input_shape, input_types = dataloader.get_input_config()
print("Input shapes and Types acquired")
ds = tf.data.Dataset.from_generator(gen_train, output_types = input_types, output_shapes = input_shape).prefetch(tf.data.AUTOTUNE)
print("Dataset extracted from DataLoader")




tf.data.experimental.save(ds, save_path, compression='GZIP') #, checkpoint_args=checkpoint_args)

print("Conversion Complete")

os.system("t-notify Conversion Complete")


