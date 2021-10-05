# The environment on Centos 7 is:
# source /cvmfs/sft.cern.ch/lcg/views/SetupViews.sh LCG_99 x86_64-centos7-gcc10-opt
import ROOT as R
import numpy as np
import time
import config_parse
import os
import yaml

R.gROOT.ProcessLine(".include ../../..")

print("Compiling Setup classes...")

with open(os.path.abspath( "../configs/trainingReco_v1.yaml")) as f:
    config = yaml.safe_load(f)
R.gInterpreter.Declare(config_parse.create_scaling_input("../configs/scaling_params_vReco_v1_stau.json", config, verbose=False))
R.gInterpreter.Declare(config_parse.create_settings(config, verbose=False))

print("Compiling DataLoader_main...")
R.gInterpreter.Declare('#include "../interface/DataLoaderReco_main.h"')

n_tau          = config["Setup"]["n_tau"]
outclass       = config["Setup"]["output_classes"]
n_features = {}
n_seq = {}
for celltype in config["Features_all"]:
    n_features[str(celltype)] = len(config["Features_all"][celltype]) - \
                                len(config["Features_disable"][celltype])
    n_seq[str(celltype)] = config["SequenceLength"][celltype]
input_grids = config["CellObjectType"]

input_files = []
for root, dirs, files in os.walk(os.path.abspath(R.Setup.input_dir)):
    for file in files:
        input_files.append(os.path.join(root, file))

data_loader = R.DataLoader()

n_batches = 1000

times = []

file_i = 0
data_loader.ReadFile(R.std.string(input_files[file_i]), 0, -1)
file_i += 1

def getdata(_obj_f, _reshape, _dtype=np.float32):
    return np.copy(np.frombuffer(_obj_f.data(),
                                 dtype=_dtype,
                                 count=_obj_f.size())).reshape(_reshape)

def getgrid(_obj_grid):
    return [ getdata(_obj_grid[getattr(R.CellObjectType,group)],
                    (n_tau, n_seq[group], n_features[group]))
                    for group in input_grids]

for i in range(n_batches):

    start = time.time()
    checker = data_loader.MoveNext()
    if checker==False:
       data_loader.ReadFile(R.std.string(input_files[file_i]), 0, -1)
       file_i += 1
       continue
    
    data = data_loader.LoadData()
    X = getgrid(data.x)
    Y = getdata(data.y, (n_tau, outclass))

    end = time.time()
    print(i, " end: ",end-start, ' s.')
    times.append(end-start)

from statistics import mean
print("Mean time: ", mean(times))
