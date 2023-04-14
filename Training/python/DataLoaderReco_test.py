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

with open(os.path.abspath( "../configs/trainingDisTauTag_v1.yaml")) as f:
    config = yaml.safe_load(f)
R.gInterpreter.Declare(config_parse.create_scaling_input("../configs/scaling_params_vDisTauTag_v1.json", config, verbose=False))
R.gInterpreter.Declare(config_parse.create_settings(config, verbose=False))

print("Compiling DataLoader_main...")
R.gInterpreter.Declare('#include "../interface/DataLoaderDisTauTag_main.h"')

n_tau          = config["Setup"]["n_tau"]
outclass       = len(config["Setup"]["jet_types_names"])
n_features = {}
n_seq = {}
for celltype in config["Features_all"]:
    n_features[str(celltype)] = len(config["Features_all"][celltype]) - \
                                len(config["Features_disable"][celltype])
    if celltype in config["SequenceLength"]: # sequence length
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

def getdata(_obj_f, tau_i, _reshape, _dtype=np.float32):
    x = np.copy(np.frombuffer(_obj_f.data(), dtype=_dtype, count=_obj_f.size()))
    return x[:tau_i] if _reshape==-1 else x.reshape(_reshape)[:tau_i]

def getsequence(_obj_grid,
                _tau_i,
                _batch_size,
                _input_grids,
                _n_features,
                _n_seq):
    return [ getdata(_obj_grid[getattr(R.CellObjectType,group)], _tau_i,
            (_batch_size, _n_seq[group], _n_features[group]))
            for group in _input_grids]

for i in range(n_batches):

    start = time.time()
    checker = data_loader.MoveNext()
    if checker==False:
       data_loader.ReadFile(R.std.string(input_files[file_i]), 0, -1)
       file_i += 1
       continue
    
    data = data_loader.LoadData(checker)
    X = getsequence(data.x, data.tau_i, n_tau, input_grids, n_features, n_seq)
    # X_glob = getdata(data.x_glob, data.tau_i, (n_tau, n_features["Global"]))
    Y = getdata(data.y, data.tau_i, (n_tau, outclass))
    W = getdata(data.weights, data.tau_i,  -1)

    # print("X valid:\n",X[0][:10,:,0])
    # print("X pt:\n",X[0][:10,:,1])
    # print("Global:\n",X_glob[:10,:])
    # print("Y:\n",Y[:10])
    # print("W:\n",W[:10])


    end = time.time()
    print(i, " end: ",end-start, ' s.')
    times.append(end-start)

from statistics import mean
print("Mean time: ", mean(times))
