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
R.gInterpreter.Declare(config_parse.create_scaling_input("../configs/scaling_params_vReco_v1(stau).json", config, verbose=False))
R.gInterpreter.Declare(config_parse.create_settings(config, verbose=False))

print("Compiling DataLoader_main...")
R.gInterpreter.Declare('#include "../interface/DataLoaderReco_main.h"')

n_tau          = R.Setup.n_tau
pfCand_n       = R.Setup.nSeq_PfCand
pfCand_fn      = R.Setup.n_PfCand
outclass       = R.Setup.output_classes

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

for i in range(n_batches):

    start = time.time()
    checker = data_loader.MoveNext()
    if checker==False:
       data_loader.ReadFile(R.std.string(input_files[file_i]), 0, -1)
       file_i += 1
       continue
    
    data = data_loader.LoadData()
    X = getdata(data.x, (n_tau, pfCand_n, pfCand_fn))
    Y = getdata(data.y, (n_tau, outclass))

    end = time.time()
    print(i, " end: ",end-start, ' s.')
    times.append(end-start)

from statistics import mean
print("Mean time: ", mean(times))
