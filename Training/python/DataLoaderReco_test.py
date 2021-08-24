# The environment on Centos 7 is:
# source /cvmfs/sft.cern.ch/lcg/views/SetupViews.sh LCG_99 x86_64-centos7-gcc10-opt
import ROOT as R
import numpy as np
import time
import config_parse
import os

R.gROOT.ProcessLine(".include ../../..")

print("Compiling Setup classes...")

R.gInterpreter.Declare(config_parse.create_scaling_input("../configs/scaling_params_vReco_v1.json", "../configs/trainingReco_v1.yaml", verbose=True))
R.gInterpreter.Declare(config_parse.create_settings("../configs/trainingReco_v1.yaml", verbose=True))
# exit()

print("Compiling DataLoader_main...")
R.gInterpreter.Declare('#include "../interface/DataLoaderReco_main.h"')
# R.gInterpreter.Declare('#include "TauMLTools/Core/interface/exception.h"')


input_files = []
for root, dirs, files in os.walk(os.path.abspath(R.Setup.input_dir)):
    for file in files:
        input_files.append(os.path.join(root, file))
# print(input_files)

data_loader = R.DataLoader()

n_batches = 1000

times = []

file_i = 0
data_loader.ReadFile(R.std.string(input_files[file_i]), 0, -1)
file_i += 1

for i in range(n_batches):

    start = time.time()
    checker = data_loader.MoveNext()
    if checker==False:
       data_loader.ReadFile(R.std.string(input_files[file_i]), 0, -1)
       file_i += 1
       continue
    data = data_loader.LoadData()

    end = time.time()
    print(i, " end: ",end-start, ' s.')
    times.append(end-start)

from statistics import mean
print("Mean time: ", mean(times))
