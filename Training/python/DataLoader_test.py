# The environment on Centos 7 is:
# source /cvmfs/sft.cern.ch/lcg/views/SetupViews.sh LCG_99 x86_64-centos7-gcc10-opt
import ROOT as R
import numpy as np
import time
import config_parse
import os
import yaml
from glob import glob

R.gROOT.ProcessLine(".include ../../..")

print("Compiling Setup classes...")

with open(os.path.abspath( "../configs/training_v1.yaml")) as f:
    config = yaml.safe_load(f)
R.gInterpreter.Declare(config_parse.create_scaling_input("../configs/scaling_params_v1.json", config, verbose=False))
R.gInterpreter.Declare(config_parse.create_settings(config, verbose=False))

print("Compiling DataLoader_main...")
R.gInterpreter.Declare('#include "../interface/DataLoader_main.h"')
R.gInterpreter.Declare('#include "TauMLTools/Core/interface/exception.h"')

n_tau          = R.Setup.n_tau
n_inner_cells  = R.Setup.n_inner_cells
n_outer_cells  = R.Setup.n_outer_cells
n_fe_tau    = R.Setup.n_TauFlat
n_pf_el     = R.Setup.n_PfCand_electron
n_pf_mu     = R.Setup.n_PfCand_muon
n_pf_chHad  = R.Setup.n_PfCand_chHad
n_pf_nHad   = R.Setup.n_PfCand_nHad
n_pf_gamma  = R.Setup.n_PfCand_gamma
n_ele       = R.Setup.n_Electron
n_muon      = R.Setup.n_Muon
tau_types   = R.Setup.tau_types_names.size()
input_files = glob(f'{R.Setup.input_dir}*.root')

n_grid_features = {
    "PfCand_electron" : n_pf_el,
    "PfCand_muon" : n_pf_mu,
    "PfCand_chHad" : n_pf_chHad,
    "PfCand_nHad" : n_pf_nHad,
    "PfCand_gamma" : n_pf_gamma,
    "Electron" : n_ele,
    "Muon" : n_muon
}

input_grids =[ [ "PfCand_electron", "PfCand_gamma", "Electron" ], [ "PfCand_muon", "Muon" ], [ "PfCand_chHad", "PfCand_nHad" ] ]

input_files = []
for root, dirs, files in os.walk(os.path.abspath(R.Setup.input_dir)):
    for file in files:
        input_files.append(os.path.join(root, file))

data_loader = R.DataLoader()

n_batches = 1000
n_batches_store = 5

from queue import Queue
data_que = Queue(maxsize = n_batches_store)

times = []

def getdata(_obj_f, _reshape, _dtype=np.float32):
    x = np.copy(np.frombuffer(_obj_f.data(), dtype=_dtype, count=_obj_f.size()))
    return x if _reshape==-1 else x.reshape(_reshape)

def getgrid(_obj_grid, _inner):
    _n_cells = n_inner_cells if _inner else n_outer_cells
    _X = []
    for group in input_grids:
        _X.append(
            np.concatenate(
                [ getdata(_obj_grid[ getattr(R.CellObjectType,fname) ][_inner],
                    (n_tau, _n_cells, _n_cells, n_grid_features[fname])) for fname in group ],
                axis=-1
                )
            )
    return _X

c = 0
data_loader.ReadFile(R.std.string(input_files[c]), 0, -1)
c+=1

for i in range(n_batches):

    start = time.time()

    if(data_que.full()):
        _ = data_que.get()

    checker = data_loader.MoveNext()

    if checker==False:
       data_loader.ReadFile(R.std.string(input_files[c]), 0, -1)
       c+=1
       continue

    data = data_loader.LoadData()


    # Flat Tau features
    X = [getdata(data.x_tau, (n_tau, n_fe_tau))]
    # Inner grid
    X += getgrid(data.x_grid, 1) # 500 11 11 176
    # Outer grid
    X += getgrid(data.x_grid, 0) # 500 21 21 176

    X = tuple(X)

    weights = getdata(data.weight, -1)
    Y = getdata(data.y_onehot, (n_tau, tau_types))


    data_que.put(X)

    end = time.time()
    print(i, " end: ",end-start, ' s.')
    times.append(end-start)

    for x in X:
        if np.isnan(x).any():
            print("Nan detected! element=",x.shape) 
            print(np.argwhere(x))
        if np.isinf(x).any():
            print("Inf detected! element=",x.shape)
        if np.amax(x)==0:
            print("Empty tuple detected! element=",x.shape)

from statistics import mean
print("Mean time: ", mean(times))

time_arr = np.asarray(times)
np.savetxt("dataloader.csv", time_arr, delimiter=",")
