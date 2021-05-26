# The environment on Centos 7 is:
# source /cvmfs/sft.cern.ch/lcg/views/SetupViews.sh LCG_99 x86_64-centos7-gcc10-opt
import ROOT as R
import numpy as np
import time
import config_parse
import os

R.gROOT.ProcessLine(".include ../../..")

print("Compiling Setup classes...")
temp_file = open("Setup_tmp.h", "w")
temp_file.write(config_parse.create_scaling_input("../configs/scaling_test.json", "../configs/training_v1.yaml", verbose=False))
temp_file.write(config_parse.create_settings("../configs/training_v1.yaml", verbose=False))
temp_file.close()
R.gInterpreter.ProcessLine('#include "Setup_tmp.h"')
os.remove("Setup_tmp.h")

print("Compiling DataLoader_main...")
R.gInterpreter.ProcessLine('#include "../interface/DataLoader_main.h"')

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

data_loader = R.DataLoader()
print("Number of all entries / batch size: ", data_loader.GetEntries()/n_tau)

n_batches = 500
n_batches_store = 1

def to_tensor(data):

    return [
        np.frombuffer(data.y_onehot.data(), dtype=np.float32, count=data.y_onehot.size()).reshape((n_tau, tau_types)),
        np.frombuffer(data.weight.data(), dtype=np.float32, count=data.weight.size()),
        np.frombuffer(data.x_tau.data(), dtype=np.float32, count=data.x_tau.size()).reshape((n_tau,n_fe_tau)),
        np.frombuffer(data.x_grid[0][0].data(), dtype=np.float32, count=data.x_grid[0][0].size()).reshape((n_tau, n_outer_cells, n_outer_cells, n_pf_el)),
        np.frombuffer(data.x_grid[1][0].data(), dtype=np.float32, count=data.x_grid[1][0].size()).reshape((n_tau, n_outer_cells, n_outer_cells, n_pf_mu)),
        np.frombuffer(data.x_grid[2][0].data(), dtype=np.float32, count=data.x_grid[2][0].size()).reshape((n_tau, n_outer_cells, n_outer_cells, n_pf_chHad)),
        np.frombuffer(data.x_grid[3][0].data(), dtype=np.float32, count=data.x_grid[3][0].size()).reshape((n_tau, n_outer_cells, n_outer_cells, n_pf_nHad)),
        np.frombuffer(data.x_grid[4][0].data(), dtype=np.float32, count=data.x_grid[4][0].size()).reshape((n_tau, n_outer_cells, n_outer_cells, n_pf_gamma)),
        np.frombuffer(data.x_grid[5][0].data(), dtype=np.float32, count=data.x_grid[5][0].size()).reshape((n_tau, n_outer_cells, n_outer_cells, n_ele)),
        np.frombuffer(data.x_grid[6][0].data(), dtype=np.float32, count=data.x_grid[6][0].size()).reshape((n_tau, n_outer_cells, n_outer_cells, n_muon)),
        np.frombuffer(data.x_grid[0][1].data(), dtype=np.float32, count=data.x_grid[0][1].size()).reshape((n_tau, n_inner_cells, n_inner_cells, n_pf_el)),
        np.frombuffer(data.x_grid[1][1].data(), dtype=np.float32, count=data.x_grid[1][1].size()).reshape((n_tau, n_inner_cells, n_inner_cells, n_pf_mu)),
        np.frombuffer(data.x_grid[2][1].data(), dtype=np.float32, count=data.x_grid[2][1].size()).reshape((n_tau, n_inner_cells, n_inner_cells, n_pf_chHad)),
        np.frombuffer(data.x_grid[3][1].data(), dtype=np.float32, count=data.x_grid[3][1].size()).reshape((n_tau, n_inner_cells, n_inner_cells, n_pf_nHad)),
        np.frombuffer(data.x_grid[4][1].data(), dtype=np.float32, count=data.x_grid[4][1].size()).reshape((n_tau, n_inner_cells, n_inner_cells, n_pf_gamma)),
        np.frombuffer(data.x_grid[5][1].data(), dtype=np.float32, count=data.x_grid[5][1].size()).reshape((n_tau, n_inner_cells, n_inner_cells, n_ele)),
        np.frombuffer(data.x_grid[6][1].data(), dtype=np.float32, count=data.x_grid[6][1].size()).reshape((n_tau, n_inner_cells, n_inner_cells, n_muon))
    ]

from queue import Queue
data_que = Queue(maxsize = n_batches_store)

times = []
for i in range(n_batches):
    start = time.time()

    data_loader.MoveNext()
    data = data_loader.LoadData()

    if data_que.full():
        data_que.get()

    data_np = to_tensor(data)
    data_que.put((data, data_np))

    end = time.time()
    print(i, " end: ",end-start, ' s.')
    times.append(end-start)

    for n in range(17):
        if np.isnan(data_que.queue[0][1][n]).any():
            print("Nan detected! element=",n)
            print(np.argwhere(np.isnan(data_que.queue[0][1][n])))
        if np.isinf(data_que.queue[0][1][n]).any():
            print("Inf detected! element=",n)
            print(np.argwhere(np.isinf(data_que.queue[0][1][n])))
        if np.amax(data_que.queue[0][1][n])==0:
            print("Empty tuple detected! element=",n)

from statistics import mean
print("Mean time: ", mean(times))

time_arr = np.asarray(times)
np.savetxt("dataloader.csv", time_arr, delimiter=",")
