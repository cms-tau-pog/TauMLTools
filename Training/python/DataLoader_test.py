# The environment on Centos 7 is:
# source /cvmfs/sft.cern.ch/lcg/views/SetupViews.sh LCG_99 x86_64-centos7-gcc10-opt
import ROOT as R
import numpy as np
import ctypes
import time
import sys
import enumecg
import json_parse

R.gROOT.ProcessLine(".include ../../..")
R.gInterpreter.ProcessLine(json_parse.create_settings("../configs/training_v1.json"))
R.gInterpreter.ProcessLine('#include "../interface/DataLoader_main.h"')

n_tau          = R.Setup.n_tau
n_inner_cells  = R.Setup.n_inner_cells
n_outer_cells  = R.Setup.n_outer_cells
n_fe_tau    = R.Setup.n_fe_tau
n_pf_el     = R.Setup.n_pf_el
n_pf_mu     = R.Setup.n_pf_mu
n_pf_chHad  = R.Setup.n_pf_chHad
n_pf_nHad   = R.Setup.n_pf_nHad
n_pf_gamma  = R.Setup.n_pf_gamma
n_ele       = R.Setup.n_ele
n_muon      = R.Setup.n_muon
tau_types   = R.Setup.tau_types

data_loader = R.DataLoader()
print("Number of all entries / batch size: ", data_loader.GetEntries()/n_tau)

n_batches = 2000
n_batches_store = 10

# To convert list to tensor
def to_tensor(data_list):

    length = len(data_list)

    y = np.zeros((length,n_tau,tau_types))
    w = np.zeros((length,n_tau))
    x_tau = np.zeros((length,n_tau,n_fe_tau))

    x_outer_1 = np.zeros((length, n_tau, n_outer_cells, n_outer_cells, n_pf_el))
    x_outer_2 = np.zeros((length, n_tau, n_outer_cells, n_outer_cells, n_pf_mu))
    x_outer_3 = np.zeros((length, n_tau, n_outer_cells, n_outer_cells, n_pf_chHad))
    x_outer_4 = np.zeros((length, n_tau, n_outer_cells, n_outer_cells, n_pf_nHad))
    x_outer_5 = np.zeros((length, n_tau, n_outer_cells, n_outer_cells, n_pf_gamma))
    x_outer_6 = np.zeros((length, n_tau, n_outer_cells, n_outer_cells, n_ele))
    x_outer_7 = np.zeros((length, n_tau, n_outer_cells, n_outer_cells, n_muon))

    x_inner_1 = np.zeros((length, n_tau, n_inner_cells, n_inner_cells, n_pf_el))
    x_inner_2 = np.zeros((length, n_tau, n_inner_cells, n_inner_cells, n_pf_mu))
    x_inner_3 = np.zeros((length, n_tau, n_inner_cells, n_inner_cells, n_pf_chHad))
    x_inner_4 = np.zeros((length, n_tau, n_inner_cells, n_inner_cells, n_pf_nHad))
    x_inner_5 = np.zeros((length, n_tau, n_inner_cells, n_inner_cells, n_pf_gamma))
    x_inner_6 = np.zeros((length, n_tau, n_inner_cells, n_inner_cells, n_ele))
    x_inner_7 = np.zeros((length, n_tau, n_inner_cells, n_inner_cells, n_muon))

    for i, data in enumerate(data_list):

        y[i] = np.frombuffer(data.y_onehot.data(), dtype=np.float32, count=data.y_onehot.size()).reshape((n_tau, tau_types))
        w[i] = np.frombuffer(data.weight.data(), dtype=np.float32, count=data.weight.size())
        x_tau[i] = np.frombuffer(data.x_tau.data(), dtype=np.float32, count=data.x_tau.size()).reshape((n_tau,n_fe_tau))

        x_outer_1[i] = np.frombuffer(data.x_grid[0][0].data(), dtype=np.float32, count=data.x_grid[0][0].size()).reshape((n_tau, n_outer_cells, n_outer_cells, n_pf_el))
        x_outer_2[i] = np.frombuffer(data.x_grid[1][0].data(), dtype=np.float32, count=data.x_grid[1][0].size()).reshape((n_tau, n_outer_cells, n_outer_cells, n_pf_mu))
        x_outer_3[i] = np.frombuffer(data.x_grid[2][0].data(), dtype=np.float32, count=data.x_grid[2][0].size()).reshape((n_tau, n_outer_cells, n_outer_cells, n_pf_chHad))
        x_outer_4[i] = np.frombuffer(data.x_grid[3][0].data(), dtype=np.float32, count=data.x_grid[3][0].size()).reshape((n_tau, n_outer_cells, n_outer_cells, n_pf_nHad))
        x_outer_5[i] = np.frombuffer(data.x_grid[4][0].data(), dtype=np.float32, count=data.x_grid[4][0].size()).reshape((n_tau, n_outer_cells, n_outer_cells, n_pf_gamma))
        x_outer_6[i] = np.frombuffer(data.x_grid[5][0].data(), dtype=np.float32, count=data.x_grid[5][0].size()).reshape((n_tau, n_outer_cells, n_outer_cells, n_ele))
        x_outer_7[i] = np.frombuffer(data.x_grid[6][0].data(), dtype=np.float32, count=data.x_grid[6][0].size()).reshape((n_tau, n_outer_cells, n_outer_cells, n_muon))

        x_inner_1[i] = np.frombuffer(data.x_grid[0][1].data(), dtype=np.float32, count=data.x_grid[0][1].size()).reshape((n_tau, n_inner_cells, n_inner_cells, n_pf_el))
        x_inner_2[i] = np.frombuffer(data.x_grid[1][1].data(), dtype=np.float32, count=data.x_grid[1][1].size()).reshape((n_tau, n_inner_cells, n_inner_cells, n_pf_mu))
        x_inner_3[i] = np.frombuffer(data.x_grid[2][1].data(), dtype=np.float32, count=data.x_grid[2][1].size()).reshape((n_tau, n_inner_cells, n_inner_cells, n_pf_chHad))
        x_inner_4[i] = np.frombuffer(data.x_grid[3][1].data(), dtype=np.float32, count=data.x_grid[3][1].size()).reshape((n_tau, n_inner_cells, n_inner_cells, n_pf_nHad))
        x_inner_5[i] = np.frombuffer(data.x_grid[4][1].data(), dtype=np.float32, count=data.x_grid[4][1].size()).reshape((n_tau, n_inner_cells, n_inner_cells, n_pf_gamma))
        x_inner_6[i] = np.frombuffer(data.x_grid[5][1].data(), dtype=np.float32, count=data.x_grid[5][1].size()).reshape((n_tau, n_inner_cells, n_inner_cells, n_ele))
        x_inner_7[i] = np.frombuffer(data.x_grid[6][1].data(), dtype=np.float32, count=data.x_grid[6][1].size()).reshape((n_tau, n_inner_cells, n_inner_cells, n_muon))

    return  y, w, x_tau, \
            x_outer_1, x_outer_2, x_outer_3, x_outer_4, x_outer_5, x_outer_6, x_outer_7, \
            x_inner_1, x_inner_2, x_inner_3, x_inner_4, x_inner_5, x_inner_6, x_inner_7

from queue import Queue
data_que = Queue(maxsize = n_batches_store)

times = []
for i in range(n_batches):
    start = time.time()

    data_loader.MoveNext()
    data = data_loader.LoadData()

    if data_que.full():
        data_que.get()
    data_que.put(data)


    end = time.time()
    print(i, " end: ",end-start, ' s.')
    times.append(end-start)

from statistics import mean
print("Mean time: ", mean(times))

time_arr = np.asarray(times)
np.savetxt("dataloader.csv", time_arr, delimiter=",")
