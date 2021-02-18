# The environment on Centos 7 is:
# source /cvmfs/sft.cern.ch/lcg/views/setupViews.sh LCG_99 x86_64-centos7-gcc10-opt
import ROOT as R
import numpy as np
import ctypes
import time
import sys

R.gROOT.ProcessLine(".include ../../..")
R.gInterpreter.ProcessLine('#include "../interface/DataLoader_setup.h"')
R.gInterpreter.ProcessLine('#include "../interface/DataLoader_main.h"')

n_tau          = R.setup.n_tau
n_inner_cells  = R.setup.n_inner_cells
n_outer_cells  = R.setup.n_outer_cells
n_fe_tau    = R.setup.n_fe_tau
n_pf_el     = R.setup.n_pf_el
n_pf_mu     = R.setup.n_pf_mu
n_pf_chHad  = R.setup.n_pf_chHad
n_pf_nHad   = R.setup.n_pf_nHad
n_pf_gamma  = R.setup.n_pf_gamma
n_ele       = R.setup.n_ele
n_muon      = R.setup.n_muon
tau_types   = R.setup.tau_types

data_loader = R.DataLoader()
print("Max. batch number: ",data_loader.GetMaxBatchNumber())

n_batches = 2000
n_batches_store = 10

# NN intput data
y = np.zeros((n_batches_store,n_tau,tau_types))
w = np.zeros((n_batches_store,n_tau))
x_tau = np.zeros((n_batches,n_tau,n_fe_tau))

x_outer_1 = np.zeros((n_batches_store, n_tau, n_outer_cells, n_outer_cells, n_pf_el))
x_outer_2 = np.zeros((n_batches_store, n_tau, n_outer_cells, n_outer_cells, n_pf_mu))
x_outer_3 = np.zeros((n_batches_store, n_tau, n_outer_cells, n_outer_cells, n_pf_chHad))
x_outer_4 = np.zeros((n_batches_store, n_tau, n_outer_cells, n_outer_cells, n_pf_nHad))
x_outer_5 = np.zeros((n_batches_store, n_tau, n_outer_cells, n_outer_cells, n_pf_gamma))
x_outer_6 = np.zeros((n_batches_store, n_tau, n_outer_cells, n_outer_cells, n_ele))
x_outer_7 = np.zeros((n_batches_store, n_tau, n_outer_cells, n_outer_cells, n_muon))

x_inner_1 = np.zeros((n_batches_store, n_tau, n_inner_cells, n_inner_cells, n_pf_el))
x_inner_2 = np.zeros((n_batches_store, n_tau, n_inner_cells, n_inner_cells, n_pf_mu))
x_inner_3 = np.zeros((n_batches_store, n_tau, n_inner_cells, n_inner_cells, n_pf_chHad))
x_inner_4 = np.zeros((n_batches_store, n_tau, n_inner_cells, n_inner_cells, n_pf_nHad))
x_inner_5 = np.zeros((n_batches_store, n_tau, n_inner_cells, n_inner_cells, n_pf_gamma))
x_inner_6 = np.zeros((n_batches_store, n_tau, n_inner_cells, n_inner_cells, n_ele))
x_inner_7 = np.zeros((n_batches_store, n_tau, n_inner_cells, n_inner_cells, n_muon))

def to_tensor(data, i):

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

data_list = []
times = []
for i in range(n_batches):
    start = time.time()

    if not(data_loader.HasNext()):
        break
    data = data_loader.LoadNext()
    to_tensor(data, i % n_batches_store)

    end = time.time()
    print(i, " end: ",end-start, ' s.')
    times.append(end-start)

from statistics import mean
print("Mean time: ", mean(times))

time_arr = np.asarray(times)
np.savetxt("dataloaderBM_fulltest.csv", time_arr, delimiter=",")
