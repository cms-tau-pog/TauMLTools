# The environment on Centos 7 is:
# source /cvmfs/sft.cern.ch/lcg/views/setupViews.sh LCG_99 x86_64-centos7-gcc10-opt
import ROOT as R
import numpy as np
import ctypes
import time

n_tau          = 500
n_inner_cells  = 11
n_outer_cells  = 21
n_fe_tau    = 43
n_pf_el     = 22
n_pf_mu     = 23
n_pf_chHad  = 27
n_pf_nHad   = 7
n_pf_gamma  = 23
n_ele       = 37
n_muon      = 37
tau_types   = 6

R.gROOT.ProcessLine(".include /afs/cern.ch/user/m/myshched/DeepTau/CMSSW_10_6_17/src")
R.gInterpreter.ProcessLine('#include "../interface/DataLoader_setup.h"')
R.gInterpreter.ProcessLine('#include "../interface/DataLoader_main.h"')

data_loader = R.DataLoader()
print("maximum batch number: ",data_loader.GetMaxBatchNumber())

times = []
for i in range(1000):
    start = time.time()

    if not(data_loader.HasNext()):
        break
    data = data_loader.LoadNext()

    y_hot = np.asarray(data.y_onehot).reshape((n_tau, tau_types))
    weights = np.asarray(data.weight)
    x_tau = np.asarray(data.x_tau).reshape((n_tau, n_fe_tau))

    x_grid_outer1 = np.asarray(data.x_grid[1][0]).reshape((n_tau, n_outer_cells, n_outer_cells, n_pf_mu))
    x_grid_outer2 = np.asarray(data.x_grid[0][0]).reshape((n_tau, n_outer_cells, n_outer_cells, n_pf_el))
    x_grid_outer3 = np.asarray(data.x_grid[2][0]).reshape((n_tau, n_outer_cells, n_outer_cells, n_pf_chHad))
    x_grid_outer4 = np.asarray(data.x_grid[3][0]).reshape((n_tau, n_outer_cells, n_outer_cells, n_pf_nHad))
    x_grid_outer5 = np.asarray(data.x_grid[4][0]).reshape((n_tau, n_outer_cells, n_outer_cells, n_pf_gamma))
    x_grid_outer6 = np.asarray(data.x_grid[5][0]).reshape((n_tau, n_outer_cells, n_outer_cells, n_ele))
    x_grid_outer7 = np.asarray(data.x_grid[6][0]).reshape((n_tau, n_outer_cells, n_outer_cells, n_muon))

    x_grid_inner1 = np.asarray(data.x_grid[0][1]).reshape((n_tau, n_inner_cells, n_inner_cells, n_pf_el))
    x_grid_inner1 = np.asarray(data.x_grid[1][1]).reshape((n_tau, n_inner_cells, n_inner_cells, n_pf_mu))
    x_grid_inner1 = np.asarray(data.x_grid[2][1]).reshape((n_tau, n_inner_cells, n_inner_cells, n_pf_chHad))
    x_grid_inner1 = np.asarray(data.x_grid[3][1]).reshape((n_tau, n_inner_cells, n_inner_cells, n_pf_nHad))
    x_grid_inner1 = np.asarray(data.x_grid[4][1]).reshape((n_tau, n_inner_cells, n_inner_cells, n_pf_gamma))
    x_grid_inner1 = np.asarray(data.x_grid[5][1]).reshape((n_tau, n_inner_cells, n_inner_cells, n_ele))
    x_grid_inner1 = np.asarray(data.x_grid[6][1]).reshape((n_tau, n_inner_cells, n_inner_cells, n_muon))

    end = time.time()
    print("end: ",end-start, ' s.')
    times.append(end-start)

from statistics import mean
print("Mean time: ", mean(times[1:]))
