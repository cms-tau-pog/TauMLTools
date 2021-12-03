import torch.multiprocessing as mp
from queue import Empty as EmptyException
from queue import Full as FullException

import math
import numpy as np
import ROOT as R
import config_parse
import tensorflow as tf
import torch
import os
import time

# class TerminateGenerator:
#     pass

def torch_to_tf(return_truth = True, return_weights = True):

    def with_both(X):
        return tuple([tuple([x.clone().numpy() for x in X[0]]),
                      X[1].clone().numpy(),
                      X[2].clone().numpy()])

    def with_truth(X):
        return tuple([tuple([x.clone().numpy() for x in X[0]]),
                     X[1].clone().numpy()])

    if return_truth and return_weights:
        return with_both
    elif return_truth:
        return with_truth
    else:
        raise RuntimeError("Error: conversion rule from torch.tensor is unknown!")

def ugly_clean(queue):
    while True:
        try:
            _ = queue.get_nowait()
        except EmptyException:
            time.sleep(0.2)
            if queue.qsize()==0:
                break
    if queue.qsize()!=0:
        raise RuntimeError("Error: queue was not clean properly.")


class QueueEx:
    def __init__(self, max_size=0, max_n_puts=math.inf):
        self.n_puts = mp.Value('i', 0)
        self.max_n_puts = max_n_puts
        if self.max_n_puts < 0:
            self.max_n_puts = math.inf
        self.mp_queue = mp.Queue(max_size)

    def put(self, item, retry_interval=0.3):
        while True:
            with self.n_puts.get_lock():
                if self.n_puts.value >= self.max_n_puts:
                    return False
                try:
                    self.mp_queue.put(item, False)
                    self.n_puts.value += 1
                    return True
                except FullException:
                    pass
            time.sleep(retry_interval)
    
    def put_terminate(self, value):
        self.mp_queue.put(value)
        
    def get(self):
        return self.mp_queue.get()
    
    def clear(self):
        while not self.mp_queue.empty():
            self.mp_queue.get()

class DataSource:
    def __init__(self, queue_files):
        self.data_loader = R.DataLoader()
        self.queue_files = queue_files
        self.require_file = True
    
    def get(self):
        while True:
            if self.require_file:
                try:
                    file_name = self.queue_files.get(False)
                    self.data_loader.ReadFile(R.std.string(file_name), 0, -1)
                    self.require_file = False
                except EmptyException:
                    return None

            if self.data_loader.MoveNext():
                return self.data_loader.LoadData()
            else:
                self.require_file = True

class DataLoaderBase:

    @staticmethod
    def compile_classes(config, file_scaling, dataloader_core):

        _rootpath = os.path.abspath(os.path.dirname(__file__)+"/../../..")
        R.gROOT.ProcessLine(".include "+_rootpath)

        if not os.path.isfile(file_scaling):
            raise RuntimeError("file_scaling do not exist")

        if not(os.path.isfile(_rootpath+"/"+dataloader_core)):
            raise RuntimeError("c++ dataloader does not exist")

        # compilation should be done in corresponding order:
        print("Compiling DataLoader headers.")
        R.gInterpreter.Declare(config_parse.create_scaling_input(file_scaling, config, verbose=False))
        R.gInterpreter.Declare(config_parse.create_settings(config, verbose=False))
        R.gInterpreter.Declare('#include "{}"'.format(dataloader_core))
        R.gInterpreter.Declare('#include "TauMLTools/Core/interface/exception.h"')

class GetData():

    @staticmethod
    def getdata(_obj_f,
                _reshape,
                _dtype=np.float32):
        x = np.copy(np.frombuffer(_obj_f.data(), dtype=_dtype, count=_obj_f.size()))
        if np.isnan(x).any():
            print("Nan detected! element=",x.shape)
            print(np.argwhere(np.isnan(x)))
            raise RuntimeError("Terminate: nans detected in the tensor.")
        return torch.from_numpy(x) if _reshape==-1 else torch.reshape(torch.from_numpy(x), _reshape)

    @staticmethod
    def getgrid(_obj_grid,
                batch_size,
                n_grid_features,
                input_grids,
                _n_cells,
                _inner):
        _X = []
        for group in input_grids:
            _X.append(
                torch.cat(
                    [ __class__.getdata(_obj_grid[ getattr(R.CellObjectType,fname) ][_inner],
                     (batch_size, _n_cells, _n_cells, n_grid_features[fname])) for fname in group ],
                    dim=-1
                    )
                )
        return _X
    
    @staticmethod
    def getsequence(_obj_grid,
                    _n_tau,
                    _input_grids,
                    _n_seq,
                    _n_features):
        return [ __class__.getdata(_obj_grid[getattr(R.CellObjectType,group)],
                (_n_tau, _n_seq[group], _n_features[group]))
                for group in _input_grids]

    @staticmethod
    def getX(data,
            batch_size,
            n_grid_features,
            n_flat_features,
            input_grids,
            n_inner_cells,
            n_outer_cells):        
        # Flat Tau features
        X_all = [ __class__.getdata(data.x_tau, (batch_size, n_flat_features)) ]
        # Inner grid
        X_all += __class__.getgrid(data.x_grid, batch_size, n_grid_features,
                                   input_grids, n_inner_cells, True) # 500 11 11 176
        # Outer grid
        X_all += __class__.getgrid(data.x_grid, batch_size, n_grid_features,
                                   input_grids, n_outer_cells, False) # 500 11 11 176
        return X_all
