import gc
import multiprocessing as mp

# from pathos.helpers import mp
# from mp import Process,Queue
# from faster_fifo import Queue
# from queue import Full, Empty
# https://stackoverflow.com/questions/47085458/why-is-multiprocessing-queue-get-so-slow
# https://pypi.org/project/quick-queue/
# from quick_queue import QQueue as Queue

import numpy as np
import ROOT as R
import config_parse
import tensorflow as tf
import os
import yaml
import copy
import time

R.gROOT.Reset()
class TerminateGenerator:
    pass

def LoaderThread(queue_out, queue_files, input_grids, batch_size, n_inner_cells, n_outer_cells,
                 n_flat_features, n_grid_features, tau_types, return_truth, return_weights):

    def getdata(_obj_f, _reshape, _dtype=np.float32):
        x = np.copy(np.frombuffer(_obj_f.data(), dtype=_dtype, count=_obj_f.size()))
        return x if _reshape==-1 else x.reshape(_reshape)
    
    def getgrid(_obj_grid, _inner):
        _n_cells = n_inner_cells if _inner else n_outer_cells
        _X = []
        for group in input_grids:
            _X.append(tf.convert_to_tensor(
                np.concatenate(
                    [ getdata(_obj_grid[ getattr(R.CellObjectType,fname) ][_inner],
                     (batch_size, _n_cells, _n_cells, n_grid_features[fname])) for fname in group ],
                    axis=-1
                    ),dtype=tf.float32)
                )
        return _X

    _dl_worker = R.DataLoader()
    _req_file = True

    while True:

        if _req_file:
            try:
                _filename = queue_files.get(False)
                _dl_worker.ReadFile(R.std.string(_filename), 0, -1)
                _req_file = False
                continue
            except: # Queue.Empty
                queue_out.put(TerminateGenerator())
                break

        if not _dl_worker.MoveNext():
            _req_file = True
            continue
        
        data = _dl_worker.LoadData()
        # Flat Tau features
        X_all = [tf.convert_to_tensor(getdata(data.x_tau, (batch_size, n_flat_features)))]
        # Inner grid
        X_all += getgrid(data.x_grid, 1) # 500 11 11 176
        # Outer grid
        X_all += getgrid(data.x_grid, 0) # 500 21 21 176

        # X_all = tuple(X_all)

        if return_weights:
            weights = getdata(data.weight, -1)
        if return_truth:
            Y = getdata(data.y_onehot, (batch_size, tau_types))

        if return_truth and return_weights:
            item = (X_all, Y, weights)
        elif return_truth:
            item = (X_all, Y)
        elif return_weights:
            item = (X_all, weights)
        else:
            item = X_all

        queue_out.put(item)

class DataLoader:

    @staticmethod
    def ListToVector(l, elem_type):
        vec = R.std.vector(elem_type)()
        for elem in l:
            vec.push_back(elem)
        return vec

    @staticmethod
    def compile_classes(file_config, file_scaling):

        _rootpath = os.path.abspath(os.path.dirname(__file__)+"/../../..")
        R.gROOT.ProcessLine(".include "+_rootpath)

        _LOADPATH = "TauMLTools/Training/interface/DataLoader_main.h"

        if not(os.path.isfile(file_config) \
           and os.path.isfile(file_scaling)):
            raise RuntimeError("file_config or file_scaling do not exist")

        if not(os.path.isfile(_rootpath+"/"+_LOADPATH)):
            raise RuntimeError("c++ dataloader does not exist")

        # compilation should be done in corresponding order:
        print("Compiling DataLoader headers.")
        R.gInterpreter.ProcessLine(config_parse.create_scaling_input(file_scaling))
        R.gInterpreter.ProcessLine(config_parse.create_settings(file_config))
        R.gInterpreter.ProcessLine('#include "{}"'.format(_LOADPATH))


    def __init__(self, file_config, file_scaling):

        DataLoader.compile_classes(file_config, file_scaling)

        with open(file_config) as file:
            self.config = yaml.safe_load(file)

        self.n_grid_features = {}
        for celltype in self.config["Features_all"]:
            if celltype!="TauFlat":
                self.n_grid_features[str(celltype)] = len(self.config["Features_all"][celltype])

        self.n_flat_features = len(self.config["Features_all"]["TauFlat"])

        # global variables after compile are read out here 
        self.batch_size     = self.config["Setup"]["n_tau"]
        self.n_inner_cells  = self.config["Setup"]["n_inner_cells"]
        self.n_outer_cells  = self.config["Setup"]["n_outer_cells"]
        self.tau_types      = len(self.config["Setup"]["tau_types_names"])
        self.n_load_workers   = self.config["SetupNN"]["n_load_workers"]
        self.n_batches        = self.config["SetupNN"]["n_batches"]
        self.n_batches_val    = self.config["SetupNN"]["n_batches_val"]
        self.validation_split = self.config["SetupNN"]["validation_split"]
        self.max_queue_size   = self.config["SetupNN"]["max_queue_size"]
        self.n_epochs         = self.config["SetupNN"]["n_epochs"]
        self.epoch         = self.config["SetupNN"]["epoch"]
        self.input_grids        = self.config["SetupNN"]["input_grids"]
        self.n_cells = { 'inner': self.n_inner_cells, 'outer': self.n_outer_cells }

        data_files = []
        for root, dirs, files in os.walk(os.path.abspath(self.config["Setup"]["input_dir"])):
            for file in files:
                data_files.append(os.path.join(root, file))

        self.train_files, self.val_files = \
             np.split(data_files, [int(len(data_files)*(1-self.validation_split))])
            
        if len(self.train_files) == 0:
            raise RuntimeError("Taining file queue is empty.")
        if len(self.val_files) == 0:
            raise RuntimeError("Validation file queue is empty.")

        print("Files for training:", len(self.train_files))
        print("Files for validation:", len(self.val_files))


    def get_generator(self, primary_set = True, return_truth = True, return_weights = False):

        _files = self.train_files if primary_set else self.val_files
        print("Number of workers in DataLoader: ", self.n_load_workers)

        def _generator():

            finish_counter = 0
            queue_files = mp.Queue()
            [ queue_files.put(file) for file in _files ]
            queue_out = mp.Queue(self.max_queue_size)

            processes = []
            for i in range(self.n_load_workers):
                processes.append(
                mp.Process(target = LoaderThread, 
                        args = (queue_out, queue_files, self.input_grids,
                                self.batch_size, self.n_inner_cells, self.n_outer_cells,
                                self.n_flat_features, self.n_grid_features, self.tau_types,
                                return_truth, return_weights)))
                processes[-1].deamon = True
                processes[-1].start()


            while True:
                item = queue_out.get()
                if isinstance(item, TerminateGenerator):
                    finish_counter+=1
                else:
                    yield item
                if finish_counter == self.n_load_workers:
                    break
                    
            for pr in processes:
                pr.join()
            gc.collect()

        return _generator


    def get_config(self):

        def get_branches(config, group):
            return list(set(config["Features_all"][group]) - \
                        set(config["Features_disable"][group]))

        # copy of feature lists is not necessery
        input_tau_branches = get_branches(self.config,"TauFlat")
        input_cell_pfCand_ele_branches = get_branches(self.config,"PfCand_electron")
        input_cell_pfCand_muon_branches = get_branches(self.config,"PfCand_muon")
        input_cell_pfCand_chHad_branches = get_branches(self.config,"PfCand_chHad")
        input_cell_pfCand_nHad_branches = get_branches(self.config,"PfCand_nHad")
        input_cell_pfCand_gamma_branches = get_branches(self.config,"PfCand_gamma")
        input_cell_ele_branches = get_branches(self.config,"Electron")
        input_cell_muon_branches = get_branches(self.config,"Muon")

        # code below is a shortened copy of common.py
        class NetConf:
            def __init__(self, name, final, tau_branches, cell_locations, component_names, component_branches):
                self.name = name
                self.final = final
                self.tau_branches = tau_branches
                self.cell_locations = cell_locations
                self.comp_names = component_names
                self.comp_branches = component_branches

        netConf_preTau = NetConf("preTau", False, input_tau_branches, [], [], [])
        netConf_preInner = NetConf("preInner", False, [], ['inner'], ['egamma', 'muon', 'hadrons'], [
            input_cell_pfCand_ele_branches + input_cell_ele_branches + input_cell_pfCand_gamma_branches,
            input_cell_pfCand_muon_branches + input_cell_muon_branches,
            input_cell_pfCand_chHad_branches + input_cell_pfCand_nHad_branches
        ])
        # netConf_preTauInter = NetConf("preTauInter", False, netConf_preTau.tau_branches, netConf_preInner.cell_locations,
        #                             netConf_preInner.comp_names, netConf_preInner.comp_branches)
        netConf_preOuter = NetConf("preOuter", False, [], ['outer'], netConf_preInner.comp_names,
                                netConf_preInner.comp_branches)
        netConf_full = NetConf("full", True, netConf_preTau.tau_branches,
                            netConf_preInner.cell_locations + netConf_preOuter.cell_locations,
                            netConf_preInner.comp_names, netConf_preInner.comp_branches)

        # Input tensor shape and type 
        input_shape, input_types = [], []
        input_shape.append(tuple([None, len(get_branches(self.config,"TauFlat"))]))
        input_types.append(tf.float32)
        for grid in ["inner","outer"]:
            for f_group in self.input_grids:
                n_f = sum([len(get_branches(self.config,cell_type)) for cell_type in f_group])
                input_shape.append(tuple([None, self.n_cells[grid], self.n_cells[grid], n_f]))
                input_types.append(tf.float32)
        input_shape = tuple([tuple(input_shape),(None, self.tau_types)])
        input_types = tuple([tuple(input_types),(tf.float32)])

        return netConf_full, input_shape, input_types
