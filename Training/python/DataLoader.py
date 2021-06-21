import gc
from multiprocessing import Process, Queue
import numpy as np
import ROOT as R
import config_parse
import tensorflow as tf
import os
import yaml

R.gROOT.Reset()

def LoaderThread(data_loader, queue, input_grids, batch_size, n_inner_cells, n_outer_cells,
                 n_flat_features, n_grid_features, tau_types, return_truth, return_weights, steps_per_epoch):
    
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
                     (batch_size, _n_cells, _n_cells, n_grid_features[fname])) for fname in group ],
                    axis=-1
                    )
                )
        return _X

    current_epoch_step = 0

    while data_loader.MoveNext() and current_epoch_step < steps_per_epoch:

        data = data_loader.LoadData()

        # Flat Tau features
        X_all = [getdata(data.x_tau, (batch_size, n_flat_features))]
        # Inner grid
        X_all += getgrid(data.x_grid, 1) # 500 11 11 176
        # Outer grid
        X_all += getgrid(data.x_grid, 0) # 500 21 21 176

        X_all = tuple(X_all)

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

        queue.put(item)
        current_epoch_step += 1


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
        self.train_files, self.val_files = list(self.train_files), list(self.val_files)

        if len(self.train_files) == 0:
            raise RuntimeError("Taining file list is empty.")
        if len(self.val_files) == 0:
            raise RuntimeError("Validation file list is empty.")

        print("Files for training:",len(self.train_files))
        print("Files for validation:",len(self.val_files))


    def get_generator(self, primary_set = True, return_truth = True, return_weights = False):

        if primary_set:
            _batch_loader = R.DataLoader(DataLoader.ListToVector(self.train_files,'string'))
            _batch_size = self.n_batches
        else:
            _batch_loader = R.DataLoader(DataLoader.ListToVector(self.val_files,'string'))
            _batch_size = self.n_batches_val

        _steps_per_epoch = int(_batch_loader.GetEntries() / self.batch_size)
        _steps_per_epoch = _steps_per_epoch if _batch_size == -1 else min(_steps_per_epoch, _batch_size)

        def _generator():

            queue = Queue(maxsize=self.max_queue_size)
            current_pass = self.epoch
            while self.n_passes < 0 or current_pass < self.n_epochs:
                _batch_loader.reset()

                process = Process(target=LoaderThread, 
                           args=(  _batch_loader, queue, self.input_grids,
                                self.batch_size, self.n_inner_cells, self.n_outer_cells,
                                self.n_flat_features, self.n_grid_features, self.tau_types,
                                return_truth, return_weights, _steps_per_epoch))
                process.daemon = True
                process.start()

                for step_id in range(_steps_per_epoch):
                    item = queue.get()
                    yield item
                
                process.join()
                gc.collect()
                current_pass += 1

        return _generator, _steps_per_epoch


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