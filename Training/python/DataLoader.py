import gc
import multiprocessing as mp
from queue import Empty as EmptyException
from queue import Full as FullException

import math
import numpy as np
import ROOT as R
import config_parse
import tensorflow as tf
import os
import yaml
import time

class TerminateGenerator:
    pass

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
    
    def put_terminate(self):
        self.mp_queue.put(TerminateGenerator())
        
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

def LoaderThread(queue_out, queue_files, batch_size, pfCand_n, pfCand_fn,
                 output_classes, return_truth, return_weights):

    def getdata(_obj_f, _reshape, _dtype=np.float32):
        return np.copy(np.frombuffer(_obj_f.data(),
                                    dtype=_dtype,
                                    count=_obj_f.size())).reshape(_reshape)

    data_source = DataSource(queue_files)
    put_next = True

    while put_next:

        data = data_source.get()
        if data is None:
            break


        X_all = getdata(data.x, (batch_size, pfCand_n, pfCand_fn))
        # if np.isnan(X_all).any() or np.isinf(X_all).any():
        #     print("Nan detected X!")
        #     continue

        # if return_weights:
        #     weights = getdata(data.weight, -1)
        if return_truth:
            Y = getdata(data.y, (batch_size, output_classes))

        # if np.isnan(Y).any() or np.isinf(Y).any():
        #     print("Nan detected Y!")
        #     continue
        # if return_truth and return_weights:
        #     item = (X_all, Y, weights)
        if return_truth:
            item = (X_all, Y)
        # elif return_weights:
        #     item = (X_all, weights)
        else:
            item = X_all
        
        put_next = queue_out.put(item)

    queue_out.put_terminate()

class DataLoader:

    @staticmethod
    def compile_classes(file_config, file_scaling):

        _rootpath = os.path.abspath(os.path.dirname(__file__)+"/../../..")
        R.gROOT.ProcessLine(".include "+_rootpath)

        _LOADPATH = "TauMLTools/Training/interface/DataLoaderReco_main.h"

        if not(os.path.isfile(file_config) \
           and os.path.isfile(file_scaling)):
            raise RuntimeError("file_config or file_scaling do not exist")

        if not(os.path.isfile(_rootpath+"/"+_LOADPATH)):
            raise RuntimeError("c++ dataloader does not exist")

        # compilation should be done in corresponding order:
        print("Compiling DataLoader headers.")
        R.gInterpreter.Declare(config_parse.create_scaling_input(file_scaling,file_config, verbose=False))
        R.gInterpreter.Declare(config_parse.create_settings(file_config, verbose=False))
        R.gInterpreter.Declare('#include "{}"'.format(_LOADPATH))
        R.gInterpreter.Declare('#include "TauMLTools/Core/interface/exception.h"')


    def __init__(self, file_config, file_scaling):

        self.compile_classes(file_config, file_scaling)

        with open(file_config) as file:
            self.config = yaml.safe_load(file)

        self.input_map = {} #[pfCand_type, feature, feature_int]
        for pfCand_type in self.config["CellObjectType"]:
            self.input_map[pfCand_type] = {}
            for f_dict in self.config["Features_all"][pfCand_type]:
                f = next(iter(f_dict))
                if f not in self.config["Features_disable"][pfCand_type]:
                    self.input_map[pfCand_type][f] = \
                        getattr(getattr(R,pfCand_type+"_Features"),f)


        # global variables after compile are read out here 
        self.batch_size       = self.config["Setup"]["n_tau"]
        self.output_n         = self.config["Setup"]["output_classes"]
        self.n_load_workers   = self.config["SetupNN"]["n_load_workers"]
        self.n_batches        = self.config["SetupNN"]["n_batches"]
        self.n_batches_val    = self.config["SetupNN"]["n_batches_val"]
        self.validation_split = self.config["SetupNN"]["validation_split"]
        self.max_queue_size   = self.config["SetupNN"]["max_queue_size"]
        self.n_epochs         = self.config["SetupNN"]["n_epochs"]
        self.epoch            = self.config["SetupNN"]["epoch"]
        self.sequence_len     = self.config["SequenceLength"]

        data_files = []
        for root, dirs, files in os.walk(os.path.abspath(self.config["Setup"]["input_dir"])):
            for file in files:
                data_files.append(os.path.join(root, file))

        self.train_files, self.val_files = \
             np.split(data_files, [int(len(data_files)*(1-self.validation_split))])

        print("Files for training:", len(self.train_files))
        print("Files for validation:", len(self.val_files))


    def get_generator(self, primary_set = True, return_truth = True, return_weights = False):

        _files = self.train_files if primary_set else self.val_files
        if len(_files)==0:
            raise RuntimeError(("Taining" if primary_set else "Validation")+\
                               " file list is empty.")

        n_batches = self.n_batches if primary_set else self.n_batches_val
        print("Number of workers in DataLoader: ", self.n_load_workers)

        def _generator():

            finish_counter = 0
            
            queue_files = mp.Queue()
            [ queue_files.put(file) for file in _files ]

            queue_out = QueueEx(max_size = self.max_queue_size, max_n_puts = n_batches)

            processes = []
            for i in range(self.n_load_workers):
                processes.append(
                mp.Process(target = LoaderThread, 
                        args = (queue_out, queue_files, self.batch_size,
                                self.sequence_len["PfCand"], len(self.input_map["PfCand"]),
                                self.output_n, return_truth, return_weights)))
                processes[-1].deamon = True
                processes[-1].start()

            while finish_counter < self.n_load_workers:
                
                item = queue_out.get()

                if isinstance(item, TerminateGenerator):
                    finish_counter+=1
                else:
                    yield item
                    
            ugly_clean(queue_files)
            queue_out.clear()

            for i, pr in enumerate(processes):
                pr.join()
            gc.collect()

        return _generator


    def get_config(self):

        '''
        At the moment get_config returns
        the config for PfCand sequence only.
        But this part is customizable
        '''
        input_shape = ( 
                        (self.batch_size, self.sequence_len["PfCand"], len(self.input_map["PfCand"])),
                        (self.batch_size, self.output_n)
                      )
        input_types = (tf.float32, tf.float32)

        return self.input_map["PfCand"], input_shape, input_types
