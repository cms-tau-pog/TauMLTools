import gc
from multiprocessing import Process, Queue
import numpy as np
import tensorflow as tf
import keras
import keras.backend as K
from keras import regularizers
from keras.models import Sequential, Model, load_model
from keras.layers import Input, Dense, Conv2D, Dropout, AlphaDropout, Activation, BatchNormalization, Flatten, \
                                    Concatenate, PReLU, TimeDistributed, LSTM, Masking
import ROOT as R
import config_parse
import os

R.gROOT.ProcessLine(".include ../../..")

def LoaderThread(data_loader, queue, batch_size, n_inner_cells, n_outer_cells,
                 n_flat_features, n_grid_features, tau_types, return_truth, return_weights, steps_per_epoch):

    def getdata(obj_f, reshape, dtype=np.float32):
        x = np.copy(np.frombuffer(obj_f.data(), dtype=dtype, count=obj_f.size()))
        return x if reshape==-1 else x.reshape(reshape)
    
    def getgrid(obj_grid, grid_features, inner):
        X_grid = []
        n_cells = n_inner_cells if inner else n_outer_cells
        for fname, n_f in grid_features:
            X_grid.append(getdata(obj_grid[ getattr(R.CellObjectType,fname) ][inner],
                                    (batch_size, n_cells, n_cells, n_f)))
        return X_grid

    current_epoch_step = 0

    while data_loader.MoveNext() and current_epoch_step < steps_per_epoch:

        data = data_loader.LoadData()

        # Test:
        X_ = [ getdata(data.x_tau, (batch_size, n_flat_features)).reshape(batch_size,-1) ]
        X_ += [ np.concatenate(getgrid(data.x_grid, n_grid_features, 0), axis=-1).reshape(batch_size,-1) ]
        X_ += [ np.concatenate(getgrid(data.x_grid, n_grid_features, 1), axis=-1).reshape(batch_size,-1) ]
        X_all = np.concatenate(X_,axis=1)

        # To be replaced with:
        # # Flat Tau features
        # X_all = [ getdata(data.x_tau, (batch_size, n_flat_features)) ]
        # # Outer grid
        # X_all += [ np.concatenate(getgrid(data.x_grid, n_grid_features, 0), axis=-1) ] # 500 21 21 176
        # # Inner grid
        # X_all += [ np.concatenate(getgrid(data.x_grid, n_grid_features, 1), axis=-1) ]

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
        
        if not(os.path.isfile(file_config) \
           and os.path.isfile(file_config) \
           and os.path.isfile("../interface/DataLoader_main.h")):
               raise RuntimeError("DataLoader file does not exists")

        # compilation should be done in corresponding order:
        print("Compiling DataLoader headers.")
        R.gInterpreter.ProcessLine(config_parse.create_scaling_input(file_scaling))
        R.gInterpreter.ProcessLine(config_parse.create_settings(file_config))
        R.gInterpreter.ProcessLine('#include "../interface/DataLoader_main.h"')


    def __init__(self, file_config, file_scaling):

        DataLoader.compile_classes(file_config, file_scaling)

        # global variables after compile are read out here 
        self.batch_size = R.Setup.n_tau
        self.n_batches  = R.Setup.n_batches
        self.n_batches_val  = R.Setup.n_batches_val
        self.n_inner_cells  = R.Setup.n_inner_cells
        self.n_outer_cells  = R.Setup.n_outer_cells
        self.n_flat_features = R.Setup.n_TauFlat     
        self.n_grid_features = [(str(cell), getattr(R.Setup, "n_"+str(cell))) for cell in R.Setup.CellObjectTypes]
        self.tau_types   = R.Setup.tau_types_names.size()

        self.validation_split = R.Setup.validation_split
        self.max_queue_size = R.Setup.max_queue_size
        self.n_passes = R.Setup.n_passes

        data_files = []
        for root, dirs, files in os.walk(os.path.abspath(R.Setup.input_dir)):
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
            current_pass = 0
            while self.n_passes < 0 or current_pass < self.n_passes:
                _batch_loader.reset()

                process = Process(target=LoaderThread, 
                           args=(  _batch_loader, queue, self.batch_size, self.n_inner_cells, self.n_outer_cells,
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


if __name__ == "__main__":

    # Create a TensorBoard callback
    logs = "logs/" + datetime.now().strftime("%Y%m%d-%H%M%S")
    tboard_callback = tf.keras.callbacks.TensorBoard(log_dir = logs,
                                                     profile_batch='1, 10')

    config   = "../configs/training_v1.yaml"
    scaling  = "../configs/scaling_test.json"
    dataloader = DataLoader(config, scaling)

    gen_train, n_steps_train = dataloader.get_generator(primary_set = True)
    gen_val, n_steps_val = dataloader.get_generator(primary_set = False)
    print("Training steps:",n_steps_train, "Validation steps:",n_steps_val)

    num_classes = 4
    input_shape = (98955,)

    data_train = tf.data.Dataset.from_generator(gen_train, (tf.float32, tf.int16), (tf.TensorShape([None,98955]), tf.TensorShape([None, num_classes])))
    data_val = tf.data.Dataset.from_generator(gen_val, (tf.float32, tf.int16), (tf.TensorShape([None,98955]), tf.TensorShape([None, num_classes])))

    model = keras.Sequential([
        keras.Input(shape=input_shape),
        Dense(100, activation="relu"),
        Dense(100, activation="relu"),
        Dense(100, activation="relu"),
        Dense(100, activation="relu"),
        Dense(num_classes, activation="softmax"),
        ])

    model.summary()

    model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
    model.fit(data_train, steps_per_epoch = n_steps_train,
              max_queue_size=1, validation_data=data_val,
              validation_steps = n_steps_val, epochs = 10,
              callbacks=[tboard_callback])