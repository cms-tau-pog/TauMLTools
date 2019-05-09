#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import gc
import sys
import glob
import time
import math
import numpy as np
import uproot
import pandas
from functools import partial
from concurrent.futures import ThreadPoolExecutor

import tensorflow as tf
import keras
import keras.backend as K
from keras import regularizers
from keras.models import Sequential, Model, load_model
from keras.layers import Input, Dense, Conv2D, Dropout, AlphaDropout, Activation, BatchNormalization, Flatten,                                     Concatenate, PReLU, TimeDistributed, LSTM, Masking, MaxPooling2D
from keras.callbacks import Callback, ModelCheckpoint, CSVLogger

sys.path.insert(0, "..")
from common import *
#from t_notify import Notify
from DataLoader import DataLoader, read_hdf_lock


config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
K.set_session(sess)

# In[2]:


class MaskedDense(Dense):
    def __init__(self, units, **kwargs):
        super(MaskedDense, self).__init__(units, **kwargs)

    def call(self, inputs, mask=None):
        base_out = super(MaskedDense, self).call(inputs)
        if mask is None:
            return base_out
        zeros = tf.zeros_like(base_out)
        return tf.where(mask, base_out, zeros)


# In[3]:


class NetSetup:
    def __init__(self, activation, activation_shared_axes, dropout_rate, first_layer_size, last_layer_size, decay_factor,
                 kernel_regularizer, time_distributed):
        self.activation = activation
        self.activation_shared_axes = activation_shared_axes
        if activation == 'relu' or activation == 'PReLU' or activation == 'tanh':
            self.DropoutType = Dropout
            self.kernel_init = 'he_uniform'
            self.apply_batch_norm = True
        elif activation == 'selu':
            self.DropoutType = AlphaDropout
            self.kernel_init = 'lecun_normal'
            self.apply_batch_norm = False
        else:
            raise RuntimeError('Activation "{}" not supported.'.format(activation))
        self.dropout_rate = dropout_rate
        self.first_layer_size = first_layer_size
        self.last_layer_size = last_layer_size
        self.decay_factor = decay_factor
        self.kernel_regularizer = kernel_regularizer
        self.time_distributed = time_distributed

    def RecalcLayerSizes(self, n_input_features, width_factor, compression_factor, consider_dropout = True):
        drop_factor = 1 + self.dropout_rate if consider_dropout else 1
        self.first_layer_size = int(math.ceil(n_input_features * drop_factor * width_factor))
        self.last_layer_size = int(math.ceil(n_input_features * drop_factor * compression_factor))


# In[4]:


def add_block_ending(net_setup, name_format, layer):
    if net_setup.apply_batch_norm:
        norm_layer = BatchNormalization(name=name_format.format('norm'))
        if net_setup.time_distributed:
            norm_layer = TimeDistributed(norm_layer, name=name_format.format('norm'))
        norm_layer = norm_layer(layer)
    else:
        norm_layer = layer
    if net_setup.activation == 'PReLU':
        activation_layer = PReLU(shared_axes=net_setup.activation_shared_axes,
                                 name=name_format.format('activation'))(norm_layer)
    else:
        activation_layer = Activation(net_setup.activation, name=name_format.format('activation'))(norm_layer)
    if net_setup.dropout_rate > 0:
        return net_setup.DropoutType(net_setup.dropout_rate, name=name_format.format('dropout'))(activation_layer)
    return activation_layer


# In[5]:


def dense_block(prev_layer, kernel_size, net_setup, block_name, n):
    DenseType = MaskedDense if net_setup.time_distributed else Dense
    dense = DenseType(kernel_size, name="{}_dense_{}".format(block_name, n),
                      kernel_initializer=net_setup.kernel_init,
                      kernel_regularizer=net_setup.kernel_regularizer)
    if net_setup.time_distributed:
        dense = TimeDistributed(dense, name="{}_dense_{}".format(block_name, n))
    dense = dense(prev_layer)
    return add_block_ending(net_setup, '{}_{{}}_{}'.format(block_name, n), dense)


# In[6]:


def reduce_n_features_1d(input_layer, net_setup, block_name):
    prev_layer = input_layer
    current_size = net_setup.first_layer_size
    n = 1
    while True:
        prev_layer = dense_block(prev_layer, current_size, net_setup, block_name, n)
        if current_size == net_setup.last_layer_size: break
        current_size = max(net_setup.last_layer_size, int(current_size / net_setup.decay_factor))
        n += 1
    return prev_layer


# In[7]:


def dense_block_sequence(input_layer, net_setup, n_layers, block_name):
    prev_layer = input_layer
    current_size = net_setup.first_layer_size
    for n in range(n_layers):
        prev_layer = dense_block(prev_layer, current_size, net_setup, block_name, n+1)
    return prev_layer


# In[8]:


def conv_block(prev_layer, filters, kernel_size, net_setup, padding, block_name, n):
    conv = Conv2D(filters, kernel_size, name="{}_conv_{}".format(block_name, n), padding=padding,
                  kernel_initializer=net_setup.kernel_init)(prev_layer)
    return add_block_ending(net_setup, '{}_{{}}_{}'.format(block_name, n), conv)


# In[9]:


def reduce_n_features_2d(input_layer, net_setup, block_name):
    conv_kernel=(1, 1)
    prev_layer = input_layer
    current_size = net_setup.first_layer_size
    n = 1
    while True:
        prev_layer = conv_block(prev_layer, current_size, conv_kernel, net_setup, 'valid', block_name, n)
        if current_size == net_setup.last_layer_size: break
        current_size = max(net_setup.last_layer_size, int(current_size / net_setup.decay_factor))
        n += 1
    return prev_layer


# In[10]:


def create_model(net_config):
    tau_net_setup = NetSetup('PReLU', None, 0.2, 128, 128, 1.4, None, False)
    comp_net_setup = NetSetup('PReLU', [1, 2], 0.2, 1024, 64, 1.6, None, False)
    #dense_net_setup = NetSetup('relu', 0, 512, 32, 1.4, keras.regularizers.l1(1e-5))
    dense_net_setup = NetSetup('PReLU', None, 0.2, 320, 64, 1.6, None, False)

    input_layers = []
    high_level_features = []

    if len(net_config.tau_branches) > 0:
        input_layer_tau = Input(name="input_tau", shape=(len(net_config.tau_branches),))
        input_layers.append(input_layer_tau)
        tau_net_setup.RecalcLayerSizes(len(net_config.tau_branches), 1.5, 1)
        processed_tau = reduce_n_features_1d(input_layer_tau, tau_net_setup, 'tau')
        #processed_tau = dense_block_sequence(input_layer_tau, tau_net_setup, 4, 'tau')
        high_level_features.append(processed_tau)

    for loc in net_config.cell_locations:
        reduced_inputs = []
        for comp_id in range(len(net_config.comp_names)):
            comp_name = net_config.comp_names[comp_id]
            n_comp_features = len(input_cell_external_branches) + len(net_config.comp_branches[comp_id])
            input_layer_comp = Input(name="input_{}_{}".format(loc, comp_name),
                                     shape=(n_cells_eta[loc], n_cells_phi[loc], n_comp_features))
            input_layers.append(input_layer_comp)
            comp_net_setup.RecalcLayerSizes(n_comp_features, 1.5, 1)
            #input_layer_comp_masked = Masking(name="input_{}_{}_masking".format(loc, comp_name))(input_layer_comp)
            #reduced_comp = dense_block_sequence(input_layer_comp_masked, comp_net_setup, 4, "{}_{}".format(loc, comp_name))
            #reduced_comp = reduce_n_features_1d(input_layer_comp_masked, comp_net_setup, "{}_{}".format(loc, comp_name))
            reduced_comp = reduce_n_features_2d(input_layer_comp, comp_net_setup, "{}_{}".format(loc, comp_name))
            reduced_inputs.append(reduced_comp)

        cell_output_size = 64
        if len(net_config.comp_names) > 1:
            conv_all_start = Concatenate(name="{}_cell_concat".format(loc), axis=3)(reduced_inputs)
            comp_net_setup.first_layer_size = conv_all_start.shape.as_list()[3]
            comp_net_setup.last_layer_size = 64
            prev_layer = reduce_n_features_2d(conv_all_start, comp_net_setup, "{}_all".format(loc))
        else:
            prev_layer = reduced_inputs[0]

        if n_cells_eta[loc] == 11:
            sequence = [ ['c', 64, 3, 2], ['p', 3], ['c', 96, 2, 2], ['p', 2], ['c', 128, 2, 1, 'valid'] ]
        elif n_cells_eta[loc] == 21:
            sequence = [ ['c', 64, 3, 2], ['p', 3], ['c', 96, 3, 2], ['p', 3], ['c', 128, 3, 1, 'valid'] ]
        else:
            raise RuntimeError("unsupported number of cells")

        conv_n = 1
        pool_n = 1

        for item in sequence:
            if item[0] == 'c':
                features = item[1]
                win_size = item[2]
                padding = item[4] if len(item) > 4 else 'same'
                for k in range(item[3]):
                    prev_layer = conv_block(prev_layer, features, win_size, comp_net_setup, padding,
                                            "{}_all_{}x{}".format(loc, win_size, win_size), conv_n)
                    conv_n += 1
            elif item[0] == 'p':
                win_size = item[1]
                prev_layer = MaxPooling2D(win_size, name="{}_all_{}x{}_pool_{}".format(loc, win_size, win_size, pool_n),
                                          padding='same')(prev_layer)
                pool_n += 1
            else:
                raise RuntimeError("unsupported item type")

        cells_flatten = Flatten(name="{}_cells_flatten".format(loc))(prev_layer)
        high_level_features.append(cells_flatten)

    if len(high_level_features) > 1:
        features_concat = Concatenate(name="features_concat", axis=1)(high_level_features)
    else:
        features_concat = high_level_features[0]
    if net_config.final:
        #print(features_concat.get_shape())
        #dense_net_setup.RecalcLayerSizes(128, 1, 0.5, False)
        final_dense = reduce_n_features_1d(features_concat, dense_net_setup, 'final')
        #final_dense = dense_block_sequence(features_concat, dense_net_setup, 4, 'final')
        output_layer = Dense(n_outputs, name="final_dense_last",
                             kernel_initializer=dense_net_setup.kernel_init)(final_dense)

    else:
        final_dense = dense_block(features_concat, 1024, dense_net_setup,
                                  'tmp_{}'.format(net_config.name), 1)
        output_layer = Dense(n_outputs, name="tmp_{}_dense_last".format(net_config.name),
                             kernel_initializer=dense_net_setup.kernel_init)(final_dense)
    softmax_output = Activation("softmax", name="main_output")(output_layer)

    model = Model(input_layers, softmax_output, name="DeepTau2017v2")
    return model


# In[11]:


model_name = "DeepTau2017v2p7"
model = create_model(netConf_full)
model.summary()


# In[12]:


def compile_model(model, learning_rate):
    #opt = keras.optimizers.Adam(lr=learning_rate)
    opt = keras.optimizers.Nadam(lr=learning_rate, schedule_decay=1e-4)
    #model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=["accuracy"])
    metrics = [
        "accuracy", TauLosses.tau_crossentropy, TauLosses.tau_crossentropy_v2,
        TauLosses.Le, TauLosses.Lmu, TauLosses.Ljet,
        TauLosses.He, TauLosses.Hmu, TauLosses.Htau, TauLosses.Hjet,
        TauLosses.Hcat_e, TauLosses.Hcat_mu, TauLosses.Hcat_jet, TauLosses.Hbin,
        TauLosses.Hcat_eInv, TauLosses.Hcat_muInv, TauLosses.Hcat_jetInv,
        TauLosses.Fe, TauLosses.Fmu, TauLosses.Fjet, TauLosses.Fcmb
    ]
    model.compile(loss=TauLosses.tau_crossentropy_v2, optimizer=opt, metrics=metrics, weighted_metrics=metrics)

def close_file(f_name):
    file_objs = [ obj for obj in gc.get_objects() if ("TextIOWrapper" in str(type(obj))) and (obj.name == f_name)]
    for obj in file_objs:
        obj.close()

class TimeCheckpoint(Callback):
    def __init__(self, time_interval, file_name_prefix):
        self.time_interval = time_interval
        self.file_name_prefix = file_name_prefix
        self.initial_time = time.time()
        self.last_check_time = self.initial_time

    def on_batch_end(self, batch, logs=None):
        if self.time_interval is None or batch % 100 != 0: return
        current_time = time.time()
        delta_t = current_time - self.last_check_time
        if delta_t >= self.time_interval:
            abs_delta_t_h = (current_time - self.initial_time) / 60. / 60.
            read_hdf_lock.acquire()
            self.model.save('{}_historic_b{}_{:.1f}h.h5'.format(self.file_name_prefix, batch, abs_delta_t_h))
            read_hdf_lock.release()
            self.last_check_time = current_time

    def on_epoch_end(self, epoch, logs=None):
        read_hdf_lock.acquire()
        self.model.save('{}_e{}.h5'.format(self.file_name_prefix, epoch))
        read_hdf_lock.release()
        #Notify("Epoch {} is ended.".format(epoch))

def run_training(train_suffix, model_name, data_loader, epoch, n_epochs):

    train_name = '%s_%s' % (model_name, train_suffix)
    log_name = "%s.log" % train_name
    if os.path.isfile(log_name):
        close_file(log_name)
        os.remove(log_name)
    csv_log = CSVLogger(log_name, append=True)

    time_checkpoint = TimeCheckpoint(12*60*60, train_name)
    callbacks = [time_checkpoint, csv_log]
    fit_hist = model.fit_generator(data_loader.generator(True), validation_data=data_loader.generator(False),
                                   steps_per_epoch=data_loader.steps_per_epoch, validation_steps=data_loader.validation_steps,
                                   callbacks=callbacks, epochs=n_epochs, initial_epoch=epoch, verbose=0)

    read_hdf_lock.acquire()
    model.save("%s_final.hdf5" % train_name)
    read_hdf_lock.release()
    return fit_hist


# In[13]:

TauLosses.SetSFs(1, 2.5, 5, 1.5)
print(TauLosses.Le_sf, TauLosses.Lmu_sf, TauLosses.Ltau_sf, TauLosses.Ljet_sf)
compile_model(model, 1e-3)


# In[14]:


loader = DataLoader('N:/tau-ml/tuples-v2-training-v2-t1/training/part_*.h5', netConf_full, 100, 2000,
                    validation_size=10000000, max_queue_size=40, n_passes=-1, return_grid=True)
print(loader.file_entries)
print(loader.total_size, loader.data_size, loader.validation_size)


# In[ ]:


fit_hist = run_training('step{}'.format(1), model_name, loader, 0, 10)


# In[ ]:
