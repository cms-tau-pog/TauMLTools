import os
import gc
import sys
import re
import glob
import time
import math
import numpy as np
import uproot
import pandas
import functools
from concurrent.futures import ThreadPoolExecutor

import tensorflow as tf
import keras
import keras.backend as K
from keras import regularizers
from keras.models import Sequential, Model, load_model
from keras.layers import Input, Dense, Conv2D, Dropout, AlphaDropout, Activation, BatchNormalization, Flatten, \
                                    Concatenate, PReLU, TimeDistributed, LSTM, Masking
from keras.callbacks import Callback, ModelCheckpoint, CSVLogger

sys.path.insert(0, "..")
from common import *
#from t_notify import Notify
from DataLoader import DataLoader, read_hdf_lock

class MaskedDense(Dense):
    def __init__(self, units, **kwargs):
        super(MaskedDense, self).__init__(units, **kwargs)

    def call(self, inputs, mask=None):
        base_out = super(MaskedDense, self).call(inputs)
        if mask is None:
            return base_out
        zeros = tf.zeros_like(base_out)
        return tf.where(mask, base_out, zeros)

class SafeModelCheckpoint(ModelCheckpoint):
    def __init__(self, filepath, **kwargs):
        super(SafeModelCheckpoint, self).__init__(filepath, **kwargs)

    def on_epoch_end(self, epoch, logs=None):
        read_hdf_lock.acquire()
        super(SafeModelCheckpoint, self).on_epoch_end(epoch, logs)
        read_hdf_lock.release()

class NetSetup:
    def __init__(self, activation, activation_shared_axes, dropout_rate, first_layer_size, last_layer_size,
                 decay_factor, kernel_regularizer, time_distributed):
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


def dense_block(prev_layer, kernel_size, net_setup, block_name, n):
    DenseType = MaskedDense if net_setup.time_distributed else Dense
    dense = DenseType(kernel_size, name="{}_dense_{}".format(block_name, n),
                      kernel_initializer=net_setup.kernel_init,
                      kernel_regularizer=net_setup.kernel_regularizer)
    if net_setup.time_distributed:
        dense = TimeDistributed(dense, name="{}_dense_{}".format(block_name, n))
    dense = dense(prev_layer)
    return add_block_ending(net_setup, '{}_{{}}_{}'.format(block_name, n), dense)

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

def dense_block_sequence(input_layer, net_setup, n_layers, block_name):
    prev_layer = input_layer
    current_size = net_setup.first_layer_size
    for n in range(n_layers):
        prev_layer = dense_block(prev_layer, current_size, net_setup, block_name, n+1)
    return prev_layer

def conv_block(prev_layer, filters, kernel_size, net_setup, block_name, n):
    conv = Conv2D(filters, kernel_size, name="{}_conv_{}".format(block_name, n),
                  kernel_initializer=net_setup.kernel_init, kernel_regularizer=net_setup.kernel_regularizer)(prev_layer)
    return add_block_ending(net_setup, '{}_{{}}_{}'.format(block_name, n), conv)

def reduce_n_features_2d(input_layer, net_setup, block_name):
    conv_kernel=(1, 1)
    prev_layer = input_layer
    current_size = net_setup.first_layer_size
    n = 1
    while True:
        prev_layer = conv_block(prev_layer, current_size, conv_kernel, net_setup, block_name, n)
        if current_size == net_setup.last_layer_size: break
        current_size = max(net_setup.last_layer_size, int(current_size / net_setup.decay_factor))
        n += 1
    return prev_layer

def create_model(net_config):
    kernel_regularizer = None # keras.regularizers.l1(1e-5)
    tau_net_setup = NetSetup('PReLU', None, 0.5, 128, 128, 1.4, kernel_regularizer, False)
    comp_net_setup = NetSetup('PReLU', [1, 2], 0.5, 1024, 64, 1.6, kernel_regularizer, False)
    #dense_net_setup = NetSetup('relu', 0, 512, 32, 1.4, keras.regularizers.l1(1e-5))
    dense_net_setup = NetSetup('PReLU', None, 0.5, 200, 64, 1.4, kernel_regularizer, False)

    input_layers = []
    high_level_features = []

    if len(net_config.tau_branches) > 0:
        input_layer_tau = Input(name="input_tau", shape=(len(net_config.tau_branches),))
        input_layers.append(input_layer_tau)
        tau_net_setup.RecalcLayerSizes(len(net_config.tau_branches), 2, 1)
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
            comp_net_setup.RecalcLayerSizes(n_comp_features, 2, 1)
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
        window_size = 3
        current_size = n_cells_eta[loc]
        n = 1
        while current_size > 1:
            win_size = min(current_size, window_size)
            prev_layer = conv_block(prev_layer, cell_output_size, (win_size, win_size), comp_net_setup,
                                    "{}_all_{}x{}".format(loc, win_size, win_size), n)
            n += 1
            current_size -= window_size - 1

        cells_flatten = Flatten(name="{}_cells_flatten".format(loc))(prev_layer)
        high_level_features.append(cells_flatten)

    if len(high_level_features) > 1:
        features_concat = Concatenate(name="features_concat", axis=1)(high_level_features)
    else:
        features_concat = high_level_features[0]
    if net_config.final:
        #print(features_concat.get_shape())
        #dense_net_setup.RecalcLayerSizes(128, 1, 0.5, False)
        #final_dense = reduce_n_features_1d(features_concat, dense_net_setup, 'final')
        final_dense = dense_block_sequence(features_concat, dense_net_setup, 4, 'final')
        output_layer = Dense(n_outputs, name="final_dense_last",
                             kernel_initializer=dense_net_setup.kernel_init)(final_dense)

    else:
        final_dense = dense_block(features_concat, 1024, dense_net_setup,
                                  'tmp_{}'.format(net_config.name), 1)
        output_layer = Dense(n_outputs, name="tmp_{}_dense_last".format(net_config.name),
                             kernel_initializer=dense_net_setup.kernel_init)(final_dense)
    if n_outputs > 1:
        softmax_output = Activation("softmax", name="main_output")(output_layer)
    else:
        softmax_output = Activation("sigmoid", name="main_output")(output_layer)

    weight_input = Input(name="weight_input", shape=(1,))
    input_layers.append(weight_input)
    model = Model(input_layers, softmax_output, name="DeepTau2017v2")
    return model

def compile_model(model, learning_rate):
    opt = keras.optimizers.Nadam(lr=learning_rate, schedule_decay=1e-4)
    weight_input = model.inputs[-1]

    def dy_mc_acc(target, output):
        return TauLosses.binary(target, output, weights=weight_input, selected=1)
    def emb_acc(target, output):
        return TauLosses.binary(target, output, weights=weight_input, selected=0)
    def avg_acc(target, output):
        return (dy_mc_acc(target, output) + emb_acc(target, output)) / 2

    metrics = [ "accuracy", avg_acc, dy_mc_acc, emb_acc ]
    model.compile(loss='binary_crossentropy', optimizer=opt, metrics=metrics)

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

def run_training(model_name, data_loader, epoch, n_epochs):

    train_name = model_name
    log_name = "%s.log" % train_name
    if os.path.isfile(log_name):
        close_file(log_name)
        os.remove(log_name)
    csv_log = CSVLogger(log_name, append=True)
    time_checkpoint = TimeCheckpoint(None, train_name)
    callbacks = [time_checkpoint, csv_log]
    fit_hist = model.fit_generator(data_loader.generator(True), validation_data=data_loader.generator(False),
                                   steps_per_epoch=data_loader.steps_per_epoch, validation_steps=data_loader.validation_steps,
                                   callbacks=callbacks, epochs=n_epochs, initial_epoch=epoch, verbose=1)

    #read_hdf_lock.acquire()
    #model.save("%s_final.h5" % train_name)
    #read_hdf_lock.release()
    return fit_hist

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
K.set_session(sess)

#netConf = netConf_preTau
#tau_br_arg = sys.argv[2] if len(sys.argv) > 2 else ''
#if len(tau_br_arg) > 0:
#    netConf.tau_branches = [ s for s in re.split(" |,|'", tau_br_arg) if len(s) > 0 ]
netConf = netConf_full
model_name = "DeepTau2017v2p6emb_noPCA_0.5drop"
#if len(sys.argv) > 1:
#    model_name += '_partial_{}'.format(sys.argv[1])
model = create_model(netConf)
compile_model(model, 1e-3)

loader = DataLoader('/data/tau-ml/tuples-v2.5-training-v1-t1/training.h5', netConf, 100, 2000,
                    validation_size=2000000, max_queue_size=40, n_passes=-1, return_grid=True)
print(loader.file_entries)
print(loader.total_size, loader.data_size, loader.validation_size)
#print("Tau branches:", netConf.tau_branches)
fit_hist = run_training(model_name, loader, 0, 100)
