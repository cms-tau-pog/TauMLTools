#!/usr/bin/env python
# coding: utf-8

import sys
import math
import tensorflow as tf
import keras
import keras.backend as K
from keras import regularizers
from keras.models import Sequential, Model
from keras.layers import Input, Dense, Conv2D, Dropout, AlphaDropout, Activation, BatchNormalization, Flatten,                                     Concatenate, PReLU, TimeDistributed, LSTM, Masking
from keras.callbacks import Callback, ModelCheckpoint, CSVLogger
#from keras_tqdm import TQDMNotebookCallback

sys.path.insert(0, "../../python")
from common import *

class MaskedDense(Dense):
    def __init__(self, units, **kwargs):
        super(MaskedDense, self).__init__(units, **kwargs)

    def call(self, inputs, mask=None):
        base_out = super(MaskedDense, self).call(inputs)
        if mask is None:
            return base_out
        zeros = tf.zeros_like(base_out)
        return tf.where(mask, base_out, zeros)

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

def add_block_ending(net_setup, name_format, layer):
    if net_setup.apply_batch_norm:
        #floatx = K.floatx()
        #K.set_floatx('float32')
        norm_layer = BatchNormalization(name=name_format.format('norm'))
        if net_setup.time_distributed:
            norm_layer = TimeDistributed(norm_layer, name=name_format.format('norm'))
        norm_layer = norm_layer(layer)
        #K.set_floatx(floatx)
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
                  kernel_initializer=net_setup.kernel_init)(prev_layer)
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

def create_cell_model(net_config, loc):
    comp_net_setup = NetSetup('PReLU', [1, 2], 0.2, 1024, 64, 1.6, None, False)

    input_layers = []
    high_level_features = []

    reduced_inputs = []
    for comp_id in range(len(net_config.comp_names)):
        comp_name = net_config.comp_names[comp_id]
        n_comp_features = len(input_cell_external_branches) + len(net_config.comp_branches[comp_id])
        input_layer_comp = Input(name="input_{}_{}".format(loc, comp_name),
                                 shape=(1, 1, n_comp_features))
        input_layers.append(input_layer_comp)
        comp_net_setup.RecalcLayerSizes(n_comp_features, 2, 1)
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

    model = Model(input_layers, prev_layer, name="DeepTau2017v2")
    return model

model_name = "DeepTau2017v2p6"
inner_model = create_cell_model(netConf_full, 'inner')
inner_model.summary()

inner_model.load_weights('../../../../output/networks/2017v2p6/DeepTau2017v2p6_step1_e6.h5', by_name=True)
inner_model.save('../../../../output/networks/2017v2p6/DeepTau2017v2p6_step1_e6_inner.h5')

model_name = "DeepTau2017v2p6"
outer_model = create_cell_model(netConf_full, 'outer')
outer_model.summary()

outer_model.load_weights('../../../../output/networks/2017v2p6/DeepTau2017v2p6_step1_e6.h5', by_name=True)
outer_model.save('../../../../output/networks/2017v2p6/DeepTau2017v2p6_step1_e6_outer.h5')

def create_core_model(net_config):
    tau_net_setup = NetSetup('PReLU', None, 0.2, 128, 128, 1.4, None, False)
    comp_net_setup = NetSetup('PReLU', [1, 2], 0.2, 1024, 64, 1.6, None, False)
    dense_net_setup = NetSetup('PReLU', None, 0.2, 200, 64, 1.4, None, False)

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
        cell_output_size = 64
        input_layer_loc = Input(name="input_{}".format(loc),
                                shape=(n_cells_eta[loc], n_cells_phi[loc], cell_output_size))
        input_layers.append(input_layer_loc)
        prev_layer = input_layer_loc

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
        final_dense = dense_block_sequence(features_concat, dense_net_setup, 4, 'final')
        output_layer = Dense(n_outputs, name="final_dense_last",
                             kernel_initializer=dense_net_setup.kernel_init)(final_dense)

    else:
        final_dense = dense_block(features_concat, 1024, dense_net_setup,
                                  'tmp_{}'.format(net_config.name), 1)
        output_layer = Dense(n_outputs, name="tmp_{}_dense_last".format(net_config.name),
                             kernel_initializer=dense_net_setup.kernel_init)(final_dense)
    softmax_output = Activation("softmax", name="main_output")(output_layer)

    print(input_layers)
    model = Model(input_layers, softmax_output, name="DeepTau2017v2")
    return model

model_name = "DeepTau2017v2p6"
core_model = create_core_model(netConf_full)
core_model.summary()

core_model.load_weights('../../../../output/networks/2017v2p6/DeepTau2017v2p6_step1_e6.h5', by_name=True)
core_model.save('../../../../output/networks/2017v2p6/DeepTau2017v2p6_step1_e6_core.h5')
