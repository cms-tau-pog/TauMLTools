import os
import yaml
import gc
import sys
import glob
import time
import math
import numpy as np
# import uproot
# import pandas
from functools import partial
from concurrent.futures import ThreadPoolExecutor

import tensorflow as tf
from tensorflow import keras
import tensorflow.keras.backend as K
from tensorflow.keras import regularizers
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import Input, Dense, Conv2D, Dropout, AlphaDropout, Activation, BatchNormalization, Flatten, \
                                    Concatenate, PReLU, TimeDistributed, LSTM, Masking
from tensorflow.keras.callbacks import Callback, ModelCheckpoint, CSVLogger
from datetime import datetime

import mlflow
mlflow.tensorflow.autolog(log_models=False)

import hydra
from hydra.utils import to_absolute_path
from omegaconf import DictConfig, OmegaConf

sys.path.insert(0, "..")
from common import *
import DataLoader

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

def create_model(net_config):
    tau_net_setup = NetSetup(*net_config.tau_net_setup)
    comp_net_setup = NetSetup(*net_config.comp_net_setup)
    dense_net_setup = NetSetup(*net_config.dense_net_setup)

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
            # n_comp_features = len(input_cell_external_branches) + len(net_config.comp_branches[comp_id])
            n_comp_features = len(net_config.comp_branches[comp_id])
            input_layer_comp = Input(name="input_{}_{}".format(loc, comp_name),
                                     shape=(net_config.n_cells_eta[loc], net_config.n_cells_phi[loc], n_comp_features))
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
        current_size = net_config.n_cells_eta[loc]
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
        output_layer = Dense(net_config.n_outputs, name="final_dense_last",
                             kernel_initializer=dense_net_setup.kernel_init)(final_dense)

    else:
        final_dense = dense_block(features_concat, 1024, dense_net_setup,
                                  'tmp_{}'.format(net_config.name), 1)
        output_layer = Dense(net_config.n_outputs, name="tmp_{}_dense_last".format(net_config.name),
                             kernel_initializer=dense_net_setup.kernel_init)(final_dense)
    softmax_output = Activation("softmax", name="main_output")(output_layer)

    model = Model(input_layers, softmax_output, name="DeepTau2017v2")
    return model

def compile_model(model, learning_rate):
    # opt = keras.optimizers.Adam(lr=learning_rate)
    opt = tf.keras.optimizers.Nadam(learning_rate=learning_rate, schedule_decay=1e-4)
    # opt = Nadam(lr=learning_rate, beta_1=1e-4)

    metrics = [
        "accuracy", TauLosses.tau_crossentropy, TauLosses.tau_crossentropy_v2,
        TauLosses.Le, TauLosses.Lmu, TauLosses.Ljet,
        TauLosses.He, TauLosses.Hmu, TauLosses.Htau, TauLosses.Hjet,
        TauLosses.Hcat_e, TauLosses.Hcat_mu, TauLosses.Hcat_jet, TauLosses.Hbin,
        TauLosses.Hcat_eInv, TauLosses.Hcat_muInv, TauLosses.Hcat_jetInv,
        TauLosses.Fe, TauLosses.Fmu, TauLosses.Fjet, TauLosses.Fcmb
    ]
    model.compile(loss=TauLosses.tau_crossentropy_v2, optimizer=opt, metrics=metrics, weighted_metrics=metrics)

    # log metric names for passing them during model loading
    metric_names = {(m if isinstance(m, str) else m.__name__): '' for m in metrics}
    mlflow.log_dict(metric_names, 'input_cfg/metric_names.json')

def run_training(model, data_loader, to_profile, log_suffix, path_to_csv_log, path_to_log_dir, path_to_model):

    gen_train = data_loader.get_generator(primary_set = True)
    gen_val = data_loader.get_generator(primary_set = False)
    input_shape, input_types = data_loader.get_input_config()

    data_train = tf.data.Dataset.from_generator(
        gen_train, output_types = input_types, output_shapes = input_shape
        ).prefetch(tf.data.AUTOTUNE)
    data_val = tf.data.Dataset.from_generator(
        gen_val, output_types = input_types, output_shapes = input_shape
        ).prefetch(tf.data.AUTOTUNE)

    model_name = data_loader.model_name
    log_name = '%s_%s' % (model_name, log_suffix)
    csv_log_file = f"{path_to_csv_log}/{log_name}.log"
    if os.path.isfile(csv_log_file):
        close_file(csv_log_file)
        os.remove(csv_log_file)
    csv_log = CSVLogger(csv_log_file, append=True)
    time_checkpoint = TimeCheckpoint(12*60*60, log_name)
    callbacks = [time_checkpoint, csv_log]

    logs = path_to_log_dir + '/' + log_name + '_' + datetime.now().strftime("%Y.%m.%d(%H:%M)")
    tboard_callback = tf.keras.callbacks.TensorBoard(log_dir = logs,
                                                     profile_batch = ('100, 300' if to_profile else 0),
                                                     update_freq = ( 0 if data_loader.n_batches_log<=0 else data_loader.n_batches_log ))
    callbacks.append(tboard_callback)

    fit_hist = model.fit(data_train, validation_data = data_val,
                         epochs = data_loader.n_epochs, initial_epoch = data_loader.epoch,
                         callbacks = callbacks)

    model_path = f"{path_to_model}/{log_name}_final.tf"
    model.save(model_path, save_format="tf")
    mlflow.log_artifacts(model_path, "model")
    return fit_hist

@hydra.main(config_path='.', config_name='train')
def main(cfg: DictConfig) -> None:
    # set up mlflow experiment id
    mlflow.set_tracking_uri(f"file://{to_absolute_path(cfg.path_to_mlflow)}")
    experiment = mlflow.get_experiment_by_name(cfg.experiment_name)
    if experiment is not None: # fetch existing experiment id
        run_kwargs = {'experiment_id': experiment.experiment_id} 
    else: # create new experiment
        experiment_id = mlflow.create_experiment(cfg.experiment_name)
        run_kwargs = {'experiment_id': experiment_id} 
    
    # run the training with mlflow tracking
    with mlflow.start_run(**run_kwargs) as active_run:
        setup_gpu(cfg.gpu_cfg)
        training_cfg = OmegaConf.to_object(cfg.training_cfg) # convert to python dictionary
        scaling_cfg = to_absolute_path(cfg.scaling_cfg)
        dataloader = DataLoader.DataLoader(training_cfg, scaling_cfg)

        TauLosses.SetSFs(*dataloader.TauLossesSFs)
        print("loss consts:",TauLosses.Le_sf, TauLosses.Lmu_sf, TauLosses.Ltau_sf, TauLosses.Ljet_sf)

        netConf_full = dataloader.get_net_config()
        model = create_model(netConf_full)
        compile_model(model, dataloader.learning_rate)
        fit_hist = run_training(model, dataloader, False, cfg.log_suffix, 
                                to_absolute_path(cfg.path_to_csv_log), to_absolute_path(cfg.path_to_log_dir), to_absolute_path(cfg.path_to_model))

        mlflow.log_dict(training_cfg, 'input_cfg/training.yaml')
        mlflow.log_artifact(scaling_cfg, 'input_cfg')
        mlflow.log_artifact(to_absolute_path("Training_v0p1.py"), 'input_cfg')
        mlflow.log_artifact(to_absolute_path("../common.py"), 'input_cfg')
        mlflow.log_param('run_id', active_run.info.run_id)
        print(f'\nTraining has finished! Corresponding MLflow experiment name (ID): {cfg.experiment_name}({run_kwargs["experiment_id"]}), and run ID: {active_run.info.run_id}\n')

if __name__ == '__main__':
    main() 