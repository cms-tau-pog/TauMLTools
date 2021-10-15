import os
import yaml
import gc
import sys
from glob import glob
import time
import math
import numpy as np
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
from mlflow.tracking.context.git_context import _get_git_commit
mlflow.tensorflow.autolog(log_models=False)

import hydra
from hydra.utils import to_absolute_path
from omegaconf import DictConfig, OmegaConf

sys.path.insert(0, "..")
from common import *
import DataLoader

class NetSetup:
    def __init__(self, activation, dropout_rate=0, reduction_rate=1, kernel_regularizer=None):
        self.activation = activation
        self.dropout_rate = dropout_rate
        self.reduction_rate = reduction_rate
        self.kernel_regularizer = kernel_regularizer

        if self.activation == 'relu' or self.activation == 'PReLU' or self.activation == 'tanh':
            self.DropoutType = Dropout
            self.kernel_init = 'he_uniform'
            self.apply_batch_norm = True
        elif self.activation == 'selu':
            self.DropoutType = AlphaDropout
            self.kernel_init = 'lecun_normal'
            self.apply_batch_norm = False
        else:
            raise RuntimeError('Activation "{}" not supported.'.format(self.activation))

class NetSetupFixed(NetSetup):
    def __init__(self, first_layer_width, last_layer_width, min_n_layers=None, max_n_layers=None, **kwargs):
        super().__init__(**kwargs)
        self.first_layer_width = first_layer_width
        self.last_layer_width = last_layer_width
        self.min_n_layers = min_n_layers
        self.max_n_layers = max_n_layers

    @staticmethod
    def GetNumberOfUnits(n_input_features, layer_width, dropout_rate):
        if type(layer_width) == int:
            return layer_width
        elif type(layer_width) == str:
            eval_res = eval(layer_width, {}, {'n': n_input_features, 'drop': dropout_rate})
            if type(eval_res) not in [ int, float ]:
                raise RuntimeError(f'Invalid formula for layer widht: "{layer_width}"')
            return int(math.ceil(eval_res))
        raise RuntimeError(f"layer width definition = '{layer_width}' is not supported")

    def ComputeLayerSizes(self, n_input_features):
        self.first_layer_size = NetSetupFixed.GetNumberOfUnits(n_input_features, self.first_layer_width,
                                                               self.dropout_rate)
        self.last_layer_size = NetSetupFixed.GetNumberOfUnits(n_input_features, self.last_layer_width,
                                                              self.dropout_rate)

class NetSetup1D(NetSetupFixed):
    def __init__(self, time_distributed=False, **kwargs):
        super().__init__(**kwargs)
        self.activation_shared_axes = None
        self.time_distributed = time_distributed

class NetSetup2D(NetSetupFixed):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.activation_shared_axes = [1, 2]
        self.time_distributed = False

class NetSetupConv2D(NetSetup):
    def __init__(self, window_size=3, **kwargs):
        super().__init__(**kwargs)
        self.activation_shared_axes = [1, 2]
        self.time_distributed = False
        self.window_size = window_size

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

def get_layer_size_sequence(net_setup):
    layer_sizes = []
    current_size = net_setup.first_layer_size
    current_size = net_setup.first_layer_size
    n = 1
    while True:
        layer_sizes.append(current_size)
        n += 1
        if current_size > net_setup.last_layer_size:
            if n == net_setup.max_n_layers:
                current_size = net_setup.last_layer_size
            else:
                current_size = max(net_setup.last_layer_size, int(current_size / net_setup.reduction_rate))
        elif net_setup.min_n_layers is None or n > net_setup.min_n_layers:
            break
    return layer_sizes

def reduce_n_features_1d(input_layer, net_setup, block_name):
    prev_layer = input_layer
    layer_sizes = get_layer_size_sequence(net_setup)
    for n, layer_size in enumerate(layer_sizes):
        prev_layer = dense_block(prev_layer, layer_size, net_setup, block_name, n+1)
    return prev_layer

def conv_block(prev_layer, filters, kernel_size, net_setup, block_name, n):
    conv = Conv2D(filters, kernel_size, name="{}_conv_{}".format(block_name, n),
                  kernel_initializer=net_setup.kernel_init)(prev_layer)
    return add_block_ending(net_setup, '{}_{{}}_{}'.format(block_name, n), conv)

def reduce_n_features_2d(input_layer, net_setup, block_name):
    conv_kernel=(1, 1)
    prev_layer = input_layer
    layer_sizes = get_layer_size_sequence(net_setup)
    for n, layer_size in enumerate(layer_sizes):
        prev_layer = conv_block(prev_layer, layer_size, conv_kernel, net_setup, block_name, n+1)
    return prev_layer

def get_n_filters_conv2d(n_input, current_size, window_size, reduction_rate):
    if reduction_rate is None:
        return n_input
    if window_size <= 1 or current_size < window_size:
        raise RuntineError("Unable to compute number of filters for the next Conv2D layer.")
    n_filters = ((float(current_size) / float(current_size - window_size + 1)) ** 2) * n_input / reduction_rate
    return int(math.ceil(n_filters))

def create_model(net_config, model_name):
    tau_net_setup = NetSetup1D(**net_config.tau_net)
    comp_net_setup = NetSetup2D(**net_config.comp_net)
    comp_merge_net_setup = NetSetup2D(**net_config.comp_merge_net)
    conv_2d_net_setup = NetSetupConv2D(**net_config.conv_2d_net)
    dense_net_setup = NetSetup1D(**net_config.dense_net)

    input_layers = []
    high_level_features = []

    if net_config.n_tau_branches > 0:
        input_layer_tau = Input(name="input_tau", shape=(net_config.n_tau_branches,))
        input_layers.append(input_layer_tau)
        tau_net_setup.ComputeLayerSizes(net_config.n_tau_branches)
        processed_tau = reduce_n_features_1d(input_layer_tau, tau_net_setup, 'tau')
        high_level_features.append(processed_tau)

    for loc in net_config.cell_locations:
        reduced_inputs = []
        for comp_id in range(len(net_config.comp_names)):
            comp_name = net_config.comp_names[comp_id]
            n_comp_features = net_config.n_comp_branches[comp_id]
            input_layer_comp = Input(name="input_{}_{}".format(loc, comp_name),
                                     shape=(net_config.n_cells[loc], net_config.n_cells[loc], n_comp_features))
            input_layers.append(input_layer_comp)
            comp_net_setup.ComputeLayerSizes(n_comp_features)
            reduced_comp = reduce_n_features_2d(input_layer_comp, comp_net_setup, "{}_{}".format(loc, comp_name))
            reduced_inputs.append(reduced_comp)

        if len(net_config.comp_names) > 1:
            conv_all_start = Concatenate(name="{}_cell_concat".format(loc), axis=3)(reduced_inputs)
            comp_merge_net_setup.ComputeLayerSizes(conv_all_start.shape.as_list()[3])
            prev_layer = reduce_n_features_2d(conv_all_start, comp_merge_net_setup, "{}_all".format(loc))
        else:
            prev_layer = reduced_inputs[0]
        current_grid_size = net_config.n_cells[loc]
        n_inputs = prev_layer.shape.as_list()[3]
        n = 1
        while current_grid_size > 1:
            win_size = min(current_grid_size, conv_2d_net_setup.window_size)
            n_filters = get_n_filters_conv2d(n_inputs, current_grid_size, win_size, conv_2d_net_setup.reduction_rate)
            prev_layer = conv_block(prev_layer, n_filters, (win_size, win_size), conv_2d_net_setup,
                                    "{}_all_{}x{}".format(loc, win_size, win_size), n)
            n += 1
            current_grid_size -= win_size - 1
            n_inputs = n_filters

        cells_flatten = Flatten(name="{}_cells_flatten".format(loc))(prev_layer)
        high_level_features.append(cells_flatten)

    if len(high_level_features) > 1:
        features_concat = Concatenate(name="features_concat", axis=1)(high_level_features)
    else:
        features_concat = high_level_features[0]

    dense_net_setup.ComputeLayerSizes(features_concat.shape.as_list()[1])
    final_dense = reduce_n_features_1d(features_concat, dense_net_setup, 'final')
    output_layer = Dense(net_config.n_outputs, name="final_dense_last",
                         kernel_initializer=dense_net_setup.kernel_init)(final_dense)
    softmax_output = Activation("softmax", name="main_output")(output_layer)

    model = Model(input_layers, softmax_output, name=model_name)
    return model

def compile_model(model, opt_name, learning_rate):
    # opt = keras.optimizers.Adam(lr=learning_rate)
    opt = getattr(tf.keras.optimizers, opt_name)(learning_rate=learning_rate)
    #opt = tf.keras.optimizers.Nadam(learning_rate=learning_rate, schedule_decay=1e-4)
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

def run_training(model, data_loader, to_profile, log_suffix):

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
    csv_log_file = "metrics.log"
    if os.path.isfile(csv_log_file):
        close_file(csv_log_file)
        os.remove(csv_log_file)
    csv_log = CSVLogger(csv_log_file, append=True)
    time_checkpoint = TimeCheckpoint(12*60*60, log_name)
    callbacks = [time_checkpoint, csv_log]

    logs = log_name + '_' + datetime.now().strftime("%Y.%m.%d(%H:%M)")
    tboard_callback = tf.keras.callbacks.TensorBoard(log_dir = logs,
                                                     profile_batch = ('100, 300' if to_profile else 0),
                                                     update_freq = ( 0 if data_loader.n_batches_log<=0 else data_loader.n_batches_log ))
    callbacks.append(tboard_callback)

    fit_hist = model.fit(data_train, validation_data = data_val,
                         epochs = data_loader.n_epochs, initial_epoch = data_loader.epoch,
                         callbacks = callbacks)

    model_path = f"{log_name}_final.tf"
    model.save(model_path, save_format="tf")

    # mlflow logs
    for checkpoint_dir in glob(f'{log_name}*.tf'):
         mlflow.log_artifacts(checkpoint_dir, f"model_checkpoints/{checkpoint_dir}")
    mlflow.log_artifacts(model_path, "model")
    mlflow.log_artifacts(logs, "custom_tensorboard_logs")
    mlflow.log_artifact(csv_log_file)
    mlflow.log_param('model_name', model_name)

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
        run_id = active_run.info.run_id
        setup_gpu(cfg.gpu_cfg)
        training_cfg = OmegaConf.to_object(cfg.training_cfg) # convert to python dictionary
        scaling_cfg = to_absolute_path(cfg.scaling_cfg)
        dataloader = DataLoader.DataLoader(training_cfg, scaling_cfg)
        setup = dataloader.config["SetupNN"]
        TauLosses.SetSFs(*setup["TauLossesSFs"])
        print("loss consts:",TauLosses.Le_sf, TauLosses.Lmu_sf, TauLosses.Ltau_sf, TauLosses.Ljet_sf)

        netConf_full = dataloader.get_net_config()
        model = create_model(netConf_full, dataloader.model_name)
        compile_model(model, setup["optimizer_name"], setup["learning_rate"])
        fit_hist = run_training(model, dataloader, False, cfg.log_suffix)

        # log NN params
        for net_type in ['tau_net', 'comp_net', 'comp_merge_net', 'conv_2d_net', 'dense_net']:
            mlflow.log_params({f'{net_type}_{k}': v for k,v in cfg.training_cfg.SetupNN[net_type].items()})
        mlflow.log_params({f'TauLossesSFs_{i}': v for i,v in enumerate(cfg.training_cfg.SetupNN.TauLossesSFs)})
        with open(to_absolute_path(f'{cfg.path_to_mlflow}/{run_kwargs["experiment_id"]}/{run_id}/artifacts/model_summary.txt')) as f:
            for l in f:
                if (s:='Trainable params: ') in l:
                    mlflow.log_param('n_train_params', int(l.split(s)[-1].replace(',', '')))

        # log training related files
        mlflow.log_dict(training_cfg, 'input_cfg/training_cfg.yaml')
        mlflow.log_artifact(scaling_cfg, 'input_cfg')
        mlflow.log_artifact(to_absolute_path("Training_v0p1.py"), 'input_cfg')
        mlflow.log_artifact(to_absolute_path("../common.py"), 'input_cfg')

        # log hydra files
        mlflow.log_artifacts('.hydra', 'input_cfg/hydra')
        mlflow.log_artifact('Training_v0p1.log', 'input_cfg/hydra')

        # log misc. info
        mlflow.log_param('run_id', run_id)
        mlflow.log_param('git_commit', _get_git_commit(to_absolute_path('.')))
        print(f'\nTraining has finished! Corresponding MLflow experiment name (ID): {cfg.experiment_name}({run_kwargs["experiment_id"]}), and run ID: {run_id}\n')

if __name__ == '__main__':
    main()
