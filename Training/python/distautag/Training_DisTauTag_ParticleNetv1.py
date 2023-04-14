import os
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
from commonReco import *
from common import setup_gpu
import DataLoaderReco

class _DotDict:
    pass

# A shape is (N, P_A, C), B shape is (N, P_B, C)
# D shape is (N, P_A, P_B)
@tf.function
def batch_distance_matrix_general(A, B):
    with tf.name_scope('dmat'):
        r_A = tf.reduce_sum(A * A, axis=2, keepdims=True)
        r_B = tf.reduce_sum(B * B, axis=2, keepdims=True)
        m = tf.matmul(A, tf.transpose(B, perm=(0, 2, 1)))
        D = r_A - 2 * m + tf.transpose(r_B, perm=(0, 2, 1))
        return D

@tf.function
def knn(num_points, k, topk_indices, features):
    # topk_indices: (N, P, K)
    # features: (N, P, C)
    with tf.name_scope('knn'):
        queries_shape = tf.shape(features)
        batch_size = queries_shape[0]
        batch_indices = tf.tile(tf.reshape(tf.range(batch_size), (-1, 1, 1, 1)), (1, num_points, k, 1))
        indices = tf.concat([batch_indices, tf.expand_dims(topk_indices, axis=3)], axis=3)  # (N, P, K, 2)
        return tf.gather_nd(features, indices)


class EdgeConv(tf.keras.layers.Layer):

    """EdgeConv
    Args:
        K: int, number of neighbors
        in_channels: # of input channels
        channels: tuple of output channels
        pooling: pooling method ('max' or 'average')
    Inputs:
        points: (N, P, C_p)
        features: (N, P, C_0)
    Returns:
        transformed points: (N, P, C_out), C_out = channels[-1]
    """

    def __init__(self, num_points, K, channels, with_bn=True,
                activation='relu', pooling='average', name='edgeconv',
                **kwargs):
        
        super(EdgeConv, self).__init__()

        self.num_points = num_points
        self.K = K
        self.channels = channels
        self.with_bn = with_bn
        self.activation = activation
        self.pooling = pooling
        self.name_ = name

        self.Conv2D_layers = []
        self.BatchNormalization_layers = []
        self.Activation_layers = []

        for idx, channel in enumerate(self.channels):
            self.Conv2D_layers.append(keras.layers.Conv2D(channel, kernel_size=(1, 1), strides=1, data_format='channels_last',
                                      use_bias=False if self.with_bn else True, kernel_initializer='glorot_normal', name='%s_conv%d' % (self.name_, idx)))
            if self.with_bn:
                self.BatchNormalization_layers.append(keras.layers.BatchNormalization(name='%s_bn%d' % (self.name_, idx)))
            if self.activation:
                self.Activation_layers.append(keras.layers.Activation(self.activation, name='%s_act%d' % (self.name_, idx)))

        self.shortcut = keras.layers.Conv2D(self.channels[-1], kernel_size=(1, 1), strides=1, data_format='channels_last',
                        use_bias=False if self.with_bn else True, kernel_initializer='glorot_normal', name='%s_sc_conv' % self.name_)
        if self.with_bn:
            self.shortcut_batchnorm = keras.layers.BatchNormalization(name='%s_sc_bn' % self.name_)
        if self.activation:
            self.shortcut_activ = keras.layers.Activation(self.activation, name='%s_sc_act' % self.name_)

    @tf.function
    def call(self, points, features):

        with tf.name_scope('edgeconv'):

            # distance
            D = batch_distance_matrix_general(points, points)  # (N, P, P)
            _, indices = tf.nn.top_k(-D, k=self.K + 1)  # (N, P, K+1)
            indices = indices[:, :, 1:]  # (N, P, K)

            fts = features
            knn_fts = knn(self.num_points, self.K, indices, fts)  # (N, P, K, C)
            knn_fts_center = tf.tile(tf.expand_dims(fts, axis=2), (1, 1, self.K, 1))  # (N, P, K, C)
            knn_fts = tf.concat([knn_fts_center, tf.subtract(knn_fts, knn_fts_center)], axis=-1)  # (N, P, K, 2*C)

            x = knn_fts
            for idx, channel in enumerate(self.channels):
                x = self.Conv2D_layers[idx](x)
                if self.with_bn:
                    x = self.BatchNormalization_layers[idx](x)
                if self.activation:
                    x = self.Activation_layers[idx](x)

            if self.pooling == 'max':
                fts = tf.reduce_max(x, axis=2)  # (N, P, C')
            else:
                fts = tf.reduce_mean(x, axis=2)  # (N, P, C')

            # shortcut
            sc = self.shortcut(tf.expand_dims(features, axis=2))
            if self.with_bn:
                sc = self.shortcut_batchnorm(sc)
            sc = tf.squeeze(sc, axis=2)

            if self.activation:
                return self.shortcut_activ (sc + fts)  # (N, P, C')
            else:
                return sc + fts

class ParticleNet(tf.keras.Model):

    # points : (N, P, C_coord)
    # features:  (N, P, C_features)
    # mask: (N, P, 1), optinal

    def __init__(self, name=None,cfg=None):

        super(ParticleNet, self).__init__()

        self.setting = _DotDict()

        self.setting.num_class = cfg["Setup"]["output_classes"]

        # conv_params: list of tuple in the format (K, (C1, C2, C3))
        self.setting.conv_params = []
        for layer_setup in cfg["SetupParticleNet"]["conv_params"]:
            self.setting.conv_params.append((layer_setup[0],tuple(layer_setup[1]),))

        # conv_pooling: 'average' or 'max'
        self.setting.conv_pooling = 'average'

        # fc_params: list of tuples in the format (C, drop_rate)
        self.setting.fc_params = []
        for layer_setup in cfg["SetupParticleNet"]["dense_params"]:
            self.setting.fc_params.append((layer_setup, cfg["SetupParticleNet"]["dropout_rate"],))

        assert(cfg["SequenceLength"]["PfCand"]==cfg["SequenceLength"]["PfCandCategorical"])
        self.setting.num_points = cfg["SequenceLength"]["PfCand"]

        self.map_features = cfg["input_map"]["PfCand"]
        self.name_ = name

        self.batch_norm = keras.layers.BatchNormalization(name='%s_fts_bn' % self.name_)

        self.edge_conv_layers = []

        for layer_idx, layer_param in enumerate(self.setting.conv_params):
            K, channels = layer_param
            self.edge_conv_layers.append(EdgeConv(self.setting.num_points, K, channels, with_bn=True, activation='relu',
                    pooling=self.setting.conv_pooling, name='%s_%s%d' % (self.name_, 'EdgeConv', layer_idx)))

        self.dense_layers = []
        self.dense_dropout = []

        if self.setting.fc_params is not None:

            for layer_idx, layer_param in enumerate(self.setting.fc_params):
                units, drop_rate = layer_param
                self.dense_layers.append(keras.layers.Dense(units, activation='relu'))
                if drop_rate is not None and drop_rate > 0:
                    self.dense_dropout.append(keras.layers.Dropout(drop_rate))

            self.out = keras.layers.Dense(self.setting.num_class, activation='softmax')



    @tf.function
    def call(self, input_):

        xx = input_[0]
        xx_mask = tf.expand_dims(xx[:,:,self.map_features['pfCand_valid']],-1)
        xx_coord = xx[:,:,-2:]
        xx_cat = input_[1]

        xx_ftr = tf.concat((xx_cat, xx),axis = 2)    

        with tf.name_scope(self.name):

            mask = tf.cast(tf.not_equal(xx_mask, 0), dtype='float32')  # 1 if valid
            coord_shift = tf.multiply(999., tf.cast(tf.equal(mask, 0), dtype='float32'))  # make non-valid positions to 99

            fts = tf.squeeze(self.batch_norm(tf.expand_dims(xx_ftr, axis=2)), axis=2)

            for layer_idx, layer_param in enumerate(self.setting.conv_params):
                pts = tf.add(coord_shift, xx_coord) if layer_idx == 0 else tf.add(coord_shift, fts)
                fts = self.edge_conv_layers[layer_idx](pts, fts)

            fts = tf.multiply(fts, mask)
            pool = tf.reduce_mean(fts, axis=1)  # (N, C)

            x = pool
            for layer_idx, layer_param in enumerate(self.setting.fc_params):
                units, drop_rate = layer_param
                x = self.dense_layers[layer_idx](x)
                if drop_rate is not None and drop_rate > 0:
                    x = self.dense_dropout[layer_idx](x)

            out = self.out(x)

            return out  # (N, num_classes)


def compile_model(model, learning_rate):

    opt = tf.keras.optimizers.Nadam(learning_rate=learning_rate, schedule_decay=1e-4)
    # opt = tf.keras.optimizers.Adam(learning_rate = learning_rate)

    metrics = ["accuracy",
               tf.keras.metrics.BinaryAccuracy(name='BinaryAccuracy'),
               tf.keras.metrics.AUC(name='AUC', curve='ROC')]
    model.compile(loss='binary_crossentropy', optimizer=opt, metrics=metrics)

    # log metric names for passing them during model loading
    # metric_names = {(m if isinstance(m, str) else m.__name__): '' for m in metrics}
    # mlflow.log_dict(metric_names, 'input_cfg/metric_names.json')


def run_training(model, data_loader, to_profile, log_suffix):

    gen_train = data_loader.get_generator(primary_set = True)
    gen_val = data_loader.get_generator(primary_set = False)
    input_shape, input_types = data_loader.get_shape()

    data_train = tf.data.Dataset.from_generator(
        gen_train, output_types = input_types, output_shapes = input_shape
        ).prefetch(tf.data.AUTOTUNE)
    data_val = tf.data.Dataset.from_generator(
        gen_val, output_types = input_types, output_shapes = input_shape
        ).prefetch(tf.data.AUTOTUNE)

    net_setups =  data_loader.config["SetupBaseNN"]
    model_name = net_setups["model_name"]
    log_name = '%s_%s' % (model_name, log_suffix)
    csv_log_file = "metrics.log"
    if os.path.isfile(csv_log_file):
        close_file(csv_log_file)
        os.remove(csv_log_file)
    csv_log = CSVLogger(csv_log_file, append=True)
    time_checkpoint = TimeCheckpoint(12*60*60, log_name)
    callbacks = [time_checkpoint, csv_log]

    # does not allow perbatch logging: 
    logs = log_name + '_' + datetime.now().strftime("%Y.%m.%d(%H:%M)")
    tboard_callback = tf.keras.callbacks.TensorBoard(log_dir = logs,
                                                     profile_batch = ('100, 300' if to_profile else 0))
                                                    #  update_freq = ( 0 if net_setups["n_batches_log"]<=0 else net_setups["n_batches_log"] ))
    callbacks.append(tboard_callback)

    my_callback = LossLogCallback(logs, period = 0 if net_setups["n_batches_log"]<=0 else net_setups["n_batches_log"],
                                  metrics_names=["loss", "accuracy", "BinaryAccuracy", "AUC"])
    callbacks.append(my_callback)

    fit_hist = model.fit(data_train, validation_data = data_val,
                         epochs = net_setups["n_epochs"], initial_epoch = net_setups["epoch"],
                         callbacks = callbacks)
    
    model_path = f"{log_name}_final.tf"
    model.save(model_path, save_format="tf")

    # mlflow logs
    for checkpoint_dir in glob.glob(f'{log_name}*.tf'):
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
        dataloader = DataLoaderReco.DataLoader(training_cfg, scaling_cfg)
        input_shape, _  = dataloader.get_shape()
        dl_config =  dataloader.config


        model = ParticleNet(name=dl_config["SetupBaseNN"]["model_name"], cfg=dl_config)
        model._name = dl_config["SetupBaseNN"]["model_name"]

        # print(input_shape[0])
        # compile_build = tf.ones(input_shape[0], dtype=tf.float32, name=None)
        model.build(list(input_shape[0]))
        compile_model(model, dl_config["SetupBaseNN"]["learning_rate"])
        model.summary()
        fit_hist = run_training(model, dataloader, False, cfg.log_suffix)

        mlflow.log_dict(training_cfg, 'input_cfg/training_cfg.yaml')
        mlflow.log_artifact(scaling_cfg, 'input_cfg')
        mlflow.log_artifact(to_absolute_path("Training_DisTauTag_ParticleNetv1.py"), 'input_cfg')
        mlflow.log_artifact(to_absolute_path("../commonReco.py"), 'input_cfg')
        mlflow.log_artifacts('.hydra', 'input_cfg/hydra')
        mlflow.log_artifact('Training_DisTauTag_ParticleNetv1.log', 'input_cfg/hydra')
        mlflow.log_param('run_id', run_id)
        print(f'\nTraining has finished! Corresponding MLflow experiment name (ID): {cfg.experiment_name}({run_kwargs["experiment_id"]}), and run ID: {run_id}\n')


if __name__ == '__main__':
    main() 