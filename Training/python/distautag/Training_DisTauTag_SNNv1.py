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
from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2_as_graph
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

class SpaceEdgeConv(tf.keras.layers.Layer):

    """SpaceEdgeConv
    Args:
        n_dim: number of dimentional features (from the end of feature vector)
        num_outputs : number of output features
    Inputs:
        features: (N, P, features + n_dim)
    Returns:
        transformed points: (N, P, num_outputs)
    """

    def __init__(self, n_dim, num_outputs, regu_rate, **kwargs):
        super(SpaceEdgeConv, self).__init__(**kwargs)
        self.n_dim        = n_dim
        self.num_outputs  = num_outputs
        self.regu_rate    = regu_rate

    def build(self, input_shape):
        self.A = self.add_weight("A", shape=(input_shape[-1]* 2, self.num_outputs),
                                regularizer=False if self.regu_rate < 0 else tf.keras.regularizers.L2(l2=self.regu_rate),
                                initializer="he_uniform", trainable=True)
        self.b = self.add_weight("b", shape=(self.num_outputs,),
                                regularizer=False if self.regu_rate < 0 else tf.keras.regularizers.L2(l2=self.regu_rate), 
                                initializer="he_uniform",trainable=True)


    def compute_output_shape(self, input_shape):
        return [input_shape[0], input_shape[1], self.num_outputs]

    @tf.function
    def call(self, x, mask):

        x_shape = tf.shape(x)
        N, P = x_shape[0], x_shape[1]

        coor = x[:,:,-self.n_dim:]
        D = batch_distance_matrix_general(coor,coor)    # (N, P, P)
        D = tf.expand_dims(D, axis=-1)  # (N, P, P, 1)
        W = tf.math.exp(-10*D)  # (N, P, P, 1)
        
        a   = tf.tile(x, (1, P, 1))   # (N, P*P, n_features)
        na   = tf.reshape(a, (N, P, P, -1))   # (N, P, P, n_features)

        mask = tf.expand_dims(mask, axis=-1)    # (N, P, 1)
        mask_dim = tf.tile(mask, (1, P ,1))    # (N, P*P, 1)
        mask_dim = tf.reshape(mask_dim, (N, P, P,1))    # (N, P, P, 1)        

        s = na * W * mask_dim   # (N, P, P, n_features)
        ss = tf.reduce_sum(s, axis=2)
        norm_ = tf.reduce_sum(mask_dim, axis = 2)      

        # need to substruct the personal features because they were counted in the 's' summ
        ss = ss - x
        ss = ss/norm_
        x = tf.concat((x, ss), axis = 2)    # (N, P, n_features*2)

        ### Ax+b:
        output = tf.matmul(x, self.A) + self.b
        output = output * mask # reapply mask to be sure

        return output

class SpaceParticleNet(tf.keras.Model):

    def __init__(self, dl_config):

        super(SpaceParticleNet, self).__init__()

        self.map_features = dl_config["input_map"]["PfCand"]

        self.conv1D_params = dl_config["SetupSNN"]["conv1D_params"]
        self.conv_params = dl_config["SetupSNN"]["conv_params"]
        self.dense_params = dl_config["SetupSNN"]["dense_params"]
        self.wiring_period = dl_config["SetupSNN"]["wiring_period"]
        self.dropout_rate = dl_config["SetupSNN"]["dropout_rate"]
        self.regu_rate = dl_config["SetupSNN"]["regu_rate"]
        self.output_labels = dl_config["Setup"]["output_classes"]

        self.embedding_n       = dl_config["n_features"]["PfCandCategorical"]
        self.embedding         = self.embedding_n * [None]

        self.EdgeConv_layers         = []
        self.EdgeConv_bnorm_layers   = []
        self.EdgeConv_acti_layers    = []
        self.dense_layers            = []
        self.dense_batch_norm_layers = []
        self.dense_acti_layers       = []
        self.Conv1D_layers           = []

        if(self.dropout_rate > 0):
            self.EdgeConv_dropout_layers = []
            self.dense_dropout_layers = []

        # enumerate embedding in the correct order
        # based on enume class PfCandCategorical
        # for var in dl_config["input_map"]["PfCandCategorical"]:
        #     self.embedding[dl_config["input_map"]["PfCandCategorical"][var]] = \
        #         tf.keras.layers.Embedding(dl_config['embedded_param']['PfCandCategorical'][var][0],
        #                                   dl_config['embedded_param']['PfCandCategorical'][var][1])

        for idx, channel in enumerate(self.conv1D_params):
            self.Conv1D_layers.append(keras.layers.Conv1D(channel, kernel_size=1, name='Conv1D_{}'.format(idx), activation = 'relu'))

        for i,(n_dim, n_output) in enumerate(self.conv_params):
            self.EdgeConv_layers.append(SpaceEdgeConv(n_dim=n_dim, num_outputs=n_output, regu_rate = self.regu_rate, name='EdgeConv_{}'.format(i)))
            self.EdgeConv_bnorm_layers.append(tf.keras.layers.BatchNormalization(name='EdgeConv_bnorm_{}'.format(i)))
            self.EdgeConv_acti_layers.append(tf.keras.layers.Activation("relu", name='EdgeConv_acti_{}'.format(i)))
            if(self.dropout_rate > 0):
                self.EdgeConv_dropout_layers.append(tf.keras.layers.Dropout(self.dropout_rate ,name='EdgeConv_dropout_{}'.format(i)))

        for i, n_dense in enumerate(self.dense_params):
            self.dense_layers.append(tf.keras.layers.Dense(n_dense, kernel_initializer="he_uniform", bias_initializer="he_uniform",
                            kernel_regularizer=None if self.regu_rate<0 else tf.keras.regularizers.L2(l2=self.regu_rate), 
                            bias_regularizer=None if self.regu_rate<0 else tf.keras.regularizers.L2(l2=self.regu_rate),
                            name='dense_{}'.format(i)))
            self.dense_batch_norm_layers.append(tf.keras.layers.BatchNormalization(name='dense_batch_normalization_{}'.format(i)))
            self.dense_acti_layers.append(tf.keras.layers.Activation("relu", name='dense_acti_{}'.format(i)))
            if(self.dropout_rate > 0):
                self.dense_dropout_layers.append(tf.keras.layers.Dropout(self.dropout_rate ,name='dropout_dense_{}'.format(i)))

        self.dense_out = tf.keras.layers.Dense(self.output_labels, kernel_initializer="he_uniform", activation='softmax',
                                               bias_initializer="he_uniform", name='dense_final')

    @tf.function
    def call(self, input_):
        x = input_[0]
        x_cat = input_[1]
        x_mask = x[:,:,self.map_features['pfCand_valid']]
        
        # xx_emb = [self.embedding[i](x_cat[:,:,i]) for i in range(self.embedding_n)]
        
        x = tf.concat((x_cat,x), axis = 2) # (N, P, n_categorical + n_pf_features)

        coor = x[:,:,-2:]
        for i in range(len(self.Conv1D_layers)):
            x = self.Conv1D_layers[i](x)
        x = tf.concat([x, coor], axis=2)    
        
        # x = tf.concat((*xx_emb,x), axis = 2) # (N, P, n_categorical + n_pf_features)
        x0 = x
        for i in range(len(self.EdgeConv_layers)):
            if(self.wiring_period>0):
                if(i % self.wiring_period == 0 and i > 0):
                    x = tf.concat([x0, x], axis=2)
            x = self.EdgeConv_layers[i](x, mask=x_mask)
            if(self.wiring_period>0):
                if(i % self.wiring_period == 0 and i > 0):
                    x0 = x
            x = self.EdgeConv_bnorm_layers[i](x)
            x = self.EdgeConv_acti_layers[i](x)
            #if(self.wiring_period>0):
             #   if(i % self.wiring_period == 0 and i > 0):
              #      x0 = x            
            if(self.dropout_rate > 0):
                x = self.EdgeConv_dropout_layers[i](x)

        x_mask = tf.expand_dims(x_mask, axis=-1)
        x_mean = tf.reduce_mean(x * x_mask, axis=1)
        x_max = tf.reduce_max(x * x_mask, axis=1)
        x = tf.concat([x_mean, x_max], axis=1)

        for i in range(len(self.dense_layers)):
            x = self.dense_layers[i](x)
            x = self.dense_batch_norm_layers[i](x)
            x = self.dense_acti_layers[i](x)
            if(self.dropout_rate > 0):
                x = self.dense_dropout_layers[i](x)

        x = self.dense_out(x)

        return x

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

        dl_config =  dataloader.config
        model = SpaceParticleNet(dl_config)
        model._name = dl_config["SetupBaseNN"]["model_name"]
        input_shape, _  = dataloader.get_shape()

        model.build(list(input_shape[0]))
        compile_model(model, dl_config["SetupBaseNN"]["learning_rate"])
        model.summary()
        exit()
        fit_hist = run_training(model, dataloader, False, cfg.log_suffix)

        mlflow.log_dict(training_cfg, 'input_cfg/training_cfg.yaml')
        mlflow.log_artifact(scaling_cfg, 'input_cfg')
        mlflow.log_artifact(to_absolute_path("Training_DisTauTag_SNNv1.py"), 'input_cfg')
        mlflow.log_artifact(to_absolute_path("../commonReco.py"), 'input_cfg')
        mlflow.log_artifacts('.hydra', 'input_cfg/hydra')
        mlflow.log_artifact('Training_DisTauTag_SNNv1.log', 'input_cfg/hydra')
        mlflow.log_param('run_id', run_id)
        
        print(f'\nTraining has finished! Corresponding MLflow experiment name (ID): {cfg.experiment_name}({run_kwargs["experiment_id"]}), and run ID: {run_id}\n')


if __name__ == '__main__':
    main() 
