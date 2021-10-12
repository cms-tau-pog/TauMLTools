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

class MyGNNLayer(tf.keras.layers.Layer):
    def __init__(self, n_dim, num_outputs, regu_rate, **kwargs):
        super(MyGNNLayer, self).__init__(**kwargs)
        self.n_dim        = n_dim
        self.num_outputs  = num_outputs
        self.supports_masking = True # to pass the mask to the next layers and not destroy it
        self.regu_rate = regu_rate

    def build(self, input_shape):
        if(self.regu_rate < 0):
            self.A = self.add_weight("A", shape=((input_shape[-1]+1) * 2 - 1, self.num_outputs),
                                    initializer="he_uniform", trainable=True)
            self.b = self.add_weight("b", shape=(self.num_outputs,), initializer="he_uniform",trainable=True)
        else:
            self.A = self.add_weight("A", shape=((input_shape[-1]+1) * 2 - 1, self.num_outputs),
                                    initializer="he_uniform", regularizer=tf.keras.regularizers.L2(l2=self.regu_rate), trainable=True)
            self.b = self.add_weight("b", shape=(self.num_outputs,), initializer="he_uniform", 
                                    regularizer=tf.keras.regularizers.L2(l2=self.regu_rate),trainable=True)

    def compute_output_shape(self, input_shape):
        return [input_shape[0], input_shape[1], self.num_outputs]

    @tf.function
    def call(self, x, mask):
        ### a and b contain copies for each pf_Cand:
        x_shape = tf.shape(x)

        ## a tensor: a[n_tau, pf_others, pf, features]
        rep = tf.stack([1,x_shape[1],1])
        a   = tf.tile(x, rep)
        a   = tf.reshape(a,(x_shape[0],x_shape[1],x_shape[1],x_shape[2]))

        ## b tensor: a[n_tau, pf, pf_others, features]
        rep = tf.stack([1,1,x_shape[1]])
        b   = tf.tile(x, rep)
        b   = tf.reshape(b,(x_shape[0],x_shape[1],x_shape[1],x_shape[2]))


        ### Compute distances:
        ca = a[:,:,:, -self.n_dim:]
        cb = b[:,:,:, -self.n_dim:]
        c_shape = tf.shape(ca)
        diff = ca-cb
        diff = tf.math.square(diff)
        dist = tf.math.reduce_sum(diff, axis = -1)
        dist = tf.reshape(dist,(c_shape[0],c_shape[1],c_shape[2],1)) # needed to concat
        na   = tf.concat((a,dist),axis=-1) #a[n_tau, pf_others, pf, features+1]


        ### Weighted sum of features:
        w = tf.math.exp(-10*na[:,:,:,-1]) # weights
        w_shape = tf.shape(w)
        w    = tf.reshape(w,(w_shape[0],w_shape[1],w_shape[2],1)) # needed for multiplication
        mask = tf.reshape(mask, (w_shape[0],w_shape[1],1)) # needed for multiplication
        ## copies of mask:
        rep  = tf.stack([1,w_shape[1],1])
        mask_copy = tf.tile(mask, rep)
        mask_copy = tf.reshape(mask_copy,(w_shape[0],w_shape[1],w_shape[2],1))
        # mask_copy = [n_tau, n_pf_others, n_pf, mask]
        s = na * w * mask_copy # weighted na
        ss = tf.math.reduce_sum(s, axis = 1) # weighted sum of features
        # ss = [n_tau, n_pf, features+1]
        self_dist = tf.zeros((x_shape[0], x_shape[1], 1))
        xx = tf.concat([x, self_dist], axis = 2) # [n_tau, n_pf, features+1]
        ss = ss - xx # difference between weighted features and original ones
        x = tf.concat((x, ss), axis = 2) # add to original features
        # print('check x shape 2: ', x) #(n_tau, n_pf, features*2+1)


        ### Ax+b:
        output = tf.matmul(x, self.A) + self.b

        # print('output.shape: ', output.shape)
        output = output * mask # reapply mask to be sure

        return output

class MyGNN(tf.keras.Model):

    def __init__(self, dl_config):
        super(MyGNN, self).__init__()

        self.map_features = dl_config["input_map"]["PfCand"]

        self.mode = dl_config["SetupNN"]["mode"]

        self.n_gnn_layers      = dl_config["SetupNN"]["n_gnn_layers"]
        self.n_dim_gnn         = dl_config["SetupNN"]["n_dim_gnn"]
        self.n_output_gnn      = dl_config["SetupNN"]["n_output_gnn"]
        self.n_output_gnn_last = dl_config["SetupNN"]["n_output_gnn_last"]
        self.n_dense_layers    = dl_config["SetupNN"]["n_dense_layers"]
        self.n_dense_nodes     = dl_config["SetupNN"]["n_dense_nodes"]
        self.wiring_mode       = dl_config["SetupNN"]["wiring_mode"]
        self.dropout_rate      = dl_config["SetupNN"]["dropout_rate"]
        self.regu_rate         = dl_config["SetupNN"]["regu_rate"]
        self.embedding_n       = dl_config["n_features"]["PfCandCategorical"]
        self.embedding         = self.embedding_n * [None]

        self.GNN_layers  = []
        self.batch_norm  = []
        self.acti_gnn    = []
        self.dense            = []
        self.dense_batch_norm = []
        self.dense_acti       = []
        if(self.dropout_rate > 0):
            self.dropout_gnn = []
            self.dropout_dense    = []

        list_outputs = [self.n_output_gnn] * (self.n_gnn_layers-1) + [self.n_output_gnn_last]
        list_n_dim   = [2] + [self.n_dim_gnn] * (self.n_gnn_layers-1)
        self.n_gnn_layers = len(list_outputs)
        self.n_dense_layers = self.n_dense_layers

        # enumerate embedding in the correct order
        # based on enume class PfCandCategorical
        for var in dl_config["input_map"]["PfCandCategorical"]:
            self.embedding[dl_config["input_map"]["PfCandCategorical"][var]] = \
                tf.keras.layers.Embedding(dl_config['embedded_param']['PfCandCategorical'][var][0],
                                          dl_config['embedded_param']['PfCandCategorical'][var][1])

        for i in range(self.n_gnn_layers):
            self.GNN_layers.append(MyGNNLayer(n_dim=list_n_dim[i], num_outputs=list_outputs[i], regu_rate = self.regu_rate, name='GNN_layer_{}'.format(i)))
            self.batch_norm.append(tf.keras.layers.BatchNormalization(name='batch_normalization_{}'.format(i)))
            self.acti_gnn.append(tf.keras.layers.Activation("tanh", name='acti_gnn_{}'.format(i)))
            if(self.dropout_rate > 0):
                self.dropout_gnn.append(tf.keras.layers.Dropout(self.dropout_rate ,name='dropout_gnn_{}'.format(i)))

        for i in range(self.n_dense_layers-1):
            if(self.regu_rate < 0):
                self.dense.append(tf.keras.layers.Dense(self.n_dense_nodes, kernel_initializer="he_uniform",
                                    bias_initializer="he_uniform", name='dense_{}'.format(i)))
            else:
                self.dense.append(tf.keras.layers.Dense(self.n_dense_nodes, kernel_initializer="he_uniform",
                                bias_initializer="he_uniform", kernel_regularizer=tf.keras.regularizers.L2(l2=self.regu_rate), 
                                bias_regularizer=tf.keras.regularizers.L2(l2=self.regu_rate), name='dense_{}'.format(i)))
            self.dense_batch_norm.append(tf.keras.layers.BatchNormalization(name='dense_batch_normalization_{}'.format(i)))
            self.dense_acti.append(tf.keras.layers.Activation("sigmoid", name='dense_acti{}'.format(i)))
            if(self.dropout_rate > 0):
                self.dropout_dense.append(tf.keras.layers.Dropout(self.dropout_rate ,name='dropout_dense_{}'.format(i)))

        n_last = 4 if self.mode == "p4_dm" else 2
        # self.dense_dm = tf.keras.layers.Dense(6, kernel_initializer="he_uniform",
        #                         bias_initializer="he_uniform", activation="softmax", name='dense_dm')
        # self.dense_p4 = tf.keras.layers.Dense(2, kernel_initializer="he_uniform",
        #                         bias_initializer="he_uniform", name='dense_p4')
        self.dense2 = tf.keras.layers.Dense(n_last, kernel_initializer="he_uniform",
                                bias_initializer="he_uniform", name='dense2')

    @tf.function
    def call(self, input_):
        xx = input_[0]
        x_mask = xx[:,:,self.map_features['pfCand_valid']]
        
        # xx_cat = tf.split(input[1], num_or_size_splits=self.embedding_n, axis=-1)
        xx_cat = input_[1]
        xx_emb = [self.embedding[i](xx_cat[:,:,i]) for i in range(self.embedding_n)]

        x = tf.concat((xx, *xx_emb),axis = 2)
        if(self.wiring_mode=="m2"):
            for i in range(self.n_gnn_layers):
                if i > 1:
                    x = tf.concat([x0, x], axis=2)
                x = self.GNN_layers[i](x, mask=x_mask)
                if i == 0:
                    x0 = x
                x = self.batch_norm[i](x)
                x = self.acti_gnn[i](x)
                if(self.dropout_rate > 0):
                    x = self.dropout_gnn[i](x)
        elif(self.wiring_mode=="m1"):
            for i in range(self.n_gnn_layers):
                x = self.GNN_layers[i](x, mask=x_mask)
                x = self.batch_norm[i](x)
                x = self.acti_gnn[i](x)
                if(self.dropout_rate > 0):
                    x = self.dropout_gnn[i](x)
        elif(self.wiring_mode=="m3"):
            for i in range(self.n_gnn_layers):
                if(i%3==0 and i > 0):
                    x = tf.concat([x0, x], axis=2)
                x = self.GNN_layers[i](x, mask=x_mask)
                if(i%3==0):
                    x0 = x
                x = self.batch_norm[i](x)
                x = self.acti_gnn[i](x)
                if(self.dropout_rate > 0):
                    x = self.dropout_gnn[i](x)


        if("p4" in self.mode):
            xx_p4 = xx[:,:,self.map_features['pfCand_px']:self.map_features['pfCand_E']+1]
            xx_p4_shape = tf.shape(xx_p4)
            xx_p4_other = xx[:,:,self.map_features['pfCand_pt']:self.map_features['pfCand_mass']+1]

            x_coor = x[:,:, -self.n_dim_gnn:]
            x_coor = tf.math.square(x_coor)
            d = tf.square(tf.math.reduce_sum(x_coor, axis = -1))
            w = tf.reshape(tf.math.exp(-10*d), (xx_p4_shape[0], xx_p4_shape[1], 1))

            x_mask_shape = tf.shape(x_mask)
            x_mask = tf.reshape(x_mask, (x_mask_shape[0], x_mask_shape[1], 1))
            sum_p4 = tf.reduce_sum(xx_p4 * w * x_mask, axis=1)
            # print('sum_p4.shape: ', sum_p4.shape) #(100,4)
            sum_p4_other = self.ToPtM2(sum_p4)

            x = tf.concat([x, xx_p4, xx_p4_other], axis = 2)

            #xx_p4 = tf.reshape(xx_p4, (xx_p4_shape[0], xx_p4_shape[1] * xx_p4_shape[2]))
            x_shape = tf.shape(x)
            x = tf.reshape(x, (x_shape[0], x_shape[1] * x_shape[2]))
            x = tf.concat([x, sum_p4, sum_p4_other], axis = 1)
            
        elif("dm"==self.mode):
            x_shape = tf.shape(x)
            x = tf.reshape(x, (x_shape[0], x_shape[1] * x_shape[2]))


        for i in range(self.n_dense_layers-1):
            x = self.dense[i](x)
            x = self.dense_batch_norm[i](x)
            x = self.dense_acti[i](x)
            if(self.dropout_rate > 0):
                x = self.dropout_dense[i](x)
        ### dm 6 outputs:
        # x_dm = self.dense_dm(x)
        # x_p4 = self.dense_p4(x)
        # return tf.concat([x_dm, x_p4], axis=1)
        ###
        
        x = self.dense2(x)

        x_zeros = tf.zeros((x_shape[0], 2))
        if(self.mode == "dm"):
            xout = tf.concat([x, x_zeros], axis=1)
        elif self.mode == "p4":
            xout = tf.concat([x_zeros, x], axis=1)
        else:
            xout = x

        # print('xout shape: ',xout)
        return xout

    def ToPtM2(self, x):
        mypx  = x[:,0]
        mypy  = x[:,1]
        mypz  = x[:,2]
        myE   = x[:,3]

        mypx2  = tf.square(mypx)
        mypy2  = tf.square(mypy)
        mypz2  = tf.square(mypz)
        myE2   = tf.square(myE)

        mypt   = tf.sqrt(mypx2 + mypy2)
        mymass = myE2 - mypx2 - mypy2 - mypz2
        absp   = tf.sqrt(mypx2 + mypy2 + mypz2)

        return tf.stack([mypt,mymass], axis=1)

def compile_model(model, mode, learning_rate):
    # opt = tf.keras.optimizers.Nadam(learning_rate=learning_rate, beta_1=1e-4)
    opt = tf.keras.optimizers.Nadam(learning_rate=learning_rate, schedule_decay=1e-4)
    # opt = tf.keras.optimizers.Adam(learning_rate = learning_rate)
    CustomMSE.mode = mode
    metrics = []
    if "dm" in mode:
        metrics.extend([my_acc, my_mse_ch, my_mse_neu])
    if "p4" in mode:
        metrics.extend([my_mse_pt, my_mse_mass, pt_res, pt_res_rel, m2_res])
    model.compile(loss=CustomMSE(), optimizer=opt, metrics=metrics)
    
    # log metric names for passing them during model loading
    metric_names = {(m if isinstance(m, str) else m.__name__): '' for m in metrics}
    mlflow.log_dict(metric_names, 'input_cfg/metric_names.json')


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

    net_setups =  data_loader.config["SetupNN"]
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
                                                     profile_batch = ('100, 300' if to_profile else 0),
                                                     update_freq = ( 0 if net_setups["n_batches_log"]<=0 else net_setups["n_batches_log"] ))
    callbacks.append(tboard_callback)

    fit_hist = model.fit(data_train, validation_data = data_val,
                         epochs = net_setups["n_epochs"], initial_epoch = net_setups["epoch"],
                         callbacks = callbacks)
    
    model_path = f"{log_name}_final.tf"
    model.save(model_path, save_format="tf")

    # mlflow logs
    for checkpoint_dir in glob(f'{log_name}*.tf'):
         mlflow.log_artifacts(checkpoint_dir, f"model_checkpoints/{checkpoint_dir}")
    mlflow.log_artifacts(model_path, "model")
    mlflow.log_artifacts(logs, "custom_tensorboard_logs")
    mlflow.log_artifact(csv_log_file)

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
        dataloader = DataLoaderReco.DataLoader(training_cfg, scaling_cfg)

        dl_config =  dataloader.config
        model = model = MyGNN(dl_config)
        input_shape, _  = dataloader.get_shape()
        # print(input_shape[0])
        # compile_build = tf.ones(input_shape[0], dtype=tf.float32, name=None)
        model.build(list(input_shape[0]))
        compile_model(model, dl_config["SetupNN"]["mode"], dl_config["SetupNN"]["learning_rate"])
        fit_hist = run_training(model, dataloader, False, cfg.log_suffix)

        mlflow.log_dict(training_cfg, 'input_cfg/training_cfg.yaml')
        mlflow.log_artifact(scaling_cfg, 'input_cfg')
        mlflow.log_artifact(to_absolute_path("Training_SNNv0.py"), 'input_cfg')
        mlflow.log_artifact(to_absolute_path("../commonReco.py"), 'input_cfg')
        mlflow.log_artifacts('.hydra', 'input_cfg/hydra')
        mlflow.log_artifact('Training_SNNv0.log', 'input_cfg/hydra')
        mlflow.log_param('run_id', active_run.info.run_id)
        print(f'\nTraining has finished! Corresponding MLflow experiment name (ID): {cfg.experiment_name}({run_kwargs["experiment_id"]}), and run ID: {active_run.info.run_id}\n')

if __name__ == '__main__':
    main() 