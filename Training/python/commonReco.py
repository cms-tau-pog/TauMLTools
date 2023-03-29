import time
import gc
import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import Callback, ModelCheckpoint, CSVLogger

class LossLogCallback(Callback):

    def __init__(self, log_dir, period = 1, metrics_names=[]):
        self.period = period
        self.metrics_names = metrics_names
        self.global_step = 0
        self.writer = tf.summary.create_file_writer(log_dir+"/steps/")

    def on_batch_end(self, batch, logs):
        if self.period == 0: return
        if batch % self.period != 0: return
        with self.writer.as_default():
          for name in self.metrics_names:
            # print('\nbatch_'+name, logs[name], batch)
            tf.summary.scalar('batch_'+name, data=logs[name], step=self.global_step)
        self.global_step = self.global_step + self.period

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
            self.model.save('{}_historic_b{}_{:.1f}h.tf'.format(self.file_name_prefix, batch, abs_delta_t_h),
                            save_format="tf")
            self.last_check_time = current_time

    def on_epoch_end(self, epoch, logs=None):
        self.model.save('{}_e{}.tf'.format(self.file_name_prefix, epoch),
                        save_format="tf")
        print("Epoch {} is ended.".format(epoch))

def close_file(f_name):
    file_objs = [ obj for obj in gc.get_objects() if ("TextIOWrapper" in str(type(obj))) and (obj.name == f_name)]
    for obj in file_objs:
        obj.close()

def load_graph(graph_filename):
    with tf.gfile.GFile(graph_filename, 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())

    with tf.Graph().as_default() as graph:
        tf.import_graph_def(graph_def, name="deepTau")
    return graph

#############################################################################################
##### Custom metrics:
### Accuracy calculation for number of charged/neutral hadrons:
@tf.function
def my_acc(y_true, y_pred):
    # print('\naccuracy calcualtion:')
    y_true = tf.math.round(y_true)
    y_true_int = tf.cast(y_true, tf.int32)
    y_pred = tf.math.round(y_pred)
    y_pred_int = tf.cast(y_pred, tf.int32)
    result = tf.math.logical_and(y_true_int[:, 0] == y_pred_int[:, 0], y_true_int[:, 1] == y_pred_int[:, 1])
    return tf.cast(result, tf.float32)


### Resolution of 4-momentum:
class MyResolution(tf.keras.metrics.Metric):
    def __init__(self, _name, var_pos, is_relative = False,**kwargs):
        super(MyResolution, self).__init__(name=_name,**kwargs)
        self.is_relative = is_relative
        self.var_pos     = var_pos
        self.sum_x       = self.add_weight(name="sum_x", initializer="zeros")
        self.sum_x2      = self.add_weight(name="sum_x2", initializer="zeros")
        self.N           = self.add_weight(name="N", initializer="zeros")

    @tf.function
    def update_state(self, y_true, y_pred, sample_weight=None):
        self.N.assign_add(tf.cast(tf.shape(y_true)[0], dtype=tf.float32))
        if(self.is_relative):
            self.sum_x.assign_add(tf.math.reduce_sum((y_pred[:,self.var_pos]   - y_true[:,self.var_pos])/y_true[:,self.var_pos]))
            self.sum_x2.assign_add(tf.math.reduce_sum(((y_pred[:,self.var_pos] - y_true[:,self.var_pos])/y_true[:,self.var_pos])**2))
        else:
            self.sum_x.assign_add(tf.math.reduce_sum(y_pred[:,self.var_pos]   - y_true[:,self.var_pos]))
            self.sum_x2.assign_add(tf.math.reduce_sum((y_pred[:,self.var_pos] - y_true[:,self.var_pos])**2))

    @tf.function
    def result(self):
        mean_x  = self.sum_x/self.N
        mean_x2 = self.sum_x2/self.N
        return mean_x2 -  mean_x**2

    def reset(self):
        self.sum_x.assign(0.)
        self.sum_x2.assign(0.)
        self.N.assign(0.)

    def get_config(self):
        config = {
            "is_relative": self.is_relative,
            "var_pos": self.var_pos,
            "name": self.name,
        }
        base_config = super(MyResolution, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    @classmethod
    def from_config(config):
        raise RuntimeError("Im here")
        return MyResolution(config["name"], config["var_pos"], is_relative=config["is_relative"])

pt_res_obj_rel = MyResolution('pt_res_rel' , 2 ,True)
pt_res_obj     = MyResolution('pt_res' , 2 ,False)
m2_res_obj     = MyResolution('m^2_res', 3 ,False)

def pt_res(y_true, y_pred, sample_weight=None):
    global pt_res_obj
    pt_res_obj.update_state(y_true, y_pred)
    return pt_res_obj.result()

def pt_res_rel(y_true, y_pred, sample_weight=None):
    global pt_res_obj_rel
    pt_res_obj_rel.update_state(y_true, y_pred)
    return pt_res_obj_rel.result()

def m2_res(y_true, y_pred, sample_weight=None):
    global m2_res_obj
    m2_res_obj.update_state(y_true, y_pred)
    return m2_res_obj.result()

@tf.function
def my_mse_ch(y_true, y_pred):
    def_mse = 0.19802300936720527
    w = tf.constant(1/def_mse)
    # new_w = tf.where(tf.logical_and(y_true[:,0] == 1, y_true[:,1] == 0), w*2, w)
    # new_w = tf.where(y_true[:,0] == 1, w*2, w)
    return w*tf.square(y_true[:,0] - y_pred[:,0])

# @tf.function
def my_mse_neu(y_true, y_pred):
    def_mse = 0.4980008353282306
    w = 1/def_mse
    # new_w = tf.where(tf.logical_and(y_true[:,0] == 1, y_true[:,1] == 0), w*2, w)
    # new_w = tf.where(y_true[:,0] == 1, w*2, w)
    return w*tf.square(y_true[:,1] - y_pred[:,1])

cat_cross = tf.keras.losses.CategoricalCrossentropy()

def my_cat_dm(y_true, y_pred):
    global cat_cross
    y_true_cat = convert_y(y_true)
    return cat_cross(y_true_cat, y_pred[:, :6])

@tf.function
def my_mse_pt(y_true, y_pred):
    def_mse = 0.022759110487849007 # relative
    w = 1/def_mse
    # return w*tf.square((y_true[:,2] - y_pred[:,6]) / y_true[:,2]) # dm 6 outputd
    return w*tf.square((y_true[:,2] - y_pred[:,2]) / y_true[:,2])

@tf.function
def my_mse_mass(y_true, y_pred):
    def_mse = 0.5968616152311431
    w = 1/def_mse
    # return w*tf.square(y_true[:,3] - y_pred[:,7]) # dm 6 outputs
    return w*tf.square(y_true[:,3] - y_pred[:,3])

#############################################################
##### new metrics
@tf.function
def log_cosh_pt(y_true, y_pred):
    # def_val = 0.01097714248585314 # without *100
    # def_val = 0.5961012601878 # with *10
    # def_val = 9.210500832319457 # with *100
    # def_val = 4.3278327706899 # *50
    def_val = 1.4785428311515774 # *20 and *30
    # def_val = 3.3668514079633973 #*40
    w = 1/def_val
    delta_pt_rel = (y_true[:,2] - y_pred[:,2]) / y_true[:,2]
    output = tf.math.log(tf.math.cosh(delta_pt_rel*20))
    return w*output

@tf.function
def log_cosh_mass(y_true, y_pred):
    # def_val = 0.20604327826842186 # without * 100
    # def_val = 45.1603939803365 # with * 100
    # def_val = 22.301361586255634 # *50
    def_val = 8.612943444728497 # *20 and *30
    # def_val = 17.733670178795087 # *40
    w = 1/def_val
    delta_mass = y_true[:,3] - y_pred[:,3]
    output = tf.math.log(tf.math.cosh(delta_mass*20))
    return w*output

@tf.function
def my_mse_ch_4(y_true, y_pred):
    def_mse = 0.6809799439884483
    w = 1/def_mse
    return w*tf.pow(y_true[:,0] - y_pred[:,0],4)

@tf.function
def my_mse_neu_4(y_true, y_pred):
    def_mse = 1.2626542147826219
    w = 1/def_mse
    return w*tf.pow(y_true[:,1] - y_pred[:,1],4)

@tf.function
def my_mse_pt_4(y_true, y_pred):
    def_mse = 0.02982611302398386
    w = 1/def_mse
    return w*tf.pow((y_true[:,2] - y_pred[:,2]) / y_true[:,2],4)

@tf.function
def my_mse_mass_4(y_true, y_pred):
    def_mse = 18.941287385879463
    w = 1/def_mse
    return w*tf.pow(y_true[:,3] - y_pred[:,3],4)

def quantile_pt(y_true, y_pred):
    a_def = 0.09867886998338264
    # b_def = 0.12375059356959137 #75-25
    b_def = 0.043592315013624615 # 60-40
    w_a = 1/a_def
    w_b = 1/b_def
    delta_pt_rel = (y_true[:,2] - y_pred[:,2]) / y_true[:,2]
    # q75 = tpf.stats.percentile(delta_pt_rel,75, interpolation='linear')
    # q25 = tpf.stats.percentile(delta_pt_rel,25, interpolation='linear')
    # return w_a*0.2*tf.math.abs(delta_pt_rel) + w_b*0.8*tf.math.abs(q75-q25)
    q60 = tpf.stats.percentile(delta_pt_rel,60, interpolation='linear')
    q40 = tpf.stats.percentile(delta_pt_rel,40, interpolation='linear')
    return w_a*0.5*tf.math.abs(delta_pt_rel) + w_b*0.5*tf.math.abs(q60-q40)

def my_mse_ch_new(y_true, y_pred):
    def_mse = 0.19802300936720527
    w = 1/def_mse # ca. 5.0505
    w2 = w+1
    return tf.where(y_true[:,0]== 1, w2*tf.square(y_true[:,0] - y_pred[:,0]), w*tf.square(y_true[:,0] - y_pred[:,0]))

@tf.function
def my_mse_neu_new(y_true, y_pred):
    def_mse = 0.4980008353282306
    w = 1/def_mse # ca. 2.008
    w2 = w+1
    return tf.where(y_true[:,1]==0,w2*tf.square(y_true[:,1] - y_pred[:,1]),w*tf.square(y_true[:,1] - y_pred[:,1]))


def convert_y(y):
    pi = tf.cast(tf.logical_and(y[:, 0] == 1, y[:, 1] == 0), tf.float32)
    pip0 = tf.cast(tf.logical_and(y[:, 0] == 1, y[:, 1] == 1), tf.float32)
    pi2p0 = tf.cast(tf.logical_and(y[:, 0] == 1, y[:, 1] == 2), tf.float32)
    pipipi = tf.cast(tf.logical_and(y[:, 0] == 3, y[:, 1] == 0), tf.float32)
    pipipip0 = tf.cast(tf.logical_and(y[:, 0] == 3, y[:, 1] == 1), tf.float32)
    other = 1 - (pi + pip0 + pi2p0 + pipipi + pipipip0)

    return tf.stack([pi, pip0, pi2p0, pipipi, pipipip0, other], axis = 1)

##### Custom loss function:
class CustomMSE(tf.keras.losses.Loss):
    mode = None

    def __init__(self, name="custom_mse", **kwargs):
        super().__init__(name=name,**kwargs)

    @tf.function
    def call(self, y_true, y_pred):
        y_shape = tf.shape(y_true)
        mse = tf.zeros(y_shape[0])
        if "dm" in CustomMSE.mode:
            mse = my_mse_ch(y_true, y_pred) + my_mse_neu(y_true, y_pred)
            # mse = my_cat_dm(y_true, y_pred)
            # mse = my_mse_ch_4(y_true, y_pred) + my_mse_neu_4(y_true, y_pred)
            # mse = my_mse_ch_new(y_true, y_pred) + my_mse_neu_new(y_true, y_pred)
        if "p4" in CustomMSE.mode:
            mse = mse + my_mse_pt(y_true, y_pred) + my_mse_mass(y_true, y_pred)
            # mse = mse + 0.01 * (my_mse_pt(y_true, y_pred) + my_mse_mass(y_true, y_pred))
            # mse = mse + 0.1 * (my_mse_pt(y_true, y_pred) + my_mse_mass(y_true, y_pred))
            # mse = mse + my_mse_pt(y_true, y_pred) + my_mse_mass(y_true, y_pred) + log_cosh_pt(y_true,y_pred) + log_cosh_mass(y_true,y_pred)
            # mse = mse + my_mse_pt(y_true, y_pred) + my_mse_mass(y_true, y_pred) + log_cosh_mass(y_true,y_pred)
            # mse = mse + my_mse_pt_4(y_true, y_pred) + my_mse_mass_4(y_true, y_pred)
            # mse = mse + my_mse_pt(y_true, y_pred) + my_mse_mass(y_true, y_pred) + quantile_pt(y_true, y_pred)
        return mse


def decay_mode_histo(x1, x2, dm_bins):
    decay_mode = np.zeros((x1.shape[0],2))
    decay_mode[:,0] = x1
    decay_mode[:,1] = x2
    h_dm, _ = np.histogramdd(decay_mode, bins=[dm_bins,dm_bins])
    h_dm[:,-1] = h_dm[:,4]+h_dm[:,-1] # sum the last and 4. column into the last column
    h_dm = np.delete(h_dm,4,1)        # delete the 4. column
    h_dm[-1,:] = h_dm[4,:]+h_dm[-1,:] # sum the last and 4. column into the last column
    h_dm = np.delete(h_dm,4,0)        # delete the 4. column
    return h_dm