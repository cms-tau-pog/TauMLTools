import time
import gc
import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.models import load_model
from tensorflow import keras

e, mu, tau, jet = 0, 1, 2, 3

def setup_gpu(gpu_cfg):
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            tf.config.set_visible_devices(gpus[gpu_cfg['gpu_index']], 'GPU')
            tf.config.experimental.set_virtual_device_configuration(gpus[gpu_cfg['gpu_index']],
                [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=gpu_cfg['gpu_mem']*1024)])
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            print(e)


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

class TauLosses:
    Le_sf = 1
    Lmu_sf = 1
    Ltau_sf = 1
    Ljet_sf = 1
    epsilon = 1e-7
    merge_thr = 0.1

    @staticmethod
    @tf.function
    def SetSFs(sf_e, sf_mu, sf_tau, sf_jet):
        sf_corr = 4. / (sf_e + sf_mu + sf_tau + sf_jet)
        TauLosses.Le_sf = sf_e * sf_corr
        TauLosses.Lmu_sf = sf_mu * sf_corr
        TauLosses.Ltau_sf = sf_tau * sf_corr
        TauLosses.Ljet_sf = sf_jet * sf_corr

    @staticmethod
    @tf.function
    def Lbase(target, output, genuine_index, fake_index):
        epsilon = tf.constant(TauLosses.epsilon, output.dtype.base_dtype)
        genuine_vs_fake = output[:, genuine_index] / (output[:, genuine_index] + output[:, fake_index] + epsilon)
        genuine_vs_fake = tf.clip_by_value(genuine_vs_fake, epsilon, 1 - epsilon)
        loss = -target[:, genuine_index] * tf.math.log(genuine_vs_fake) - target[:, fake_index] * tf.math.log(1 - genuine_vs_fake)
        return loss

    @staticmethod
    @tf.function
    def Hbase(target, output, index, inverse):
        epsilon = tf.constant(TauLosses.epsilon, output.dtype.base_dtype)
        x = tf.clip_by_value(output[:, index], epsilon, 1 - epsilon)
        if inverse:
            return - (1 - target[:, index]) * tf.math.log(1 - x)
        return - target[:, index] * tf.math.log(x)

    @staticmethod
    @tf.function
    def Fbase(target, output, index, gamma, apply_decay, inverse):
        decay_factor = (tf.math.tanh(70 * (output[:, tau] - 0.1)) + 1) / 2 if apply_decay else 1
        if gamma <= 0:
            raise RuntimeError("Focal Loss requires gamma > 0.")
        epsilon = tf.constant(TauLosses.epsilon, output.dtype.base_dtype)
        gamma_t = tf.constant(gamma, output.dtype.base_dtype)
        x = tf.clip_by_value(output[:, index], epsilon, 1 - epsilon)
        if inverse:
            return - decay_factor * (1 - target[:, index]) * tf.pow(x, gamma_t) * tf.math.log(1 - x)
        return - decay_factor * target[:, index] * tf.pow(1-x, gamma_t) * tf.math.log(x)

    @staticmethod
    @tf.function
    def Le(target, output):
        return TauLosses.Lbase(target, output, tau, e)

    @staticmethod
    @tf.function
    def Lmu(target, output):
        return TauLosses.Lbase(target, output, tau, mu)

    @staticmethod
    @tf.function
    def Ljet(target, output):
        return TauLosses.Lbase(target, output, tau, jet)

    @staticmethod
    @tf.function
    def sLe(target, output):
        sf = tf.constant(TauLosses.Le_sf, output.dtype.base_dtype)
        return sf * TauLosses.Le(target, output)

    @staticmethod
    @tf.function
    def sLmu(target, output):
        sf = tf.constant(TauLosses.Lmu_sf, output.dtype.base_dtype)
        return sf * TauLosses.Lmu(target, output)

    @staticmethod
    @tf.function
    def sLjet(target, output):
        sf = tf.constant(TauLosses.Ljet_sf, output.dtype.base_dtype)
        return sf * TauLosses.Ljet(target, output)

    @staticmethod
    @tf.function
    def He(target, output):
        return TauLosses.Hbase(target, output, e, False)

    @staticmethod
    @tf.function
    def Hmu(target, output):
        return TauLosses.Hbase(target, output, mu, False)

    @staticmethod
    @tf.function
    def Htau(target, output):
        return TauLosses.Hbase(target, output, tau, False)

    @staticmethod
    @tf.function
    def Hjet(target, output):
        return TauLosses.Hbase(target, output, jet, False)

    @staticmethod
    @tf.function
    def Hcat_base(target, output, index, inverse):
        decay_factor = (tf.math.tanh(70 * (output[:, tau] - 0.1)) + 1) / 2
        if inverse:
            decay_factor = 1 - decay_factor
        return decay_factor * TauLosses.Hbase(target, output, index, False)

    @staticmethod
    @tf.function
    def Hcat_e(target, output):
        return TauLosses.Hcat_base(target, output, e, False)

    @staticmethod
    @tf.function
    def Hcat_mu(target, output):
        return TauLosses.Hcat_base(target, output, mu, False)

    @staticmethod
    @tf.function
    def Hcat_jet(target, output):
        return TauLosses.Hcat_base(target, output, jet, False)

    @staticmethod
    @tf.function
    def Hcat_eInv(target, output):
        return TauLosses.Hcat_base(target, output, e, True)

    @staticmethod
    @tf.function
    def Hcat_muInv(target, output):
        return TauLosses.Hcat_base(target, output, mu, True)

    @staticmethod
    @tf.function
    def Hcat_jetInv(target, output):
        return TauLosses.Hcat_base(target, output, jet, True)

    @staticmethod
    @tf.function
    def Hbin(target, output):
        return TauLosses.Hbase(target, output, tau, True)

    @staticmethod
    @tf.function
    def Fe(target, output):
        F_factor = tf.constant(1.63636, dtype=output.dtype.base_dtype)
        return F_factor * TauLosses.Fbase(target, output, e, 2, True, False)

    @staticmethod
    @tf.function
    def Fmu(target, output):
        F_factor = tf.constant(1.63636, dtype=output.dtype.base_dtype)
        return F_factor * TauLosses.Fbase(target, output, mu, 2, True, False)

    @staticmethod
    @tf.function
    def Fjet(target, output):
        F_factor = tf.constant(1.63636, dtype=output.dtype.base_dtype)
        return F_factor * TauLosses.Fbase(target, output, jet, 2, True, False)

    @staticmethod
    @tf.function
    def Fcmb(target, output):
        F_factor = tf.constant(1.17153, dtype=output.dtype.base_dtype)
        return F_factor * TauLosses.Fbase(target, output, tau, 0.5, False, True)

    @staticmethod
    @tf.function
    def tau_crossentropy(target, output):
        return TauLosses.sLe(target, output) + TauLosses.sLmu(target, output) + TauLosses.sLjet(target, output)

    @staticmethod
    @tf.function
    def tau_crossentropy_v2(target, output):
        F_factor = tf.constant(5, dtype=output.dtype.base_dtype)
        sf = tf.constant([TauLosses.Le_sf, TauLosses.Lmu_sf, TauLosses.Ltau_sf, TauLosses.Ljet_sf],
                         dtype=output.dtype.base_dtype)
        return sf[tau] * TauLosses.Htau(target, output) + (sf[e] + sf[mu] + sf[jet]) * TauLosses.Fcmb(target, output) \
               + F_factor * (sf[e] * TauLosses.Fe(target, output) + sf[mu] * TauLosses.Fmu(target, output) \
                             + sf[jet] * TauLosses.Fjet(target, output))

    WPs_e = [0.0630386, 0.1686942, 0.3628130, 0.681543, 0.8847544, 0.9675541, 0.9859251, 0.9928449]
    WPs_mu = [0.1058354, 0.2158633, 0.5551894, 0.8754835]
    WPs_jet = [0.2599605, 0.4249705, 0.5983682, 0.7848675, 0.8834768, 0.930868, 0.9573137, 0.9733927]

    @staticmethod
    @tf.function
    def SetWPs(wps_e, wps_mu, wps_jet):
        TauLosses.WPs_e = wps_e
        TauLosses.WPs_mu = wps_mu
        TauLosses.WPs_jet = wps_jet

    @staticmethod
    @tf.function
    def discri(target, output, index):
        epsilon = tf.constant(TauLosses.epsilon, output.dtype.base_dtype)
        discriminator = output[:, tau] / (output[:, tau] + output[:, index] + epsilon)
        discriminator = tf.clip_by_value(discriminator, epsilon, 1 - epsilon)
        return discriminator

    @staticmethod
    @tf.function
    def Wbase(target, output, index, WPs, gamma):
        discriminator = TauLosses.discri(target, output, index)
        gamma_t = tf.constant(gamma, output.dtype.base_dtype)
        decay_factor = 1.0
        decay_factor_inv = 1.0
        for wp in WPs:
          wp_t = tf.constant(wp, output.dtype.base_dtype)
          decay_factor += ( ( -tf.math.tanh(40*(discriminator-wp_t)) + 1) / 2 ) / len(WPs)
          decay_factor_inv += ( ( tf.math.tanh(40*(discriminator-wp_t)) + 1) / 2 ) / len(WPs)
        return -decay_factor_inv * target[:, index] * tf.pow(discriminator, gamma_t) * tf.math.log(1 - discriminator) -decay_factor * target[:, tau] * tf.pow(1-discriminator, gamma_t) * tf.math.log(discriminator)

    @staticmethod
    @tf.function
    def We(target, output):
        return TauLosses.Wbase(target, output, e, TauLosses.WPs_e, 2)

    @staticmethod
    @tf.function
    def Wmu(target, output):
        return TauLosses.Wbase(target, output, mu, TauLosses.WPs_mu, 2)

    @staticmethod
    @tf.function
    def Wjet(target, output):
        return TauLosses.Wbase(target, output, jet, TauLosses.WPs_jet, 2)

    @staticmethod
    @tf.function
    def sWe(target, output):
        sf = tf.constant(TauLosses.Le_sf, output.dtype.base_dtype)
        return sf * TauLosses.We(target, output)

    @staticmethod
    @tf.function
    def sWmu(target, output):
        sf = tf.constant(TauLosses.Lmu_sf, output.dtype.base_dtype)
        return sf * TauLosses.Wmu(target, output)

    @staticmethod
    @tf.function
    def sWjet(target, output):
        sf = tf.constant(TauLosses.Ljet_sf, output.dtype.base_dtype)
        return sf * TauLosses.Wjet(target, output)

    @staticmethod
    @tf.function
    def WPloss(target, output):
        return TauLosses.sWe(target, output) + TauLosses.sWmu(target, output) + TauLosses.sWjet(target, output)

    @staticmethod
    @tf.function
    def WPloss2(target, output):
        TC_factor = tf.constant(9, dtype=output.dtype.base_dtype)
        return (TauLosses.sWe(target, output) + TauLosses.sWmu(target, output) + TauLosses.sWjet(target, output)) + TauLosses.tau_crossentropy_v2(target, output) / TC_factor

    @staticmethod
    @tf.function
    def tau_vs_other(prob_tau, prob_other):
        #return np.where(prob_tau > TauLosses.merge_thr, prob_tau / (prob_tau + prob_other), prob_tau)
        #return np.where(prob_tau > TauLosses.merge_thr, prob_tau / np.exp(prob_other), prob_tau)
        return np.where(prob_tau > 0, prob_tau / (prob_tau + prob_other), np.zeros(prob_tau.shape))
        #return prob_tau / (prob_tau + prob_other + TauLosses.epsilon)

    @staticmethod
    @tf.function
    def binary(target, output, weights, selected):
        if selected == 1:
            cmp_target = target > 0.5
            cmp_output = output > 0.5
        else:
            cmp_target = target < 0.5
            cmp_output = output < 0.5
        shape = tf.shape(target)
        w_all = tf.where(cmp_target, weights, tf.zeros(shape))
        w_correct = tf.where(tf.math.logical_and(cmp_target, cmp_output), weights, tf.zeros(shape))
        return tf.reduce_sum(w_correct) / tf.reduce_sum(w_all)
        #return tf.where(tf.math.logical_and(target == 1, output > 0.5), tf.ones(shape), tf.zeros(shape))
        #return tf.where(target > 0.5, tf.ones(shape), tf.zeros(shape))

    @staticmethod
    @tf.function
    def binary_negative(target, output):
        zero = tf.constant(0, dtype=target.dtype.base_dtype)
        shape = tf.shape(target)
        return tf.where(target < 0.5, tf.ones(shape), tf.zeros(shape))

    @staticmethod
    @tf.function
    def crossentropy_adversarial(target, adv_output):
        tau_target = target # MC_tau->0, data_tau->1, this is done by setting y_onehot in main DataLoader 
        tau_output = adv_output #  given output from adversarial
        loss = tf.keras.losses.binary_crossentropy(tau_target, tau_output) # Standard cross entropy loss function
        return loss

    @staticmethod
    @tf.function
    def focal_adversarial(target, adv_output, gamma):
        if gamma <= 0:
            raise RuntimeError("Focal Loss requires gamma > 0.")
        loss = tf.keras.losses.binary_focal_crossentropy(target, adv_output, gamma=gamma)
        return loss

def LoadModel(model_file, compile=True):
    if compile:
        return load_model(model_file, custom_objects = {
            'tau_crossentropy': TauLosses.tau_crossentropy, 'tau_crossentropy_v2': TauLosses.tau_crossentropy_v2,
            'Le': TauLosses.Le, 'Lmu': TauLosses.Lmu, 'Ljet': TauLosses.Ljet,
            'sLe': TauLosses.sLe, 'sLmu': TauLosses.sLmu, 'sLjet': TauLosses.sLjet,
            'He': TauLosses.He, 'Hmu': TauLosses.Hmu, 'Htau': TauLosses.Htau, 'Hjet': TauLosses.Hjet,
            'Hcat_e': TauLosses.Hcat_e, 'Hcat_mu': TauLosses.Hcat_mu, 'Hcat_jet': TauLosses.Hcat_jet,
            'Hcat_eInv': TauLosses.Hcat_eInv, 'Hcat_muInv': TauLosses.Hcat_muInv, 'Hcat_jetInv': TauLosses.Hcat_jetInv,
            'Hbin': TauLosses.Hbin, 'HbinInv': TauLosses.Hbin,
            'Fe': TauLosses.Fe, 'Fmu': TauLosses.Fmu, 'Fjet': TauLosses.Fjet, 'Fcmb': TauLosses.Fcmb,
            'We': TauLosses.We, 'Wmu': TauLosses.Wmu, 'Wjet': TauLosses.Wjet,
            'WPloss': TauLosses.WPloss, 'WPloss2': TauLosses.WPloss2
        })
    else:
        return load_model(model_file, compile = False)


def quantile_ex(data, quantiles, weights):
    quantiles = np.array(quantiles)
    indices = np.argsort(data)
    data_sorted = data[indices]
    weights_sorted = weights[indices]
    prob = np.cumsum(weights_sorted) - weights_sorted / 2
    prob = (prob[:] - prob[0]) / (prob[-1] - prob[0])
    return np.interp(quantiles, prob, data_sorted)
