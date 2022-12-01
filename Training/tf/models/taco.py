from omegaconf import OmegaConf, DictConfig
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense, Conv1D, Softmax, LayerNormalization, Dropout
from tensorflow.keras.layers import ReLU, BatchNormalization
from models.embedding import FeatureEmbedding

class RadialFrequencies(tf.keras.layers.Layer):
    def __init__(self, hidden_dim, n_freqs):
        super().__init__()
        self.dense_1 = Dense(hidden_dim, activation=tf.nn.relu)
        # self.dense_2 = Dense(hidden_dim, activation=tf.nn.relu)
        self.r_real = Dense(n_freqs+1, activation=tf.nn.relu) # bias_initializer=tf.keras.initializers.ones
        self.r_imag = Dense(n_freqs, activation=tf.nn.relu)  
        
    def call(self, inputs):
        x_hidden = self.dense_1(inputs)
        # x_hidden = self.dense_2(x_hidden)
        r_real = self.r_real(x_hidden)
        r_imag = self.r_imag(x_hidden)
        return r_real, r_imag

class Conv1DBlock(tf.keras.layers.Layer):
    def __init__(self, n_conv_layers=2, kernel_size=3, n_conv_filters=10):
        super().__init__()
        self.n_conv_layers = n_conv_layers
        self.kernel_size = kernel_size
        self.non_linearity = ReLU()
        self.batch_norm = BatchNormalization(axis=2)
        self.conv_layers = [Conv1D(n_conv_filters, kernel_size, padding='valid',
                                    data_format='channels_first', activation='relu') 
                                    for i in range(self.n_conv_layers)]

    def pad_waveforms(self, x):
        n_add_left = int(np.floor((self.kernel_size-1)/2))
        n_add_right = int(np.ceil((self.kernel_size-1)/2))
        x_padded = tf.concat([x[..., -n_add_left:], x, x[..., :n_add_right]], axis=-1)
        return x_padded

    def call(self, inputs):
        x_conv_in = self.batch_norm(self.non_linearity(inputs))
        for i in range(self.n_conv_layers):
            x_conv_in = self.pad_waveforms(x_conv_in)
            x_conv_in = self.conv_layers[i](x_conv_in)
        return x_conv_in

class WaveformEncoder(tf.keras.Model):
    def __init__(self, feature_name_to_idx, embedding_kwargs, hidden_dim_radial=16, n_freqs=4, n_filters=10, n_rotations=32):
        super().__init__()
        self.feature_name_to_idx = feature_name_to_idx
        self.n_freqs = n_freqs
        self.n_filters = n_filters
        self.n_rotations = n_rotations
        self.radial_models = [RadialFrequencies(hidden_dim=hidden_dim_radial, n_freqs=self.n_freqs) 
                                                    for _ in range(self.n_filters)]
        
        if isinstance(embedding_kwargs, DictConfig):
            embedding_kwargs = OmegaConf.to_object(embedding_kwargs)
        cat_emb_feature = embedding_kwargs.pop('cat_emb_feature')
        embedding_kwargs['cat_emb_feature_idx'] = self.feature_name_to_idx[cat_emb_feature]
        self.feature_embedding = FeatureEmbedding(**embedding_kwargs)
    
    @staticmethod    
    def to_complex(real, imag):
        real = tf.concat([tf.reverse(real[..., 1:], axis=[-1]), real], axis=-1)
        zeros = tf.expand_dims(tf.zeros_like(imag[..., 0]), axis=-1)
        imag = tf.concat([-tf.reverse(imag, axis=[-1]), zeros, imag], axis=-1)
        z = tf.dtypes.complex(real, imag)
        return z

    def get_radial_spectrum(self, inputs):
        r = tf.expand_dims(inputs[..., self.feature_name_to_idx['r']], axis=-1)
        r_freqs = []
        for radial_model in self.radial_models:
            r_freqs_real, r_freqs_imag = radial_model(r)
            r_freqs.append(self.to_complex(r_freqs_real, r_freqs_imag))
        r_freqs = tf.stack(r_freqs, axis=-2)
        return r_freqs

    def get_azim_spectrum(self, inputs):
        theta = inputs[..., self.feature_name_to_idx['theta']]
        azim_freqs = [tf.math.exp(tf.dtypes.complex(0, m*theta)) for m in range(-self.n_freqs, self.n_freqs+1)]
        azim_freqs = tf.stack(azim_freqs, axis=-1)[..., tf.newaxis, :] # additional axis for filter dimension
        return azim_freqs

    def get_rotation_spectrum(self):
        rotations = tf.constant(2*np.pi/self.n_rotations, dtype=tf.float32)*tf.range(0, self.n_rotations, dtype=tf.float32)
        rotation_freqs = tf.tensordot(tf.range(-self.n_freqs, self.n_freqs+1, dtype=tf.float32), rotations, axes=0) # tensor product with output dim [2*n_freqs+1, n_rotations]
        rotation_freqs = tf.math.exp(tf.dtypes.complex(tf.constant(0., dtype=tf.float32), rotation_freqs))
        return rotation_freqs

    def sample_waveforms(self, proj_freqs):  
        rotation_freqs = self.get_rotation_spectrum()
        waveforms = tf.tensordot(proj_freqs, rotation_freqs, axes=[[3], [0]]) # axes 3 and 0 are m dimension (frequency)
        # if not tf.math.reduce_all((imag_part:=tf.math.abs(tf.math.imag(waveforms))) < 1.e-5):
        #     print(waveforms[imag_part > 1.e-5])
        #     raise RuntimeError('Found large elements in imaginary part of waveforms')
        waveforms = tf.math.abs(waveforms)
        return waveforms

    def project_onto_filters(self, inputs, filter_freqs):
        inputs_emb = self.feature_embedding(inputs) # [batch, None, emb]
        z = tf.dtypes.complex(inputs_emb, 0)[..., tf.newaxis,  tf.newaxis] # axis for filter and frequency dimensions (respectively) 
        proj_freqs = tf.multiply(z, filter_freqs) # [batch, None, emb, filter, freq]
        proj_freqs = tf.math.reduce_sum(proj_freqs, axis=1) # [batch, emb, filter, freq] - sum over constituents 
        # assert tf.math.reduce_all(tf.math.imag(proj_freqs + tf.reverse(proj_freqs, axis=[-1])) == 0)
        return proj_freqs

    def call(self, inputs):
        r_freqs = self.get_radial_spectrum(inputs) # [batch, None, filter, freq]
        azim_freqs = self.get_azim_spectrum(inputs) # [batch, None, 1, freq]
        filter_freqs = tf.math.multiply(r_freqs, azim_freqs) # [batch, None, filter, freq]
        filter_freqs = filter_freqs[..., tf.newaxis, :, :] # [batch, None, 1, filter, freq]
        proj_freqs = self.project_onto_filters(inputs, filter_freqs) # [batch, emb, filter, freq]
        waveforms = self.sample_waveforms(proj_freqs) # [batch, emb, filter, rotation]
        return waveforms

class WaveformDecoder(tf.keras.Model):
    def __init__(self, n_conv_blocks=5, n_conv_layers=2, kernel_size=3, n_conv_filters=10, hidden_dim=10, n_outputs=2):
        super().__init__()
        self.n_conv_blocks = n_conv_blocks
        self.kernel_size = kernel_size
        self.conv_blocks = [Conv1DBlock(n_conv_layers=n_conv_layers, kernel_size=kernel_size, n_conv_filters=n_conv_filters) 
                                    for _ in range(self.n_conv_blocks)]

        # self.dropout = Dropout(0.2)
        # self.layer_norm = LayerNormalization(axis=[1,2,3])
        self.dense_1 = Dense(hidden_dim, activation=tf.nn.relu)
        # self.dense_2 = Dense(hidden_dim//2, activation=tf.nn.relu)
        self.output_dense = Dense(n_outputs, activation=None)
        self.output_softmax = Softmax()

    def call(self, inputs):
        x_block_in = inputs 
        for i in range(self.n_conv_blocks):
            x_block_out = self.conv_blocks[i](x_block_in)
            x_block_in = x_block_in + x_block_out
        
        # x_block_in = self.dropout(x_block_in)
        # x_block_in = self.layer_norm(x_block_in)
        x_block_out = tf.math.reduce_mean(x_block_in, axis=-1) # pulling by mean over rotation dim
        x_block_out = tf.math.reduce_mean(x_block_out, axis=-1) # pulling by mean over filter dim
        
        x_dense = self.dense_1(x_block_out)
        # x_dense = self.dense_2(x_dense)
        outputs = self.output_softmax(self.output_dense(x_dense))
        return outputs

class TacoNet(tf.keras.Model):
    def __init__(self, feature_name_to_idx, encoder_kwargs, decoder_kwargs):
        super().__init__()
        self.wave_encoder = WaveformEncoder(feature_name_to_idx, **encoder_kwargs)
        self.wave_decoder = WaveformDecoder(**decoder_kwargs)
        
    def call(self, inputs):
        waveforms = self.wave_encoder(inputs)
        outputs = self.wave_decoder(waveforms)
        return outputs