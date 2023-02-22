from tkinter import W
from omegaconf import OmegaConf, DictConfig
import tensorflow as tf
from tensorflow.keras.layers import BatchNormalization, Dense, Dropout, Conv2D, Activation
from models.embedding import FeatureEmbedding
from utils.training import create_padding_mask

class ParticleNet(tf.keras.Model):
    def __init__(self, feature_name_to_idx, encoder_cfg, decoder_cfg):
        super(ParticleNet, self).__init__()

        self.feature_name_to_idx = feature_name_to_idx

        self.masking = encoder_cfg['masking']
        self.coord_type = encoder_cfg['coordinate_type']
        self.spatial_features = encoder_cfg['spatial_features']
        assert len(self.spatial_features) == 2 # but can be easily generalised to auxiliary number
        if self.coord_type == 'polar':
            assert self.spatial_features == ['r', 'theta']

        # here assume the correspondence of coord1/2_idx and yielded inputs 
        self.coord1_idx = [feature_to_idx[self.spatial_features[0]] if particle_type != 'global' else None for particle_type, feature_to_idx in self.feature_name_to_idx.items()]
        self.coord2_idx = [feature_to_idx[self.spatial_features[1]] if particle_type != 'global' else None for particle_type, feature_to_idx in self.feature_name_to_idx.items()]

        self.conv_params = [(layer_setup[0], tuple(layer_setup[1]),) for layer_setup in encoder_cfg['conv_params']]
        self.conv_pooling = encoder_cfg['conv_pooling']

        self.use_global_features = decoder_cfg['use_global_features']
        self.fc_params = [(layer_setup, decoder_cfg['dropout_rate'],) for layer_setup in decoder_cfg['dense_params']]
        self.num_classes = decoder_cfg['n_outputs']

        self.particle_blocks_to_drop = [i for i, feature_names in enumerate(encoder_cfg['embedding_kwargs']['features_to_drop'].values())
                                                     if feature_names=='all']

        self.global_block_id = list(feature_name_to_idx.keys()).index('global')
        assert self.global_block_id in self.particle_blocks_to_drop

        embedding_kwargs = encoder_cfg['embedding_kwargs']
        if isinstance(embedding_kwargs, DictConfig):
            embedding_kwargs = OmegaConf.to_object(embedding_kwargs)
            self.r_cut = embedding_kwargs.pop('r_cut') # r_cut currently not implemented, but r_cut needs to be removed form embedding_kwargs
        shared_cat_feature = embedding_kwargs.pop('shared_cat_feature')
        features_to_drop = embedding_kwargs.pop('features_to_drop')
        embedding_kwargs['shared_cat_feature_idx'], embedding_kwargs['feature_idx_to_select'] = [], []

        # extract indices of feature to embedded and features to be used in the training 
        for particle_type, names_to_idx in feature_name_to_idx.items():
            if features_to_drop[particle_type] == "all": continue
            embedding_kwargs['shared_cat_feature_idx'].append(names_to_idx[shared_cat_feature])
            embedding_kwargs['feature_idx_to_select'].append([i for f, i in names_to_idx.items() 
                                                                if f not in features_to_drop[particle_type] and f != shared_cat_feature])

        self.feature_embedding = FeatureEmbedding(**embedding_kwargs) 

        self.batch_norm = BatchNormalization()
        
        self.edge_conv_layers = []
        for layer_idx, layer_param in enumerate(self.conv_params):
            K, channels = layer_param
            self.edge_conv_layers.append(
                EdgeConv(K, channels, with_bn=True, activation=encoder_cfg['activation'], pooling=self.conv_pooling, name=f'{self.name}_EdgeConv_{layer_idx}')
            )

        self.global_idx_to_select = [i for f,i in self.feature_name_to_idx['global'].items()
                                            if f not in decoder_cfg['global_features_to_drop'] ]
        self.decoder_layers = tf.keras.Sequential()

        if self.fc_params is not None:
            for layer_idx, layer_param in enumerate(self.fc_params):
                units, dropout_rate = layer_param

                self.decoder_layers.add(Dense(units, activation=decoder_cfg['activation']))                
                if dropout_rate is not None and dropout_rate > 0:
                    self.decoder_layers.add(Dropout(dropout_rate))

        self.decoder_layers.add(Dense(self.num_classes, activation='softmax'))

    @tf.function
    def call(self, inputs):
        padded_inputs = []
        mask = []
        coord_shift = []
        points = []
        
        global_fts = None
        for input_id, input_ in enumerate(inputs):
            if input_id == self.global_block_id and self.use_global_features: 
                global_fts = tf.gather(input_, indices=self.global_idx_to_select, axis=-1); continue
            if input_id in self.particle_blocks_to_drop: continue

            input_ = input_.to_tensor()
            padded_inputs.append(input_)

            # masks per particle type
            mask_ = create_padding_mask(input_)
            mask_ = tf.cast(mask_[:, :, tf.newaxis], dtype='float32')
            mask.append(mask_)
            coord_shift.append(tf.multiply(1e9, tf.cast(tf.equal(mask_, 0), dtype='float32')))
        
            # compute cloud coordinates for the first layer
            if self.coord_type == 'polar':
                eta = input_[:,:,self.coord1_idx[input_id]] * tf.math.cos(input_[:,:,self.coord2_idx[input_id]]) 
                phi = input_[:,:,self.coord1_idx[input_id]] * tf.math.sin(input_[:,:,self.coord2_idx[input_id]])
                eta = eta[:, :, tf.newaxis]
                phi = phi[:, :, tf.newaxis]
                points.append(tf.concat([eta, phi], -1))
            else:
                points.append(tf.concat([input_[:,:,self.coord1_idx[input_id],tf.newaxis], input_[:,:,self.coord2_idx[input_id],tf.newaxis]], -1))

        # concat across particle types
        mask = tf.concat(mask, axis=1)
        coord_shift = tf.concat(coord_shift, axis=1)
        points = tf.concat(points, axis=1)
        
        features = self.feature_embedding(padded_inputs)        
        fts = tf.squeeze(self.batch_norm(tf.expand_dims(features, axis=2)), axis=2)

        for layer_idx, layer_param in enumerate(self.conv_params):
            if self.masking:
                pts = tf.add(coord_shift, points) if layer_idx == 0 else tf.add(coord_shift, fts)
            else:
                pts = points if layer_idx == 0 else fts

            fts = self.edge_conv_layers[layer_idx](pts, fts)

        fts = tf.math.multiply(fts, mask) if self.masking else fts
        pool = tf.reduce_mean(fts, axis=1)  # (N, C)

        pool_comb = tf.concat([pool, global_fts], axis=1) if self.use_global_features else pool

        out = self.decoder_layers(pool_comb)

        return out  # (N, num_classes)


class EdgeConv(tf.keras.layers.Layer):
    def __init__(self, K, channels, with_bn=True, activation='relu', pooling='mean', **kwargs):
        super(EdgeConv, self).__init__()

        self.K = K
        self.channels = channels
        self.with_bn = with_bn
        self.activation = activation
        self.pooling = pooling

        self.conv2d_layers = []
        self.batchnorm_layers = []
        self.activation_layers = []

        for idx, channel in enumerate(self.channels):
            self.conv2d_layers.append(
                Conv2D(channel, kernel_size=(1,1), strides=1, data_format='channels_last', use_bias=False if self.with_bn else True, kernel_initializer='glorot_normal')
            )

            if self.with_bn:
                self.batchnorm_layers.append(BatchNormalization())
            if self.activation:
                self.activation_layers.append(Activation(self.activation))

        self.shortcut = Conv2D(self.channels[-1], kernel_size=(1,1), strides=1, data_format='channels_last', use_bias=False if self.with_bn else True, kernel_initializer='glorot_normal')
        if self.with_bn:
            self.shortcut_batchnorm = BatchNormalization()
        if self.activation:
            self.shortcut_activation = Activation(self.activation)

    @tf.function
    def call(self, points, features):
        d = self.batch_distance_matrix_general(points, points)
        k = self.K if tf.shape(features)[1] > self.K else tf.shape(features)[1] - 1
        _, indicies = tf.nn.top_k(-d, k=k + 1)
        indicies = indicies[:,:,1:]

        fts = features
        knn_fts = self.knn(k, indicies, fts)
        knn_fts_center = tf.tile(tf.expand_dims(fts, axis=2), (1, 1, k, 1))  # (N, P, K, C)
        knn_fts = tf.concat([knn_fts_center, tf.subtract(knn_fts, knn_fts_center)], axis=-1)  # (N, P, K, 2*C)

        x = knn_fts
        for idx, channel in enumerate(self.channels):
            x = self.conv2d_layers[idx](x)
            if self.with_bn:
                x = self.batchnorm_layers[idx](x)
            if self.activation:
                x = self.activation_layers[idx](x)

        if self.pooling == 'max':
            fts = tf.reduce_max(x, axis=2)  # (N, P, C')
        elif self.pooling == 'mean':
            fts = tf.reduce_mean(x, axis=2)  # (N, P, C')
        else:
            raise RuntimeError('Pooling parameter should be either max or mean')

        # shortcut
        sc = self.shortcut(tf.expand_dims(features, axis=2))
        if self.with_bn:
            sc = self.shortcut_batchnorm(sc)
        sc = tf.squeeze(sc, axis=2)

        if self.activation:
            return self.shortcut_activation(sc + fts)  # (N, P, C')
        else:
            return sc + fts

    def knn(self, k, topk_indices, features):
        # topk_indices: (N, P, K)
        # features: (N, P, C)
        with tf.name_scope('knn'):
            queries_shape = tf.shape(features)
            batch_size = queries_shape[0]
            num_points = queries_shape[1]
            batch_indices = tf.tile(tf.reshape(tf.range(batch_size), (-1, 1, 1, 1)), (1, num_points, k, 1))
            indices = tf.concat([batch_indices, tf.expand_dims(topk_indices, axis=3)], axis=3)  # (N, P, K, 2)
            return tf.gather_nd(features, indices)

    def batch_distance_matrix_general(self, A, B):
        with tf.name_scope('dmat'):
            r_A = tf.reduce_sum(A * A, axis=2, keepdims=True)
            r_B = tf.reduce_sum(B * B, axis=2, keepdims=True)
            m = tf.matmul(A, tf.transpose(B, perm=(0, 2, 1)))
            D = r_A - 2 * m + tf.transpose(r_B, perm=(0, 2, 1))
            return D