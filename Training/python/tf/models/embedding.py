from tensorflow.keras.layers import Dense, Embedding
import tensorflow as tf

class FeatureEmbedding(tf.keras.layers.Layer):
    def __init__(self, feature_idx_to_select, shared_cat_feature_idx, shared_cat_dim_in, shared_cat_dim_out, hidden_dim, activation, out_dim):
        super().__init__()
        assert len(feature_idx_to_select) == len(shared_cat_feature_idx)
        self.n_particle_types = len(feature_idx_to_select)
        self.shared_cat_feature_idx = shared_cat_feature_idx
        self.feature_idx_to_select = feature_idx_to_select # should not include categorical features 
        
        assert len(hidden_dim) == self.n_particle_types
        self.dense_hidden = [Dense(hidden_dim[i], activation=activation) for i in range(self.n_particle_types)]
        self.dense_out = [Dense(out_dim, activation=activation) for _ in range(self.n_particle_types)]
        self.shared_embedding = Embedding(shared_cat_dim_in, shared_cat_dim_out)
        
    def call(self, inputs):
        outputs = []
        for i, x in enumerate(inputs): # NB: assumes that len(inputs)==len(feature_idx_to_select), so make sure to align this in the model call()  
            x_emb = self.shared_embedding(x[..., self.shared_cat_feature_idx[i]])
            x_inputs = tf.gather(x, indices=self.feature_idx_to_select[i], axis=-1)
            x_inputs = tf.concat([x_inputs, x_emb], axis=-1)
            x_output = self.dense_out[i](self.dense_hidden[i](x_inputs))
            outputs.append(x_output)
        outputs = tf.concat(outputs, axis=1) # concat across constituent dimension
        return outputs