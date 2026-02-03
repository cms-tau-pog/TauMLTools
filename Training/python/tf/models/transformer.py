from omegaconf import OmegaConf, DictConfig
import tensorflow as tf
from tensorflow.keras.layers import Dense, MultiHeadAttention, LayerNormalization, Dropout, Softmax, Activation, Normalization
from models.embedding import FeatureEmbedding

relax_shapes = False

class MaskedMultiHeadAttention(tf.keras.layers.Layer):
    def __init__(self, num_heads, dim_head_key, dim_head_value, dim_out):
        super(MaskedMultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.dim_head_key = dim_head_key
        self.dim_head_value = dim_head_value
        self.dim_out = dim_out
        self.att_logit_norm = tf.sqrt(dim_head_key)
        # self.att_logit_norm = tf.sqrt(tf.cast(dim_head_key, tf.float32))

        # head projection layers, separate dim  
        self.wq = tf.keras.layers.Dense(self.dim_head_key*self.num_heads) 
        self.wk = tf.keras.layers.Dense(self.dim_head_key*self.num_heads)
        self.wv = tf.keras.layers.Dense(self.dim_head_value*self.num_heads)

        self.dense = tf.keras.layers.Dense(dim_out)

    @tf.function(experimental_relax_shapes=relax_shapes)
    def split_heads(self, x, batch_size, head_dim):
        x = tf.reshape(x, (batch_size, -1, self.num_heads, head_dim))
        return tf.transpose(x, perm=[0, 2, 1, 3]) # (batch_size, num_heads, seq_len, depth)

    @tf.function(experimental_relax_shapes=relax_shapes)
    def masked_attention(self, q, k, v, mask=None):
        # q * k^T
        matmul_qk = tf.matmul(q, k, transpose_b=True)  # (..., seq_len_q, seq_len_k)

        # scale matmul_qk
        scaled_attention_logits = matmul_qk / self.att_logit_norm
        
        # # masked logits -> softmax 
        # scaled_attention_logits -= tf.math.reduce_max(scaled_attention_logits, axis=-1, keepdims=True) & subtract max value (to prevent nans after softmax)
        # inputs_exp = tf.exp(scaled_attention_logits)
        # if mask is not None:
        #     inputs_exp *= mask
        # inputs_sum = tf.reduce_sum(inputs_exp, axis=-1, keepdims=True)
        # attention_weights = tf.where(tf.math.not_equal(inputs_sum, 0), inputs_exp/inputs_sum, 0) 

        # add the mask to the scaled tensor.
        if mask is not None:
            scaled_attention_logits += (mask * -1e9)

        # softmax normalized on the last axis (seq_len_k)
        attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)  # (..., seq_len_q, seq_len_k)

        # if mask is not None:
        #     attention_weights *= tf.cast(~tf.cast(mask, tf.bool), tf.float32)

        # att * v
        output = tf.matmul(attention_weights, v)  # (..., seq_len_q, depth_v)

        return output, attention_weights

    @tf.function(experimental_relax_shapes=relax_shapes)
    def call(self, query, key, value, attention_mask, return_attention_scores=False):
        batch_size = tf.shape(query)[0]

        # project to heads space
        # -> (batch_size, seq_len, dim_head_(k/v) * num_heads)
        query = self.wq(query)  
        key = self.wk(key)
        value = self.wv(value)

        # split away head dimension and transpose
        # -> (batch_size, num_heads, seq_len_(q,k,v), dim_head_(k,v))
        query = self.split_heads(query, batch_size, self.dim_head_key)
        key = self.split_heads(key, batch_size, self.dim_head_key)
        value = self.split_heads(value, batch_size, self.dim_head_value)

        ## compute per-head attention 
        scaled_attention, attention_weights = self.masked_attention(query, key, value, attention_mask) # att=(batch_size, num_heads, seq_len_q, dim_head_value)
        
        # combine heads together
        scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])  # (batch_size, seq_len_q, num_heads, dim_head_value)
        concat_attention = tf.reshape(scaled_attention,
                                      (batch_size, -1, self.dim_head_value * self.num_heads))

        # project onto dim_out
        output = self.dense(concat_attention)  # (batch_size, seq_len_q, dim_out)

        if return_attention_scores:
            return output, attention_weights
        return output

        
class EncoderLayer(tf.keras.layers.Layer):
    def __init__(self, use_masked_mha, num_heads, dim_model, dim_head_key, dim_head_value, dim_ff, activation, dropout_rate):
        super(EncoderLayer, self).__init__()

        if use_masked_mha:
            self.mha = MaskedMultiHeadAttention(num_heads=num_heads, dim_head_key=dim_head_key, dim_head_value=dim_head_value, dim_out=dim_model)
        else:
            self.mha = MultiHeadAttention(num_heads=num_heads, key_dim=dim_head_key, value_dim=dim_head_value, dropout=0)

        self.ffn = tf.keras.Sequential([
                          Dense(dim_ff, activation=activation),  
                          Dense(dim_model)
                         ])
        self.layernorm1 = LayerNormalization(epsilon=1e-6)
        self.layernorm2 = LayerNormalization(epsilon=1e-6)
        self.dropout1 = Dropout(dropout_rate)
        self.dropout2 = Dropout(dropout_rate)

    @tf.function(experimental_relax_shapes=relax_shapes)
    def call(self, x, mask, training, return_attention_scores=False):
        if return_attention_scores:
            attn_output, attn_score = self.mha(query=x, value=x, key=x, attention_mask=mask, return_attention_scores=False)
        else:
            attn_output = self.mha(query=x, value=x, key=x, attention_mask=mask, return_attention_scores=False)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(x + attn_output)  

        ffn_output = self.ffn(out1) 
        ffn_output = self.dropout2(ffn_output, training=training)
        out2 = self.layernorm2(out1 + ffn_output)

        if return_attention_scores:
            return out2, attn_score
        else:
            return out2
class Encoder(tf.keras.layers.Layer):
    def __init__(self, feature_name_to_idx, embedding_kwargs, use_masked_mha, num_layers, num_heads, 
                        dim_model, dim_head_key, dim_head_value, dim_ff, activation, dropout_rate):
        super(Encoder, self).__init__()

        self.num_layers = num_layers
        self.enc_layers = [EncoderLayer(use_masked_mha=use_masked_mha, num_heads=num_heads, dim_model=dim_model, 
                                        dim_head_key=dim_head_key, dim_head_value=dim_head_value, dim_ff=dim_ff, 
                                        activation=activation, dropout_rate=dropout_rate) 
                                            for _ in range(self.num_layers)]
        self.dropout = tf.keras.layers.Dropout(dropout_rate)

        if isinstance(embedding_kwargs, DictConfig):
            embedding_kwargs = OmegaConf.to_object(embedding_kwargs)
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

    @tf.function(experimental_relax_shapes=relax_shapes)
    def call(self, x, mask, training, return_attention_scores=False):
        x = self.feature_embedding(x)
        x = self.dropout(x, training=training)
        
        attn_scores = []
        for i in range(self.num_layers):
            if return_attention_scores:
                x, attn_score = self.enc_layers[i](x, mask=mask, training=training, return_attention_scores=return_attention_scores)
                attn_scores.append(attn_score)
            else:
                x = self.enc_layers[i](x, mask=mask, training=training, return_attention_scores=return_attention_scores)
        if return_attention_scores:
            return x, attn_scores
        else:
            return x

class Transformer(tf.keras.Model):
    def __init__(self, feature_name_to_idx, encoder_kwargs, decoder_kwargs):
        super().__init__()
        encoder_kwargs = OmegaConf.to_object(encoder_kwargs)
        self.use_masked_mha = encoder_kwargs["use_masked_mha"]
        # self.particle_blocks_to_drop = [i for i, feature_names in enumerate(encoder_kwargs['embedding_kwargs']['features_to_drop'].values())
        #                                              if feature_names=='all']
        self.global_block_id = list(feature_name_to_idx.keys()).index('global')
        self.r_cut = encoder_kwargs['embedding_kwargs'].pop('r_cut')
        if self.r_cut is not None:
            self.r_indices = [feature_indices['r'] if particle_block_name != 'global' else None for particle_block_name, feature_indices in feature_name_to_idx.items()]

        self.encoder = Encoder(feature_name_to_idx, **encoder_kwargs)
        self.decoder_dense = [Dense(n_nodes, activation=decoder_kwargs['activation']) for n_nodes in decoder_kwargs['dim_ff_layers']]
        self.output_dense = Dense(decoder_kwargs['n_outputs'], activation=None)
        self.output_pred = Softmax()
        self.output_attn = decoder_kwargs['output_attn']
    
    @tf.function(experimental_relax_shapes=relax_shapes) 
    def call(self, inputs, training):
        mask = tf.concat([tf.math.reduce_any(tf.math.not_equal(input_, 0), axis=-1) for input_ in inputs], axis=1) 

        padding_mask = tf.math.logical_and(tf.expand_dims(mask, axis=1), tf.expand_dims(mask, axis=-1)) # [batch, seq, seq], symmetric block-diagonal
        if self.use_masked_mha: # invert mask, 0 -> constituent, 1 -> padding
            padding_mask = ~padding_mask
        padding_mask = tf.expand_dims(padding_mask, axis=1) # additional axis for head dimension 

        # propagate through encoder
        if self.output_attn:
            enc_output, attn_scores = self.encoder(inputs, mask=padding_mask, training=training, return_attention_scores=self.output_attn)
        else:
            enc_output = self.encoder(inputs, mask=padding_mask, training=training, return_attention_scores=self.output_attn)

         # mask padded tokens before pooling 
        enc_output *= tf.expand_dims(tf.cast(mask, tf.float32), axis=-1)
        # enc_output *= tf.expand_dims(tf.cast(mask, tf.float16), axis=-1)
        
        # pooling by summing over constituent dimension
        enc_output = tf.math.reduce_sum(enc_output, axis=1) 

        # decoder
        output = enc_output
        for i in range(len(self.decoder_dense)):
            output = self.decoder_dense[i](output)
        output = self.output_pred(self.output_dense(output))

        if self.output_attn:
            return output, attn_scores
        else:
            return output

class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, d_model, warmup_steps, lr_multiplier):
        super(CustomSchedule, self).__init__()

        self.lr_multiplier = lr_multiplier
        self.d_model = d_model
        self.warmup_steps = warmup_steps

    @tf.function(experimental_relax_shapes=relax_shapes)
    def get_config(self):
        config = {
        'd_model': self.d_model,
        'warmup_steps': self.warmup_steps,
        'lr_multiplier': self.lr_multiplier
        }
        return config

    @tf.function(experimental_relax_shapes=relax_shapes)
    def __call__(self, step):
        # step = tf.cast(step, tf.float32) # needed for serialisation during model saving
        arg1 = tf.math.rsqrt(step)
        arg2 = step * (self.warmup_steps ** -1.5)

        return self.lr_multiplier * tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)