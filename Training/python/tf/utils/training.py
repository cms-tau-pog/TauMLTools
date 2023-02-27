from glob import glob
from collections import defaultdict
from omegaconf import OmegaConf
from hydra.core.hydra_config import HydraConfig

import tensorflow as tf
from tensorflow.python.ops import math_ops, array_ops
import numpy as np
import mlflow

def compose_datasets(datasets, tf_dataset_cfg, input_dataset_cfg):
    if tf_dataset_cfg['combine_via'] == 'sampling': # compose final dataset as sampling from the set of loaded input TF datasets
        datasets_for_training, sample_probas = _combine_datasets(datasets, load=True), None
        train_data = tf.data.Dataset.sample_from_datasets(datasets=datasets_for_training['train'], weights=sample_probas, seed=1234, stop_on_empty_dataset=False) # True so that the last batches are not purely of one class
        val_data = tf.data.Dataset.sample_from_datasets(datasets=datasets_for_training['val'], seed=1234, stop_on_empty_dataset=False)
    elif tf_dataset_cfg['combine_via'] == 'interleave': # compose final dataset as consecutive (cycle_length=1) loading of input TF datasets
        datasets_for_training = _combine_datasets(datasets, load=False)
        element_spec = tf.data.Dataset.load(datasets_for_training['train'][0], compression='GZIP').element_spec

        train_data = tf.data.Dataset.from_tensor_slices(datasets_for_training['train'])
        train_data = train_data.interleave(lambda x: tf.data.Dataset.load(x, element_spec=element_spec, compression='GZIP'),
                                                cycle_length=1, num_parallel_calls=tf.data.AUTOTUNE, deterministic=False)

        val_data = tf.data.Dataset.from_tensor_slices(datasets_for_training['val'])
        val_data = val_data.interleave(lambda x: tf.data.Dataset.load(x, element_spec=element_spec, compression='GZIP'),
                                                cycle_length=1, num_parallel_calls=tf.data.AUTOTUNE, deterministic=False)
    else:
        raise ValueError("`combine_via` should be either 'sampling' or 'interleave'")

    # shuffle/cache
    if tf_dataset_cfg["shuffle_buffer_size"] is not None:
        train_data = train_data.shuffle(tf_dataset_cfg["shuffle_buffer_size"])
    if tf_dataset_cfg["cache"]:
        train_data = train_data.cache()

    # batch/smart batch
    if tf_dataset_cfg['smart_batching_step'] is None:
        train_data = train_data.batch(tf_dataset_cfg["train_batch_size"])
        val_data = val_data.batch(tf_dataset_cfg["val_batch_size"])
    else:
        train_data, val_data = _smart_batch(train_data, val_data, tf_dataset_cfg)
        
    # prefetch
    train_data = train_data.prefetch(tf.data.experimental.AUTOTUNE)
    val_data = val_data.prefetch(tf.data.experimental.AUTOTUNE)

    # select from stored labels only those classes which are specified in the training cfg 
    class_idx = [input_dataset_cfg['label_columns'].index(f'label_{c}') for c in tf_dataset_cfg["classes"]]
    train_data = train_data.map(lambda *inputs: (inputs[:-1], tf.gather(inputs[-1], indices=class_idx, axis=-1)),
                                num_parallel_calls=tf.data.AUTOTUNE) # assume that labels tensor is yielded last
    val_data = val_data.map(lambda *inputs: (inputs[:-1], tf.gather(inputs[-1], indices=class_idx, axis=-1)),  
                                num_parallel_calls=tf.data.AUTOTUNE) 

    # limit number of threads, otherwise (n_threads=-1) error pops up (tf.__version__ == 2.9.1)
    options = tf.data.Options()
    options.threading.private_threadpool_size = tf_dataset_cfg["n_threads"]
    train_data = train_data.with_options(options)
    val_data = val_data.with_options(options)

    return train_data, val_data

def _combine_datasets(datasets, load=False):
    datasets_for_training = {'train': [], 'val': []} # to accumulate final datasets
    for dataset_type in datasets_for_training.keys():
        if dataset_type not in datasets:
            raise RuntimeError(f'key ({dataset_type}) should be present in dataset yaml configuration')
        for dataset_name, dataset_cfg in datasets[dataset_type].items(): # loop over specified train/val datasets
            for p in glob(f'{dataset_cfg["path_to_dataset"]}/{dataset_name}/{dataset_type}/*/'): # loop over all globbed files in the dataset
                if load:
                    _dataset = tf.data.Dataset.load(p, compression='GZIP')
                    datasets_for_training[dataset_type].append(_dataset) 
                else:   
                    datasets_for_training[dataset_type].append(p)    
    return datasets_for_training

def _combine_for_sampling(datasets):
    # NB: this is a deprecated function
    # keeping it as an example of uniform sampling across training classes
    sample_probas = [] # to store sampling probabilites on training datasets
    datasets_for_training = {'train': [], 'val': []} # to accumulate final datasets
    for dataset_type in datasets_for_training.keys():
        ds_per_tau_type = defaultdict(list)
        if dataset_type not in datasets:
            raise RuntimeError(f'key ({dataset_type}) should be present in dataset yaml configuration')
        for dataset_name, dataset_cfg in datasets[dataset_type].items(): # loop over specified train/val datasets
            for tau_type in dataset_cfg["tau_types"]: # loop over tau types specified for this dataset
                for p in glob(f'{dataset_cfg["path_to_dataset"]}/{dataset_name}/{dataset_type}/*/{tau_type}'): # loop over all globbed files in the dataset
                    dataset = tf.data.experimental.load(p) 
                    ds_per_tau_type[tau_type].append(dataset) # add TF dataset (1 input file, 1 tau type) to the map  
        
        n_tau_types = len(ds_per_tau_type.keys())
        for tau_type, ds_list in ds_per_tau_type.items():
            datasets_for_training[dataset_type] += ds_list # add datasets to the final list
            if dataset_type == "train": # for training dataset also keep corresponding sampling probas over input files
                n_files = len(ds_list)
                sample_probas += n_files*[1./(n_tau_types*n_files) ]
    
    return datasets_for_training, sample_probas

def _smart_batch(train_data, val_data, tf_dataset_cfg):
    def _element_to_bucket_id(*args):
        seq_length = element_length_func(*args)

        boundaries = list(bucket_boundaries)
        buckets_min = [np.iinfo(np.int32).min] + boundaries
        buckets_max = boundaries + [np.iinfo(np.int32).max]
        conditions_c = math_ops.logical_and(
        math_ops.less_equal(buckets_min, seq_length),
        math_ops.less(seq_length, buckets_max))
        bucket_id = math_ops.reduce_min(array_ops.where(conditions_c))

        return bucket_id

    def _reduce_func(unused_arg, dataset, batch_size):
        return dataset.batch(batch_size)

    # will do smart batching based only on the sequence lengths of the **first** element (assume it to be PF candidate block)
    # NB: careful when dropping whole blocks in `embedding.yaml` -> change smart batching id here accordingly
    element_length_func = lambda *elements: tf.shape(elements[0])[0]

    bucket_boundaries = np.arange(
        tf_dataset_cfg['sequence_length_dist_start'],
        tf_dataset_cfg['sequence_length_dist_end'],
        tf_dataset_cfg['smart_batching_step']
    )

    train_data = train_data.group_by_window(
        key_func=_element_to_bucket_id,
        reduce_func=lambda unused_arg, dataset: _reduce_func(unused_arg, dataset, tf_dataset_cfg['train_batch_size']),
        window_size=tf_dataset_cfg['train_batch_size']
    ).shuffle(tf_dataset_cfg['shuffle_smart_buffer_size'])

    val_data = val_data.group_by_window(
        key_func=_element_to_bucket_id,
        reduce_func=lambda unused_arg, dataset: _reduce_func(unused_arg, dataset, tf_dataset_cfg['val_batch_size']),
        window_size=tf_dataset_cfg['val_batch_size']
    ).shuffle(tf_dataset_cfg['shuffle_smart_buffer_size'])

    return train_data, val_data

def create_padding_mask(seq):
    mask = tf.math.reduce_any(tf.math.not_equal(seq, 0), axis=-1) # [batch, seq], 0 -> padding, 1 -> constituent
    return mask

def log_to_mlflow(model, cfg):
    # save model & print summary
    print("\n-> Saving model")
    path_to_hydra_logs = HydraConfig.get().run.dir
    model.save((f'{path_to_hydra_logs}/{cfg["model"]["name"]}.tf'), save_format="tf") # save to hydra logs
    mlflow.log_artifacts(f'{path_to_hydra_logs}/{cfg["model"]["name"]}.tf', 'model') # and also to mlflow artifacts
    if cfg["model"]["type"] == 'taco_net':
        print(model.wave_encoder.summary())
        print(model.wave_decoder.summary())
        summary_list_encoder, summary_list_decoder = [], []
        model.wave_encoder.summary(print_fn=summary_list_encoder.append)
        model.wave_decoder.summary(print_fn=summary_list_decoder.append)
        summary_encoder, summary_decoder = "\n".join(summary_list_encoder), "\n".join(summary_list_decoder)
        mlflow.log_text(summary_encoder, artifact_file="encoder_summary.txt")
        mlflow.log_text(summary_decoder, artifact_file="decoder_summary.txt") 
    elif cfg["model"]["type"] == 'transformer':
        print(model.summary())
    elif cfg['model']['type'] == 'particle_net':
        print(model.summary())

    # log data params
    mlflow.log_param('dataset_name', cfg["dataset_name"])
    mlflow.log_param('datasets_train', cfg["datasets"]["train"].keys())
    mlflow.log_param('datasets_val', cfg["datasets"]["val"].keys())
    mlflow.log_params(cfg['tf_dataset_cfg'])

    # log model params
    params_encoder = OmegaConf.to_object(cfg["model"]["kwargs"]["encoder"])
    params_embedding = params_encoder.pop('embedding_kwargs')
    params_embedding = {f'emb_{p}': v for p,v in params_embedding.items()}
    mlflow.log_param('model_name', cfg["model"]["name"])
    mlflow.log_params(params_encoder)
    for ptype, feature_list in params_embedding['emb_features_to_drop'].items():
        if len(feature_list)>5:
            params_embedding['emb_features_to_drop'][ptype] = ['too_long_to_log']
    mlflow.log_params(params_embedding)
    mlflow.log_params(cfg["model"]["kwargs"]["decoder"])
    mlflow.log_params({f'model_node_{i}': c for i,c in enumerate(cfg["tf_dataset_cfg"]["classes"])})
    if cfg['schedule']=='decrease':
        mlflow.log_param('decrease_every', cfg['decrease_every'])
        mlflow.log_param('decrease_by', cfg['decrease_by'])
    
    # log N trainable params 
    summary_list = []
    model.summary(print_fn=summary_list.append)
    for l in summary_list:
        if (s:='Trainable params: ') in l:
            mlflow.log_param('n_train_params', int(l.split(s)[-1].replace(',', '')))