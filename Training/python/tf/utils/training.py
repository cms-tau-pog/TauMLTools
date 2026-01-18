"""
Training utilities for TensorFlow-based neural network models.
This module provides functions for dataset composition, loading, and preprocessing,
with support for distributed training across multiple GPUs.

Main components:
- Dataset composition and batching strategies
- Data scaling and normalization
- TFRecord parsing and loading
- Distributed training support
- MLflow logging utilities
"""

from omegaconf import OmegaConf
from hydra.core.hydra_config import HydraConfig
import tensorflow as tf
from tensorflow.python.ops import math_ops, array_ops
import numpy as np
import mlflow
import os
import yaml

def filter_by_features(data, filter_cfg, input_dataset_cfg, token_selection_mode="first"):
    """
    Filter individual tokens in dataset based on feature values and token counts.
    !Does not work with the global scope!

    Args:
        data (tf.data.Dataset): Input dataset to filter
        filter_cfg (dict): Dictionary containing filter specifications like:
            {
                "collection_name": {
                    "feature_name": {"min": val, "max": val},
                    "max_tokens": val,  # Optional: Maximum number of tokens to keep
                    "order_by": "feature_name"  # Optional: Feature to order tokens by before filtering
                }
            }
        input_dataset_cfg (dict): Dataset configuration with feature names
        token_selection_mode (str): Mode for token selection when applying max_tokens filter.
            Options:
                - "first": Keep the first N tokens (default).
                - "last": Keep the last N tokens.
                - "random": Randomly select N tokens.

    Returns:
        tf.data.Dataset: Dataset with filtered tokens
    """
    def filter_tokens(*inputs):
        filtered_inputs = list(inputs)
        for collection_idx, collection_name in enumerate(input_dataset_cfg['feature_names']):
            if collection_name in filter_cfg and collection_name != "global":
                collection_filters = filter_cfg[collection_name]
                features = inputs[collection_idx]
                mask = tf.ones_like(features[:, 0], dtype=tf.bool)
                # Feature-based filtering
                for feature_name, bounds in collection_filters.items():
                    if feature_name not in ["max_tokens", "order_by"]:
                        feature_idx = input_dataset_cfg['feature_names'][collection_name].index(feature_name)
                        values = features[..., feature_idx]
                        if "min" in bounds:
                            mask = tf.logical_and(mask, values >= bounds["min"])
                        if "max" in bounds:
                            mask = tf.logical_and(mask, values <= bounds["max"])
                # Token count filtering
                if "max_tokens" in collection_filters and collection_filters["max_tokens"] != 'none':
                    max_tokens = collection_filters["max_tokens"]
                    valid_indices = tf.where(mask)[:, 0]
                    # Order by feature if requested
                    order_by_feature = collection_filters.get("order_by", None)
                    if order_by_feature and token_selection_mode in ["first", "last"]:
                        order_by_idx = input_dataset_cfg['feature_names'][collection_name].index(order_by_feature)
                        order_values = tf.gather(features[..., order_by_idx], valid_indices)
                        sorted_indices = tf.argsort(order_values, axis=0, direction='ASCENDING')
                        valid_indices = tf.gather(valid_indices, sorted_indices)
                        if token_selection_mode == "last":
                            valid_indices = valid_indices[-max_tokens:]
                        else:
                            valid_indices = valid_indices[:max_tokens]
                    elif token_selection_mode == "random":
                        valid_indices = tf.random.shuffle(valid_indices)[:max_tokens]
                    else:
                        if token_selection_mode == "last":
                            valid_indices = valid_indices[-max_tokens:]
                        else:
                            valid_indices = valid_indices[:max_tokens]
                    filtered = tf.gather(features, valid_indices)
                else:
                    filtered = tf.boolean_mask(features, mask)
                filtered_inputs[collection_idx] = filtered
        return tuple(filtered_inputs)
    return data.map(filter_tokens, num_parallel_calls=tf.data.AUTOTUNE)

def compose_datasets_train_val(cfg, n_gpu, input_dataset_cfg, use_strategy):
    """
    Create training and validation datasets for keras.model.fit with distributed training support.

    Args:
        cfg (dict): Configuration dictionary containing model and training parameters
        n_gpu (int): Number of GPUs to use for distributed training
        input_dataset_cfg (dict): Dataset configuration parameters
        use_strategy (tf.distribute.Strategy): Distribution strategy for multi-GPU training

    Returns:
        tuple: Contains:
            - train_data (tf.data.Dataset): Training dataset
            - val_data (tf.data.Dataset): Validation dataset
            - num_train_steps (int): Number of training steps per epoch (None for single GPU)
            - num_val_steps (int): Number of validation steps per epoch (None for single GPU)
    """
    # Get scaling data and/or num of events in training dataset from dataset configs
    # Distributed datasets (train and val) require a number of steps with keras.model.fit
    if n_gpu > 1 or cfg["tf_dataset_cfg"]["scaler"]:
        scaling_data, num_train_events = setup_scaler(cfg, input_dataset_cfg, "train")
    else:
        scaling_data = None
        num_train_events = None
    if n_gpu > 1:
        _, num_val_events = setup_scaler(cfg, input_dataset_cfg, "val")
        num_train_steps = int(num_train_events / n_gpu / cfg["tf_dataset_cfg"]["train_batch_size"])
        num_val_steps = int(num_val_events / n_gpu / cfg["tf_dataset_cfg"]["val_batch_size"])
        # Make distributed dataset with one replica of input pipeline per GPU
        def dataset_train_fn(input_context):
            train_data = compose_datasets(cfg, input_dataset_cfg, "train", scaling_data, input_context).repeat()
            return train_data
        def dataset_val_fn(input_context):
            val_data = compose_datasets(cfg, input_dataset_cfg, "val", scaling_data, input_context).repeat()
            return val_data
        train_data = use_strategy.distribute_datasets_from_function(
            dataset_train_fn,
            tf.distribute.InputOptions(
                experimental_replication_mode=tf.distribute.InputReplicationMode.PER_REPLICA
            )
        )
        val_data = use_strategy.distribute_datasets_from_function(
            dataset_val_fn,
            tf.distribute.InputOptions(
                experimental_replication_mode=tf.distribute.InputReplicationMode.PER_REPLICA
            )
        )
    else:
        # Use normal dataset with one GPU
        train_data = compose_datasets(cfg, input_dataset_cfg, "train", scaling_data)
        val_data = compose_datasets(cfg, input_dataset_cfg, "val", scaling_data)
        num_train_steps = None
        num_val_steps = None
    return train_data, val_data, num_train_steps, num_val_steps #, scaling_data

# Experimental
def merge_dicts(d1, d2):
    """ Recursively merges d2 into d1 without overwriting existing nested keys. """
    for key, value in d2.items():
        if key in d1 and isinstance(d1[key], dict) and isinstance(value, dict):
            merge_dicts(d1[key], value)  # Recursively merge dictionaries
        elif key in d1 and isinstance(d1[key], list) and isinstance(value, list):
            d1[key].extend(value)  # Merge lists
        else:
            d1[key] = value  # Assign value (or overwrite if not a dict)
    return d1

def compose_datasets(cfg, input_dataset_cfg, data_type, scaling_data, input_context=None):
    """Create dataset pipeline for train/val dataset"""
    if not data_type in ["train", "val"]:
        raise RuntimeError(f"Invalid data kind {data_type}")
    tf_dataset_cfg=cfg["tf_dataset_cfg"]
    # Load data from files
    data = load_data(cfg, input_dataset_cfg, data_type, input_context)
    # Filter data by token properties and num tokens
    max_tokens = cfg.get("global_max_tokens")
    filter_config = None
    if max_tokens and max_tokens!="none":
        filter_config = {collection: {"max_tokens": max_tokens} for collection in input_dataset_cfg['feature_names']}
        print(f"global_max_tokens set to {max_tokens}")
    else:
        print("No global_max_tokens set")
    if cfg.get("filters") and cfg.get("filters")!="none":
        filter_config_prelim = cfg.get("filters")
        if not max_tokens:
            filter_config = filter_config_prelim
        else:
            filter_config = merge_dicts(filter_config, filter_config_prelim)
        print(f"Filters set to {filter_config}")
    else:
        print("No other filters set")
    if filter_config:
        filter_type = cfg.get("filter_type")
        if not filter_type:
            filter_type = "first"
        print(f"Apply filer: {filter_config} with filter_type: {filter_type}")
        data = filter_by_features(data, filter_config, input_dataset_cfg, filter_type)


    # shuffle/cache
    if data_type == "train":
        if tf_dataset_cfg["shuffle_buffer_size"] is not None:
            data = data.shuffle(tf_dataset_cfg["shuffle_buffer_size"])
        if tf_dataset_cfg["cache"]:
            data = data.cache()

    # batch/smart batch
    if tf_dataset_cfg['batching'] == "standard":
        data = data.batch(tf_dataset_cfg[f"{data_type}_batch_size"])
    elif tf_dataset_cfg['batching'] == "smart":
        data = _smart_batch_V2(data, tf_dataset_cfg, data_type)
        # train_data, val_data = _smart_batch_V1(train_data, val_data, tf_dataset_cfg)
    elif tf_dataset_cfg['batching'] == "token":
        data = _token_batch(data, tf_dataset_cfg, data_type)
        # if tf_dataset_cfg['smart_batching_step'] is not None:
        #     train_data, val_data = _add_weights_by_size(train_data, val_data)
    else:
        raise ValueError(f"Unsupported batching method: {tf_dataset_cfg['batching']}")

    # Add axis to global collection
    if "global" in list(input_dataset_cfg['feature_names'].keys()):
        glob_index = list(input_dataset_cfg['feature_names'].keys()).index("global")
        data = data.map(lambda *inputs: (*inputs[:glob_index], tf.expand_dims(inputs[glob_index],axis=-2), *inputs[glob_index+1:]))

    # Apply scaler from training data to data if requested
    if tf_dataset_cfg["scaler"]:
        scaling_means, scaling_stds = scaling_data
        data = scale_data(data, scaling_means, scaling_stds)

    @tf.function(experimental_relax_shapes=True)
    def ragged_to_dense_and_select_classes(*inputs, class_idx):
        # First convert ragged tensors to dense
        dense_tensors = tuple(input_tensor.to_tensor() if isinstance(input_tensor, tf.RaggedTensor) else input_tensor
                            for input_tensor in inputs)
        # Split features and labels
        features = dense_tensors[:-1]  # All tensors except the last one
        labels = tf.gather(dense_tensors[-1], indices=class_idx, axis=-1)  # Get selected classes from last tensor
        return (features, labels)  # Return as tuple of (features_tuple, labels)


    # Apply the combined mapping function to the dataset
    class_idx = [input_dataset_cfg['label_columns'].index(f'label_{c}') for c in tf_dataset_cfg["classes"]]
    data = data.map(
        lambda *inputs: ragged_to_dense_and_select_classes(*inputs, class_idx=class_idx),
        num_parallel_calls=1
    )
    # Switch of autosharding as we use manual sharding
    options = tf.data.Options()
    options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.OFF
    data = data.with_options(options)
    # prefetch
    # data = data.prefetch(4) # Limit prefetching due to RAM concerns
    data = data.prefetch(1)

    return data

def log_to_mlflow(model, cfg, scaler_data=None):
    """
    Log model artifacts, parameters, and metrics to MLflow.

    Records:
    - Model architecture and weights
    - Dataset configurations
    - Training parameters
    - Model summary statistics

    Args:
        model (tf.keras.Model): Trained model to log
        cfg (dict): Configuration dictionary containing all parameters
    """
    # save model & print summary
    print("\n-> Saving model")
    path_to_hydra_logs = HydraConfig.get().run.dir
    model.save((f'{path_to_hydra_logs}/{cfg["model"]["name"]}'), save_format="tf") # save to hydra logs
    # model.save((f'{path_to_hydra_logs}/{cfg["model"]["name"]}.keras')) # save to hydra logs
    mlflow.log_artifacts(f'{path_to_hydra_logs}/{cfg["model"]["name"]}', 'model') # and also to mlflow artifacts
    if cfg["model"]["type"] == 'taco_net':
        print(model.wave_encoder.summary())
        summary_list_encoder, summary_list_decoder = [], []
        model.wave_encoder.summary(print_fn=summary_list_encoder.append)
        summary_encoder, summary_decoder = "\n".join(summary_list_encoder), "\n".join(summary_list_decoder)
        mlflow.log_text(summary_encoder, artifact_file="encoder_summary.txt")
    elif cfg["model"]["type"] == 'transformer':
        print(model.summary())
    elif cfg['model']['type'] == 'particle_net':
        print(model.summary())

    # log data params
    mlflow.log_param('dataset_name', cfg["dataset_name"])
    mlflow.log_param('datasets_cfg',  cfg["input_files"]["cfg"])
    mlflow.log_param('datasets_train',  cfg["input_files"]["train"])
    mlflow.log_param('datasets_val',  cfg["input_files"]["val"])
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

    # # Save scaler as artifact if provided
    # if scaler_data is not None:
    #     scaler_path = os.path.join(path_to_hydra_logs, "scaler.pkl")
    #     with open(scaler_path, "wb") as f:
    #         pickle.dump(scaler_data, f)
    #     mlflow.log_artifact(scaler_path, artifact_path="scaler")
    #     print(f"Scaler saved and logged to MLflow: {scaler_path}")

    # log N trainable params
    summary_list = []
    model.summary(print_fn=summary_list.append)
    for l in summary_list:
        if (s:='Trainable params: ') in l:
            mlflow.log_param('n_train_params', int(l.split(s)[-1].replace(',', '')))

# Currently hard-coded for 4 collections with one global one
def element_length_fn(*seq):
    # Sum of tokens in non-global + 1
    return tf.reduce_sum([tf.shape(seq[i])[0] for i in range(3)]) + 1

def _token_batch(data, tf_dataset_cfg, data_type):
    """
    Batch data based on token count to maintain consistent memory usage.

    Args:
        train_data (tf.data.Dataset): Training dataset
        val_data (tf.data.Dataset): Validation dataset
        tf_dataset_cfg (dict): Configuration containing token batching parameters

    Returns:
        tuple: (bucketed_train_dataset, bucketed_val_dataset)

    Note: Not up to date with the rest of the batching functions. Needs rework of input/output
    """
    bucket_boundaries = np.arange(
        tf_dataset_cfg['sequence_length_dist_start']+tf_dataset_cfg['smart_batching_step'],
        tf_dataset_cfg['sequence_length_dist_end']+tf_dataset_cfg['smart_batching_step'],
        tf_dataset_cfg['smart_batching_step']
    )
    # train_batch_sizes = (tf_dataset_cfg['train_tokens_per_batch']/bucket_boundaries).astype(int)
    # val_batch_sizes = (tf_dataset_cfg['val_tokens_per_batch']/bucket_boundaries).astype(int)
    batch_sizes = (tf_dataset_cfg['tokens_per_batch']/bucket_boundaries).astype(int)
    # train_batch_sizes = np.append(train_batch_sizes, int(train_batch_sizes[-1]/2))
    # val_batch_sizes = np.append(val_batch_sizes, int(val_batch_sizes[-1]/2))
    batch_sizes = np.append(batch_sizes, int(batch_sizes[-1]/2)).tolist()
    # Bucket the dataset by sequence length
    @tf.function(experimental_relax_shapes=True)
    def apply_bucketing(dataset, batch_sizes, is_training=True):
        bucketed_dataset = dataset.bucket_by_sequence_length(
            element_length_fn,
            bucket_boundaries=bucket_boundaries,
            bucket_batch_sizes=batch_sizes,
            no_padding=True,
            drop_remainder=True  # Drop incomplete batches
        )
        # Only shuffle for training
        if is_training:
            shuffle_size = tf_dataset_cfg['shuffle_smart_buffer_size']
            bucketed_dataset = bucketed_dataset.shuffle(shuffle_size)
        return bucketed_dataset

    # Apply bucketing
    is_train = data_type == "train"
    dataset = apply_bucketing(data, batch_sizes, is_train)
    return dataset

def _smart_batch_V1(train_data, val_data, tf_dataset_cfg):
    """
    Legacy smart batching implementation based on first element sequence lengths.

    Args:
        train_data (tf.data.Dataset): Training dataset
        val_data (tf.data.Dataset): Validation dataset
        tf_dataset_cfg (dict): Configuration for batching parameters

    Returns:
        tuple: (batched_train_data, batched_val_data)

    Note:
        This is the V1 implementation kept for reference. Use _smart_batch_V2 instead.
    """
    # will do smart batching based only on the sequence lengths of the **first** element (assume it to be PF candidate block)
    # NB: careful when dropping whole blocks in `embedding.yaml` -> change smart batching id here accordingly
    element_length_func = lambda *elements: tf.shape(elements[0])[0]

    bucket_boundaries = np.arange(
        tf_dataset_cfg['sequence_length_dist_start'],
        tf_dataset_cfg['sequence_length_dist_end'],
        tf_dataset_cfg['smart_batching_step']
    )

    def _element_to_bucket_id(*args):
        seq_length = element_length_func(*args)
        boundaries = list(bucket_boundaries)
        buckets_min = [np.iinfo(np.int32).min] + boundaries
        buckets_max = boundaries + [np.iinfo(np.int32).max]
        conditions_c = math_ops.logical_and(
            math_ops.less_equal(buckets_min, seq_length),
            math_ops.less(seq_length, buckets_max)
        )
        bucket_id = math_ops.reduce_min(array_ops.where(conditions_c))
        return bucket_id

    def _reduce_func(unused_arg, dataset, batch_size):
        return dataset.batch(batch_size)

    train_data = train_data.group_by_window(
        key_func=_element_to_bucket_id,
        reduce_func=lambda unused_arg, dataset: _reduce_func(unused_arg, dataset, tf_dataset_cfg['train_batch_size']),
        window_size=tf_dataset_cfg['train_batch_size']
    ).shuffle(int(tf_dataset_cfg['shuffle_smart_buffer_size']))

    val_data = val_data.group_by_window(
        key_func=_element_to_bucket_id,
        reduce_func=lambda unused_arg, dataset: _reduce_func(unused_arg, dataset, tf_dataset_cfg['val_batch_size']),
        window_size=tf_dataset_cfg['val_batch_size']
    ).shuffle(int(tf_dataset_cfg['shuffle_smart_buffer_size']))

    return train_data, val_data

def _smart_batch_V2(data, tf_dataset_cfg, data_type):
    """
    Implement smart batching strategy V2 with bucket-based sequence length grouping.

    This function groups sequences of similar lengths together to minimize padding
    and improve training efficiency.

    Args:
        data (tf.data.Dataset): Input dataset to batch
        tf_dataset_cfg (dict): Configuration for batching parameters
        data_type (str): Type of dataset ('train' or 'val')

    Returns:
        tf.data.Dataset: Batched dataset with smart bucketing applied

    Raises:
        RuntimeError: If data_type is invalid
    """
    if not data_type in ["train", "val"]:
        raise RuntimeError(f"Invalid data kind {data_type}")
    # Get buckets
    bucket_boundaries = np.arange(
        tf_dataset_cfg['sequence_length_dist_start']+tf_dataset_cfg['smart_batching_step'],
        tf_dataset_cfg['sequence_length_dist_end']+tf_dataset_cfg['smart_batching_step'],
        tf_dataset_cfg['smart_batching_step']
    )
    # Define uniform batch size for each bucket
    batch_sizes = [tf_dataset_cfg[f'{data_type}_batch_size']] * (len(bucket_boundaries) + 1)

    @tf.function(experimental_relax_shapes=True)
    def apply_bucketing(dataset, batch_sizes, is_training=True):
        bucketed_dataset = dataset.bucket_by_sequence_length(
            element_length_fn,
            bucket_boundaries=bucket_boundaries,
            bucket_batch_sizes=batch_sizes,
            no_padding=True,
            drop_remainder=True  # Drop incomplete batches
        )
        # Only shuffle for training
        if is_training:
            shuffle_size = tf_dataset_cfg['shuffle_smart_buffer_size']
            bucketed_dataset = bucketed_dataset.shuffle(shuffle_size)
        return bucketed_dataset

    # Apply bucketing
    is_train = data_type == "train"
    dataset = apply_bucketing(data, batch_sizes, is_train)

    return dataset

def load_data(cfg, input_dataset_cfg, data_type, input_context=None):
    """
    Load and preprocess data from files with support for distributed loading.

    Args:
        cfg (dict): Configuration dictionary
        input_dataset_cfg (dict): Dataset configuration parameters
        data_type (str): Type of dataset to load ('train' or 'val')
        input_context (tf.distribute.InputContext, optional): Context for distributed loading

    Returns:
        tf.data.Dataset: Loaded and preprocessed dataset

    Raises:
        ValueError: If no datasets are found or file format is unsupported
        RuntimeError: If data_type is invalid
    """
    if not data_type in ["train", "val"]:
        raise RuntimeError(f"Invalid data kind {data_type}")
    if data_type == "train":
        alt_type = "training"
    else:
        alt_type = "validation"
    # Get file list
    tf_dataset_cfg=cfg["tf_dataset_cfg"]
    # file_limit = 45 if data_type=="train" else 5
    files = [os.path.join(tf_dataset_cfg["datasets_location"][alt_type], os.path.basename(f)) for f in cfg["input_files"][data_type]] #[:file_limit] # Cut because of time, remove later

    if len(files) == 0:
        raise ValueError(f'No datasets found in {tf_dataset_cfg["datasets_location"][alt_type]}')
    else:
        print(f'Found {len(files)} datasets in {tf_dataset_cfg["datasets_location"][alt_type]}')

    # NOTE: it is necessary to overwrite the reader_func of the loader if used in combination with interleave
    #   as both the load function AND the interleave function attempt to use all available cores otherwise.
    #   This leads to an exponential creation of threads for machines with many cores,
    if input_dataset_cfg.get("file_format") == "tfrecord":
        element_spec = input_dataset_cfg.get("element_spec")
    elif input_dataset_cfg.get("file_format") == "tfsave":
        element_spec = tf.data.Dataset.load(files[0], compression='GZIP').element_spec
    else:
        raise ValueError(f"Unsupported file format: {input_dataset_cfg.get('file_format')}")

    if tf_dataset_cfg['combine_via'] == 'sampling':
        if input_dataset_cfg.get("file_format") == "tfrecord":
            file_names = [f"{f}/data.tfrecord" for f in files]
            _dataset = _read_tfrecord(file_names, element_spec)
        elif input_dataset_cfg.get("file_format") == "tfsave":
            _dataset = _tfsave_load_datasets(files, element_spec)
        # True so that the last batches are not purely of one class
        data = tf.data.Dataset.sample_from_datasets(datasets=_dataset, seed=1234, stop_on_empty_dataset=False)

    elif tf_dataset_cfg['combine_via'] == 'interleave': # compose final dataset as consecutive (cycle_length=1) loading of input TF datasets
        if input_dataset_cfg.get("file_format") == "tfrecord":
            file_names = [f"{f}/data.tfrecord" for f in files]
            data = _read_tfrecord(file_names, element_spec, input_context)
        elif input_dataset_cfg.get("file_format") == "tfsave":
            data = _tfsave_interleave(files, element_spec)
    else:
        raise ValueError("`combine_via` should be either 'sampling' or 'interleave'")

    return data

def scale_data(data, scaling_means, scaling_stds):
    """
    Apply standardization scaling to input features.

    Args:
        data (tf.data.Dataset): Dataset to scale
        scaling_means (list): List of mean values for each feature
        scaling_stds (list): List of standard deviation values for each feature

    Returns:
        tf.data.Dataset: Scaled dataset with original labels
    """
    @tf.function(experimental_relax_shapes=True)
    def scale_tensors(*tensors):
        # Process each feature tensor separately (Axis=-1)
        scaled_features = []
        for feature, mean, std in zip(tensors[:-1], scaling_means, scaling_stds):
            # Scale the feature tensor
            scaled_features.append((feature - mean) / std)
        # Return scaled features and original labels
        return tuple(scaled_features) + (tensors[-1],)

    # Apply scaling with multiple parallel calls
    data = data.map(
        scale_tensors,
        num_parallel_calls=1
    )
    return data

def _tfsave_load_datasets(files, element_spec=None):
    _dataset = []
    for p in files:
        _dataset.append(
            tf.data.Dataset.load(
                p,
                compression='GZIP',
                reader_func=lambda dataset: dataset.interleave(
                    lambda x: x,
                    cycle_length=1,
                    num_parallel_calls=tf.data.AUTOTUNE
                    ),
                element_spec=element_spec
                )
            )
    return _dataset

# This code only works up to tf 2.10
# there is no way to use the load function with interleave after that
def _tfsave_interleave(files, val_files, element_spec):
    cycle_length = 40
    block_length = 1
    data_ds = tf.data.Dataset.from_tensor_slices(files)
    loaded_data = data_ds.interleave(
        lambda x: tf.data.Dataset.load(
            x,
            element_spec=element_spec,
            compression='GZIP',
            reader_func=lambda dataset: dataset.interleave(
                lambda x: x, cycle_length=1, num_parallel_calls=tf.data.AUTOTUNE)
            ),
        cycle_length=cycle_length,
        num_parallel_calls=tf.data.AUTOTUNE,
        deterministic=False,
        block_length=block_length)
    return loaded_data

def _load_datasets2(files, element_spec=None):
    return [tf.data.Dataset.load(p, compression='GZIP', element_spec=element_spec,reader_func=lambda dataset: dataset.interleave(lambda x: x, cycle_length=tf.data.AUTOTUNE, num_parallel_calls=tf.data.AUTOTUNE, deterministic=False)) for p in files]

def _merge_datasets2(datasets):
    train_data_ = tf.data.Dataset.from_tensor_slices(datasets)
    train_data = train_data_.interleave(lambda x: x,cycle_length=tf.data.AUTOTUNE, num_parallel_calls=tf.data.AUTOTUNE, deterministic=False)
    return train_data

def setup_scaler(cfg, input_dataset_cfg, data_type):
    """
    Calculate mean, standard deviation and count of data features.

    Args:
        cfg (dict): Configuration dictionary containing dataset parameters
        input_dataset_cfg (dict): Input dataset configuration
        data_type (str): Type of dataset ('train' or 'val')

    Returns:
        tuple: Contains:
            - tuple or None: (scaling_means, scaling_stds) for train data, None for val data
            - int: Number of events in the dataset

    Raises:
        RuntimeError: If data_type is invalid
    """
    if not data_type in ["train", "val"]:
        raise RuntimeError(f"Invalid data kind {data_type}")
    if data_type == "train":
        alt_type = "training"
    else:
        alt_type = "validation"
    # Get list of training cfg files
    tf_dataset_cfg=cfg["tf_dataset_cfg"]
    scaler_data = {}
    files = [os.path.join(tf_dataset_cfg["datasets_location"][alt_type], os.path.basename(f)) for f in cfg["input_files"][data_type]][:45]
    for file in files:
        key = os.path.basename(file)
        with open(f"{file}/cfg.yaml", 'rb') as f:
            scaler_data[key] = yaml.safe_load(f)["scaling_data"]
    # Merge statistics
    merged_scaler_data = merge_statistics(scaler_data, input_dataset_cfg['feature_names'])
    # Sum of event counts
    num_events = merged_scaler_data["global"]["particle_type"]["count"]
    # Only calc stat data for training data
    if data_type == "val":
        return None, num_events
    else:
        # Special treatment for "particle_type" feature as it is categorical and shared between collections
        for collection in input_dataset_cfg['feature_names']:
            if "particle_type" in input_dataset_cfg['feature_names'][collection]:
                merged_scaler_data[collection]["particle_type"]["mean"] = 0.
                merged_scaler_data[collection]["particle_type"]["std"] = 1.
        # Turn merged statistics into lists
        scaling_means = []
        scaling_stds = []
        for i_col, collection in enumerate(input_dataset_cfg['feature_names']):
            scaling_means.append([])
            scaling_stds.append([])
            for variable in input_dataset_cfg['feature_names'][collection]:
                scaling_means[i_col].append(merged_scaler_data[collection][variable]["mean"])
                scaling_stds[i_col].append(merged_scaler_data[collection][variable]["std"])
        # Ensure proper shape
        scaling_means = [tf.reshape(tf.constant(mean, dtype=tf.float32), [1, 1, -1]) for mean in scaling_means]
        scaling_stds = [tf.reshape(tf.constant(std, dtype=tf.float32), [1, 1, -1]) for std in scaling_stds]
        return (scaling_means, scaling_stds), num_events

def merge_statistics(scaler_data, feature_names):
    """
    Merge multiple statistics dictionaries for feature scaling.

    Args:
        scaler_data (dict): Dictionary of statistics from multiple files
        feature_names (dict): Dictionary mapping collection names to feature lists

    Returns:
        dict: Merged statistics containing mean, std, min, max and count per feature
             organized by collection
    """
    merged_stats = {}
    for collection in feature_names:
        merged_stats[collection] = {}
        for variable in feature_names[collection]:
            merged_stats[collection][variable] = {
                    "mean": 0.0, "std": 0.0, "min": float("inf"), "max": float("-inf"), "count": 0
                }
            for file_stats in scaler_data.values():
                # Get existing values
                values = file_stats[collection][variable]
                count_old = merged_stats[collection][variable]["count"]
                count_new = values["count"]
                total_count = count_old + count_new
                # Compute new mean using weighted average
                mean_old = merged_stats[collection][variable]["mean"]
                mean_new = values["mean"]
                merged_mean = (mean_old * count_old + mean_new * count_new) / total_count if total_count > 0 else 0.0
                # Compute new std using pooled variance formula
                std_old = merged_stats[collection][variable]["std"]
                std_new = values["std"]
                merged_variance = (
                    (count_old * (std_old ** 2 + mean_old ** 2) + count_new * (std_new ** 2 + mean_new ** 2))
                    / total_count
                ) - merged_mean ** 2
                merged_std = np.sqrt(max(merged_variance, 0))  # Ensure non-negative variance
                # Update statistics
                merged_stats[collection][variable]["mean"] = merged_mean
                merged_stats[collection][variable]["std"] = merged_std
                merged_stats[collection][variable]["min"] = min(merged_stats[collection][variable]["min"], values["min"])
                merged_stats[collection][variable]["max"] = max(merged_stats[collection][variable]["max"], values["max"])
                merged_stats[collection][variable]["count"] = total_count
    return merged_stats

@tf.function(experimental_relax_shapes=True)
def _parse_tfrecord(example_proto, dataset_spec):
    """
    Parse a single TFRecord example into feature tensors and targets.

    Args:
        example_proto (tf.train.Example): Protocol buffer of the TF example
        dataset_spec (list): Specifications for each collection in the dataset,
                           including shapes and whether features are ragged

    Returns:
        tuple: Contains:
            - list[tf.Tensor or tf.RaggedTensor]: Feature tensors for each collection
            - tf.Tensor: Target labels

    Raises:
        ValueError: If the dataset specification contains unexpected formats

    Note:
        The function handles both regular and ragged tensors based on the collection
        specifications. Each collection can be either:
        - Ragged: 2D shape with specified ragged_rank
        - Dense: 1D shape with no ragged_rank
    """
    feature_description = {
        'targets': tf.io.VarLenFeature(tf.int64)
    }
    is_ragged = []
    dimensions = []
    for col_i, collection_spec in enumerate(dataset_spec[:-1]):
        dimensions.append(collection_spec["shape"][-1])
        if (len(collection_spec["shape"]) == 2) and not (collection_spec.get("ragged_rank") == None):
            is_ragged.append(True)
            feature_description.update({
                f'collection_{col_i}_rowlens': tf.io.VarLenFeature(tf.int64),
            })
        elif (len(collection_spec["shape"]) == 1) and (collection_spec.get("ragged_rank") == None):
            is_ragged.append(False)
        else:
            raise ValueError(f"Unexpected element specification: {collection_spec}")
        feature_description.update({
            f'collection_{col_i}_values': tf.io.VarLenFeature(tf.float32),
        })
    # Add global features and targets
    parsed = tf.io.parse_single_example(example_proto, feature_description)
    feature_tensors = []
    for col_i, (dim, is_ragged) in enumerate(zip(dimensions, is_ragged)):
        if is_ragged:
            values = parsed[f'collection_{col_i}_values'].values
            rowlens = tf.cast(parsed[f'collection_{col_i}_rowlens'].values, tf.int64)
            tensor = tf.RaggedTensor.from_tensor(tf.squeeze(tf.RaggedTensor.from_row_lengths(tf.reshape(values, [-1, dim]),rowlens), axis=0))
            feature_tensors.append(tensor)
        else:
            feature_tensors.append(tf.sparse.to_dense(parsed[f'collection_{col_i}_values']))
    targets = tf.sparse.to_dense(parsed['targets'])
    return tuple(feature_tensors + [targets])

@tf.function(experimental_relax_shapes=True)
def _read_tfrecord(filenames, element_spec, input_context):
    """
    Create a dataset from TFRecord files with optional sharding for distributed training.

    Args:
        filenames (list): List of TFRecord file paths to read
        element_spec (list): Specifications for parsing the TFRecord elements
        input_context (tf.distribute.InputContext): Context for distributed data loading,
                                                  None for single-worker training

    Returns:
        tf.data.Dataset: Dataset containing parsed feature tensors and labels

    Note:
        - Files are shuffled before reading
        - Uses GZIP compression
        - Automatically shards data when using distributed training
        - Enables parallel reading and parsing for better performance
    """
    filenames_ds = tf.data.Dataset.from_tensor_slices(filenames)
    if input_context:
        filenames_ds = filenames_ds.shard(num_shards=input_context.num_input_pipelines, index=input_context.input_pipeline_id)
        shuffle_size = int(len(filenames) / input_context.num_input_pipelines)
    else:
        shuffle_size = len(filenames)
    filenames_ds = filenames_ds.shuffle(shuffle_size)

    record_dataset = tf.data.TFRecordDataset(
        filenames_ds,
        compression_type='GZIP',
        num_parallel_reads=tf.data.AUTOTUNE
    )
    parsed_dataset = record_dataset.map(
        lambda x: _parse_tfrecord(x, element_spec),
        num_parallel_calls=1,
    )
    return parsed_dataset
