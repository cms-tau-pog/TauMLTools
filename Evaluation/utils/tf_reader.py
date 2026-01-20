import tensorflow as tf
import numpy as np
import yaml
import os

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
        num_parallel_reads=1
    )
    parsed_dataset = record_dataset.map(
        lambda x: _parse_tfrecord(x, element_spec),
        num_parallel_calls=1,
    )
    return parsed_dataset


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
        'targets': tf.io.VarLenFeature(tf.int64),
        'added_columns': tf.io.VarLenFeature(tf.float32),
    }
    is_ragged = []
    dimensions = []
    for col_i, collection_spec in enumerate(dataset_spec[:-2]): # Exclude targets and added_columns
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
    # Add global features, targets and added_columns
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
    added_columns = tf.sparse.to_dense(parsed['added_columns'])
    return tuple(feature_tensors + [targets, added_columns])

def setup_scaler(cfg, input_dataset_cfg, data_type, train_dat_dict=None):
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
    if train_dat_dict:
        inp_files = train_dat_dict
    else:
        inp_files = cfg["input_files"]
        
    files = [os.path.join(tf_dataset_cfg["datasets_location"][alt_type], os.path.basename(f)) for f in inp_files[data_type]][:45]
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
        for feature, mean, std in zip(tensors[:-2], scaling_means, scaling_stds):
            # Scale the feature tensor
            scaled_features.append((feature - mean) / std)
        # Return scaled features and original labels
        return tuple(scaled_features) + (tensors[-2], tensors[-1],)
            
    # Apply scaling with multiple parallel calls
    data = data.map(
        scale_tensors,
        num_parallel_calls=4
    )
    return data

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
                    "max_tokens": val  # Optional: Maximum number of tokens to keep
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
    # @tf.function(experimental_relax_shapes=True)
    def filter_tokens(*inputs):
        filtered_inputs = list(inputs)
        
        for collection_idx, collection_name in enumerate(input_dataset_cfg['feature_names']):
            if collection_name in filter_cfg and collection_name != "global":
                collection_filters = filter_cfg[collection_name]
                mask = tf.ones_like(inputs[collection_idx][:, 0], dtype=tf.bool)
                collection_feature_size = inputs[collection_idx].shape[1]
                
                # Apply feature-based filters
                for feature_name, bounds in collection_filters.items():
                    if feature_name != "max_tokens":
                        feature_idx = input_dataset_cfg['feature_names'][collection_name].index(feature_name)
                        feature_values = inputs[collection_idx][..., feature_idx]
                        
                        if "min" in bounds:
                            mask = tf.logical_and(mask, feature_values >= bounds["min"])
                        if "max" in bounds:
                            mask = tf.logical_and(mask, feature_values <= bounds["max"])
                
                # Apply token count filter if specified
                if "max_tokens" in collection_filters and not collection_filters.get("max_tokens") == 'none':
                    max_tokens = collection_filters["max_tokens"]
                    # Count remaining valid tokens after feature filtering
                    valid_indices = tf.where(mask)
                    valid_count = tf.shape(valid_indices)[0]
                    
                    if valid_count > max_tokens:
                        if token_selection_mode == "first":
                            selected_indices = valid_indices[:max_tokens]
                        elif token_selection_mode == "last":
                            selected_indices = valid_indices[-max_tokens:]
                        elif token_selection_mode == "random":
                            selected_indices = tf.random.shuffle(valid_indices)[:max_tokens]
                        else:
                            raise ValueError(f"Invalid token_selection_mode: {token_selection_mode}")
                        
                        # Create new mask with selected indices set to True
                        new_mask = tf.zeros_like(mask)
                        new_mask = tf.tensor_scatter_nd_update(
                            new_mask,
                            selected_indices,
                            tf.ones(tf.shape(selected_indices)[0], dtype=tf.bool)
                        )
                        mask = new_mask
                
                # Apply final mask to keep only valid tokens
                filtered_inputs[collection_idx] = tf.RaggedTensor.from_uniform_row_length(
                    tf.ragged.boolean_mask(inputs[collection_idx], mask).values, 
                    collection_feature_size,
                )
        
        return tuple(filtered_inputs)
    
    return data.map(filter_tokens, num_parallel_calls=1)
