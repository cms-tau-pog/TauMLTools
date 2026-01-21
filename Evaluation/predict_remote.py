import os
import gc
import yaml
import hydra
from omegaconf import DictConfig
import pickle
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from utils.tf_reader import _read_tfrecord, scale_data, filter_by_features

@hydra.main(config_path='..', config_name='predict_cfg')
def main(cfg: DictConfig) -> None:

    flat_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    print(flat_dir)
    # Batch job setup
    print('\n-> Loading model\n')
    path_to_model = os.path.abspath(f'{flat_dir}/artifacts/model/')
    model = load_model(path_to_model)
    print(model)

    # Use the first dataset_name for element_spec
    data_path_0 = os.path.join(flat_dir, f"Evaluation/data/test/{list(cfg['test_datasets'].keys())[0]}")
    data_cfg = os.path.join(data_path_0, os.listdir(data_path_0)[0], "cfg.yaml")
    with open(data_cfg, 'r') as f:
        dataset_cfg = yaml.safe_load(f)
    in_shapes = [inp["shape"][-1] for inp in dataset_cfg["element_spec"][:-2]] # Exclude targets and added_columns
    inputs = []
    for in_i, inp in enumerate(in_shapes):
        inputs.append(tf.keras.Input(shape=(None, inp), name=f"input_{in_i}"))
    # Necessary due to dynamic input shape during training
    outputs = model(tuple(inputs), training=False)
    model = tf.keras.Model(
        inputs=inputs,
        outputs=outputs
    )
    # Scaler
    scaler_arg = cfg["scaler"]
    if scaler_arg:
        scaler_save_path = f"{flat_dir}/artifacts/" + [i for i in  os.listdir(f"{flat_dir}/artifacts/") if i.endswith(".pkl")][0]
        with open(scaler_save_path, "rb") as f:
            scaling_data = pickle.load(f)
        print(f"Loaded scaler from {scaler_save_path}")

    # Collect all tfrecord files in the data dir
    for dataset_name, dataset_files in cfg['test_datasets'].items():
        # For each dataset, get tfrecord files
        data_path = os.path.join(flat_dir, f"Evaluation/data/test/{dataset_name}")
        data_file_dirs = [os.path.basename(i) for i in dataset_files]
        data_record_files = [os.path.join(flat_dir, data_path, f, "data.tfrecord")  for f in data_file_dirs]
        for file_name in data_record_files:
            dataset = _read_tfrecord([file_name], dataset_cfg["element_spec"], None)
            # Filter if necessary
            filter_config = cfg.get("filter_config", None)
            if filter_config:
                filter_type = cfg.get("filter_type")
                if not filter_type:
                    filter_type = "first"
                print(f"Apply filer: {filter_config} with filter_type: {filter_type}")
                dataset = filter_by_features(dataset, filter_config, dataset_cfg, filter_type)
            # Data loading pipeline
            dataset = dataset.batch(cfg["batch_size"])
            glob_index = list(dataset_cfg['feature_names'].keys()).index("global")
            dataset = dataset.map(lambda *inputs: (*inputs[:glob_index], tf.expand_dims(inputs[glob_index],axis=-2), *inputs[glob_index+1:]))
            if cfg["scaler"]:
                scaling_means, scaling_stds = scaling_data
                dataset = scale_data(dataset, scaling_means, scaling_stds)
            @tf.function(experimental_relax_shapes=True)
            def ragged_to_dense(*inputs):
                dense_tensors = tuple(input_tensor.to_tensor() if isinstance(input_tensor, tf.RaggedTensor) else input_tensor
                                    for input_tensor in inputs)
                features = dense_tensors[:-2]
                labels = dense_tensors[-2]
                added_columns = dense_tensors[-1]
                return (features, labels, added_columns)
            dataset = dataset.map(
                lambda *inputs: ragged_to_dense(*inputs),
                num_parallel_calls=1
            )
            dataset = dataset.prefetch(1)
            # Run test data through model
            predictions, labels, add_columns = [], [], []
            for (X, y, add_data) in dataset:
                predictions.append(model.predict(X, verbose=2))
                labels.append(y)
                add_columns.append(add_data)
            predictions = tf.concat(predictions, axis=0).numpy()
            labels = tf.concat(labels, axis=0).numpy()
            add_columns = tf.concat(add_columns, axis=0).numpy()
            predictions = pd.DataFrame(data=predictions, columns=[f'pred_{tau_type}' for tau_type in cfg["classes"]])
            labels = pd.DataFrame(data=labels, columns=dataset_cfg["label_columns"], dtype=np.int64)
            add_columns = pd.DataFrame(data=add_columns, columns=dataset_cfg["add_columns"])
            out_file = os.path.join(flat_dir, "predictions", dataset_name, os.path.basename(os.path.dirname(file_name)), "predictions.h5")
            os.makedirs(os.path.dirname(out_file), exist_ok=True)
            predictions.to_hdf(out_file, key='predictions', mode='w', format='fixed', complevel=1, complib='zlib')
            labels.to_hdf(out_file, key='labels', mode='r+', format='fixed', complevel=1, complib='zlib')
            add_columns.to_hdf(out_file, key='add_columns', mode='r+', format='fixed', complevel=1, complib='zlib')
            print(f"Saved predictions to {out_file}")
            del predictions
            del labels
            del dataset
            gc.collect()
            tf.keras.backend.clear_session()
    return

if __name__ == '__main__':
    main()
