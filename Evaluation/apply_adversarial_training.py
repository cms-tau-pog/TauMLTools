from tensorflow.keras.models import load_model
import json
import os
import yaml
import gc
import sys
from glob import glob
import math
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd
import argparse
os.environ["CUDA_VISIBLE_DEVICES"]="-1"
sys.path.insert(0, "..")


parser = argparse.ArgumentParser(description='Generate predictions for adversarial training')
parser.add_argument('--expID', required=True, type=str, help="Experiment ID")
parser.add_argument('--mlpath', required=True, type=str, help="Path to experiment")
parser.add_argument('--eval_ds', required=True, type=str, help="Dataset to evaluate on")
parser.add_argument('--n_tau', required=False, type=int, default=100, help="Number of taus/batch")
parser.add_argument('--n_batches', required=False, type=int, default=750, help="Number of batches")
# By default this script attempts to compute adversarial predictions (y_adv), these are not available for models which were not trained with the adversarial 
# subnetwork, however DeepTau VSjet distributions for data and MC can still be computed if the argument below is specified. This is useful to compare the 
# agreement before and after adversarial training.
parser.add_argument('--not_adversarial', action='store_true') 

args = parser.parse_args()
size = args.n_tau*args.n_batches



# Model prediction function
def test(data, model):
        # Unpack the data
        x, y, sample_weight = data
        y = y.numpy()[:,0] # Data/MC truth stored at this index (cf dataloader)
        if args.not_adversarial:
            # Only classification predictions are available
            y_pred_class = model(x, training=False) 
            return y_pred_class, y, sample_weight
        else:
            # Compute predictions for classification and adversarial subnetworks
            y_pred = model(x, training=False)
            y_pred_class = y_pred[0].numpy() # classification prediction
            y_pred_adv = y_pred[1].numpy() # y_adv prediction
            return y_pred_class, y_pred_adv, y, sample_weight
        

# DeepTau VS jet score computation function
def DeepTauVSjet(y_pred_class):
    DTvsjet = y_pred_class[:, 2]/(y_pred_class[:, 2]+y_pred_class[:, 3])
    return DTvsjet


# Store the relevant evaluated properties in arrays
def evaluate(ds, model):
    scores = np.zeros(size)
    weights = np.zeros(size)
    truth = np.zeros(size)
    y_pred_adv = np.zeros(size)

    i_batch = 0
    i_tau = 0
    
    for elem in ds:
        if args.not_adversarial:
            batch_y_pred_class, batch_y, batch_sample_weight = test(elem, model)
        else:
            batch_y_pred_class, batch_y_pred_adv, batch_y, batch_sample_weight = test(elem, model)
            y_pred_adv[i_tau:i_tau+args.n_tau] = batch_y_pred_adv[:,0]
        scores[i_tau:i_tau+args.n_tau] = DeepTauVSjet(batch_y_pred_class)
        truth[i_tau:i_tau+args.n_tau] = batch_y 
        weights[i_tau:i_tau+args.n_tau] = batch_sample_weight
        
        i_batch += 1
        if (i_batch%50 ==0):
            print("Batch: ", i_batch)
        i_tau += args.n_tau
        if i_batch >= args.n_batches:
            break
    return scores, truth, weights, y_pred_adv


# Load evaluation dataset:
test_ds = tf.data.experimental.load(args.eval_ds, compression="GZIP")

# Load model
path_to_exp = args.mlpath + "/" + args.expID 
path_to_artifacts = path_to_exp+"/artifacts"

with open(f'{path_to_artifacts}/input_cfg/metric_names.json') as f:
    metric_names = json.load(f)
    path_to_model = f'{path_to_artifacts}/model'

model = load_model(path_to_model, {name: lambda _: None for name in metric_names.keys()})


if args.not_adversarial:
    print("Warning, y_adv output not available for model trained without adversarial subnetwork.")
    scores, truth, weights, y_pred_adv = evaluate(test_ds, model) # y_pred_adv is only zeros here

    df = pd.DataFrame({"Djet": scores, "y": truth, "weights": weights})

else:
    scores, truth, weights, y_pred_adv = evaluate(test_ds, model)

    df = pd.DataFrame({"Djet": scores, "y": truth, "weights": weights, "y_pred_adv": y_pred_adv})

save_path = path_to_artifacts + "/predictions/adversarial_predictions.csv"
df.to_csv(save_path)
print(f"Predictions saved in {save_path}")
