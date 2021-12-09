
import argparse
import sys, os, yaml
from tensorflow.keras.metrics import BinaryAccuracy, AUC
from tensorflow.keras.models import load_model
from itertools import chain

sys.path.insert(0, "..")
from commonReco import *
from common import setup_gpu
import DataLoaderReco


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('-model','--model_path', help='Full path to the model', required=True)
    parser.add_argument('-data','--data_path', help='Full path to the data file', required=True)
    parser.add_argument('-scaling','--scaling_path', help='Full path to the scaling file', required=True)
    parser.add_argument('-training','--training_path', help='Full path to the training config', required=True)
    args = parser.parse_args()

    model = load_model(args.model_path, custom_objects={'binary_accuracy':BinaryAccuracy(), 'auc':AUC()})
    setup_gpu({"gpu_mem": 7, "gpu_index": 0})

    with open(os.path.abspath(args.training_path)) as f:
        config = yaml.safe_load(f)
    config["SetupNN"]["validation_split"] = 0.0
    config["Setup"]["n_tau"] = 500
    config["Setup"]["input_dir"] = args.data_path
    scaling  = os.path.abspath(args.scaling_path)
    dataloader = DataLoaderReco.DataLoader(config, scaling)

    gen_train = dataloader.get_generator(primary_set = True)

    input_shape, input_types  = dataloader.get_shape()

    data_train = tf.data.Dataset.from_generator(
        gen_train, output_types = input_types, output_shapes = input_shape
        ).prefetch(10)

    start = time.time()
    time_checkpoints = []

    y_pred = []
    y_true = []
    for i,(x,y) in enumerate(data_train):
        time_checkpoints.append(time.time()-start)
        print(i, " ", time_checkpoints[-1], "s.")
        start = time.time()
        y_pred.extend(list(chain.from_iterable(model.predict(x).tolist())))
        y_true.extend(list(chain.from_iterable(y.numpy().tolist())))


    # plots
    import matplotlib.pyplot as plt
    from sklearn.metrics import roc_curve

    if not os.path.exists("./eval"):
        os.makedirs("./eval")

    y_pred = np.array(y_pred)
    y_true = np.array(y_true)

    print("Plotting prob.")
    plt.figure(figsize=(10,10))
    plt.hist(y_pred[y_true==0.0], range=(0.0, 1.0), bins=100, label='QCD')
    plt.hist(y_pred[y_true==1.0], range=(0.0, 1.0), bins=100, label='Signal', alpha=0.8, color='r')
    plt.xlabel('Probability', fontsize=25)
    plt.ylabel('arb. units', fontsize=25)
    plt.legend(fontsize=15)
    plt.tick_params(axis='both', labelsize=25, pad=5)
    plt.savefig('./eval/prob.png')
    plt.yscale('log')
    plt.savefig('./eval/prob_log.png')

    print("Plotting auc.")
    fpr, tpr, _ = roc_curve(y_true, y_pred)
    plt.figure(figsize=(10,10))
    plt.plot(fpr, tpr) 
    plt.xlabel('false positive rate', fontsize=25)
    plt.ylabel('true positive rate', fontsize=25)
    plt.tick_params(axis='both', labelsize=25, pad=5)
    plt.savefig('./eval/auc.png')
    plt.yscale('log')
    plt.savefig('./eval/auc_log.png')

    exit()