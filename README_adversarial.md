# Adversarial Machine Learning Documentation

## Introduction

This file contains instructions for reproducing the adversarial domain adaptation techniques implemented in DeepTau v2p5 to reduce data/MC discrepancies in the DeepTau VSjet discriminator score distribution. The steps necessary for adversarial dataset production, training and evaluation of the DeepTau VSjet score distributions are outlined.


## Adversarial Dataset Production
Adversarial training requires a control region dataset consisting of a mix of data and MC event samples. The production of this dataset is outlined below.

### Event Selection

Most contributions to the adversarial dataset were produced using the `MuTau` selector in `TauMLTools/Production/src/Selectors.cc` with configurations specified in `TauMLTools/Production/crab/configs/MuTau2018`. 
The jobs were launched using `crab_submit.py` in the same way as described in the main `README` file for this repository. 


After this first level selection, additional cuts are applied using `TauMLTools/Analysis/skim_tree.py`. An example of this script being used to skim through a list of ROOT files containing collider data events is shown below:
```sh
python skim_tree.py --input="/eos/cms/store/group/phys_tau/lrussell/Prod2018_MuTau/SingleMuon.txt" --output="/eos/cms/store/group/phys_tau/lrussell/Prod2018_MuTau/SingleMuon_skimmed.root" --tree="taus" --sel="tau_byDeepTau2017v2p1VSjetraw>0.90 && tau_pt>30 && std::abs(tau_eta)<2.3 && tau_decayModeFindingNewDMs && (tau_decayMode==0 || tau_decayMode==1 || tau_decayMode==10 || tau_decayMode==11) && tau_byDeepTau2017v2p1VSeraw>0.1686942 && tau_byDeepTau2017v2p1VSmuraw>0.8754835 && tau_index>=0  && has_extramuon==0 && has_extraelectron==0 && has_dimuon==0"
```

For MC event skimming, requirements on tau type (e, mu, tau or jet) should be imposed, using the `recompute_tautype.py` script when calling `skim_tree.py`, and imposing the cut on the `mytauType` that it creates, for example, selecting true taus from Drell-Yan events:
```sh
python skim_tree.py --input="/eos/cms/store/group/phys_tau/lrussell/Prod2018_MuTau/DY.txt" --output="/eos/cms/store/group/phys_tau/lrussell/Prod2018_MuTau/DY_taus_skimmed.root" --tree="taus" --sel="tau_byDeepTau2017v2p1VSjetraw>0.90 && tau_pt>30 && std::abs(tau_eta)<2.3 && tau_decayModeFindingNewDMs && (tau_decayMode==0 || tau_decayMode==1 || tau_decayMode==10 || tau_decayMode==11) && tau_byDeepTau2017v2p1VSeraw>0.1686942 && tau_byDeepTau2017v2p1VSmuraw>0.8754835 && tau_index>=0 && mytauType==2 && has_extramuon==0 && has_extraelectron==0 && has_dimuon==0" --processing-module="recompute_tautype.py:compute" 
```

### Reweighting 

For MC events where the $p_T$ spectrum is very poorly balanced, to avoid inordinately large sample weights, `TauMLTools/Training/python/EventDrop.py` was written to randomly drop events in overpopulated bins to match a target spectrum more closely. This script which uses the `DatasetDrop` structure in `TauMLTools/Training/interface/DataMixer.h`, takes a target histogram from a datacard, and calculates the probability that events in each bin should be kept or dropped. The input file, target file and histogram, number of taus to be processed and save path must be specified. For example:

```sh
python EventDrop.py --input_file="/eos/cms/store/group/phys_tau/TauML/prod_2018_v2/adversarial_datasets/SkimmedTuples/MuTau_2018/DY_taus_skimmed_R6p26.root" --target_file="/home/russell/histograms/datacard_pt_2_inclusive_mt_2018_0p9VSjet.root" --target_histo="mt_inclusive/EMB" --n_tau=20000 --save_path="/home/russell/reweighted/testdrop.root"
```

Straightforward $p_T$ reweighting for each MC event type should then be done separately, by assigning weights to events in each bin to match the counts in target histograms. Weights for each bin should be stored as histograms in a ROOT file for generating tensorflow datasets in later steps.

### Mixing

Once proportions of different MC events are chosen, `TauMLTools/Training/python/EventMix.py` can be used to mix the different event types to form the adversarial control region dataset. Proportions should be specified in `EventMix.py`, and labelling of different contributions (for dataloading and evaluation) can be modified in the `DatasetMix` structure within the data mixer `TauMLTools/Training/interface/DataMixer.h` by modifying dataset IDs. The files are currently set up for the dataset used to reduce discrepancies in the DeepTauVSjet distribution.

Two ROOT files should be generated, one for training/validation and one for evaluation.

## Adversarial Training and Evaluation


### Generating tensorflow dataset

Once the adversarial control region ROOT files are created, they must be converted into tensorflow format using `TauMLTools/Training/python/AdversarialGenerate_tf.py` which runs the DeepTau DataLoader in `AdversarialGenerate` mode. This assigns each tau a label of 1 if data and 0 if MC, as well as the $p_T$ weights calculated previously. The ROOT file containing histograms with weights for each contribution should be specified in `TauMLTools/Training/configs/training_v1.yaml`. The DataLoader maps these weights to the dataset ID specified previously. An example of generating a tf dataset is shown below:
```sh
python AdversarialGenerate_tf.py --n_batches=200 --input_dir="/home/russell/MuTau_dataset/Train_R6p26" --save_path="/home/russell/MuTau_dataset/adv_trainval" --training_cfg="../configs/training_v1.yaml"
```
This should be done for a training/validation and an evaluation dataset.

### Training

Adversarial training should be performed using a default model as a starting point to have a good tau type classification baseline. Training is done using the same script as for default models: `TauMLTools/Training/python/Training_CNN.py`, with the input type in the `TauMLTools/Training/configs/training_v1.yaml` configuration file modified to `Adversarial` so that the network architecture and backpropagation are modified, and the network is trained on a mix of taus from the default and adversarial control region datasets.

The path to the adversarial tensorflow dataset and other adversarial training parameters are specified in the same configuration file, for example:
```sh
adversarial_dataset  : "/eos/cms/store/group/phys_tau/TauML/prod_2018_v2/adversarial_datasets/AdvTraining"
adv_parameter        : [1, 10] # k1, k2 for gradients in common layers
n_adv_tau            : 100 # number of candidates per adversarial batch
adv_learning_rate    : 0.01
use_previous_opt     : False
```


### Evaluation

The adversarial model is evaluated in two steps, predictions are computed on the adversarial tensorflow testing dataset with `TauMLTools/Evaluation/apply_adversarial_training.py`, these predictions are then plotted using the jupyter notebook script `evaluate_adversarial.ipnyb`. Two plot types can be produced, discriminator score distributions (DeepTau VSjet) and the output of the adversarial subnetwork (y$_{adv}$).

An example of running `apply_adversarial_training.py` for a model trained with the adversarial techniques enabled is shown below:

```sh
python apply_adversarial_training.py --expID="e1f3ddb3c4c94128a34a7635c56322eb" --mlpath="/home/russe
ll/AdversarialTauML/TauMLTools/Training/python/2018v1/mlruns/12" --eval_ds="/home/russell/tfdata/AdvEval"
```
The experiment ID, path to the mlflow experiment, and a dataset to evaluate on must be specified. The predictions are stored in the model artifacts (`artifacts/predictions/adversarial_predictions.csv`) and contain the DeepTau VSjet scores, truth (data/MC), weights, and adversarial output.

It is useful to compare the discriminator score distributions before and after adversarial training, however, since default models (before adversarial training) are not trained with the adversarial subnetwork, so only the classification predictions are available. The discriminator scores can still be calculated and stored (along with data/MC truth and weights) for taus from the adversarial evaluation dataset by specifying `--not_adversarial` when calling `apply_adversarial_training.py`:

```sh
python apply_adversarial_training.py --expID="5371f5d7080846c1a213f0e648471c11" --mlpath="/home/russell/AdversarialTauML/TauMLTools/Training/python/2018v1/mlruns/12" --eval_ds="/home/russell/tfdata/AdvEval" --not_adversarial
```

The script `evaluate_adversarial.ipynb` plots the DeepTau VSjet distributions of the MC contributions and data and saves them in the model artifacts `/artifacts/plots/adversarial_Djet.pdf`. Similarly $y_{adv}$ distributions for adversarial models are saved as "/artifacts/plots/adversarial_yadv.pdf". The notebook contains examples for an adversarial trained model as well as the default model that it started from.

The classification performance (ROC curves) should be evaluated using the methods described in the main README file for this repository.