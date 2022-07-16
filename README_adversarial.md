# Adversarial Machine Learning Documentation

## Introduction

This file contains instructions for reproducing the adversarial domain adaptation techniques implemented in DeepTau v2p5 to reduce data/MC discrepancies in the DeepTau VSjet discriminator score distribution. The steps necessary for adversarial dataset production, training and evaluation of the DeepTau VSjet score distributions are outlined.




## Adversarial Dataset Production

### Production

- Briefly describe the selector (point to it) and what files were generated, 


### Selection 


- Describe additional selection (skim tree)


### Reweighting 

- Event dropping:

```sh
python EventDrop.py --input_file="/home/russell/skimmed_tuples/MuTau_prod2018/DY_taus_skimmed_R6p26.root" --target_file="/home/russell/histograms/datacard_pt_2_inclusive_mt_2018_0p9VSjet.root" --target_histo="mt_inclusive/EMB" --n_tau=200 --save_path="/home/russell/testingPR/testdrop.root"
```

- pT reweighting

### Mixing

Mixer and proportions

## Adversarial Training and Evaluation

### Generating tensorflow dataset

- DataLoader file to run

### Training

- Load previous model etc
- Parameters that can be varied


### Evaluation

Similarly to default training, the adversarial model is evaluated in two steps, predictions are computed with `apply_adversarial_training.py` found in `TauMLTools\Evaluation`, these predictions are then plotted using the jupyter notebook script `evaluate_adversarial.ipnyb`. Two plot types can be produced, discriminator score distributions (DeepTau VSjet) and y$_{adv}$ the output of the adversarial subnetwork.

An example of running `apple_adversarial_training.py` for a model trained with the adversarial techniques enabled is shown below:

```sh
python apply_adversarial_training.py --expID="e1f3ddb3c4c94128a34a7635c56322eb" --mlpath="/home/russe
ll/AdversarialTauML/TauMLTools/Training/python/2018v1/mlruns/12" --eval_ds="/home/russell/tfdata/AdvEval"
```
The experiment ID, path to the mlflow experiment, and a dataset to evaluate on must be specified. The predictions are stored in the model `artifacts/predictions/adversarial_evaluation.csv` and contain the DeepTau VSjet scores, truth (data/MC), weights, and adversarial output.

It is useful to compare the discriminator score distributions before and after adversarial training, however, since these models were not trained with the adversarial subnetwork, only the classification predictions are available. The discriminator scores can be calculated and stored (along with data/MC truth and weights) for taus from the adversarial dataset by specifyin `--not_adversarial` when calling `apply_adversarial_training.py`:

```sh
python apply_adversarial_training.py --expID="5371f5d7080846c1a213f0e648471c11" --mlpath="/home/russell/AdversarialTauML/TauMLTools/Training/python/2018v1/mlruns/12" --eval_ds="/home/russell/tfdata/AdvEval" --not_adversarial
```