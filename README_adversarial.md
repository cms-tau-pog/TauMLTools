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
