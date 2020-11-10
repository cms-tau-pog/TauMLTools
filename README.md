# TauMLTools

## Introduction

Tools to perform machine learning studies for tau lepton reconstruction and identification at CMS.

## How to install

### Setup for pre-processing
Root-tuples production steps (both big-tuple and training tuple) require CMSSW environment
```sh
export SCRAM_ARCH=slc7_amd64_gcc700
cmsrel CMSSW_10_6_13
cd CMSSW_10_6_13/src
cmsenv
git clone -o cms-tau-pog git@github.com:cms-tau-pog/TauMLTools.git
scram b -j8
```

### Setup for training and testing
The following steps (training, testing, etc.) should be run using LCG or conda-based environment setup.

#### LCG environment
On sites where LCG software distribution is available (e.g. lxplus) it is enough to source `setup.sh`:
```sh
source /cvmfs/sft.cern.ch/lcg/views/LCG_97apython3/x86_64-centos7-clang10-opt/setup.sh
```
Currently supported LCG distribution version is LCG_97apython3.

#### conda environment
In cases when LCG software is not available, we recommend to setup a dedicated conda environment:
```sh
conda create --name tau-ml python=3.7
conda install tensorflow==1.14 pandas scikit-learn matplotlib statsmodels scipy pytables root==6.20.6 uproot lz4 xxhash cython
```

To activate it use `conda activate tau-ml`.

N.B. All of the packages are available through the conda-forge channel, which should be added to conda configuration before creating the environment:
```sh
conda config --add channels conda-forge
```

## How to produce inputs

Steps below describe how to process input datasets starting from MiniAOD up to the representation that can be directly used as an input for the training.

### Big root-tuples production

The big root-tuples, `TauTuple`, contain an extensive information about reconstructed tau, seeding jet, and all reco objects within tau signal and isolation cones: PF candidates, pat::Electrons and pat::Muons, and tracks.
`TauTuple` also contain the MC truth that can be used to identify the tau decay.
Each `TauTuple` entry corresponds to a single tau candidate.
Branches available within `TauTuple` are defined in [Analysis/interface/TauTuple.h](https://github.com/cms-tau-pog/TauMLTools/blob/master/Analysis/interface/TauTuple.h)
These branches are filled in CMSSW module [Production/plugins/TauTupleProducer.cc](https://github.com/cms-tau-pog/TauMLTools/blob/master/Production/plugins/TauTupleProducer.cc) that converts information stored in MiniAOD events into `TauTuple` format.

[Production/python/Production.py](https://github.com/cms-tau-pog/TauMLTools/blob/master/Production/python/Production.py) contains the configuration that allows to run `TauTupleProducer` with `cmsRun`.
Here is an example how to run `TauTupleProducer` on 1000 DY events using one MiniAOD file as an input:
```sh
cmsRun TauMLTools/Production/python/Production.py sampleType=MC_18 inputFiles=/store/mc/RunIIAutumn18MiniAOD/DYJetsToLL_M-50_TuneCP5_13TeV-madgraphMLM-pythia8/MINIAODSIM/102X_upgrade2018_realistic_v15-v1/00000/A788C40A-03F7-4547-B5BA-C1E01CEBB8D8.root maxEvents=1000 rerunTauReco=True
```

In order to run a large-scale production for the entire datasets, the CMS computing grid should be used via CRAB interface.
Submission and status control can be performed using [crab_submit.py](https://github.com/cms-tau-pog/TauMLTools/blob/master/Production/scripts/crab_submit.py) and [crab_cmd.py](https://github.com/cms-tau-pog/TauMLTools/blob/master/Production/scripts/crab_cmd.py) commands.

#### 2018 root-tuple production steps
1. Go to `$CMSSW_BASE/src` and load CMS and environments:
   ```sh
   cd $CMSSW_BASE/src
   cmsenv
   source /cvmfs/cms.cern.ch/common/crab-setup.sh
   ```
1. Enable VOMS proxy:
   ```sh
   voms-proxy-init -rfc -voms cms -valid 192:00
   export X509_USER_PROXY=`voms-proxy-info -path`
   ```
1. Submit task in a config file (or a set of config files) using `crab_submit.py`:
   ```sh
   crab_submit.py --workArea work-area --cfg TauMLTools/Production/python/Production.py --site T2_CH_CERN --output /store/group/phys_tau/TauML/prod_2018_v1/crab_output TauMLTools/Production/crab/configs/2018/CONFIG1.txt TauMLTools/Production/crab/configs/2018/CONFIG2.txt ...
   ```
   * For more command line options use `crab_submit.py --help`.
   * For big dataset file-based splitting should be used
   * For Embedded samples a custom DBS should be specified
1. Regularly check task status using `crab_cmd.py`:
   ```sh
   crab_cmd.py --workArea work-area --cmd status
   ```
1. Once all jobs within a given task are finished you can move it from `work-area` to `finished` folder (to avoid rerunning status each time) and set `done` for the dataset in the production google-doc.
1. If some jobs are failed: try to understand the reason and use standard crab tools to solve the problem (e.g. `crab resubmit` with additional arguments). In very problematic cases a recovery task could be created.
1. Once production is over, all produced `TauTuples` will be moved in `/eos/cms/store/group/phys_tau/TauML/prod_2018_v1/full_tuples`.



### Big tuples splitting

In order to be able to uniformly mix tau candidates from various datasets with various pt, eta and ground truth, the big tuples should be splitted into binned files.

For each input dataset run [CreateBinnedTuples](https://github.com/cms-tau-pog/TauMLTools/blob/master/Analysis/bin/CreateBinnedTuples.cxx):
```sh
DS=DY; CreateBinnedTuples --output tuples-v2/$DS --input-dir raw-tuples-v2/crab_$DS --n-threads 12 --pt-bins "20, 25, 30, 35, 40, 45, 50, 55, 60, 70, 80, 90, 100, 120, 140, 160, 180, 200, 225, 250, 275, 300, 350, 400, 450, 500, 600, 700, 800, 900, 1000" --eta-bins "0., 0.575, 1.15, 1.725, 2.3"
```

Alternatively, you can use python script that iterates over all datasets in the directory:
```sh
python -u TauMLTools/Analysis/python/CreateBinnedTuples.py --input raw-tuples-v2 --output tuples-v2 --n-threads 12
```

Once all datasets are split, the raw files are no longer needed and can be removed to save the disc space.

In the root directory (e.g. `tuples-v2`) txt file with number of tau candidates in each bin should be created:
```sh
python -u TauMLTools/Analysis/python/CreateTupleSizeList.py --input /data/tau-ml/tuples-v2/ > /data/tau-ml/tuples-v2/size_list.txt
```

In case you need to update `size_list.txt` (for example you added new dataset), you can specify `--prev-output` to retrieve bin statistics from the previous run of `CreateTupleSizeList.py`:
```sh
python -u TauMLTools/Analysis/python/CreateTupleSizeList.py --input /data/tau-ml/tuples-v2/ --prev-output /data/tau-ml/tuples-v2/size_list.txt > size_list_new.txt
mv size_list_new.txt /data/tau-ml/tuples-v2/size_list.txt
```

Each split root file represents a signle bin.
The file naming convention is the following: *tauType*\_pt\_*minPtValue*\_eta\_*minEtaValue*.root, where *tauType* is { e, mu, tau, jet }, *minPtValue* and *minEtaValue* are lower pt and eta edges of the bin.
Please, note that the implementation of the following steps requires that this naming convention is respected, and the code will not function properly otherwise.

### Shuffle and merge

The goal of this step create input root file that contains uniform contribution from all input dataset, tau types, tau pt and tau eta bins in any interval of entries.
It is needed in order to have a smooth gradient descent during the training.

Steps described below are defined in [TauMLTools/Analysis/scripts/prepare_TauTuples.sh](https://github.com/cms-tau-pog/TauMLTools/blob/master/Analysis/scripts/prepare_TauTuples.sh).


#### Training inputs

Due to considerable number of bins (and therefore input files) merging is split into two steps.
1. Inputs from various datasets are merged together for each tau type. How the merging process should proceed is defined in `TauMLTools/Analysis/config/training_inputs_{e,mu,tau,jet}.cfg`.
   For each configuration [ShuffleMerge](https://github.com/cms-tau-pog/TauMLTools/blob/master/Analysis/bin/ShuffleMerge.cxx) is executed. E.g.:
   ```sh
   ShuffleMerge --cfg TauMLTools/Analysis/config/training_inputs_tau.cfg --input tuples-v2 \
                --output output/training_preparation/all/tau_pt_20_eta_0.000.root \
                --pt-bins "20, 25, 30, 35, 40, 45, 50, 55, 60, 70, 80, 90, 100, 120, 140, 160, 180, 200, 225, 250, 275, 300, 350, 400, 450, 500, 600, 700, 800, 900, 1000" \
                --eta-bins "0., 0.575, 1.15, 1.725, 2.3" --mode MergeAll --calc-weights true --ensure-uniformity true \
                --max-bin-occupancy 500000 --n-threads 12  --disabled-branches "trainingWeight"
   ```
   Once finished, use `CreateTupleSizeList.py` to create `size_list.txt` in `output/training_preparation`.
1. Merge all 4 tau types together.
   ```sh
   ShuffleMerge --cfg TauMLTools/Analysis/config/training_inputs_step2.cfg --input output/training_preparation \
                --output output/training_preparation/training_tauTuple.root --pt-bins "20, 1000" --eta-bins "0., 2.3" --mode MergeAll \
                --calc-weights false --ensure-uniformity true --max-bin-occupancy 100000000 --n-threads 12
   ```

#### Testing inputs

For testing inputs there is no need to merge different datasets and tau types. Therefore, merging can be done in a single step.
```sh
ShuffleMerge --cfg TauML/Analysis/config/testing_inputs.cfg --input tuples-v2 --output output/training_preparation/testing \
             --pt-bins "20, 25, 30, 35, 40, 45, 50, 55, 60, 70, 80, 90, 100, 120, 140, 160, 180, 200, 225, 250, 275, 300, 350, 400, 450, 500, 600, 700, 800, 900, 1000" \
             --eta-bins "0., 0.575, 1.15, 1.725, 2.3" --mode MergePerEntry \
             --calc-weights false --ensure-uniformity false --max-bin-occupancy 20000 \
             --n-threads 12 --disabled-branches "trainingWeight"
```

#### Validation
A validation can be run on shuffled samples to ensure that different parts of the training set have compatible distributions.
To run the validation tool, a ROOT version greater or equal to 6.16 is needed:
```
source /cvmfs/sft.cern.ch/lcg/views/LCG_97apython3/x86_64-centos7-clang10-opt/setup.sh
```
Then, run:
```
python TauMLTools/Production/scripts/validation_tool.py  --input input_directory \
                                                         --id_json /path/to/dataset_id_json_file \
                                                         --group_id_json /path/to/dataset_group_id_json_file \
                                                         --output output_directory \
                                                         --n_threads n_threads \
                                                         --legend > results.txt
```
The *id_json*  (*group_id_json*) points to a json file containing the list of datasets names (dataset group names) and their hash values, used to identify them inside the shuffled ROOT tuples. These files are needed in order to create a unique identifier which can be handled by ROOT. These files are produced at the shuffle and merge step. 
The script will create the directory "output_directory" containing the results of the test.
Validation is run on the following ditributions with a Kolmogorov-Smirnov test:

- dataset_id, dataset_group_id, lepton_gen_match, sampleType 
- tau_pt and tau_eta for each bin of the previous
- dataset_id for each bin of dataset_group_id

If a KS test is not successful, a warning message is print on screen.

Optional arguments are available running:
```
python TauMLTools/Production/scripts/validation_tool.py --help
```

A time benchmark is available [here](https://github.com/cms-tau-pog/TauMLTools/pull/31#issue-510206277).

### Production of flat inputs

In this stage, `TauTuple`s are transformed into flat [TrainingTuples](https://github.com/cms-tau-pog/TauMLTools/blob/master/Analysis/interface/TrainingTuple.h) that are suitable as an input for the training.

The script to that performs conversion is defined in [TauMLTools/Analysis/scripts/prepare_TrainingTuples.sh](https://github.com/cms-tau-pog/TauMLTools/blob/master/Analysis/scripts/prepare_TrainingTuples.sh).

1. Run [TrainingTupleProducer](https://github.com/cms-tau-pog/TauMLTools/blob/master/Analysis/bin/TrainingTupleProducer.cxx) to produce root files with the flat tuples. In `prepare_TrainingTuples.sh` several instances of `TrainingTupleProducer` are run in parallel specifying `--start-entry` and `--end-entry`.
   ```sh
   TrainingTupleProducer --input /data/tau-ml/tuples-v2-training-v2/training_tauTuple.root --output output/tuples-v2-training-v2-t1-root/training-root/part_0.root
   ```
1. Convert root files into HDF5 format:
   ```sh
   python TauMLTools/Analysis/python/root_to_hdf.py --input output/tuples-v2-training-v2-t1-root/training-root/part_0.root \
                                                    --output output/tuples-v2-training-v2-t1-root/training/part_0.h5 \
                                                    --trees taus,inner_cells,outer_cells
   ```

## Training NN

1. The code to read the input grids from the file is implemented in cython in order to provide acceptable I/O performance.
   Cython code should be compiled before starting the training. To do that, you should go to `TauMLTools/Training/python/` directory and run
   ```sh
   python _fill_grid_setup.py build
   ```
1. DeepTau v2 training is defined in [TauMLTools/Training/python/2017v2/Training_p6.py](https://github.com/cms-tau-pog/TauMLTools/blob/master/Training/python/2017v2/Training_p6.py). You should modify input path, number of epoch and other parameters according to your needs in [L286](https://github.com/cms-tau-pog/TauMLTools/blob/master/Training/python/2017v2/Training_p6.py#L286) and then run the training in `TauMLTools/Training/python/2017v2` directory:
   ```sh
   python Training_p6.py
   ```
1. Once training is finished, the model can be converted to the constant graph suitable for inference:
   ```sh
   python TauMLTools/Analysis/python/deploy_model.py --input MODEL_FILE.hdf5
   ```

## Testing NN performance

1. Apply training for all testing dataset using [TauMLTools/Training/python/apply_training.py](https://github.com/cms-tau-pog/TauMLTools/blob/master/Training/python/apply_training.py):
   ```sh
   python TauMLTools/Training/python/apply_training.py --input "TUPLES_DIR" --output "PRED_DIR" \
           --model "MODEL_FILE.pb" --chunk-size 1000 --batch-size 100 --max-queue-size 20
   ```
1. Run [TauMLTools/Training/python/evaluate_performance.py](https://github.com/cms-tau-pog/TauMLTools/blob/master/Training/python/evaluate_performance.py) to produce ROC curves for each testing dataset and tau type:
   ```sh
   python TauMLTools/Training/python/evaluate_performance.py --input-taus "INPUT_TRUE_TAUS.h5" \
           --input-other "INPUT_FAKE_TAUS.h5" --other-type FAKE_TAUS_TYPE \
           --deep-results "PRED_DIR" --deep-results-label "RESULTS_LABEL" \
           --prev-deep-results "PREV_PRED_DIR" --prev-deep-results-label "PREV_LABEL" \
           --output "OUTPUT.pdf"
   ```
   Or modify [TauMLTools/Training/scripts/eval_perf.sh](https://github.com/cms-tau-pog/TauMLTools/blob/master/Training/scripts/eval_perf.sh) according to your needs to produce plots for multiple datasets.

#### Examples

Evaluate performance for Run 2:
```sh
python TauMLTools/Training/python/evaluate_performance.py --input-taus /eos/cms/store/group/phys_tau/TauML/DeepTau_v2/tuples-v2-training-v2-t1/testing/tau_HTT.h5 --input-other /eos/cms/store/group/phys_tau/TauML/DeepTau_v2/tuples-v2-training-v2-t1/testing/jet_TT.h5 --other-type jet --deep-results /eos/cms/store/group/phys_tau/TauML/DeepTau_v2/predictions/2017v2p6/step1_e6_noPCA --output output/tau_vs_jet_TT.pdf --draw-wp --setup TauMLTools/Training/python/plot_setups/run2.py
```

Evaluate performance for Phase 2 HLT:
```sh
python TauMLTools/Training/python/evaluate_performance.py --input-taus DeepTauTraining/training-hdf5/even-events/even_pt_20_eta_0.000.h5 --other-type jet --deep-results DeepTauTraining/training-hdf5/even-events-classified-by-DeepTau_odd --output output/tau_vs_jet_Phase2.pdf --setup TauMLTools/Training/python/plot_setups/phase2_hlt.py
```
