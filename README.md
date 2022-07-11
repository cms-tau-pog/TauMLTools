# TauMLTools

## Introduction

Tools to perform machine learning studies for tau lepton reconstruction and identification at CMS.

## How to install

1. Clone package from the github without loading any additional environment (like CMSSW):
   ```sh
   git clone -o cms-tau-pog -b master git@github.com:cms-tau-pog/TauMLTools.git
   ```
2. Go to `TauMLTools` directory and load appropriate environment by running `env.sh`:
   ```sh
   source env.sh ENV_NAME
   ```
   where supported `ENV_NAME` are:
   - `prod2018`: for production of root-tuples for 2018 pre-UL datasets and pre-processing steps up to Shuffle&Merge
   - `prod2018UL`: for production of root-tuples for 2018 UL datasets and pre-processing steps up to Shuffle&Merge
   - `phase2`: for production of root-tuples for Phase 2 datasets and pre-processing steps up to Shuffle&Merge
   - `lcg`: using LCG environment, if it is available (e.g. lxplus)
   - `conda`: using tau-ml conda environment -- this is the recommended environment to perform an actual NN training

   N.B. If you want to use existing `conda` installation, make sure that it is activated and the path to the `conda` executable is included in `PATH` variable. If `conda` installation not found, `env.sh` will make install it from the official web site and configure it to work with the current TauMLTools installation.

The second step (`source env.sh ENV_NAME`) should be repeated each time you open a new shell session.

### Installing law
Some parts of the ntuple production and network training can be split into different jobs and run on the HTCondor cluster. To do so, we use the [law](https://github.com/riga/law) package. Law can be installed via conda's pip (reccommended if running jobs which run in a conda environent) using the [environment yaml file](https://github.com/cms-tau-pog/TauMLTools/blob/master/tau-ml-env.yaml), or via standard pip (recommended when running jobs in the CMSSW environment) running the command
```bash
python -m pip install law
```
If installing with standard pip, be sure that law is added to the PATH variable and that the LAW libraries (e.g. $HOME/.local/lib/pythonX.Y/site-packages/) are added to the PYTHONPATH variable. It may also be needed to change the shebang of the law executable (e.g. e.g. $HOME/.local/bin/law) from **#!/some/path/to/python3** to **#!/usr/bin/env python3** in order to access CMSSW python modules. As an alternative to this last step, one can install law with the command below (not fully tested)
```bash
LAW_INSTALL_EXECUTABLE="/usr/bin/env python3" python3 -m pip install law --user --no-binary :all:
```
which automatically sets the correct shebang.

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
cmsRun TauMLTools/Production/python/Production.py sampleType=MC_18 inputFiles=/store/mc/RunIIAutumn18MiniAOD/DYJetsToLL_M-50_TuneCP5_13TeV-madgraphMLM-pythia8/MINIAODSIM/102X_upgrade2018_realistic_v15-v1/00000/A788C40A-03F7-4547-B5BA-C1E01CEBB8D8.root maxEvents=1000
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
   crab_submit.py --workArea work-area --cfg TauMLTools/Production/python/Production.py --site T2_CH_CERN --output /store/group/phys_tau/TauML/prod_2018_v2/crab_output TauMLTools/Production/crab/configs/2018/CONFIG1.txt TauMLTools/Production/crab/configs/2018/CONFIG2.txt ...
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
1. Once production is over, all produced `TauTuples` will be moved in `/eos/cms/store/group/phys_tau/TauML/prod_2018_v2/full_tuples`.



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
### ShuffleMergeSpectral (Alternative shuffle and merge)

This step represents alternative Shuffle and Merge procedure that aims to minimize usage of physicical memory as well as provide direct control over the shape of final (pt,eta) spectrum in a stochastic manner. Before running the following step on HTCondor for all the samples it is essential to analyze and tune the parameters of ShuffleMergeSpectral.

Spectrum of the initial data is needed to calculate the probability to take tau candidate of some genuine tau-type and from some pt-eta bin. To generate the specturum histograms for a dataset and lists with number of entries per datafile, run:
```sh
CreateSpectralHists --output "spectrum_file.root" \
                    --output_entries "entires_file.txt" \
                    --input-dir "path/to/dataset/dir" \
                    --pt-hist "n_bins_pt, pt_min, pt_max" \
                    --eta-hist "n_bins_eta, |eta|_min, |eta|_max" \
                    --n-threads 1
```
on this step it is important to have high granularity binning to be able later to re-bin into custom, non-uniform pt-eta bins on the next step.

Alternatively, if one wants to process several datasets the following python script can be used (parameters of pt and eta binning to be hardcoded in the script):
```sh
python Analysis/python/CreateSpectralHists.py --input /path/to/input/dir/ \
                                              --output /path/to/output/dir/ \
                                              --filter ".*(DY).*" \
                                              --rewrite
```
After the following step spectrums and .txt files with the number of entries will be created in the output folder. To merge all the .txt files into one and mix the lines:
```
cat <path_to_spectrums>/*.txt | shuf - > filelist_mix.txt
```
Mixing is needed to have shuffled files within one data group. After this step, filelist_mix.txt should contain filenames and the number of entries in the corresponding files:
```
./DYJetsToLL_M-50/eventTuple_1-70.root 249243
./TTJets_ext1/eventTuple_1-308.root 59414
./TTToHadronic/eventTuple_467.root 142781
./WJetsToLNu/eventTuple_25.root 400874
...
```
After spectrums are created for all datasets, the final procedure of Shuffle and Merge can be performed with:
```sh
ShuffleMergeSpectral --cfg Analysis/config/2018/training_inputs_MC.cfg
                     --input filelist_mix.txt
                     --prefix prefix_string
                     --output <path_to_output_file.root>
                     --mode MergeAll
                     --n-threads 1
                     --disabled-branches "trainingWeight, tauType"
                     --input-spec /afs/cern.ch/work/m/myshched/public/root-tuple-v2-hists
                     --pt-bins "20, 30, 40, 50, 60, 70, 80, 90, 100, 1000"
                     --eta-bins "0., 0.6, 1.2, 1.8, 2.4"
                     --tau-ratio "jet:1, e:1, mu:1, tau:1"
                     --lastbin-disbalance 100.0
                     --lastbin-takeall true
                     --refill-spectrum true
                     --enable-emptybin true
                     --job-idx 0 --n-jobs 500
```
- `--cfg` is a configuration file where data groups are constructed, each line represents data group in the format of (name, dataset_dir_regexp, file_regexp, tau_types to consider in this group), e.g: `Analysis/config/2018/training_inputs_MC.cfg`
- `--input` is the file containing the list of input files and entries (read line-by-line). The abdolute path is not taken from this file, only `../dataset/datafile.root n_events` is read. The rest of the path (the path to the dataset folders) should be specified in the `--prefix` argument.
- `--prefix` is the prefix which will be placed before the path of each file read form `--input`. Please note that this prefix will *not* be placed before the `--input-spec` value. This value can include a remote path compatible with xrootd.
- the last pt bin is taken as a high pt region, all entries from it are taken without rejection.
- `--tau-ratio "jet:1, e:1, mu:1, tau:1"` defines proportion of TauTypes in final root-tuple.
- `--refill-spectrum` to recalculated spectrums of the input data on flight, only events and files that correspond to the current job `--job-idx` will be considered. It was studied that in case of heterogeneity within a data group, the common spectrum of the whole data group poorly represents the spectrum of the sub-chunk of this data group if we use `--n-jobs 500`, so it is needed to fill spectrum histograms in the beginning of every job, to obtain required uniformity of the output.
- `--lastbin-takeall` due to the poor statistic in the high-pt region the option to take all candidates from last pt-bin is included (contrary to `--lastbin-takeall false`, in this case we put the requirement on n_entries in the last bin to be equal to n_entries in other bins)
- `--lastbin-disbalance` the argument is relevant in case of `-lastbin-takeall true`, it put the requirement on the ratio of (all entries up to the last bin) to the (number of entries in the last bin) not to be greater than required number, otherwise less events will be taken from all bins up to the last.
- `--enable-emptybin` in case of empty pt-eta bin, the probability in this bin will be set to 0 (that is needed in cases when datasets are statistically poor or the number of jobs `--n-jobs` is big in case of `--refill-spectrum true` mode)
- `--n-jobs 500 --job-idx 0` defines into how many parts to split the input files and the index of corresponding job

In order to find appropriate binning and `--tau-ratio` in correspondence to the present statistics it might be useful to execute one job in `--refill-spectrum false --lastbin-takeall false` mode and study the output of `./out/*.root` files. In the \<DataGroupName>_n.root files the number of entries in required  `--pt-bins --eta-bins` can be found. \<DataGroupName>.root files show the probability of accepting candidate from corresponding pt-eta bin.

#### ShuffleMergeSpectral on HTCondor
ShuffleMergeSpectral can be executed on condor through the [law](https://github.com/riga/law) package. To run it, first install law following the instructions in the first section. Then, set up the environment
```sh
cd $CMSSW_BASE/src
cmsenv
cd TauMLTools/Analysis/law
source setup.sh
law index
```
Jobs can be submitted running
```
law run ShuffleMergeSpectral --version vx --params --n-jobs N
```
where *--params* are the [ShuffleMergeSpectral](https://github.com/cms-tau-pog/TauMLTools/blob/master/Analysis/bin/ShuffleMergeSpectral.cxx#L28-L52) parameters. In particular

   - *--input* has been renamed to *--input-path*
   - *--output* has been renamed to *--output-path*

**NOTA BENE**: *--output-path* should be a full path, otherwise the output will be lost
The full list of parameters accepted by law and ShuffleMergeSpectral can be printed with the commands

```
ShuffleMergeSpectral --help
law run ShuffleMergeSpectral --help
```

Jobs are created by the script using the *--start-entry* and *--end-entry* parameters.

Additional arguments can be used to control the condor submission:

   - *--workflow local* will run locally. If omitted, condor will be used
   - *--max-runtime* condor runtime in hours
   - *--max-memory* condor RAM request in MB
   - *--batch-name* batch name to be used on condor. Default is "TauML_law"

At this point, the worker will start reporting the job status. As long as the worker is alive, it will automatically resubmit failed jobs. The worker can be killed with **ctrl+C** once all jobs have been submitted. Failed jobs can be resubmitted running the same command used in the first submission (from the same working directory).

A *data* directory is created. This directory contains information about the jobs as well as the log, output and erorr files created by condor.

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

#### Shuffled ntuples spectrum
For the following training step, the pT-eta spectrum of these input events is needed and can be computed with the command:
```bash
CreateSpectralHists \
   --outputfile output_spectrum_file.root \
   --output_entries entries_per_file.txt  \
   --input-dir /path/to/shuffle/and/merge/files
   --pt-hist "980, 20, 1000"
   --eta-hist "250, 0.0, 2.5"
   --file-name-pattern ".*2_rerun\/Shuffle.*.root"
   --n-threads 1
```
where:
   - *--outputfile* will store the pT-eta spectrum needed for the training
   - *--output_entries* will store the list of the shuffle-and-merge files and the number of entries per file
   - *--input-dir* is the location of the shuffle-and-merge files
   - *--pt-hist* and *--eta-hist* are the binnings used for the pT and eta variables
   - *--file-name-pattern* is the regex pattern used to match the input file names

The path to the output file has to be specified in the yaml configuration file described in the following sections.

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

### Feature scaling
#### Computation
During the loading of data into the model, values for each feature in the input grid tensor will undergo a sequential transformation with a following order: subtract `mean`, divide by `std`, clamp to a range `[lim_min, lim_max]` (see [`Scale()`](https://github.com/cms-tau-pog/TauMLTools/blob/45babb742f1a15f96950e84d60c95acc9b65f7e9/Training/interface/DataLoader_main.h#L314) function in `DataLoader_main.h`). Here, `mean`, `std`, `lim_min`, `lim_max` are parameters unique for each input feature and they have to be precomputed for a given input data set prior to running the training. This can be done with a script [`Training/python/feature_scaling.py`](https://github.com/cms-tau-pog/TauMLTools/blob/master/Training/python/feature_scaling.py), for example, as follows:
```python
python feature_scaling.py --cfg ../configs/training_v1.yaml --var_types TauFlat
```     
- `--cfg` (str, required) is a relative path to a main yaml configuration file used for the training
- `--var_types` (list of str, optional, default: -1) is a list of variable types to be run computation on. Should be the ones from field `Features_all` in the main training cfg file.

The scaling procedure is further configured in the dedicated `Scaling_setup` field of the training yaml cfg file (`../configs/training_v1.yaml` in the example above). There, one needs to specify the following parameters:
- `file_path`, path to input ROOT files (e.g. after Shuffle & Merge step) which are used for the training
- `output_json_folder`, path to an output directory for storing json files with scaling coefficients
- `file_range`, list indicating the range of input files to be used in the computation, -1 to run on all the files in `file_path`
- `tree_name`, name of TTree with data branches inside of input ROOT files
- `log_step`, number of files after which make a log of currently computed scaling params
- `version`, string added as a postfix to the output file names

Then, there are `cone_definition` and `cone_selection` fields which define the configuration for cone splitting. Scaling parameters are computed separately for constituents in the inner cone of the tau candidate (`constituent_dR <= dR_signal_cone`) and in the outer (`(constituent_dR > dR_tau_signal_cone) & (constituent_dR < dR_tau_outer_cone)`). Therefore, in `cone_definition` one should define the inner/outer cone dimensions and in `cone_selection` variable names (per variable type) in input `TTree` to be used to compute dR. Also `cone_types` field allows to specify the cones per variable type for which the script should compute the scaling parameters.

When it comes to variables, scaling module shares the list of ones to be used with the main training module via `Features_all` field of `configs/training_v1.yaml` cfg file. Under this field, each variable type (TauFlat, etc.) stores a list of corresponding variables. Each entry in the list is a dictionary, where the key is the variable name, and the list has the format `(selection_cut, aliases, scaling_type, *lim_params)`:
- `selection_cut`, string, cuts to be applied on-the-fly by `uproot` when loading (done on feature-by-feature basis) data into `awkward` array. Since `yaml` format allows for the usage of aliases, there are a few of those defined in the `selection` field of the cfg file.
- `aliases`, dictionary, definitions of variables not present in original file but needed for e.g. applying selection. Added on-the-fly by `uproot` when loading data.
- `scaling_type`, string, one of `["no scaling", "categorical", "linear", "normal"]`, see below for their definition.
- `lim_params`, if passed, should be either a list of two numbers or a dictionary. Depending on the scaling type (see below), specifies the range to which each variable should be clamped.

**NB:** Please note, that as of now selection criteria `selection_cut` are used **only** by the scaling module and are a duplicate of those specified internally inside of the `DataLoader` class. One needs to **ensure that those two are in synch with each other**.

The following scaling types are supported:
- for `no scaling` and `categorical` cases it automatically fills in the output json file with `mean=0`, `std=1`, `lim_min=-inf`, `lim_max=inf` (no linear transformation and no clamping). Both cases are treated similarly and "categorical" label is introduced only to distinguish those variables in the cfg file for a possible dedicated preprocessing in the future.
- `linear` case assumes the variable needs to be first clamped to a specified range, and then linearly transformed to the range `[-1,1]`. Here one should pass as `lim_params` in the cfg exactly those range, for which they would want to clamp the data *first* (recall that in `DataLoader` it is done as the last step). The corresponding `mean`, `std`, `lim_min`/`lim_max` to meet the DataLoader notation are derived and filled in the output json automatically in the function `init_dictionaries()` of `scaling_utils.py`
- `normal` case assumes the variable to be standardised by `mean` (subtract) and `std` (divide) and clamped to a specified range (in number of sigmas). Here one should pass `lim_params` as the number of sigmas to which they want to clamp the data after being scaled by mean and std (e.g. [-5, 5]). It is also possible to skip `lim_params` argument. In that case, it is automatically filled `lim_min=-inf`, `lim_max=inf`

Also note that `lim_params` (for both linear and normal cases) can be a dictionary with keys "inner" and/or "outer" and values as lists of two elements as before. In that case `lim_min`/`lim_max` will be derived separately for each specified cone.

As one may notice, it is "normal" features which require actual computation, since for other types scaling parameters can be derived easily based on specified `lim_params`. The computation of means and stds in that case is performed in an iterative manner, where the input files are opened one after another and for each variable the sum of its values, squared sum of values and counts of entries are being aggregated as the files are being read. Then, every `log_step` number of files, means/stds are computed based on so far aggregated sums and counts and together with other scaling parameters are logged into a json file. This cumulative nature of the approach also allows for a more flexible scan of the data for its validation (e.g. by comparison of aggregated statistics, not necessarily mean/std across file ranges). Moreover, for every file every variable's quantiles are stored, allowing for a validation of the scaling procedure (see section below).

The result of running the scaling script will be a set of log json files further referred to as *snapshots* (e.g. `scaling_params_v*_log_i.json`), where each file corresponds to computation of mean/std/lim_min/lim_max *after* having accumulated sums/sums2/counts for `i*log_step` files; the json file (`scaling_params_v*.json`) which corresponds to processing of all given files; json file storing variables' quantiles per file (`quantile_params_v*.json`). `scaling_params_v*.json` should be further provided to `DataLoader` in the training step to perform the scaling of inputs.  

#### Validation
Since the feature scaling computation follows a cumulative approach, one can be interested to see how the estimates of mean/std are converging to stable values from one snapshot to another as more data is being added. The convergence of the approach can be validated with [`Training/python/plot_scaling_convergence.py`](https://github.com/cms-tau-pog/TauMLTools/blob/master/Training/python/plot_scaling_convergence.py), e.g. for TauFlat variable type:
```python
python plot_scaling_convergence.py --train-cfg ../configs/training_v1.yaml -p /afs/cern.ch/work/o/ofilatov/public/scaling_v2/json/all_types_10files_log1 -pr scaling_params_v2 -n 1 -t -1 -c global -c inner -c outer -o scaling_plots
```
- `--train-cfg`, path to yaml configuration file used for training
- `-p/--path-to-snapshots`, path to the directory with scaling json snapshot files
- `-pr/--snapshot-prefix`, prefix used in the name of json files to tag a scaling version
- `-n/--nfiles-step`, number of processed files per log step (`log_step` parameter), as it was set while running the scaling computation
- `-t/--var-types`, variable types for which scaling parameters are to be plotted, -1 to run on all of them
- `-c/--cone-types`, cone types for which scaling parameters are to be plotted
- `-o/--output-folder`, output directory to save plots

This will produce a series of plots which combine all the variables of the given type into a *boxplot representation* and show its evolution as an ensemble throughout the scaling computation. Please have a look at [this presentation](https://indico.cern.ch/event/1038235/contributions/4359970/attachments/2242374/3803434/scaling_update_11May.pdf) for more details on the method.

Furthermore, once the scaling parameters are computed, one might be interested to see how the computed clamped range for a given feature relates to the actual quantiles of the feature distribution. Indeed, clamping might significantly distort the distribution if the derived (in "normal" case) or specified (in "linear" case) clamped range is not "wide enough", i.e. doesn't cover the most bulk of the distribution. The comparison of clamped and quantile ranges can be performed with [`Training/python/plot_quantile_ranges.py`](https://github.com/cms-tau-pog/TauMLTools/blob/master/Training/python/plot_quantile_ranges.py):
```python
python plot_quantile_ranges.py --train-cfg ../configs/training_v1.yaml --scaling-file /afs/cern.ch/work/o/ofilatov/public/scaling_v2/scaling_params_v2.json --quantile-file /afs/cern.ch/work/o/ofilatov/public/scaling_v2/quantile_params_v2.json --file-id 0 --output-folder quantile_plots/fid_0 --only-suspicious False
```
- `--train-cfg`, path to yaml configuration file used for training
- `--scaling-file`, path to json file with scaling parameters
- `--quantile-file`, path to json file with variables' quantiles
- `--file-id`, id (integer from `range(len(input_files))`) of the input ROOT file. Since quantile range of variables are derived per input file, will use for plotting those of `file-id`-th file.
- `--output-folder`, output directory to save plots
- `--only-suspicious`, whether to save only suspicious plots. Please have a look at the suspiciousness criteria in `plot_quantile_ranges.py`

This will produce a plot of the clamped range plotted side-by-side with quantile ranges for the corresponding feature's distribution, so that it's possible to compare if the former overshoots/undershoots the actual distribution. However, please note that this representation is dependant on the input file (`--file-id` argument), so it is advisable to check whether quantiles are robust to a file change. For more details please have a look at [this presentation](https://indico.cern.ch/event/1044740/contributions/4389426/attachments/2255446/3827568/scaling_update_1June.pdf) of the method.

#### Running on HTCondor
As an alternative to the interactive mode described above, the scaling parameter computation can be distributed to multiple jobs and run on HTCondor. The code run on HTCondor is the same that would be run interactivelly.
To run with law, first activate the right conda environment (note: it's recommended to have law installed via conda's pip), then:
```bash
cd TauMLTools/Analysis/law
source setup.sh
law index
```
then run the job submission
```bash
law run FeatureScaling --version version_tag --environment conda --cfg /path/to/yaml/cfg.yaml --output-path /path/to/dir/ --file-per-job M --n-jobs N
```
where ```--version``` specifies the job submission version (used by law to monitor the status of the jobs), ```--environment``` specifies the shell environment to be used (*conda* in this case, see [TauMLTools/Analysis/law/bootstrap.sh](https://github.com/cms-tau-pog/TauMLTools/blob/master/Analysis/law/bootstrap.sh)), ```--cfg``` is the yaml configuration file (the same used in interactive mode), ```--files-per-job``` specifies the number of files processed by each job, ```--n-jobs``` specifies the number of jobs to run, and ```--output-path``` specifies the path to the output directory, used to save the results. 
The law task run by the code above (stored in [TauMLTools/Analysis/law/FeatureScaling/task.py](https://github.com/cms-tau-pog/TauMLTools/blob/master/Analysis/law/FeatureScaling/tasks.py)) implements the same script used locally, with the following caveat:

- the number of files per job is specified by the ```--files-per-job``` argument
- the total number of jobs is defined by the ```--n-jobs``` argument, which, together with the ```--files-per-job``` argument, determines the total number of files read from the input path. Leave this to the default value (0) to run on all the input files with ```--files-per-job``` files per job
- the yaml configuration parameters ```Scaling_setup/file_range``` and ```Scaling_setup/log_step``` are ignored when running on law
- the output directory is specified using an argument, so the configuration in the .yaml file if ignored.    

The results from each single job can be merged together to obtain the final set of scaling parameters, using the interactive python script [TauMLTools/Training/python/merge_feature_scaling_jobs.py](https://github.com/cms-tau-pog/TauMLTools/blob/master/Training/python/merge_feature_scaling_jobs.py), as follows:
```bash
python merge_feature_scaling_jobs.py --output /path/to/dir --input "job*/scaling_params_v5.json"
```
where ```--output``` determines the output directory which stores the merged results and ```--input``` is a string pointing to the results of the single jobs (using a glob pattern, as in the example above. **NOTE** use the quotation marks!).  
In order to create convergence plots using the [`Training/python/plot_scaling_convergence.py`](https://github.com/cms-tau-pog/TauMLTools/blob/master/Training/python/plot_scaling_convergence.py) script described in the main section above, *merge_feature_scaling_jobs.py* accepts the ```--step``` *int* argument. If this value is specified, the script will create intermediate output files merging an increasing number of jobs with step equal to ```--step``` (e.g., if ```--step``` is equal to 10, the first intermediate output file will merge 10 jobs, the second 20, the third 30 and so on). 

### Single run
In order to have organized storage of models and its associated files, [mlflow](https://mlflow.org/docs/latest/index.html) is used from the training step onwards. At the training step, it takes care of logging necessary configuration parameters used to run the given training, plus additional artifacts, i.e. associated files like model/cfg files or output logs). Conceptually, mlflow augments the training code with additional logging of requested parameters/files whenever it is requested. Please note that hereafter mlflow notions of __run__ (a single training) and __experiment__ (a group of runs) will be used. 

The script to perform the training is [TauMLTools/Training/python/2018v1/Training_v0p1.py](https://github.com/cms-tau-pog/TauMLTools/blob/master/Training/python/2018v1/Training_v0p1.py). It is supplied with a corresponding [TauMLTools/Training/python/2018v1/train.yaml](https://github.com/cms-tau-pog/TauMLTools/blob/master/Training/python/2018v1/train.yaml) configuration file which specifies input configuration parameters. Here, [hydra](https://hydra.cc/docs/intro/) is used to compose the final configuration given the specification inside of `train.yaml` and, optionally, from a command line (see below), which is then passed as [OmegaConf](https://omegaconf.readthedocs.io/) dictionary to `main()` function. 

The specification consists of:
* the main training configuration file [`TauMLTools/Training/configs/training_v1.yaml`](https://github.com/cms-tau-pog/TauMLTools/blob/master/Training/configs/training_v1.yaml) describing `DataLoader` and model configurations, which is "imported" by hydra as a [defaults list](https://hydra.cc/docs/advanced/defaults_list/)
* mlflow parameters, specifically `path_to_mlflow` (to the folder storing mlflow runs, usually its default name is `mlruns`) and `experiment_name`.  These two parameters are needed to either create an mlflow experiment with a given `experiment_name` (if it doesn't exist) or to find the experiment to which attribute the current run
* `scaling_cfg` which specifies the path to the json file with feature scaling parameters
* `gpu_cfg` which sets the gpu related parameters 
* `log_suffix` used as a postfix in output directory names

Also note that one can conveniently [add/remove/override](https://hydra.cc/docs/advanced/override_grammar/basic/) __any__ configuration items from the command line. Otherwise, running `python Training_v0p1.py` will use default values as they are composed by hydra based on `train.yaml`.

An example of running the training would be:
```sh
python Training_v0p1.py experiment_name=run3_cnn_ho2 'training_cfg.SetupNN.tau_net={ activation: "PReLU", dropout_rate: 0.2, reduction_rate: 1.4, first_layer_width: "2*n*(1+drop)", last_layer_width: "n*(1+drop)" }' 'training_cfg.SetupNN.comp_net={ activation: "PReLU", dropout_rate: 0.2, reduction_rate: 1.6, first_layer_width: "2*n*(1+drop)", last_layer_width: "n*(1+drop)" }' 'training_cfg.SetupNN.comp_merge_net={ activation: "PReLU", dropout_rate: 0.2, reduction_rate: 1.6, first_layer_width: "n", last_layer_width: 64 }' 'training_cfg.SetupNN.conv_2d_net={ activation: "PReLU", dropout_rate: 0.2, reduction_rate: null, window_size: 3 }' 'training_cfg.SetupNN.dense_net={ activation: "PReLU", dropout_rate: 0.2, reduction_rate: 1, first_layer_width: 200, last_layer_width: 200, min_n_layers: 4 }'
```

Here, a new mlflow run will be created under the "run3_cnn_ho2" experiment (if the experiment doesn't exist, it will be created). Then, the model is composed and compiled and the training proceeds via a usual `fit()` method with `DataLoader` class instance used as a batch yielder. Several callbacks are also implemented to monitor the training process, specifically `CSVLogger`, `TimeCheckpoint` and `TensorBoard` callback. Lastly, note that the parameters related to the NN setup and initially specified in `training_v1.yaml` are overriden via the command line (`training_cfg.SetupNN.{param}=...`)

Furthermore, for the sake of convenience, submission of multiple trainings in parallel to the batch system is implemented as a dedicated law task. As an example, running the following commands will set up law and submit the trainings specified in `TauMLTools/Training/configs/input_run3_cnn_ho1.txt` to `htcondor`: 

```sh
source setup.sh
law index
law run Training --version input_run3_cnn_ho1 --workflow htcondor --working-dir /nfs/dust/cms/user/mykytaua/softDeepTau/DeepTau_master_hyperparam-law/TauMLTools/Training/python/2018v1 --input-cmds /nfs/dust/cms/user/mykytaua/softDeepTau/DeepTau_master_hyperparam-law/TauMLTools/Training/configs/input_run3_cnn_ho1.txt --conda-path /nfs/dust/cms/user/mykytaua/softML/miniconda3/bin/conda --max-memory 40000 --max-runtime 20
```

### Combining multiple runs
It might be the case that trainings from a common mlflow experiment are distributed amongst several people (e.g. for the purpose of hyperparameter optimisation), so that each person runs their fraction on their personal/institute hardware, and then all the mlflow runs are to be combined together under the same folder. Naive copying and merging of `mlruns` folders under the common folder wouldn't work because several parameters in `meta.yaml` files (mlflow internal cfg files stored per run and per experiment in corresponding folders) are not synchronised accross the team in their mlflow experiments. Specifically, in each of those files the following parameters require changes:

1. `artifact_location` (`meta.yaml` for each experiment) needs to be set to a common merged directory and the same change needs to be propagated to `artifact_uri` (`meta.yaml` for each run).
1. `experiment_id` needs to be set to a new common experiment ID (`meta.yaml` for each experiment and for each run).

These points are not a problem if the end goal is a simple file storage. However, mlflow is being used in the current training procedure mostly to improve UX with dedicated [`mlflow UI`](https://www.mlflow.org/docs/latest/tracking.html#tracking-ui) for a better navigation through the trainings and their convenient comparison. In that case, mlflow UI requires a properly structured `mlruns` folder to correctly read and visualise run-related information.

For the purpose of preparing individual's `mlruns` folder for merging, [`set_mlflow_paths.py`](https://github.com/cms-tau-pog/TauMLTools/blob/master/Training/python/set_mlflow_paths.py) script was written. What it does is:
1. recursively going through the folders in user-specified `path_to_mlflow` and corresponding experiment `exp_id` therein, opening firstly the experiment-related `meta.yaml` and resetting there `artifact_location` to `new_path_to_exp`, which is defined as a concatenation of user-specified `new_path_to_mlflow` and `exp_id` or `new_exp_id` (if passed)
1. opening run-related `meta.yaml` files and resetting `artifact_uri` key to `new_path_to_exp`, plus resetting `experiment_id` to `new_exp_id` (if passed)
1. in case `new_exp_id` argument is provided, the whole experiment folder is renamed to this new ID  

It has the following input arguments:
* `-p`, `--path-to-mlflow`: (required) Path to local folder with mlflow experiments
* `-id`, `--exp-id`: (required) Experiment id in the specified mlflow folder to be modified.
* `-np`, `--new-path-to-mlflow`: (required) New path to be set throughout `meta.yaml` configs for a specified mlflow experiment.
* `-nid`, `--new-exp-id`: (optional, default=None) If passed, will also reset the current experiment id.
* `-nn`, `--new-exp-name`: (optional, default=None) If passed, will also reset the current experiment name.

Therefore, the following workflow is suggested:
1. One runs their fraction of a common mlflow experiment at machine(s) of their choice. At this point it is not really important to have a common experiment name/experiment ID across all the team since it will be anyway unified afterwards. The outcome of this step is one has got locally a folder `${WORKDIR}/mlruns/${EXP_ID}` which contains the trainings as mlflow run folders + experiment-related `meta.yaml`.
1. The team aggrees that the combined experiment name is `${NEW_EXP_NAME}` with a corresponding ID `${NEW_EXP_ID}` and the directory where the runs distributed across the team are to be stored is `${SHARED_DIR}/mlruns`. Below will assume that this directory is located on `lxplus`.
1. Everyone in the team runs `set_mlflow_paths.py` script at the machine where the training was happening:
   ```sh
   python set_mlflow_paths.py -p ${WORKDIR}/mlruns -id ${exp_id} -np ${SHARED_DIR}/mlruns -nid ${NEW_EXP_ID} -nn ${NEW_EXP_NAME}
   ```
1. Everyone in the team copies runs from the shared experiment from their local machines to `${SHARED_DIR}`, e.g. using `rsync` (note that below `--dry-run` option is on, remove it to do the actual synching):
   ```sh
   rsync --archive --compress --verbose --human-readable --progress --inplace --dry-run ${WORKDIR}/mlruns ${USERNAME}@lxplus.cern.ch:${SHARED_DIR}
   ```
   **NB:** this command will synchronise **all** experiment folders inside of `${WORKDIR}/mlruns` with those of `${SHARED_DIR}/mlruns`. Make sure that this won't cause troubles and clashes for experiments which are not `${EXP_ID}`.
1. Now, it should be possible to bring up mlflow UI and inspect the merged runs altogether:
   ```sh
   cd ${SHARED_DIR}
   mlflow ui -p 5000 # 5000 is the default port
   # forward this port to a machine with graphics
   ```
### Additional helping commands 

In order to check the learning process stability and convergence, tensorboard monitoring is utilized in the training script. In order to compare all existing runs in the `mlruns` folder, the following command can be used:

```sh
./Training/script/run_tensorboard.sh <PATH_TO_MLRUNS_FOLDER>
```

In order to check the final training/validation loss function and corresponding model info:
```sh
./Training/script/print_results.sh <PATH_TO_MLRUNS_FOLDER>
```

## Testing NN performance

### Producing predictions

The first step in the performance evaluation pipeline is producing predictions for a given model and a given data sample. This can be achieved with [Evaluation/apply_training.py](https://github.com/cms-tau-pog/TauMLTools/blob/master/Evaluation/apply_training.py) script, where its input arguments are configured in [Evaluation/configs/apply_training.yaml](https://github.com/cms-tau-pog/TauMLTools/blob/master/Evaluation/configs/apply_training.yaml). It is important to mention that at this step `DataLoader` class is also used to yield batches for the model, therefore there are few parameters which configure its behaviour: 

* `training_cfg_upd` describes parameters which will be overriden in the `DataLoader` class initialisation. For example, for evaluation all taus should be included in batches, hence `include_mismatched=True`. Or, train/val split is not needed comparing to the train step, so `validation_split=0.`. Note that the base configuration by default is taken from mlflow logs and therefore is the one which was used for the training (`path_to_training_cfg` key).  

* `scaling_cfg` describes the path to a file with json scaling parameters and by default points to mlflow logs, i.e. the file which used for the training.

* `checkout_train_repo` describes whether a checkout of the git repo state as it was during the training (fetched from mlflow logs) should be made. This might be needed because the current repo state can include developments in `DataLoader` not present at the training step, therefore `import DataLoader` statement may not work. Moreover, this is needed to ensure that the same `DataLoader` is used for training and evaluation.

The next group of parameters in `apply_training.yaml` is the mlflow group (`path_to_mlflow/experiment_id/run_id`), which describes which mlflow run ID should be used to retrieve the associated model and to store the resulting predictions.

The remaining two arguments `path_to_file` and `sample_alias` describe I/O naming. `apply_training.py` works in a single file mode, so it expects one input ROOT file located in `path_to_file` and will output one h5 prediction file which will be stored under specified mlflow run ID in `artifacts/predictions/{sample_alias}/{basename(input_file_name)}_pred.h5`. So `sample_alias` here describes the sample to which the file belongs to (e.g. DY, or ggH). This `sample_alias` will be needed for the reference in the following eval pipeline steps.

As the last remark, `apply_training.py` automatically stores the mapping between input file and prediction file. This is kept in `artifacts/predictions/{sample_alias}/pred_input_filemap.json` and this mapping will be used downstream to automatically retrieve corresponding input files (not logged to mlflow) for mlflow-logged predictions.

An example of usage `apply_training.py` would be:
```sh
python apply_training.py path_to_mlflow=../Training/python/2018v1/mlruns experiment_id=2 run_id=1e6b4fa83d874cf8bc68857049d7371d path_to_file=eval_data/GluGluHToTauTau_M125/GluGluHToTauTau_M125_1.root sample_alias=GluGluHToTauTau_M125
```

### Evaluating metrics
The second step in the evaluation pipeline is [Evaluation/evaluate_performance.py](https://github.com/cms-tau-pog/TauMLTools/blob/master/Evaluation/evaluate_performance.py) with the corresponding main hydra cfg file being [Evaluation/configs/eval/run3.yaml](https://github.com/cms-tau-pog/TauMLTools/blob/master/Evaluation/configs/eval/run3.yaml). This step involves complex configuration setting in order to cover the variety of use cases in a generalised way. For a history of its developments and more details please refer to these presentations [[1]](https://indico.cern.ch/event/1066000/contributions/4482034/attachments/2293019/3898990/DeepTau_eval.pdf), [[2]](https://indico.cern.ch/event/1067541/contributions/4489002/attachments/2295081/3903198/DeepTau_eval_1.pdf), [[3]](https://indico.cern.ch/event/1068999/contributions/4495448/attachments/2297277/3907278/DeepTau_eval_2.pdf). 

#### Configuration description

The first definition in `run3.yaml` is the one of default lists `discriminator/plot_setup/selection`. These are essentially another yaml files from `configs` folder with their content being a part of the final hydra configuration. `discriminator` is in fact a subfolder in `configs` which contains yaml configurations to be selected from. These configurations are used to initialise `Discriminator` class which is used internally to uniformly manage ROC curve construction and metrics/param storage for various (heterogeneous) discriminator types. Likewise, `plot_setup.yaml` is used to instantiate `PlotSetup` class which is used to manage the styling of plots. `selection.yaml` stores aliases for different selections to be applied to input files so that they can be easily referred to in `run3.yaml` (see `cuts` parameter).

As usual, the next section in `run3.yaml` describes mlflow parameters (`path_to_mlflow/experiment_id/run_id`) to communicate reading/writing of necessary files/parameters for a specified mlflow run ID.

Then there is a section defining a so-called "phase space region". Conceptually, what `evaluate_performance.py` step does is takes a given model and computes the metrics (currently, ROC curves only) for a specified region of parameters. These are at the moment `vs_type`, `dataset_alias` and `pt_bins`:

* `vs_type` conventionally is used to defined tau types against which ROC curve should be constructed, e.g. tau vs jet (`vs_type=jet`). Based on this label, different selection/plotting criteria will be triggered across the module, please check them throughout discriminator/plotting/ROC curve composition steps. Moreover, as described below, in principle a mixture of particle types is also possible to be served as a "vs leg". 

* `dataset_alias` defines how to refer to a mixture of `input_samples` at the downstream plotting steps.

* `pt_bins` specifies in which bins the ROC curve should be computed, which will internally apply a corresponding cut on a combined mixed dataframe. This bin definition will also be used as an alias to retrieve corresponding ROC curve for plotting. 

The last important section is dedicated to I/O. There are three ingredients to it: inputs, predictions and targets, which can differ depending on the use case.

* In order to compute ROC curves a dataset containing genuine taus and e/mu/jet fakes is needed. The idea behind construction of such a dataset is to combine together various sources with selecting specific tau types from each of those. `input_samples` dictionary describes exactly this kind of mapping in a form `sample_alias` -> `['tau_type_1', 'tau_type_2', ...]`. It assumes by default that `sample_alias` is an alias defined at `apply_training.py` step (to refer to mlflow logged predictions automatically), but this don't have to be necessarily the case. `tau_type_*` specified in the mapping are used to apply matching to a feature with a name `gen_{tau_type}`. If `path_to_target` is provided, this will be taken from there (see `add_group()` function in `eval_tools.py`), otherwise it is assumed that this branch is already present in the input files (and added via `input_branches` argument in `run3.yaml`).

* Having defined a sample mixture, in `evaluate_performance.py` there is a loop over those samples and then for each `sample_alias` file lists with input/prediction/target files are constructed. The common template for file paths per `sample_alias` are specified via `path_to_input/path_to_pred/path_to_target` which has the option of filling placeholders in its name (`{sample_alias}`) and support for "*" asterisk. 

* These file lists are constructed in function `prepare_filelists()` of `eval_tools.py`. Conceptually, if present, it will expand `path_to_*` to a file list via `glob` and sort it according to a number id present in the file name (see `path_splitter()` function). The only non-trivial exception is the case when `path_to_pred` points to mlflow artifacts. Here, a mapping input<-> prediction from `artifacts/predictions/{sample_alias}/pred_input_filemap.json` will be used to fetch input samples corresponding to specified `path_to_pred` (and hence `path_to_input` is ignored).

Given all the information above, the metrics (e.g. ROC curve) are computed within `Discriminator` and `RocCurve` classes (see `eval_tools.py`) for a specified phase space region (`vs_type/dataset_alias/pt_bins`). They are then represented as a dictionary and dumped into `artifacts/performance.json` file which is used to collect all the "snapshots" of model evaluation across various input configurations, e.g. ROC curve TPR/FPR, plotting style, discriminator info. It is this file where the dowstream plotting modules will search for entries to be retrieved and visualised.    

#### Examples

Let's consider a comparison of a new Run3 training with already deployed DeepTau_v2 / MVA models. We will represent DeepTau Run3 training as a ROC curve, DeepTau_v2 as a ROC curve with manually defined working points (WPs) and for MVA we will take already centrally defined WPs. It should be noted that for the two latter cases the scores/WP bin codes are already present in the original input ROOT files used for DeepTau Run3 training. That means that there is nothing to `apply_training.py` to. However, we still need to introduce this models to the mlflow ecosystem under common `experiment_name=run3_cnn_ho2`, which can be done with `Training/python/log_to_mlflow.py`:

```sh
python log_to_mlflow.py path_to_mlflow=2018v1/mlruns experiment_name=run3_cnn_ho2 files_to_log=null params_to_log=null
```

This will create a new (empty) run ID under `run3_cnn_ho2` mlflow experiment (this is where DeepTau_run3 run ID is located), and we need to execute the command two times in total: one to be identified with DeepTau_v2 and the other with MVA model. After this, let's consider the case of plotting a ROC curve for `vs_type=jet`, where genuine taus are to be taken from `sample_alias=GluGluHToTauTau_M125` and jets from `sample_alias=TTToSemiLeptonic`. These are the data samples with corresponding process excluded from the training and set aside to evaluate the training performance. Suppose the corresponding input ROOT files are located in `path_to_input=Evaluation/eval_data/{sample_alias}/*.root` and predictions for DeepTau_run3 model were already produced for them with `apply_training.py` (hence stored in mlflow artifacts).

Then, since there are three models in total which are moreover fundamentally different, we will identify them with three seperate `discriminator` cfg files. Corresponding templates can be found in `configs/discriminator` folder with the parameters steering the behaviour of `Discriminator` class (defined in `eval_tools.py`). Please have a look there to understand the behaviour under the hood for each of the parameters' values. Also note that at this point it might be useful to create a new yaml cfg file for a discriminator if its behaviour isn't described by these template config files.

Finally, given that `run3_cnn_ho2` -> `experiment_id=2` and input samples cfg in `run3.yaml` look like:
```yaml
input_samples: 
  GluGluHToTauTau_M125: ['tau']
  TTToSemiLeptonic: ["jet"]
```

evaluating the performance for DeepTau_run3 can be done with (run IDs/paths below are specific to this example only):
```sh
python evaluate_performance.py path_to_mlflow=../Training/python/2018v1/mlruns experiment_id=2 run_id=06f9305d6e0b478a88af8ea234bcec20 discriminator=DeepTau_run3 path_to_input=null 'path_to_pred="${path_to_mlflow}/${experiment_id}/${run_id}/artifacts/predictions/{sample_alias}/*_pred.h5"' 'path_to_target="${path_to_mlflow}/${experiment_id}/${run_id}/artifacts/predictions/{sample_alias}/*_pred.h5"' vs_type=jet dataset_alias=ggH_TT
```

for DeepTau_v2 (note the changed paths where inputs are manually specified and targets are taken from DeepTau_run3 logged predictions, and also a change to `wp_from=pred_column` to use manually defined WPs from `working_points_thrs`):
```sh
python evaluate_performance.py path_to_mlflow=../Training/python/2018v1/mlruns experiment_id=2 run_id=90c83841fe224d48b1581061eba46e86 discriminator=DeepTau_v2p1 'path_to_input="eval_data/{sample_alias}/*.root"' path_to_pred=null 'path_to_target="${path_to_mlflow}/${experiment_id}/06f9305d6e0b478a88af8ea234bcec20/artifacts/predictions/{sample_alias}/*_pred.h5"' vs_type=jet dataset_alias=ggH_TT discriminator.wp_from=pred_column
```

for MVA:
```sh
python evaluate_performance.py path_to_mlflow=../Training/python/2018v1/mlruns experiment_id=2 run_id=d2ec6115624d44c9bf60f88460b09b54 discriminator=MVA_jinst_vs_jet 'path_to_input="eval_data/{sample_alias}/*.root"' path_to_pred=null 'path_to_target="${path_to_mlflow}/${experiment_id}/06f9305d6e0b478a88af8ea234bcec20/artifacts/predictions/{sample_alias}/*_pred.h5"' vs_type=jet dataset_alias=ggH_TT
```

Now one can inspect `performance.json` files in corresponding mlflow run artifacts to get the intuition of how the skimmed performance info looks like. For example, since internally WP and ROC curve are defined and treated as instances of the same `RocCurve` class, output in `performance.json` for MVA model looks structurally the same as for DeepTau_run3, although for the former we just plot a set of working points, and for the latter the whole ROC curve.

### Plotting ROC curves
The third step in the evaluation pipeline is [Evaluation/plot_roc.py](https://github.com/cms-tau-pog/TauMLTools/blob/master/Evaluation/plot_roc.py) with the corresponding [Evaluation/configs/plot_roc.yaml](https://github.com/cms-tau-pog/TauMLTools/blob/master/Evaluation/configs/plot_roc.yaml) cfg file. In the latter one need to specify:

* mlflow `experiment_id`, which assumes that all run IDs below belong to this experiment ID
* `discriminators`: a dictionary mapping `run_id` -> `[curve_type_1, 'curve_type_2']`, where `curve_type_*` is either `roc_curve` or `roc_wp` and describes which types of ROC curves should be plotted.
* `reference`: a pair `run_id`: `curve_type` which will be used as the reference curve to plot the ratio for other discriminants. 
* `vs_type/dataset_alias/pt_bin`: parameters identifying the region of interest. These are used to retrieve the corresponding entries in `performance.json` file for each of the runs to be plotted, see `eval_tools.select_curve()` function which does that.
* `output_name`: the name of the output pdf file.

Continuing the example of the previous section, setting the following in `plot_roc.yaml`:
```yaml
path_to_mlflow: ../Training/python/2018v1/mlruns
experiment_id: 2
discriminators:
  06f9305d6e0b478a88af8ea234bcec20: ['roc_curve']
  90c83841fe224d48b1581061eba46e86: ['roc_curve', 'roc_wp']
  d2ec6115624d44c9bf60f88460b09b54: ['roc_wp']
reference: 
  06f9305d6e0b478a88af8ea234bcec20: 'roc_curve'
```

and running:
```sh
python plot_roc.py vs_type=jet dataset_alias=ggH_TT 'pt_bin=[20,100]'
```

will produce the plot with specified models plotted side-by-side and will store it in `mlruns/2/06f9305d6e0b478a88af8ea234bcec20/artifacts/plots/`, which then can be viewed directly or through mlflow UI.

### Computing working points

For a given model, one usually has to define working points (WPs): that is, to derive thresholds on the model output scores yielding classification with predefined genuine tau efficiency, in the specified region of the phase space. This can be done with the script [Evaluation/derive_wp.py](https://github.com/cms-tau-pog/TauMLTools/blob/master/Evaluation/derive_wp.py) and the corresponding cfg file [Evaluation/configs/derive_wp.yaml](https://github.com/cms-tau-pog/TauMLTools/blob/master/Evaluation/configs/derive_wp.yaml). The script contains the definition of a `WPMaker` class which takes care of the computation procedure + the code to run it, while the yaml file specifies the arguments to the class and to `create_df()` function defined in [Evaluation/utils/data_processing.py](https://github.com/cms-tau-pog/TauMLTools/blob/master/Evaluation/utils/data_processing.py). The computation follows the logic:

* Take genuine taus (`create_df.tau_type_to_select=tau`) from all the samples in the dictionary `create_df.pred_samples` (will read files based on `create_df.pred_samples.{sample_name}.filename_pattern`) within `create_df.path_to_preds` (by default assumed to be within mlflow artifacts, but can be specified differently) which pass `create_df.selection`. Feature values for selection (e.g. `tau_pt`) will be taken from the input file corresponding to every prediction file, where the mapping is obtained from `pred_input_filemap.json` (assumed to be present in `pred_path/{sample_name}`).

* `score_vs_{type}` is computed for each tau as usual `score_tau / (score_tau + score_vs_type)`. If `reweight_to_lumi` isn't set to `null`, will also assign a weight to taus from the sample according to the fraction of luminosity `sample_lumi` assigned to it (i.e. `lumi_weight = sample_lumi / (reweight_to_lumi * N_taus_in_sample)`).

* Function `WPMaker.update_thrs()` for each `score_vs_{type}` computes thresholds corresponding to its weighted quantiles given by `tpr` values. `tpr` is defined as a grid of evenly-spaced values with `step=wp_maker.tpr_step`. Thresholds which yield efficiencies closest to the values specified in `wp_maker.wp_definitions.{type}.wp_eff` are selected.

* The procedure is repeated until convergence of threshold values within `wp_maker.epsilon` for all `vs_types`. If `wp_maker.require_wp_vs_others=False`, it should take only 2 iterations (no self-dependancy). Otherwise, the dependancy on passing the loosest WPs vs remaining tau types is introduced in the definition of WP vs current_type, so convergence will require >2 iterations (usually, it takes 5-8 iterations). 

* Resulting json file with thresholds is logged to mlflow artifacts of the specified `run_id`.

An example of running the script - assuming `pred_samples` are already specified as needed in `derive_wp.yaml` - would be:

```bash
python derive_wp.py 'create_df.path_to_mlflow=../Training/python/2018v1/mlruns/' create_df.experiment_id=4 create_df.run_id=e1f3ddb3c4c94128a34a7635c56322eb
```

### Plotting working point efficiencies

Once working points are derived, it is important to understand how tau/fake efficiency look like as a function of tau-related variables for each of the working points. Such plots can be produced with a script [Evaluation/plot_wp_eff.py](https://github.com/cms-tau-pog/TauMLTools/blob/master/Evaluation/plot_wp_eff.py) + config [Evaluation/configs/plot_wp_eff.yaml](https://github.com/cms-tau-pog/TauMLTools/blob/master/Evaluation/configs/plot_wp_eff.yaml). 

What it does is firstly composes two datasets, one with genuine taus and the other with `vs_type` fakes. This uses the same `create_df()` function from `Evaluation/utils/data_processing.py`, which in this case however is called in a [partial manner](https://docs.python.org/3/library/functools.html#functools.partial). For that, the core arguments `create_df.*` in the config, specifying general paths/selection, are shared for each `tau_type`, while `pred_samples` argument (dictionary of dictionaries, `tau_type -> samples -> sample_cfg`) is factorised from it, allowing for a flexible separate calls per `tau_type`. If `from_skims=False`, those datasets will be logged to mlflow artifacts for the specified run, in order to remove the need in producing the datasets each time (hence, for `from_skims=True` it will instead search for such datasets in `output_skim_folder` within the mlflow artifacts).

Then, WP thresholds are retrieved from the json file from mlflow artifacts and `differential_efficiency()` function from `Evaluation/utils/wp_eff.py` is called. It computes separately for both genuine taus and fakes the efficiency of passing WPs in `var_bins` of `wp_eff_var`. Note that this parameter in the main `plot_wp_eff.yaml` is itself the name of another yaml config to be imported into the final hydra configuration from `Evaluation/configs/wp_eff_var` folder. This folder contains a template set of yaml files where each describes binning & plotting parameters for a corresponding variable.  

It is `efficiency()` function which counts tau/fake objects passing WPs and returns the corresponding efficiency with a confidence interval (`eff, eff_down, eff_up`). It should be mentioned that this computation can be done both with and without assumption of passing some user-defined WPs. So it is the arguments `require_WPs_in_numerator`/ `require_WPs_in_denominator` which set if objects entering numerator/denominator of the efficiency formula should be required to pass `WPs_to_require`.

Once differential efficiencies are calculated, `plot_efficiency()` is called partially, with the core arguments taken from `wp_eff_var` config and the other (efficiencies) passed as partial arguments. This creates a corresponding plot and stores it within mlflow artifacts as `output_filename`.

An example of running the script (assuming `pred_samples` are already specified as needed in `plot_wp_eff.yaml`) would be:

```bash
python plot_wp_eff.py vs_type=jet from_skims=True output_skim_folder=wp_eff/data/Run3 wp_eff_var=tau_eta 'create_df.path_to_mlflow=../Training/python/2018v1/mlruns/' create_df.experiment_id=4 create_df.run_id=e1f3ddb3c4c94128a34a7635c56322eb
```