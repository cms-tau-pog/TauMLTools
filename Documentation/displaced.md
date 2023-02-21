# Displaced Tau Tagging Documentation

This file contains instructions for reproducing the displaced tau tagging implemented within TauMLTools machinery.

## How to produce inputs for the Displaced Taus tagging

Steps below describe how to process input datasets starting from MiniAOD updated with the functionality for displaced tau tagging specific.

### Big root-tuples production

[Production/python/Production.py](https://github.com/cms-tau-pog/TauMLTools/blob/master/Production/python/Production.py) contains the configuration that allows to run `TauTupleProducer` with `cmsRun`.
The config script should be run with the specification of the `rerunTauReco` and `selector`. Also disabling unnecessary branches can be used for saving disk disk space. Here is an example how to run `TauTupleProducer` on 1000 long-lived SUSY-TAU events using one MiniAOD file as an input:
```sh
cmsRun ./TauMLTools/Production/python/Production.py sampleType=MC era=Run2_2018 inputFiles=/store/mc/RunIISummer20UL18MiniAODv2/TTToSemiLeptonic_TuneCP5_13TeV-powheg-pythia8/MINIAODSIM/106X_upgrade2018_realistic_v16_L1v1-v2/120000/4AFA2D6F-F6CD-6C46-8962-13C479C00E8B.root maxEvents=1000 disabledBranches="boostedTau_.*,fatJet_.*" rerunTauReco="displacedTau"
```

In order to run a large-scale production for the entire datasets, the CMS computing grid should be used via CRAB interface.
Submission and status control can be performed using [crab_submit.py](https://github.com/cms-tau-pog/TauMLTools/blob/master/Production/scripts/crab_submit.py) and [crab_cmd.py](https://github.com/cms-tau-pog/TauMLTools/blob/master/Production/scripts/crab_cmd.py) commands.

#### UL2018 tuple production


The production is identical to the [2018 tuples production](https://github.com/cms-tau-pog/TauMLTools/edit/master/README.md#2018-root-tuple-production-steps).

The main change is related to the specifying additional function in config files:
`disabledBranches=boostedTau_.*,fatJet_.*` - allows to disable branches that are not used.
`rerunTauReco=displacedTau` - rerun jet clustering and reconstruction with modified IP criterion to be more sensitive to displaced taus/
`selector=TauJetTag` - selects events with genJet or recoJet or genLepton  


### Merging the ntuples

#### Spectrum creation
This step merge and mix the training files.

Spectrum of the initial data is needed to calculate the probability to take tau candidate of some genuine tau-type and from some pt-eta bin. To generate the spectrum histograms for a dataset and lists with number of entries per datafile, run:
```sh
CreateSpectralHists --output "spectrum_file.root" \
                    --output_entries "entires_file.txt" \
                    --input-dir "path/to/dataset/dir" \
                    --pt-hist "n_bins_pt, pt_min, pt_max" \
                    --eta-hist "n_bins_eta, |eta|_min, |eta|_max" \
	        –mode “jet”
                    --n-threads 1
```
on this step it is important to have high granularity binning to be able later to re-bin into custom, non-uniform pt-eta bins on the next step.

**S&M spectrum is not used for the current version of SimpleMix script however to be considered in the further preprocessing step developments.**

Alternatively, if one wants to process several datasets the following python script can be used (parameters of pt and eta binning to be hardcoded in the script):
```sh
python Analysis/python/CreateSpectralHists.py --input /path/to/input/dir/ \
                                              --output /path/to/output/dir/ \
                                              --filter ".*(DY).*" \
                                              --rewrite
                                              --mode "jet"
```
Mode `jet` is added specifically for using displaced tau object definitions [here](Analysis/interface/DisTauTagSelection.h).

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

#### Simple mix:
After spectrums are created for all datasets, the final procedure of Shuffle and Merge can be performed with:
```sh
SimpleMergeMix --cfg Analysis/config/2018_UL_simpl/training_input_test.cfg
                --input filelist_mix.txt
                --output <path_to_output_file.root>
                --input-spec <path_to_spectrum>
                --n-jobs 1
                --job-idx 0
                --prefix <path_to_data>
                --disabled-branches "boostedTau_.*,fatJet_.*"

```
The plugin arguments are analogical to the [S&M](https://github.com/cms-tau-pog/TauMLTools#shuffling--merging).
