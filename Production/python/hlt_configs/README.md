Getting and Running the HLT:
https://twiki.cern.ch/twiki/bin/view/CMSPublic/SWGuideGlobalHLT

## For 2025 studies

CMSSW setup with the tracking changes:
```sh
cmsrel CMSSW_15_0_2
cd CMSSW_15_0_2/src
eval `scramv1 runtime -sh`
scram b -j8
```

MC HLT menu
```sh
cmsEnv hltGetConfiguration /dev/CMSSW_15_0_0/GRun/V22 --globaltag 142X_mcRun3_2025_realistic_v7 --mc --unprescale --output none --max-events 10 --eras Run3_2025 --l1-emulator uGT --l1 L1Menu_Collisions2024_v1_3_0_xml --input /store/mc/Run3Winter25Digi/GluGluHto2Tau_Par-MH-125_TuneCP5_13p6TeV_powhegMINLO-pythia8/GEN-SIM-RAW/142X_mcRun3_2025_realistic_v7_ext1-v2/140002/008a4477-721e-4793-801e-00465c4c5de5.root > hltMC_orig.py
```

data HLT menu
```sh
cmsEnv hltGetConfiguration /dev/CMSSW_14_0_0/GRun/V58 --globaltag 140X_dataRun3_HLT_for2024TSGStudies_v1 --data --unprescale --output none --max-events 10 --eras Run3 --l1-emulator uGT --l1 L1Menu_Collisions2024_v1_0_0_xml --input /store/data/Run2023D/EphemeralHLTPhysics0/RAW/v1/000/370/293/00000/2ef73d2a-1fb7-4dac-9961-149525f9e887.root > hltData_v58.py

cmsEnv hltGetConfiguration /dev/CMSSW_15_0_0/GRun/V22 --globaltag 150X_dataRun3_HLT_v1 --data --unprescale --output none --max-events 10 --eras Run3_2024 --l1-emulator uGT --l1 L1Menu_Collisions2024_v1_3_0_xml --input /store/data/Run2024I/EphemeralHLTPhysics0/RAW/v1/000/386/593/00000/91a08676-199e-404c-9957-f72772ef1354.root > hltData_orig.py
```

Test MC
```sh
mkdir -p tmp && cd tmp
cmsEnv python3 $ANALYSIS_PATH/RunKit/cmsRunWrapper.py cmsRunCfg=$ANALYSIS_PATH/Production/python/hlt_configs/hltMC.py maxEvents=100 inputFiles=/store/mc/Run3Winter25Digi/GluGluHto2Tau_Par-MH-125_TuneCP5_13p6TeV_powhegMINLO-pythia8/GEN-SIM-RAW/142X_mcRun3_2025_realistic_v7_ext1-v2/140002/008a4477-721e-4793-801e-00465c4c5de5.root writePSet=True copyInputsToLocal=False 'output=nano.root;./output'
cmsEnv $ANALYSIS_PATH/RunKit/crabJob.sh &> crabJob.log
```

Test data
```sh
mkdir -p tmp && cd tmp
cmsEnv python3 $ANALYSIS_PATH/RunKit/cmsRunWrapper.py cmsRunCfg=$ANALYSIS_PATH/Production/python/hlt_configs/hltData.py maxEvents=100 inputFiles=/store/data/Run2024I/EphemeralHLTPhysics0/RAW/v1/000/386/593/00000/91a08676-199e-404c-9957-f72772ef1354.root writePSet=True copyInputsToLocal=False 'output=nano.root;./output'
cmsEnv $ANALYSIS_PATH/RunKit/crabJob.sh &> crabJob.log
```