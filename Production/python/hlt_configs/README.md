Getting and Running the HLT:
https://twiki.cern.ch/twiki/bin/view/CMSPublic/SWGuideGlobalHLT

For MC
```sh
cmsEnv hltGetConfiguration /dev/CMSSW_13_0_0/GRun/V24 --globaltag 126X_mcRun3_2023_forPU65_v1 --mc --unprescale --output none --max-events 10 --input /store/mc/Run3Winter23Digi/TT_TuneCP5_13p6TeV_powheg-pythia8/GEN-SIM-RAW/126X_mcRun3_2023_forPU65_v1_ext1-v2/40002/cbcb2b23-174a-4e7f-a385-152d9c5c5b87.root --customise HLTrigger/Configuration/customizeHLTFor2023.customizeHCALFor2023 --eras Run3 --l1-emulator FullMC --l1 L1Menu_Collisions2022_v1_4_0-d1_xml > hltMC.py
```

For data
```sh
cmsEnv hltGetConfiguration /dev/CMSSW_13_0_0/GRun/V24 --globaltag 126X_dataRun3_HLT_v1 --data --unprescale --output none --max-events 100 --eras Run3 --l1-emulator Full --l1 L1Menu_Collisions2022_v1_4_0-d1_xml --input /store/data/Run2022G/EphemeralHLTPhysics3/RAW/v1/000/362/720/00000/850a6b3c-6eef-424c-9dad-da1e678188f3.root > hltData_L1.py
```