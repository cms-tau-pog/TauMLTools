Getting and Running the HLT:
https://twiki.cern.ch/twiki/bin/view/CMSPublic/SWGuideGlobalHLT

For MC HLT menu V1.1 (2023)
```sh
cmsEnv hltGetConfiguration /dev/CMSSW_13_0_0/GRun/V115 --globaltag 126X_mcRun3_2023_forPU65_v5 --mc --unprescale --output none --max-events 10 --input /store/mc/Run3Winter23Digi/TT_TuneCP5_13p6TeV_powheg-pythia8/GEN-SIM-RAW/126X_mcRun3_2023_forPU65_v1_ext1-v2/40002/cbcb2b23-174a-4e7f-a385-152d9c5c5b87.root --eras Run3 --l1-emulator FullMC --l1 L1Menu_Collisions2023_v1_1_0-v2_xml > hltMC_v115.py
```

For 2022 data HLT menu V1.1 (2023)
```sh
cmsEnv hltGetConfiguration /dev/CMSSW_13_0_0/GRun/V115 --globaltag 130X_dataRun3_HLT_v2 --data --unprescale --output none --max-events 100 --eras Run3 --l1-emulator Full --l1 L1Menu_Collisions2023_v1_1_0-v2_xml --input /store/data/Run2022G/EphemeralHLTPhysics3/RAW/v1/000/362/720/00000/850a6b3c-6eef-424c-9dad-da1e678188f3.root > hltData_v115.py
```

Outdated recommendations for 2023: https://github.com/silviodonato/cmssw/tree/customizeHLTFor2023/HLTrigger/Configuration/python#hlt-customization-functions-for-2023-run-3-studies