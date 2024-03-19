Getting and Running the HLT:
https://twiki.cern.ch/twiki/bin/view/CMSPublic/SWGuideGlobalHLT

For MC HLT menu (2024)

```sh
cmsEnv hltGetConfiguration /dev/CMSSW_14_0_0/GRun/V58 --globaltag auto:phase1_2024_realistic --mc --unprescale --output none --max-events 10 --eras Run3 --l1-emulator uGT --l1 L1Menu_Collisions2024_v1_0_0_xml --input /store/mc/Run3Winter24Digi/GluGluHToTauTau_M-125_TuneCP5_13p6TeV_powheg-pythia8/GEN-SIM-RAW/133X_mcRun3_2024_realistic_v8-v2/50000/000d40c2-6549-4879-a6fe-b71e3b1e3a57.root  > hltMC_v58.py
```

For 2023 data HLT menu (2024)
```sh
cmsEnv hltGetConfiguration /dev/CMSSW_14_0_0/GRun/V58 --globaltag 140X_dataRun3_HLT_for2024TSGStudies_v1 --data --unprescale --output none --max-events 10 --eras Run3 --l1-emulator uGT --l1 L1Menu_Collisions2024_v1_0_0_xml --input /store/data/Run2023D/EphemeralHLTPhysics0/RAW/v1/000/370/293/00000/2ef73d2a-1fb7-4dac-9961-149525f9e887.root > hltData_v58.py
```
