Samples are on EOS (`/eos/cms/store/group/upgrade/RTB/Iter6/11_3/RAW/`), not DAS, so it's better to run interactively than to submit with crab.

Here's an example command:

```
cmsRun python/Production.py sampleType=MC_Phase2_113X fileNamePrefix="file:" fileList=/afs/cern.ch/work/d/dmroy/TauMLTools/Production/ListToProcess_QCD_Pt-15to7000_pt1.txt tupleOutput="/eos/user/d/dmroy/TauML/QCD_Pt-15to7000_Flat_PU0-200/eventTuple_1.root"
```
