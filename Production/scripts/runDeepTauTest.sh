#!/bin/bash

cd $CMSSW_BASE/src/RecoTauTag/RecoTau/data/
cat download.url | xargs -n 1 wget
cd -

cmsRun -j FrameworkJobReport.xml -p PSet.py
