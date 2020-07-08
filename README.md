# TauMLTools

## Introduction

Tools to perform machine learning studies for tau lepton reconstruction and identification at CMS.

## How to install

```sh
cmsrel CMSSW_10_6_13
cd CMSSW_10_6_13/src
cmsenv
git cms-addpkg RecoTauTag/RecoTau
git clone -o cms-tau-pog git@github.com:cms-tau-pog/TauMLTools.git
scram b -j8
```
