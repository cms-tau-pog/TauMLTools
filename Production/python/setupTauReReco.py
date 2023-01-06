import FWCore.ParameterSet.Config as cms

# Customization of Tau ReReco at MiniAOD is introduced in this file.
# Current customization is added after initialization of 
# cmssw/RecoTauTag/Configuration/python/tools/adaptToRunAtMiniAOD.py

# Available options:
#   - Custom signal cone reconstruction
#   - Displaced tau reconstruction

def reReco_SigCone(process):
    
    process.combinatoricRecoTaus.builders[0].signalConeSize = cms.string('max(min(0.2, 4.528/(pt()^0.8982)), 0.03)') ## change to quantile 0.95
    process.selectedPatTaus.cut = cms.string('pt > 18.')   ## remove DMFinding filter (was pt > 18. && tauID(\'decayModeFindingNewDMs\')> 0.5)

def reReco_DisTau(process):

    # The following study is considered as a starting point for modifications:
    # https://indico.cern.ch/event/868576/contributions/3696661/attachments/1979084/3295013/tau_displaced_ll_ws_31_1_2020.pdf

    process.selectedPatTaus.cut = cms.string('pt > 18.')   ## remove DMFinding filter (was pt > 18. && tauID(\'decayModeFindingNewDMs\')> 0.5)
    for builder in process.combinatoricRecoTaus.builders \
                 + process.ak4PFJetsRecoTauChargedHadrons.builders:
        builder.qualityCuts.signalQualityCuts.maxDeltaZ = cms.double(100.)
        builder.qualityCuts.signalQualityCuts.maxTrackChi2 = cms.double(1000.)
        builder.qualityCuts.signalQualityCuts.maxTransverseImpactParameter = cms.double(100.)
        builder.qualityCuts.vxAssocQualityCuts.maxTransverseImpactParameter = cms.double(100.)
        builder.qualityCuts.pvFindingAlgo = cms.string('highestPtInEvent')