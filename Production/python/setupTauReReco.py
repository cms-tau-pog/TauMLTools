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


_HPS_MODULES = ("hpsSelectionDiscriminator",
                "hpsPFTauDiscriminationByDecayModeFindingNewDMs")

def _dm_psets(process):
    """Yield {(nCharged, nPiZeros): PSet} for each module holding the HPS windows."""
    for name in _HPS_MODULES:
        mod = getattr(process, name)
        yield {(p.nCharged.value(), p.nPiZeros.value()): p for p in mod.decayModes}

def _loosen_dm10(dm):   # 3 prongs, 0 strips
    dm[(3, 0)].minMass = 0.6
    dm[(3, 0)].maxMass = "max(1.5, min(1.5*sqrt(pt/100.), 4.0))"

def _loosen_dm11(dm):   # 3 prongs, 1 strip; maxMass also bounds the charged-hadron mass
    dm[(3, 1)].maxMass = "max(1.6, min(1.6*sqrt(pt/100.), 4.0))"

def _loosen_dm2(dm):    # 1 prong, 2 strips: open the strip-pair mass window
    dm[(1, 2)].minPi0Mass = -1.e3
    dm[(1, 2)].maxPi0Mass = 1.e9

def reReco_Default(process):
    pass   # rerun only: closure check and baseline

def reReco_NoDMF(process):
    # The pt/DM-finding cut lives in selectedPatTausNoNewIDs (PATTauSelector); selectedPatTaus
    # is the PATTauIDEmbedder that reads from it, so setting .cut there would have no effect.
    process.selectedPatTausNoNewIDs.cut = cms.string('pt > 18.')   # keep taus failing DM finding

def _make_loose(*loosen):
    def apply(process):
        for dm in _dm_psets(process):
            for f in loosen:
                f(dm)
    return apply

reReco_LooseDM10 = _make_loose(_loosen_dm10)
reReco_LooseDM11 = _make_loose(_loosen_dm11)
reReco_LooseDM2  = _make_loose(_loosen_dm2)
reReco_LooseAll  = _make_loose(_loosen_dm10, _loosen_dm11, _loosen_dm2)
