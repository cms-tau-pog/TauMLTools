# Produce TauTuple.

import re
import importlib
import FWCore.ParameterSet.Config as cms
from FWCore.ParameterSet.VarParsing import VarParsing
import RecoTauTag.Configuration.tools.adaptToRunAtMiniAOD as tauAtMiniTools
import os


options = VarParsing('analysis')
options.register('sampleType', '', VarParsing.multiplicity.singleton, VarParsing.varType.string,
                 "Indicates the sample type: MC_18, Run2018ABC, ...")
options.register('fileList', '', VarParsing.multiplicity.singleton, VarParsing.varType.string,
                 "List of root files to process.")
options.register('fileNamePrefix', '', VarParsing.multiplicity.singleton, VarParsing.varType.string,
                 "Prefix to add to input file names.")
options.register('tupleOutput', 'eventTuple.root', VarParsing.multiplicity.singleton, VarParsing.varType.string,
                 "Event tuple file.")
options.register('lumiFile', '', VarParsing.multiplicity.singleton, VarParsing.varType.string,
                 "JSON file with lumi mask.")
options.register('eventList', '', VarParsing.multiplicity.singleton, VarParsing.varType.string,
                 "List of events to process.")
options.register('dumpPython', False, VarParsing.multiplicity.singleton, VarParsing.varType.bool,
                 "Dump full config into stdout.")
options.register('numberOfThreads', 1, VarParsing.multiplicity.singleton, VarParsing.varType.int,
                 "Number of threads.")
options.register('storeJetsWithoutTau', False, VarParsing.multiplicity.singleton, VarParsing.varType.bool,
                 "Store jets that don't match to any pat::Tau.")
options.register('requireGenMatch', True, VarParsing.multiplicity.singleton, VarParsing.varType.bool,
                 "Store only tau jets that have genLepton_index >= 0 or genJet_index >= 0.")
options.register('requireGenORRecoTauMatch', True, VarParsing.multiplicity.singleton, VarParsing.varType.bool,
                 """Store only tau jets that satisfy the following condition:
                    tau_index >= 0 || boostedTau_index >= 0 || (genLepton_index >= 0 && genLepton_kind == 5)""")
options.register('applyRecoPtSieve', False, VarParsing.multiplicity.singleton, VarParsing.varType.bool,
                 "Randomly drop jet->tau fakes depending on reco tau pt to balance contributions from low and higt pt.")
options.register('reclusterJets', True, VarParsing.multiplicity.singleton, VarParsing.varType.bool,
                " If 'reclusterJets' set true a new collection of uncorrected ak4PFJets is built to seed taus (as at RECO), otherwise standard slimmedJets are used")
options.register('rerunTauReco', False, VarParsing.multiplicity.singleton, VarParsing.varType.bool,
                "If true, tau reconstruction is re-run on MINIAOD with a larger signal cone and no DM finding filter")
options.parseArguments()

sampleConfig = importlib.import_module('TauMLTools.Production.sampleConfig')
isData = sampleConfig.IsData(options.sampleType)
isEmbedded = sampleConfig.IsEmbedded(options.sampleType)
isRun2UL = sampleConfig.isRun2UL(options.sampleType)
isRun2PreUL = sampleConfig.isRun2PreUL(options.sampleType)
isPhase2 = sampleConfig.isPhase2(options.sampleType)
period = sampleConfig.GetPeriod(options.sampleType)
period_cfg = sampleConfig.GetPeriodCfg(options.sampleType)

processName = 'tupleProduction'
process = cms.Process(processName, period_cfg)
process.options = cms.untracked.PSet()
process.options.wantSummary = cms.untracked.bool(False)
process.options.allowUnscheduled = cms.untracked.bool(True)
process.options.numberOfThreads = cms.untracked.uint32(options.numberOfThreads)
process.options.numberOfStreams = cms.untracked.uint32(0)

process.load('Configuration.StandardSequences.MagneticField_cff')
# include Phase2 specific configuration only after 11_0_X
if isPhase2:
    process.load('Configuration.Geometry.GeometryExtended2026D49Reco_cff')
    process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')
    from Configuration.AlCa.GlobalTag import GlobalTag
    process.GlobalTag = GlobalTag(process.GlobalTag, sampleConfig.GetGlobalTag(options.sampleType), '')
else:
    process.load('Configuration.Geometry.GeometryRecoDB_cff')
    process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_condDBv2_cff')
    process.GlobalTag.globaltag = sampleConfig.GetGlobalTag(options.sampleType)
process.source = cms.Source('PoolSource', fileNames = cms.untracked.vstring())
process.TFileService = cms.Service('TFileService', fileName = cms.string(options.tupleOutput) )
process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(-1) )

from TauMLTools.Production.readFileList import *
if len(options.fileList) > 0:
    readFileList(process.source.fileNames, options.fileList, options.fileNamePrefix)
elif len(options.inputFiles) > 0:
    addFilesToList(process.source.fileNames, options.inputFiles, options.fileNamePrefix)

if options.maxEvents > 0:
    process.maxEvents.input = options.maxEvents

if len(options.lumiFile) > 0:
    import FWCore.PythonUtilities.LumiList as LumiList
    process.source.lumisToProcess = LumiList.LumiList(filename = options.lumiFile).getVLuminosityBlockRange()

if options.eventList != '':
    process.source.eventsToProcess = cms.untracked.VEventRange(re.split(',', options.eventList))

tau_collection = 'slimmedTaus'
if options.rerunTauReco:
    tau_collection = 'selectedPatTaus'

    tauAtMiniTools.addTauReReco(process)
    tauAtMiniTools.adaptTauToMiniAODReReco(process, options.reclusterJets)

    if isData:
        from PhysicsTools.PatAlgos.tools.coreTools import runOnData
        runOnData(process, names = ['Taus'], outputModules = [])

    process.combinatoricRecoTaus.builders[0].signalConeSize = cms.string('max(min(0.2, 4.528/(pt()^0.8982)), 0.03)') ## change to quantile 0.95
    process.selectedPatTaus.cut = cms.string('pt > 18.')   ## remove DMFinding filter (was pt > 18. && tauID(\'decayModeFindingNewDMs\')> 0.5)

# include Phase2 specific configuration only after 11_0_X
if isPhase2:
    tauIdConfig = importlib.import_module('RecoTauTag.RecoTau.tools.runTauIdMVA')
    updatedTauName = "slimmedTausNewID"
    tauIdEmbedder = tauIdConfig.TauIDEmbedder(
        process, cms, updatedTauName = updatedTauName,
        toKeep = [ "2017v2", "dR0p32017v2", "newDM2017v2", "deepTau2017v2p1", "newDMPhase2v1"]
    )
    tauIdEmbedder.runTauID() # note here, that with the official CMSSW version of 'runTauIdMVA' slimmedTaus are hardcoded as input tau collection
    boostedTaus_InputTag = cms.InputTag('slimmedTausBoosted')
elif isRun2UL:
    boostedTaus_InputTag = cms.InputTag('slimmedTausBoosted')
else:
    from TauMLTools.Production.runTauIdMVA import runTauID
    updatedTauName = "slimmedTausNewID"
    runTauID(process, outputTauCollection = updatedTauName, inputTauCollection = tau_collection,
             toKeep = [ "2017v2", "dR0p32017v2", "newDM2017v2", "deepTau2017v2p1" ])
    tauIdConfig = importlib.import_module('TauMLTools.Production.runTauIdMVA')

    from RecoTauTag.Configuration.boostedHPSPFTaus_cff import ca8PFJetsCHSprunedForBoostedTaus
    process.ca8PFJetsCHSprunedForBoostedTausPAT = ca8PFJetsCHSprunedForBoostedTaus.clone(
        src = cms.InputTag("packedPFCandidates"),
        jetCollInstanceName = cms.string('subJetsForSeedingBoostedTausPAT')
    )
    process.cleanedSlimmedTausBoosted = cms.EDProducer("PATBoostedTauCleaner",
        src = cms.InputTag('slimmedTausBoosted'),
        pfcands = cms.InputTag('packedPFCandidates'),
        vtxLabel= cms.InputTag('offlineSlimmedPrimaryVertices'),
        ca8JetSrc = cms.InputTag('ca8PFJetsCHSprunedForBoostedTausPAT','subJetsForSeedingBoostedTausPAT'),
        removeOverLap = cms.bool(True),
    )

    updatedBoostedTauName = "slimmedBoostedTausNewID"
    runTauID(process, outputTauCollection=updatedBoostedTauName, inputTauCollection="cleanedSlimmedTausBoosted",
             toKeep = [ "2017v2", "dR0p32017v2", "newDM2017v2", "deepTau2017v2p1" ])

    process.boostedSequence = cms.Sequence(
        process.ca8PFJetsCHSprunedForBoostedTausPAT *
        process.cleanedSlimmedTausBoosted *
        getattr(process, updatedBoostedTauName + 'rerunMvaIsolationSequence') *
        getattr(process, updatedBoostedTauName))
    boostedTaus_InputTag = cms.InputTag(updatedBoostedTauName)

# boostedTaus_InputTag = cms.InputTag('slimmedTausBoosted')
if isRun2UL:
    taus_InputTag = cms.InputTag('slimmedTaus')
else:
    taus_InputTag = cms.InputTag('slimmedTausNewID')

if isPhase2:
    process.slimmedElectronsMerged = cms.EDProducer("SlimmedElectronMerger",
    src = cms.VInputTag("slimmedElectrons","slimmedElectronsFromMultiCl")
    )
    electrons_InputTag = cms.InputTag('slimmedElectronsMerged')
    vtx_InputTag = cms.InputTag('offlineSlimmedPrimaryVertices4D')
else:
    electrons_InputTag = cms.InputTag('slimmedElectrons')
    vtx_InputTag = cms.InputTag('offlineSlimmedPrimaryVertices')


tauJetBuilderSetup = cms.PSet(
    genLepton_genJet_dR     = cms.double(0.4),
    genLepton_tau_dR        = cms.double(0.2),
    genLepton_boostedTau_dR = cms.double(0.2),
    genLepton_jet_dR        = cms.double(0.4),
    genLepton_fatJet_dR     = cms.double(0.8),
    genJet_tau_dR           = cms.double(0.4),
    genJet_boostedTau_dR    = cms.double(0.4),
    genJet_jet_dR           = cms.double(0.4),
    genJet_fatJet_dR        = cms.double(0.8),
    tau_boostedTau_dR       = cms.double(0.2),
    tau_jet_dR              = cms.double(0.4),
    tau_fatJet_dR           = cms.double(0.8),
    jet_fatJet_dR           = cms.double(0.8),
    jet_maxAbsEta           = cms.double(3.4),
    fatJet_maxAbsEta        = cms.double(3.8),
    genLepton_cone          = cms.double(0.5),
    genJet_cone             = cms.double(0.5),
    tau_cone                = cms.double(0.5),
    boostedTau_cone         = cms.double(0.5),
    jet_cone                = cms.double(0.8),
    fatJet_cone             = cms.double(0.8),
)

process.tauTupleProducer = cms.EDAnalyzer('TauTupleProducer',
    isMC                     = cms.bool(not isData),
    isEmbedded               = cms.bool(isEmbedded),
    requireGenMatch          = cms.bool(options.requireGenMatch),
    requireGenORRecoTauMatch = cms.bool(options.requireGenORRecoTauMatch),
    applyRecoPtSieve         = cms.bool(options.applyRecoPtSieve),
    tauJetBuilderSetup       = tauJetBuilderSetup,

    lheEventProduct    = cms.InputTag('externalLHEProducer'),
    genEvent           = cms.InputTag('generator'),
    genParticles       = cms.InputTag('prunedGenParticles'),
    puInfo             = cms.InputTag('slimmedAddPileupInfo'),
    vertices           = vtx_InputTag,
    rho                = cms.InputTag('fixedGridRhoAll'),
    electrons          = electrons_InputTag,
    muons              = cms.InputTag('slimmedMuons'),
    taus               = taus_InputTag,
    boostedTaus        = boostedTaus_InputTag,
    jets               = cms.InputTag('slimmedJets'),
    fatJets            = cms.InputTag('slimmedJetsAK8'),
    pfCandidates       = cms.InputTag('packedPFCandidates'),
    isoTracks          = cms.InputTag('isolatedTracks'),
    lostTracks         = cms.InputTag('lostTracks'),
    genJets            = cms.InputTag('slimmedGenJets'),
    genJetFlavourInfos = cms.InputTag('slimmedGenJetsFlavourInfos'),
)

process.tupleProductionSequence = cms.Sequence(process.tauTupleProducer)

if isPhase2:
    process.p = cms.Path(
        getattr(process, 'rerunMvaIsolationSequence') *
        getattr(process, updatedTauName) *
        process.tupleProductionSequence
    )
elif isRun2UL:
    process.p = cms.Path(process.tupleProductionSequence)
else:
    process.p = cms.Path(
        getattr(process, updatedTauName + 'rerunMvaIsolationSequence') *
        getattr(process, updatedTauName) *
        process.tupleProductionSequence
    )

if isPhase2:
    process.p.insert(0, process.slimmedElectronsMerged)

if isRun2PreUL:
    process.p.insert(2, process.boostedSequence)

process.load('FWCore.MessageLogger.MessageLogger_cfi')
x = process.maxEvents.input.value()
x = x if x >= 0 else 10000
process.MessageLogger.cerr.FwkReport.reportEvery = max(1, min(1000, x // 10))

if options.dumpPython:
    print process.dumpPython()
