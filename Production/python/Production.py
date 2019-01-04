# Produce TauTuple.

import re
import importlib
import FWCore.ParameterSet.Config as cms
from FWCore.ParameterSet.VarParsing import VarParsing

options = VarParsing('analysis')
options.register('globalTag', '', VarParsing.multiplicity.singleton, VarParsing.varType.string,
                 "Global Tag to use.")
options.register('sampleType', '', VarParsing.multiplicity.singleton, VarParsing.varType.string,
                 "Indicates the sample type: Summer16MC, Run2016, ...")
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
options.register('saveGenTopInfo', False, VarParsing.multiplicity.singleton, VarParsing.varType.bool,
                 "Save generator-level information for top quarks.")
options.register('dumpPython', False, VarParsing.multiplicity.singleton, VarParsing.varType.bool,
                 "Dump full config into stdout.")
options.register('numberOfThreads', 1, VarParsing.multiplicity.singleton, VarParsing.varType.int,
                 "Number of threads.")
options.register('storeJetsWithoutTau', False, VarParsing.multiplicity.singleton, VarParsing.varType.bool,
                 "Store jets that don't match to any pat::Tau.")

options.parseArguments()

sampleConfig = importlib.import_module('TauML.Production.sampleConfig')
isData = sampleConfig.IsData(options.sampleType)
period = sampleConfig.GetPeriod(options.sampleType)

processName = 'tupleProduction'
process = cms.Process(processName)
process.options = cms.untracked.PSet()
process.options.wantSummary = cms.untracked.bool(False)
process.options.allowUnscheduled = cms.untracked.bool(True)
process.options.numberOfThreads = cms.untracked.uint32(options.numberOfThreads)
process.options.numberOfStreams = cms.untracked.uint32(0)

process.load('FWCore.MessageLogger.MessageLogger_cfi')
process.MessageLogger.cerr.FwkReport.reportEvery = 100

process.load('Configuration.StandardSequences.MagneticField_cff')
process.load('Configuration.Geometry.GeometryRecoDB_cff')
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_condDBv2_cff')

process.GlobalTag.globaltag = options.globalTag
process.source = cms.Source('PoolSource', fileNames = cms.untracked.vstring())
process.TFileService = cms.Service('TFileService', fileName = cms.string(options.tupleOutput) )
process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(0) )

if len(options.fileList) > 0:
    from AnalysisTools.Run.readFileList import *
    readFileList(process.source.fileNames, options.fileList, options.fileNamePrefix)
    process.maxEvents.input = options.maxEvents

if len(options.lumiFile) > 0:
    import FWCore.PythonUtilities.LumiList as LumiList
    process.source.lumisToProcess = LumiList.LumiList(filename = options.lumiFile).getVLuminosityBlockRange()

if options.eventList != '':
    process.source.eventsToProcess = cms.untracked.VEventRange(re.split(',', options.eventList))

if period == 'Run2016':
    tauSrc_InputTag = cms.InputTag('slimmedTaus')

if period == 'Run2017':
    import RecoTauTag.RecoTau.tools.runTauIdMVA as tauIdConfig
    updatedTauName = "slimmedTausNewID"
    tauIdEmbedder = tauIdConfig.TauIDEmbedder(
        process, cms, debug = False, updatedTauName = updatedTauName,
        toKeep = [ "2017v2", "dR0p32017v2", "newDM2017v2", "2016v1", "newDM2016v1",
                   "deepTau2017v1", "DPFTau_2016_v0", "againstEle2018", ]
    )
    tauIdEmbedder.runTauID()
    tauSrc_InputTag = cms.InputTag('slimmedTausNewID')

process.topGenSequence = cms.Sequence()
if options.saveGenTopInfo:
    process.load("TopQuarkAnalysis.TopEventProducers.sequences.ttGenEvent_cff")
    process.decaySubset.fillMode = cms.string("kME")
    process.initSubset.src = cms.InputTag('prunedGenParticles')
    process.decaySubset.src = cms.InputTag('prunedGenParticles')
    process.decaySubset.runMode = cms.string("Run2")
    process.topGenSequence += process.makeGenEvt

tauJetdR = 0.2
objectdR = 0.5

process.tauTupleProducer = cms.EDAnalyzer('TauTupleProducer',
    isMC                            = cms.bool(not isData),
    minJetPt                        = cms.double(10.),
    maxJetEta                       = cms.double(3.),
    forceTauJetMatch                = cms.bool(False),
    storeJetsWithoutTau             = cms.bool(options.storeJetsWithoutTau),
    tauJetMatchDeltaRThreshold      = cms.double(tauJetdR),
    objectMatchDeltaR2ThresholdTau  = cms.double(objectdR),
    objectMatchDeltaR2ThresholdJet  = cms.double(tauJetdR + objectdR),

    lheEventProduct = cms.InputTag('externalLHEProducer'),
    genEvent        = cms.InputTag('generator'),
    genParticles    = cms.InputTag('prunedGenParticles'),
    puInfo          = cms.InputTag('slimmedAddPileupInfo'),
    vertices        = cms.InputTag('offlineSlimmedPrimaryVertices'),
    rho             = cms.InputTag('fixedGridRhoAll'),
    electrons       = cms.InputTag('slimmedElectrons'),
    muons           = cms.InputTag('slimmedMuons'),
    taus            = tauSrc_InputTag,
    jets            = cms.InputTag('slimmedJets'),
    pfCandidates    = cms.InputTag('packedPFCandidates'),
)

process.tupleProductionSequence = cms.Sequence(process.tauTupleProducer)

if period == 'Run2016':
    process.p = cms.Path(
        process.topGenSequence *
        process.tupleProductionSequence
    )

if period == 'Run2017':
    process.p = cms.Path(
        process.rerunMvaIsolationSequence *
        getattr(process, updatedTauName) *
        process.topGenSequence *
        process.tupleProductionSequence
    )

if options.dumpPython:
    print process.dumpPython()
