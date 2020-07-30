# Produce TauTuple.

import re
import importlib
import FWCore.ParameterSet.Config as cms
from FWCore.ParameterSet.VarParsing import VarParsing
import RecoTauTag.Configuration.tools.adaptToRunAtMiniAOD as tauAtMiniTools


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
                 "Store only taus/jets that have GenLeptonMatch or GenQcdMatch.")
options.register('reclusterJets', True, VarParsing.multiplicity.singleton, VarParsing.varType.bool,
                " If 'reclusterJets' set true a new collection of uncorrected ak4PFJets is built to seed taus (as at RECO), otherwise standard slimmedJets are used")
options.register('rerunTauReco', False, VarParsing.multiplicity.singleton, VarParsing.varType.bool,
                "If true, tau reconstruction is re-run on MINIAOD with a larger signal cone and no DM finding filter")
options.parseArguments()

sampleConfig = importlib.import_module('TauMLTools.Production.sampleConfig')
isData = sampleConfig.IsData(options.sampleType)
period = sampleConfig.GetPeriod(options.sampleType)
period_cfg = sampleConfig.GetPeriodCfg(options.sampleType)

processName = 'tupleProduction'
process = cms.Process(processName, period_cfg)
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

    process.combinatoricRecoTaus.builders[0].signalConeSize = cms.string('max(min(0.2, 4.528/(pt()^0.8982)), 0.03)') ## change to quantile 0.95
    process.selectedPatTaus.cut = cms.string('pt > 18.')   ## remove DMFinding filter (was pt > 18. && tauID(\'decayModeFindingNewDMs\')> 0.5)

tauIdConfig = importlib.import_module('TauMLTools.Production.runTauIdMVA')
updatedTauName = "slimmedTausNewID"
tauIdEmbedder = tauIdConfig.TauIDEmbedder(
    process, cms, debug = False, updatedTauName = updatedTauName,
    toKeep = [ "2017v2", "dR0p32017v2", "newDM2017v2", "deepTau2017v2p1"]
)
tauIdEmbedder.runTauID(tau_collection = tau_collection)
tauSrc_InputTag = cms.InputTag('slimmedTausNewID')

tauJetdR = 0.3
objectdR = 0.5

process.tauTupleProducer = cms.EDAnalyzer('TauTupleProducer',
    isMC                            = cms.bool(not isData),
    minJetPt                        = cms.double(10.),
    maxJetEta                       = cms.double(3.),
    forceTauJetMatch                = cms.bool(False),
    storeJetsWithoutTau             = cms.bool(options.storeJetsWithoutTau),
    tauJetMatchDeltaRThreshold      = cms.double(tauJetdR),
    objectMatchDeltaRThresholdTau   = cms.double(objectdR),
    objectMatchDeltaRThresholdJet   = cms.double(tauJetdR + objectdR),
    requireGenMatch                 = cms.bool(options.requireGenMatch),

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
    tracks          = cms.InputTag('isolatedTracks'),
)

process.tupleProductionSequence = cms.Sequence(process.tauTupleProducer)

process.p = cms.Path(
    process.rerunMvaIsolationSequence *
    getattr(process, updatedTauName) *
    process.tupleProductionSequence
)

if options.dumpPython:
    print process.dumpPython()
