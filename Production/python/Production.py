# Produce TauTuple.

import re
import importlib
import FWCore.ParameterSet.Config as cms
from FWCore.ParameterSet.VarParsing import VarParsing
from RecoMET.METPUSubtraction.deepMETProducer_cfi import deepMETProducer
import os


options = VarParsing('analysis')
options.register('sampleType', '', VarParsing.multiplicity.singleton, VarParsing.varType.string,
                 "Indicates the sample type")
options.register('era', '', VarParsing.multiplicity.singleton, VarParsing.varType.string,
                 "Indicates the era")
options.register('fileList', '', VarParsing.multiplicity.singleton, VarParsing.varType.string,
                 "List of root files to process.")
options.register('fileNamePrefix', '', VarParsing.multiplicity.singleton, VarParsing.varType.string,
                 "Prefix to add to input file names.")
options.register('output', 'eventTuple.root', VarParsing.multiplicity.singleton, VarParsing.varType.string,
                     "Event tuple file.")
options.register('lumiFile', '', VarParsing.multiplicity.singleton, VarParsing.varType.string,
                 "JSON file with lumi mask.")
options.register('eventList', '', VarParsing.multiplicity.singleton, VarParsing.varType.string,
                 "List of events to process.")
options.register('dumpPython', False, VarParsing.multiplicity.singleton, VarParsing.varType.bool,
                 "Dump full config into stdout.")
options.register('numberOfThreads', 1, VarParsing.multiplicity.singleton, VarParsing.varType.int,
                 "Number of threads.")
options.register('selector', 'None', VarParsing.multiplicity.singleton, VarParsing.varType.string,
                 "Name of the tauJet selector.")
options.register('triggers', '', VarParsing.multiplicity.singleton, VarParsing.varType.string,
                 "Store only events that pass the specified HLT paths.")
options.register("disabledBranches", [], VarParsing.multiplicity.list, VarParsing.varType.string,
                 "Not store following branches in tupleOutput file.")
options.register("enabledBranches", [], VarParsing.multiplicity.list, VarParsing.varType.string,
                 "Branches to store in tupleOutput file (if empty list: stores all the branches).")
options.register('storeJetsWithoutTau', False, VarParsing.multiplicity.singleton, VarParsing.varType.bool,
                 "Store jets that don't match to any pat::Tau.")
options.register('requireGenMatch', True, VarParsing.multiplicity.singleton, VarParsing.varType.bool,
                 "Store only tau jets that have genLepton_index >= 0 or genJet_index >= 0.")
options.register('requireGenORRecoTauMatch', False, VarParsing.multiplicity.singleton, VarParsing.varType.bool,
                 """Store only tau jets that satisfy the following condition:
                    tau_index >= 0 || boostedTau_index >= 0 || (genLepton_index >= 0 && genLepton_kind == 5)""")
options.register('applyRecoPtSieve', False, VarParsing.multiplicity.singleton, VarParsing.varType.bool,
                 "Randomly drop jet->tau fakes depending on reco tau pt to balance contributions from low and higt pt.")
options.register('reclusterJets', True, VarParsing.multiplicity.singleton, VarParsing.varType.bool,
                " If 'reclusterJets' set true a new collection of uncorrected ak4PFJets is built to seed taus (as at RECO), otherwise standard slimmedJets are used")
options.register('rerunTauReco', '', VarParsing.multiplicity.singleton, VarParsing.varType.string,
                """If not empty, tau reconstruction is re-run on MINIAOD (available modes:
                   1) signalCone - larger signal cone and no DM finding filter,
                   2) displacedTau - loose max Impact Parameter and signal quality cuts)""")
options.register('runTauSpinner', False, VarParsing.multiplicity.singleton, VarParsing.varType.bool,
                 "Run TauPOGSpinner to store CP weights")

options.parseArguments()

import importlib.util
import os
import sys

def load(module_file, default_path):
  module_path = os.path.join(default_path, module_file)
  if not os.path.exists(module_path):
    module_path = os.path.join(os.path.dirname(__file__), module_file)
    if not os.path.exists(module_path):
      module_path = os.path.join(os.getenv("CMSSW_BASE"), 'src', module_file)
      if not os.path.exists(module_path):
        raise RuntimeError(f"Cannot find path to {module_file}.")

  module_name, module_ext = os.path.splitext(module_file)
  spec = importlib.util.spec_from_file_location(module_name, module_path)
  module = importlib.util.module_from_spec(spec)
  sys.modules[module_name] = module
  spec.loader.exec_module(module)
  return module


sampleConfig  = load('sampleConfig.py', os.path.join(os.getenv("CMSSW_BASE"), 'src/TauMLTools/Production/python'))
from sampleConfig import Era, SampleType

sampleType = SampleType[options.sampleType]
era = Era[options.era]
isData = sampleType == SampleType.Data
isEmbedded = sampleType == SampleType.Embedded
isRun2 = sampleConfig.isRun2(era)
isRun3 = sampleConfig.isRun3(era)
isPhase2 = sampleConfig.isPhase2(era)
era_cfg = sampleConfig.getEraCfg(era)
globalTag = sampleConfig.getGlobalTag(era, sampleType)

if not (isRun2 or isRun3 or isPhase2):
    raise RuntimeError("Support for era = {} is not implemented".format(era.name))

processName = 'tupleProduction'
process = cms.Process(processName, era_cfg)
process.options = cms.untracked.PSet()
process.options.wantSummary = cms.untracked.bool(False)
process.options.allowUnscheduled = cms.untracked.bool(True)
process.options.numberOfThreads = cms.untracked.uint32(options.numberOfThreads)
process.options.numberOfStreams = cms.untracked.uint32(0)

process.load('Configuration.StandardSequences.MagneticField_cff')

# TauSpinner
process.TauSpinnerReco = cms.EDProducer( "TauPOGSpinner",
    isReco = cms.bool(True),
    isTauolaConfigured = cms.bool(False),
    isLHPDFConfigured = cms.bool(False),
    LHAPDFname = cms.untracked.string('NNPDF30_nlo_as_0118'),
    CMSEnergy = cms.double(13000.0),
    gensrc = cms.InputTag('prunedGenParticles')
)

process.RandomNumberGeneratorService = cms.Service('RandomNumberGeneratorService',
                                                   TauSpinnerReco = cms.PSet(
    initialSeed = cms.untracked.uint32(123456789),
    engineName = cms.untracked.string('HepJamesRandom')
    )
)

# DeepMET
process.deepMETProducer = deepMETProducer.clone()

# include Phase2 specific configuration only after 11_0_X
if isPhase2:
    process.load('Configuration.Geometry.GeometryExtended2026D49Reco_cff')
elif isRun2 or isRun3:
    process.load('Configuration.Geometry.GeometryRecoDB_cff')

process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')
from Configuration.AlCa.GlobalTag import GlobalTag
process.GlobalTag = GlobalTag(process.GlobalTag, globalTag, '')

process.source = cms.Source('PoolSource', fileNames = cms.untracked.vstring())
process.TFileService = cms.Service('TFileService', fileName = cms.string(options.output) )
process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(-1) )


readFileList  = load('readFileList.py', os.path.join(os.getenv("CMSSW_BASE"), 'src/TauMLTools/Production/python'))
from readFileList import *

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

    import RecoTauTag.Configuration.tools.adaptToRunAtMiniAOD as tauAtMiniConfig
    tauAtMiniTools = tauAtMiniConfig.adaptToRunAtMiniAOD(process, runBoosted=False)
    tauAtMiniTools.addTauReReco()
    tauAtMiniTools.adaptTauToMiniAODReReco(reclusterJets = options.reclusterJets)

    if isData:
        from PhysicsTools.PatAlgos.tools.coreTools import runOnData
        runOnData(process, names = ['Taus'], outputModules = [])

    import TauMLTools.Production.setupTauReReco as setupTauReReco
    if options.rerunTauReco == "signalCone":
        setupTauReReco.reReco_SigCone(process)
    elif options.rerunTauReco == "displacedTau":
        setupTauReReco.reReco_DisTau(process)
    else:
        raise RuntimeError('rerunTauReco = "{}" mode is not supported.'.format(options.rerunTauReco))

# include Phase2 specific configuration only after 11_0_X
if isPhase2:
    tauIdConfig = importlib.import_module('RecoTauTag.RecoTau.tools.runTauIdMVA')
    updatedTauName = "slimmedTausNewID"
    tauIdEmbedder = tauIdConfig.TauIDEmbedder(
        process, cms, originalTauName = tau_collection, updatedTauName = updatedTauName,
        toKeep = [ "2017v2", "dR0p32017v2", "newDM2017v2", "deepTau2017v2p1", "newDMPhase2v1"]
    )
    tauIdEmbedder.runTauID() # note here, that with the official CMSSW version of 'runTauIdMVA' slimmedTaus are hardcoded as input tau collection
elif isRun2 or isRun3:
    tauIdConfig = importlib.import_module('RecoTauTag.RecoTau.tools.runTauIdMVA')
    updatedTauName = "slimmedTausNewID"
    tauIdEmbedder = tauIdConfig.TauIDEmbedder(
        process, cms, originalTauName = tau_collection, updatedTauName = updatedTauName,
        toKeep = [ "deepTau2017v2p1", "deepTau2018v2p5" ]
    )
    tauIdEmbedder.runTauID()

boostedTaus_InputTag = cms.InputTag('slimmedTausBoosted')
taus_InputTag = cms.InputTag('slimmedTausNewID')

if isPhase2:
    process.slimmedElectronsMerged = cms.EDProducer("SlimmedElectronMerger",
        src = cms.VInputTag("slimmedElectrons","slimmedElectronsHGC")
    )
    electrons_InputTag = cms.InputTag('slimmedElectronsMerged')
    vtx_InputTag = cms.InputTag('offlineSlimmedPrimaryVertices4D')
elif isRun2 or isRun3:
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
    disabledBranches         = cms.vstring(options.disabledBranches),
    enabledBranches          = cms.vstring(options.enabledBranches),
    tauJetBuilderSetup       = tauJetBuilderSetup,
    selector		     = cms.string(options.selector),

    lheEventProduct    = cms.InputTag('externalLHEProducer'),
    genEvent           = cms.InputTag('generator'),
    genParticles       = cms.InputTag('prunedGenParticles'),
    puInfo             = cms.InputTag('slimmedAddPileupInfo'),
    vertices           = vtx_InputTag,
    secondVertices     = cms.InputTag('slimmedSecondaryVertices'),
    rho                = cms.InputTag('fixedGridRhoAll'),
    electrons          = electrons_InputTag,
    photons	       = cms.InputTag('slimmedPhotons'),
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
    METs               = cms.InputTag('slimmedMETs'),
    puppiMETs	       = cms.InputTag('slimmedMETsPuppi'),
    deepMETs           = cms.InputTag('deepMETProducer', ''),
    genMETs	       = cms.InputTag('genMetTrue'),
    triggerResults     = cms.InputTag('TriggerResults', '', 'HLT'),
    triggerObjects     = cms.InputTag('slimmedPatTrigger'),
    tauSpinnerWTEven   = cms.InputTag('TauSpinnerReco','TauSpinnerWTEven'),
    tauSpinnerWTOdd    = cms.InputTag('TauSpinnerReco','TauSpinnerWTOdd'),
    tauSpinnerWTMM     = cms.InputTag('TauSpinnerReco','TauSpinnerWTMM'),

)

process.tupleProductionSequence = cms.Sequence(process.tauTupleProducer)

if isPhase2:
    process.p = cms.Path(
        process.slimmedElectronsMerged +
        getattr(process, 'rerunMvaIsolationSequence') +
        getattr(process, updatedTauName) +
        process.tupleProductionSequence
    )
elif isRun2 or isRun3:
    process.p = cms.Path(
        process.deepMETProducer +
        getattr(process, 'rerunMvaIsolationSequence') +
        getattr(process, updatedTauName) +
        process.tupleProductionSequence
    )
    if options.runTauSpinner:
        process.p.insert(0, process.TauSpinnerReco)
    if options.rerunTauReco:
        process.p.insert(0, getattr(process,'miniAODTausSequence'))

if len(options.triggers) > 0:
    hlt_paths = options.triggers.split(',')
    process.hltFilter = cms.EDFilter('TriggerResultsFilter',
        hltResults = cms.InputTag('TriggerResults', '', 'HLT'),
        l1tResults = cms.InputTag(''),
        l1tIgnoreMaskAndPrescale = cms.bool(False),
        throw = cms.bool(True),
        triggerConditions = cms.vstring(hlt_paths),
    )
    process.p.insert(0, process.hltFilter)

process.load('FWCore.MessageLogger.MessageLogger_cfi')
x = process.maxEvents.input.value()
x = x if x >= 0 else 10000
process.MessageLogger.cerr.FwkReport.reportEvery = max(1, min(1000, x // 10))

if options.dumpPython:
    print(process.dumpPython())
