import importlib
import FWCore.ParameterSet.Config as cms
from FWCore.ParameterSet.VarParsing import VarParsing
from Configuration.Eras.Era_Run3_cff import Run3
import os

# to keep sampleType, fileList, fileNamePrefix, tupleOutput, lumiFile, eventList and dumpPython

options = VarParsing('analysis')
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
options.register('dumpPython', False, VarParsing.multiplicity.singleton, VarParsing.varType.bool,
                 "Dump full config into stdout.")
options.parseArguments()

sampleConfig = importlib.import_module('TauMLTools.Production.sampleConfig')
isData = sampleConfig.IsData(options.sampleType)
isEmbedded = sampleConfig.IsEmbedded(options.sampleType)
period = sampleConfig.GetPeriod(options.sampleType)
period_cfg = sampleConfig.GetPeriodCfg(options.sampleType)

processName = 'tupleProduction'
process = cms.Process(processName, period_cfg)
process.options = cms.untracked.PSet()
process.options.wantSummary = cms.untracked.bool(False)
process.options.allowUnscheduled = cms.untracked.bool(True)
process.options.numberOfStreams = cms.untracked.uint32(0)

# import of standard configurations
process.load('Configuration.StandardSequences.Services_cff')
process.load('SimGeneral.HepPDTESSource.pythiapdt_cfi')
process.load('FWCore.MessageService.MessageLogger_cfi')
process.load('Configuration.EventContent.EventContent_cff')
process.load('SimGeneral.MixingModule.mixNoPU_cfi')
process.load('Configuration.StandardSequences.GeometryRecoDB_cff')
process.load('Configuration.StandardSequences.MagneticField_cff')
process.load('HLTrigger.Configuration.HLT_User_cff')
process.load('Configuration.StandardSequences.EndOfProcess_cff')
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')


process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(-1),
    output = cms.optional.untracked.allowed(cms.int32,cms.PSet)
)

if isData:
    process.source = cms.Source("PoolSource",
        fileNames = cms.untracked.vstring('file:tau_hlt_RAW.root'),
        secondaryFileNames = cms.untracked.vstring()
    )
else:
    process.source = cms.Source("PoolSource",
        fileNames = cms.untracked.vstring('file:/eos/home-v/vdamante/F2760E46-A3DB-DA4F-A6EC-525C10EDCBC7.root'),
        #fileNames = cms.untracked.vstring('/store/mc/Run3Winter20DRPremixMiniAOD/VBFHToTauTau_M125_TuneCUETP8M1_14TeV_powheg_pythia8/GEN-SIM-RAW/110X_mcRun3_2021_realistic_v6-v1/20000/F2760E46-A3DB-DA4F-A6EC-525C10EDCBC7.root'),
        secondaryFileNames = cms.untracked.vstring()
    )

process.options = cms.untracked.PSet(
    FailPath = cms.untracked.vstring(),
    IgnoreCompletely = cms.untracked.vstring(),
    Rethrow = cms.untracked.vstring(),
    SkipEvent = cms.untracked.vstring(),
    allowUnscheduled = cms.obsolete.untracked.bool,
    canDeleteEarly = cms.untracked.vstring(),
    emptyRunLumiMode = cms.obsolete.untracked.string,
    eventSetup = cms.untracked.PSet(
        forceNumberOfConcurrentIOVs = cms.untracked.PSet(
            allowAnyLabel_=cms.required.untracked.uint32
        ),
        numberOfConcurrentIOVs = cms.untracked.uint32(1)
    ),
    fileMode = cms.untracked.string('FULLMERGE'),
    forceEventSetupCacheClearOnNewRun = cms.untracked.bool(False),
    makeTriggerResults = cms.obsolete.untracked.bool,
    numberOfConcurrentLuminosityBlocks = cms.untracked.uint32(1),
    numberOfConcurrentRuns = cms.untracked.uint32(1),
    numberOfStreams = cms.untracked.uint32(0),
    numberOfThreads = cms.untracked.uint32(1),
    printDependencies = cms.untracked.bool(False),
    sizeOfStackForThreadsInKB = cms.optional.untracked.uint32,
    throwIfIllegalParameter = cms.untracked.bool(True),
    wantSummary = cms.untracked.bool(False)
)

process.configurationMetadata = cms.untracked.PSet(
    annotation = cms.untracked.string('tau_hlt nevts:-1'),
    name = cms.untracked.string('Applications'),
    version = cms.untracked.string('$Revision: 1.19 $')
)

if isData:
    process.RECOSIMoutput = cms.OutputModule("PoolOutputModule",
        dataset = cms.untracked.PSet(
            dataTier = cms.untracked.string(''),
            filterName = cms.untracked.string('')
        ),
        fileName = cms.untracked.string('tau_hlt_HLT.root'),
        outputCommands = process.RECOSIMEventContent.outputCommands,
        splitLevel = cms.untracked.int32(0)
    )

from HLTrigger.Configuration.CustomConfigs import ProcessName
process = ProcessName(process)

from Configuration.AlCa.GlobalTag import GlobalTag
process.GlobalTag = GlobalTag(process.GlobalTag, 'auto:run3_data_GRun', '')

process.endjob_step = cms.EndPath(process.endOfProcess)
if isData:
    process.RECOSIMoutput_step = cms.EndPath(process.RECOSIMoutput)

process.schedule = cms.Schedule()
process.schedule.extend(process.HLTSchedule)
if isData;
    process.schedule.extend([process.endjob_step,process.RECOSIMoutput_step])
else:
    process.schedule.extend([process.endjob_step])
from PhysicsTools.PatAlgos.tools.helpers import associatePatAlgosToolsTask
associatePatAlgosToolsTask(process)

from TauMLTools.Production.myHLT_Train import update
process = update(process)

if isData:
    from HLTrigger.Configuration.customizeHLTforCMSSW import customiseFor2018Input
    process = customiseFor2018Input(process)
else:
    from HLTrigger.Configuration.customizeHLTforMC import customizeHLTforMC
    process = customizeHLTforMC(process)

from Configuration.StandardSequences.earlyDeleteSettings_cff import customiseEarlyDelete
process = customiseEarlyDelete(process)


process.load('FWCore.MessageLogger.MessageLogger_cfi')
x = process.maxEvents.input.value()
x = x if x >= 0 else 10000
process.MessageLogger.cerr.FwkReport.reportEvery = max(1, min(1000, x // 10))

if options.dumpPython:
    print process.dumpPython()
