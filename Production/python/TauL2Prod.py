import FWCore.ParameterSet.Config as cms
from FWCore.ParameterSet.VarParsing import VarParsing
from Configuration.Eras.Era_Run3_cff import Run3
import os

txtList = {
    "DY":["DYToLL_M-50_TuneCP5_14TeV-pythia8.txt"],
    "QCD":["QCD_Pt-15to3000_TuneCP5_Flat_14TeV_pythia8.txt","QCD_Pt-15to7000_TuneCP5_Flat_14TeV_pythia8.txt","QCD_Pt-15to7000_TuneCP5_Flat_14TeV_pythia8_ext.txt","QCD_Pt_120to170_TuneCP5_14TeV_pythia8.txt","QCD_Pt_170to300_TuneCP5_14TeV_pythia8.txt","QCD_Pt_300to470_TuneCP5_14TeV_pythia8.txt","QCD_Pt_30to50_TuneCP5_14TeV_pythia8.txt","QCD_Pt_470to600_TuneCP5_14TeV_pythia8.txt","QCD_Pt_50to80_TuneCP5_14TeV_pythia8.txt","QCD_Pt_600oInf_TuneCP5_14TeV_pythia8.txt","QCD_Pt_80to120_TuneCP5_14TeV_pythia8.txt"],
    "TT":["TTToSemiLeptonic_TuneCP5_14TeV-powheg-pythia8.txt","TT_TuneCP5_14TeV-powheg-pythia8.txt","TT_TuneCP5_14TeV-powheg-pythia8_ext.txt"],
    "VBF":["VBFHToTauTau_M125_TuneCUETP8M1_14TeV_powheg_pythia80.txt"],
    "WJets":["WJetsToLNu_TuneCP5_14TeV-amcatnloFXFX-pythia8.txt"],
    "ZPrime":["ZprimeToTauTau_M-4000_TuneCP5_14TeV-pythia8-tauola.txt"]
}

options = VarParsing('analysis')
options.register('sampleType', '', VarParsing.multiplicity.singleton, VarParsing.varType.string,
                 "Indicates the sample type: Summer16MC, Run2016, ...")
options.register('txtList', '', VarParsing.multiplicity.singleton, VarParsing.varType.string,
                 "List of txt of list of root files to process.")
options.register('txtListPrefix', '', VarParsing.multiplicity.singleton, VarParsing.varType.string,
                 "Directory of list of txt of list of root files to process.")
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


dataDict = {"Run3MC" : False, "Run2Data" : True}
isData = dataDict[options.sampleType]
processName = 'reHLT'
process = cms.Process(processName, Run3)

# import of standard configurations
process.load('Configuration.StandardSequences.Services_cff')
process.load('SimGeneral.HepPDTESSource.pythiapdt_cfi')
process.load('FWCore.MessageService.MessageLogger_cfi')
process.load('Configuration.EventContent.EventContent_cff')
process.load('Configuration.StandardSequences.GeometryRecoDB_cff')
process.load('Configuration.StandardSequences.MagneticField_cff')
process.load('HLTrigger.Configuration.HLT_User_cff')
process.load('Configuration.StandardSequences.EndOfProcess_cff')
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')

if not isData:
    process.load('SimGeneral.MixingModule.mixNoPU_cfi')

process.source = cms.Source('PoolSource', fileNames = cms.untracked.vstring())
# Input source

from TauMLTools.Production.readFileList import *
if len(options.fileList) > 0:
    readFileList(process.source.fileNames, options.fileList, options.fileNamePrefix)
elif len(options.inputFiles) > 0:
    addFilesToList(process.source.fileNames, options.inputFiles, options.fileNamePrefix)
elif len(options.txtList)>0:
    readFilesFileList(process.source.fileNames, txtList[options.txtList], options.txtListPrefix, options.fileNamePrefix)
else:
    if isData:
        process.source.fileNames = cms.untracked.vstring('/store/data/Run2018D/EphemeralHLTPhysics1/RAW/v1/000/325/113/00000/EA32A985-9BAF-D044-96FF-D59033C22A09.root')
    else:
        process.source.fileNames = cms.untracked.vstring('file:/eos/home-v/vdamante/F2760E46-A3DB-DA4F-A6EC-525C10EDCBC7.root')

if len(options.lumiFile) > 0:
    import FWCore.PythonUtilities.LumiList as LumiList
    process.source.lumisToProcess = LumiList.LumiList(filename = options.lumiFile).getVLuminosityBlockRange()

if options.eventList != '':
    process.source.eventsToProcess = cms.untracked.VEventRange(re.split(',', options.eventList))


process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(options.maxEvents),
    output = cms.optional.untracked.allowed(cms.int32,cms.PSet)
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

# Production Info
process.configurationMetadata = cms.untracked.PSet(
    annotation = cms.untracked.string('tau_hlt nevts:-1'),
    name = cms.untracked.string('Applications'),
    version = cms.untracked.string('$Revision: 1.19 $')
)

# Output definition

# Additional output definition

# Other statements
from HLTrigger.Configuration.CustomConfigs import ProcessName
process = ProcessName(process)

from Configuration.AlCa.GlobalTag import GlobalTag
GlobalTagName = 'auto:run3_data_GRun' if isData else 'auto:run3_mc_GRun'
process.GlobalTag = GlobalTag(process.GlobalTag, GlobalTagName, '')

# Path and EndPath definitions
process.endjob_step = cms.EndPath(process.endOfProcess)

# Schedule definition
process.schedule = cms.Schedule()
process.schedule.extend(process.HLTSchedule)
process.schedule.extend([process.endjob_step])
from PhysicsTools.PatAlgos.tools.helpers import associatePatAlgosToolsTask
associatePatAlgosToolsTask(process)

# customisation of the process.


# Automatic addition of the customisation function

if isData:
    # Customisation from command line
    from HLTrigger.Configuration.customizeHLTforCMSSW import customisePixelGainForRun2Input,synchronizeHCALHLTofflineRun3on2018data
    process = customisePixelGainForRun2Input(process)
    process = synchronizeHCALHLTofflineRun3on2018data(process)
else:
    from HLTrigger.Configuration.customizeHLTforMC import customizeHLTforMC
    process = customizeHLTforMC(process)

from TauMLTools.Production.myHLT_Train import update
process = update(process,isData, options.outputFile)
print options.outputFile

# End of customisation functions

# Customisation from command line

process.load('FWCore.MessageLogger.MessageLogger_cfi')
x = process.maxEvents.input.value()
x = x if x >= 0 else 10000
process.MessageLogger.cerr.FwkReport.reportEvery = max(1, min(1000, x // 10))

# Add early deletion of temporary data products to reduce peak memory need
from Configuration.StandardSequences.earlyDeleteSettings_cff import customiseEarlyDelete
process = customiseEarlyDelete(process)

if options.dumpPython:
    print process.dumpPython()

# End adding early deletion
