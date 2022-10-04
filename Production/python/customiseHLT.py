# How to run:
# hltGetConfiguration /dev/CMSSW_12_4_0/GRun --globaltag auto:phase1_2022_realistic --mc --unprescale --no-output --max-events 100 --eras Run3 --l1-emulator FullMC --l1 L1Menu_Collisions2022_v1_3_0-d1_xml --customise TauMLTools/Production/customiseHLT.customise --input /store/mc/Run3Summer21DRPremix/TT_TuneCP5_14TeV-powheg-pythia8/GEN-SIM-DIGI-RAW/120X_mcRun3_2021_realistic_v6-v2/2540000/b354245e-d8bc-424d-b527-58815586a6a5.root > hltRun3Summer21MC.py
# cmsRun hltRun3Summer21MC.py

import FWCore.ParameterSet.Config as cms

def customise(process):
  process.NANOAODSIMoutput = cms.OutputModule("NanoAODOutputModule",
      compressionAlgorithm = cms.untracked.string('LZMA'),
      compressionLevel = cms.untracked.int32(9),
      dataset = cms.untracked.PSet(
          dataTier = cms.untracked.string('NANOAODSIM'),
          filterName = cms.untracked.string('')
      ),
      fileName = cms.untracked.string('file:nano.root'),
      outputCommands  = cms.untracked.vstring(
          'drop *',
          'keep nanoaodFlatTable_*Table_*_*',
      )
  )
  process.load('PhysicsTools.NanoAOD.nano_cff')
  from PhysicsTools.NanoAOD.nano_cff import nanoAOD_customizeMC

  #call to customisation function nanoAOD_customizeMC imported from PhysicsTools.NanoAOD.nano_cff
  process = nanoAOD_customizeMC(process)

  process.nanoAOD_step = cms.Path(process.nanoSequenceMC)
  process.NANOAODSIMoutput_step = cms.EndPath(process.NANOAODSIMoutput)

  process.nanoTableTaskFS = cms.Task(process.genParticleTablesTask, process.genParticleTask)
  process.nanoSequenceFS = cms.Sequence(process.nanoTableTaskFS)
  process.nanoSequenceMC = cms.Sequence(process.nanoTableTaskFS)
  process.finalGenParticles.src = cms.InputTag("genParticles")

  process.MessageLogger.cerr.FwkReport.reportEvery = 100
  process.finalGenParticles.select = cms.vstring(
      "drop *",
      "keep++ abs(pdgId) == 15", # keep full decay chain for taus
      "+keep abs(pdgId) == 11 || abs(pdgId) == 13 || abs(pdgId) == 15", #keep leptons, with at most one mother back in the history
      "+keep+ abs(pdgId) == 6 || abs(pdgId) == 23 || abs(pdgId) == 24 || abs(pdgId) == 25 || abs(pdgId) == 35 || abs(pdgId) == 39  || abs(pdgId) == 9990012 || abs(pdgId) == 9900012",   # keep VIP particles
      "drop abs(pdgId)= 2212 && abs(pz) > 1000", #drop LHC protons accidentally added by previous keeps
  )

  from PhysicsTools.NanoAOD.common_cff import Var
  process.genParticleTable.variables.vertex_x = Var('vertex().x', float, precision=10, doc='x coordinate of the gen particle production vertex')
  process.genParticleTable.variables.vertex_y = Var('vertex().y', float, precision=10, doc='y coordinate of the gen particle production vertex')
  process.genParticleTable.variables.vertex_z = Var('vertex().z', float, precision=10, doc='z coordinate of the gen particle production vertex')
  process.genParticleTable.variables.mass.expr = cms.string('mass')
  process.genParticleTable.variables.mass.doc = cms.string('mass')

  process.FastTimerService.printEventSummary = False
  process.FastTimerService.printRunSummary = False
  process.FastTimerService.printJobSummary = False
  process.hltL1TGlobalSummary.DumpTrigSummary = False

  del process.MessageLogger.TriggerSummaryProducerAOD
  del process.MessageLogger.L1GtTrigReport
  del process.MessageLogger.L1TGlobalSummary
  del process.MessageLogger.HLTrigReport
  del process.MessageLogger.FastReport
  del process.MessageLogger.ThroughputService


  process.options.wantSummary = False

  process.schedule.insert(1000000, process.nanoAOD_step)
  process.schedule.insert(1000000, process.NANOAODSIMoutput_step)
  return process