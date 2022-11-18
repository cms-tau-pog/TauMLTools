# How to run:
# hltGetConfiguration /dev/CMSSW_12_4_0/GRun --globaltag auto:phase1_2022_realistic --mc --unprescale --no-output --max-events 100 --eras Run3 --l1-emulator FullMC --l1 L1Menu_Collisions2022_v1_3_0-d1_xml --customise TauMLTools/Production/customiseHLT.customise --input /store/mc/Run3Summer21DRPremix/TT_TuneCP5_14TeV-powheg-pythia8/GEN-SIM-DIGI-RAW/120X_mcRun3_2021_realistic_v6-v2/2540000/b354245e-d8bc-424d-b527-58815586a6a5.root > hltRun3Summer21MC.py
# cmsRun hltRun3Summer21MC.py
# file dataset=/store/mc/Run3Summer21DRPremix/TT_TuneCP5_14TeV-powheg-pythia8/GEN-SIM-DIGI-RAW/120X_mcRun3_2021_realistic_v6-v2/2540000/00c642d1-bf7e-477d-91e1-4dd9ce2c8099.root
import FWCore.ParameterSet.Config as cms
from PhysicsTools.NanoAOD.common_cff import Var, P4Vars

def customiseGenParticles(process):
  def pdgOR(pdgs):
    abs_pdgs = [ f'abs(pdgId) == {pdg}' for pdg in pdgs ]
    return '( ' + ' || '.join(abs_pdgs) + ' )'

  leptons = pdgOR([ 11, 13, 15 ])
  important_particles = pdgOR([ 6, 23, 24, 25, 35, 39, 9990012, 9900012 ])
  process.finalGenParticles.select = [
    'drop *',
    'keep++ statusFlags().isLastCopy() && ' + leptons,
    '+keep statusFlags().isFirstCopy() && ' + leptons,
    'keep+ statusFlags().isLastCopy() && ' + important_particles,
    '+keep statusFlags().isFirstCopy() && ' + important_particles,
    "drop abs(pdgId) == 2212 && abs(pz) > 1000", #drop LHC protons accidentally added by previous keeps
  ]

  for coord in [ 'x', 'y', 'z']:
    setattr(process.genParticleTable.variables, 'v'+ coord,
            Var(f'vertex().{coord}', float, precision=10,
                doc=f'{coord} coordinate of the gen particle production vertex'))
  process.genParticleTable.variables.mass.expr = cms.string('mass')
  process.genParticleTable.variables.mass.doc = cms.string('mass')

  return process


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

  process.nanoAOD_step = cms.Path(process.HLTBeginSequence \
    + process.HLTL2TauTagNNSequence \
    + process.HLTGlobalPFTauHPSSequence \
    + process.HLTHPSDeepTauPFTauSequenceForVBFIsoTau \
    + process.nanoSequenceMC)
  process.NANOAODSIMoutput_step = cms.EndPath(process.NANOAODSIMoutput)

  process.tauTable = cms.EDProducer("SimpleCandidateFlatTableProducer",
    src = cms.InputTag( "hltHpsPFTauProducer" ),
    cut = cms.string(""),
    name= cms.string("Tau"),
    doc = cms.string("HLT taus"),
    singleton = cms.bool(False), # the number of entries is variable
    extension = cms.bool(False), # this is the main table
    variables = cms.PSet(
      P4Vars,
      charge = Var("charge", int, doc="electric charge"),
    )
  )
  # cms.EDProducer : an object that produces a new data object

  process.pfCandTable = cms.EDProducer("SimpleCandidateFlatTableProducer",
    src = cms.InputTag( "hltParticleFlowForTaus" ),
    cut = cms.string(""),
    name= cms.string("PFCand"),
    doc = cms.string("HLT PF candidates for taus"),
    singleton = cms.bool(False), # the number of entries is variable
    extension = cms.bool(False), # this is the main table
    variables = cms.PSet(
      P4Vars,
      charge = Var("charge", int, doc="electric charge"),

      vx = Var("vx", float, doc='x coordinate of vertex position'),
      vy = Var("vy", float, doc='y coordinate of vertex position'),
      vz = Var("vz", float, doc='z coordinate of vertex position'),
      vertexChi2 = Var("vertexChi2", float, doc='chi-squares'),
      vertexNdof = Var("vertexNdof", float, doc='Number of degrees of freedom,  Meant to be Double32_t for soft-assignment fitters'),
      pdgId = Var("pdgId", int, doc='PDG identifier'),
      status = Var("status", int, doc='status word'),
      longLived = Var("longLived", bool, doc='is long lived?'),
      massConstraint = Var("massConstraint", bool, doc='do mass constraint?'),
      # source: cmssw/PhysicsTools/NanoAOD/plugins/SimpleFlatTableProducerPlugins.cc 


    )
  )

  process.tauTablesTask = cms.Task(process.tauTable)
  process.pfCandTablesTask = cms.Task(process.pfCandTable)
  process.nanoTableTaskFS = cms.Task(process.genParticleTablesTask, process.genParticleTask,
                                     process.tauTablesTask, process.pfCandTablesTask)
  process.nanoSequenceMC = cms.Sequence(process.nanoTableTaskFS)
  process.finalGenParticles.src = cms.InputTag("genParticles")


  process.MessageLogger.cerr.FwkReport.reportEvery = 100
  process = customiseGenParticles(process)

  process.genParticleTable.variables.vx = Var('vertex().x', float, precision=10, doc='x coordinate of the gen particle production vertex')
  process.genParticleTable.variables.vy = Var('vertex().y', float, precision=10, doc='y coordinate of the gen particle production vertex')
  process.genParticleTable.variables.vz = Var('vertex().z', float, precision=10, doc='z coordinate of the gen particle production vertex')
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