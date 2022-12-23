# How to run:
# hltGetConfiguration /dev/CMSSW_12_4_0/GRun --globaltag auto:phase1_2022_realistic --mc --unprescale --no-output --max-events 100 --eras Run3 --l1-emulator FullMC --l1 L1Menu_Collisions2022_v1_3_0-d1_xml --customise TauMLTools/Production/customiseHLT.customise --input file:00c642d1-bf7e-477d-91e1-4dd9ce2c8099.root > hltRun3Summer21MC.py
# hltGetConfiguration /dev/CMSSW_12_4_0/GRun --globaltag auto:phase1_2022_realistic --mc --unprescale --no-output --max-events 100 --eras Run3 --l1-emulator FullMC --l1 L1Menu_Collisions2022_v1_3_0-d1_xml --customise TauMLTools/Production/customiseHLT.customise --input /store/mc/Run3Summer21DRPremix/TT_TuneCP5_14TeV-powheg-pythia8/GEN-SIM-DIGI-RAW/120X_mcRun3_2021_realistic_v6-v2/2540000/b354245e-d8bc-424d-b527-58815586a6a5.root > hltRun3Summer21MC.py

# cmsRun hltRun3Summer21MC.py
# file dataset=/store/mc/Run3Summer21DRPremix/TT_TuneCP5_14TeV-powheg-pythia8/GEN-SIM-DIGI-RAW/120X_mcRun3_2021_realistic_v6-v2/2540000/00c642d1-bf7e-477d-91e1-4dd9ce2c8099.root
# file dataset=/TT_TuneCP5_14TeV-powheg-pythia8/Run3Summer21DRPremix-120X_mcRun3_2021_realistic_v6-v2/GEN-SIM-DIGI-RAW
import FWCore.ParameterSet.Config as cms
from PhysicsTools.NanoAOD.common_cff import Var, P4Vars
from RecoJets.JetProducers.ak4GenJets_cfi import ak4GenJets
# from PhysicsTools.NanoAOD.jetMC_cff import Var, P4Vars
from PhysicsTools.JetMCAlgos.AK4PFJetsMCFlavourInfos_cfi import ak4JetFlavourInfos



def customiseGenParticles(process):
  def pdgOR(pdgs):
    abs_pdgs = [ f'abs(pdgId) == {pdg}' for pdg in pdgs ]
    return '( ' + ' || '.join(abs_pdgs) + ' )'

  leptons = pdgOR([ 11, 13, 15 ])
  important_particles = pdgOR([ 6, 23, 24, 25, 35, 39, 9990012, 9900012, 1000015 ])
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
  process.load('RecoJets.JetProducers.ak4GenJets_cfi')
  from PhysicsTools.NanoAOD.nano_cff import nanoAOD_customizeMC
  from RecoJets.JetProducers.ak4GenJets_cfi import ak4GenJets
  #call to customisation function nanoAOD_customizeMC imported from PhysicsTools.NanoAOD.nano_cff
  process = nanoAOD_customizeMC(process)

  process.nanoAOD_step = cms.Path(process.HLTBeginSequence \
    + process.HLTL2TauTagNNSequence \
    + process.HLTGlobalPFTauHPSSequence \
    + process.HLTHPSDeepTauPFTauSequenceForVBFIsoTau \
    + process.nanoSequenceMC)
  process.NANOAODSIMoutput_step = cms.EndPath(process.NANOAODSIMoutput)

  from PhysicsTools.JetMCAlgos.HadronAndPartonSelector_cfi import selectedHadronsAndPartons
  process.selectedHadronsAndPartons = selectedHadronsAndPartons.clone(src = cms.InputTag("genParticlesForJetsNoNu"))
  # set ak4GenJets producer
  process.ak4GenJetsNoNu = ak4GenJets.clone( src = "genParticlesForJetsNoNu")
  process.genJetFlavourInfos = ak4JetFlavourInfos.clone(
    jets = cms.InputTag( "ak4GenJetsNoNu" ),
    bHadrons= cms.InputTag("selectedHadronsAndPartons","bHadrons"),
    cHadrons= cms.InputTag("selectedHadronsAndPartons","cHadrons"),
    partons= cms.InputTag("selectedHadronsAndPartons","physicsPartons"), )

  process.load('RecoJets.Configuration.GenJetParticles_cff')

  process.GenJetTable = cms.EDProducer("SimpleCandidateFlatTableProducer",
    src = cms.InputTag( "ak4GenJetsNoNu" ),
    cut = cms.string(""),
    name= cms.string("GenJet"),
    doc = cms.string("HLT ak4 GenJet"),
    singleton = cms.bool(False), # the number of entries is variable
    extension = cms.bool(False), # this is the main table
    variables = cms.PSet(
      P4Vars,
      emEnergy = Var("emEnergy", float, doc="energy of electromagnetic particles "),
      hadEnergy = Var("hadEnergy", float, doc="energy of hadronic particles "),
      invisibleEnergy = Var("invisibleEnergy", float, doc="invisible energy "),
      auxiliaryEnergy = Var("auxiliaryEnergy", float, doc=" other energy (undecayed Sigmas etc.) "),
      chargedHadronEnergy = Var("chargedHadronEnergy", float, doc="energy of charged hadrons "),
      neutralHadronEnergy = Var("neutralHadronEnergy", float, doc="energy of neutral hadrons "),
      chargedEmEnergy = Var("chargedEmEnergy", float, doc="energy of charged electromagnetic particles "),
      neutralEmEnergy = Var("neutralEmEnergy", float, doc="energy of neutral electromagnetic particles "),
      muonEnergy = Var("muonEnergy", float, doc="energy of muons "),
      chargedHadronMultiplicity = Var("chargedHadronMultiplicity", int, doc="number of charged hadrons "),
      neutralHadronMultiplicity = Var("neutralHadronMultiplicity", int, doc="number of neutral hadrons "),
      chargedEmMultiplicity = Var("chargedEmMultiplicity", int, doc="number of charged electromagnetic particles "),
      neutralEmMultiplicity = Var("neutralEmMultiplicity", int, doc="number of neutral electromagnetic particles "),
      muonMultiplicity = Var("muonMultiplicity", int, doc="number of muons "),
      )
  )
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
      vx = Var("vx", float, doc='x coordinate of vertex position'),
      vy = Var("vy", float, doc='y coordinate of vertex position'),
      vz = Var("vz", float, doc='z coordinate of vertex position'),
      pdgId = Var("pdgId", int, doc='PDG identifier'),
      dz = Var("? leadPFCand.trackRef.isNonnull && leadPFCand.trackRef.isAvailable ? leadPFCand.trackRef.dz : -999 ", float, doc='lead PF Candidate dz'),
      dzError = Var("? leadPFCand.trackRef.isNonnull && leadPFCand.trackRef.isAvailable ? leadPFCand.trackRef.dzError : -999 ", float, doc='lead PF Candidate dz Error'),
      decayMode = Var("decayMode", int, doc='tau decay mode'),
      tauIsValid = Var("jetRef.isNonnull && jetRef.isAvailable", bool, doc = "tau is valid"),
      # variables available in PF jets
      # source: DataFormats/JetReco/interface/PFJet.h
      chargedHadronEnergy = Var("? jetRef.isNonnull && jetRef.isAvailable ? jetRef.chargedHadronEnergy : -999.", float, doc = "chargedHadronEnergy"),
      neutralHadronEnergy = Var("? jetRef.isNonnull && jetRef.isAvailable ? jetRef.neutralHadronEnergy : -999.", float, doc = "neutralHadronEnergy"),
      photonEnergy = Var("? jetRef.isNonnull && jetRef.isAvailable ? jetRef.photonEnergy : -999.", float, doc = "photonEnergy"),
      electronEnergy = Var("? jetRef.isNonnull && jetRef.isAvailable ? jetRef.electronEnergy : -999.", float, doc = "electronEnergy"),
      muonEnergy = Var("? jetRef.isNonnull && jetRef.isAvailable ? jetRef.muonEnergy : -999.", float, doc = "muonEnergy"),
      HFHadronEnergy = Var("? jetRef.isNonnull && jetRef.isAvailable ? jetRef.HFHadronEnergy : -999.", float, doc = "HFHadronEnergy"),
      HFEMEnergy = Var("? jetRef.isNonnull && jetRef.isAvailable ? jetRef.HFEMEnergy : -999.", float, doc = "HFEMEnergy"),
      chargedHadronMultiplicity = Var("? jetRef.isNonnull && jetRef.isAvailable ? jetRef.chargedHadronMultiplicity : -999.", float, doc = "chargedHadronMultiplicity"),
      neutralHadronMultiplicity = Var("? jetRef.isNonnull && jetRef.isAvailable ? jetRef.neutralHadronMultiplicity : -999.", float, doc = "neutralHadronMultiplicity"),
      photonMultiplicity = Var("? jetRef.isNonnull && jetRef.isAvailable ? jetRef.photonMultiplicity : -999.", float, doc = "photonMultiplicity"),
      electronMultiplicity = Var("? jetRef.isNonnull && jetRef.isAvailable ? jetRef.electronMultiplicity : -999.", float, doc = "electronMultiplicity"),
      muonMultiplicity = Var("? jetRef.isNonnull && jetRef.isAvailable ? jetRef.muonMultiplicity : -999.", float, doc = "muonMultiplicity"),
      HFHadronMultiplicity = Var("? jetRef.isNonnull && jetRef.isAvailable ? jetRef.HFHadronMultiplicity : -999.", float, doc = "HFHadronMultiplicity"),
      HFEMMultiplicity = Var("? jetRef.isNonnull && jetRef.isAvailable ? jetRef.HFEMMultiplicity : -999.", float, doc = "HFEMMultiplicity"),
      chargedEmEnergy = Var("? jetRef.isNonnull && jetRef.isAvailable ? jetRef.chargedEmEnergy : -999.", float, doc = "chargedEmEnergy"),
      chargedMuEnergy = Var("? jetRef.isNonnull && jetRef.isAvailable ? jetRef.chargedMuEnergy : -999.", float, doc = "chargedMuEnergy"),
      neutralEmEnergy = Var("? jetRef.isNonnull && jetRef.isAvailable ? jetRef.neutralEmEnergy : -999.", float, doc = "neutralEmEnergy"),
      chargedMultiplicity = Var("? jetRef.isNonnull && jetRef.isAvailable ? jetRef.chargedMultiplicity : -999.", float, doc = "chargedMultiplicity"),
      neutralMultiplicity = Var("? jetRef.isNonnull && jetRef.isAvailable ? jetRef.neutralMultiplicity : -999.", float, doc = "neutralMultiplicity"),
      # # source: DataFormats/TauReco/interface/PFTau.h
      # ## variables available in PF tau
      emFraction = Var("emFraction", float, doc = " Ecal/Hcal Cluster Energy"),
      hcalTotOverPLead = Var("hcalTotOverPLead", float, doc = " total Hcal Cluster E / leadPFChargedHadron P"),
      signalConeSize = Var("signalConeSize", float, doc = "Size of signal cone"),
      # lepton decision variables
      electronPreIDDecision = Var("electronPreIDDecision", bool, doc = " Decision from Electron PreID"),
      muonDecision = Var("muonDecision", bool, doc = "muonDecision"),
    )
  )
  # cms.EDProducer : an object that produces a new data object
  process.tauExtTable = cms.EDProducer("TauTableProducerHLT",
    taus = cms.InputTag( "hltHpsPFTauProducer" ),
    deepTauVSe = cms.InputTag("hltHpsPFTauDeepTauProducerForVBFIsoTau", "VSe"),
    deepTauVSmu = cms.InputTag("hltHpsPFTauDeepTauProducerForVBFIsoTau", "VSmu"),
    deepTauVSjet = cms.InputTag("hltHpsPFTauDeepTauProducerForVBFIsoTau", "VSjet"),
    tauTransverseImpactParameters = cms.InputTag( "hltHpsPFTauTransverseImpactParametersForDeepTauForVBFIsoTau" ),
    precision = cms.int32(7),
  )
  process.ak4GenJetsNoNuExtTable = cms.EDProducer("GenJetFlavourTableProducer",
    name = cms.string("GenJet"),
    src = cms.InputTag( "ak4GenJetsNoNu" ),
    cut = cms.string(""),
    deltaR = cms.double(0.4),
    jetFlavourInfos = cms.InputTag( "genJetFlavourInfos" ),
  )
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
      trackDz = Var("? trackRef.isNonnull && trackRef.isAvailable ? trackRef.dz : -999.", float, doc = "track Dz"),
      trackDxy = Var("? trackRef.isNonnull && trackRef.isAvailable ? trackRef.dxy : -999.", float, doc = "track Dxy"),
      trackIsValid = Var("trackRef.isNonnull && trackRef.isAvailable", bool, doc = "track is valid"),
      trackDzError = Var("? trackRef.isNonnull && trackRef.isAvailable ? trackRef.dzError : -999.", float, doc = "track DzError"),
      trackDxyError = Var("? trackRef.isNonnull && trackRef.isAvailable ? trackRef.dxyError : -999.", float, doc = "track DxyError"),
      trackPt = Var("? trackRef.isNonnull && trackRef.isAvailable ? trackRef.pt : -999.", float, doc = "track Pt"),
      trackEta = Var("? trackRef.isNonnull && trackRef.isAvailable ? trackRef.eta : -999.", float, doc = "track Eta"),
      trackPhi = Var("? trackRef.isNonnull && trackRef.isAvailable ? trackRef.phi : -999.", float, doc = "track Phi"),
      trackPtError = Var("? trackRef.isNonnull && trackRef.isAvailable ? trackRef.ptError : -999.", float, doc = "track PtError"),
      trackEtaError = Var("? trackRef.isNonnull && trackRef.isAvailable ? trackRef.etaError : -999.", float, doc = "track PtaError"),
      trackPhiError = Var("? trackRef.isNonnull && trackRef.isAvailable ? trackRef.phiError : -999.", float, doc = "track PhiError"),
      trackChi2 = Var("? trackRef.isNonnull && trackRef.isAvailable ? trackRef.chi2 : -999.", float, doc = "track Chi2"),
      trackNdof = Var("? trackRef.isNonnull && trackRef.isAvailable ? trackRef.ndof : -999.", float, doc = "track Ndof"),
      # source: DataFormats/TrackReco/interface/TrackBase.h
      trackNumberOfValidHits = Var("? trackRef.isNonnull && trackRef.isAvailable ? trackRef.numberOfValidHits : -999.", float, doc = "number of valid hits found"),
      trackNumberOfLostHits = Var("? trackRef.isNonnull && trackRef.isAvailable ? trackRef.numberOfLostHits : -999.", float, doc = " number of cases where track crossed a layer without getting a hit."),
      trackHitsValidFraction= Var("? trackRef.isNonnull && trackRef.isAvailable ? trackRef.validFraction : -999.", float, doc = "fraction of valid hits on the track"),
      rawHcalEnergy = Var("rawHcalEnergy", float, doc='rawHcalEnergy'),
      rawEcalEnergy = Var("rawEcalEnergy", float, doc='rawEcalEnergy'),
      EcalEnergy = Var("ecalEnergy", float, doc='EcalEnergy'),
      HcalEnergy = Var("hcalEnergy", float, doc='HcalEnergy'),
    )
  )
  process.AK4PFJetsTable = cms.EDProducer("SimpleCandidateFlatTableProducer",
    src = cms.InputTag( "hltAK4PFJets" ),
    cut = cms.string(""),
    name= cms.string("Jet"),
    doc = cms.string("HLT AK4 PF Jets"),
    singleton = cms.bool(False), # the number of entries is variable
    extension = cms.bool(False), # this is the main table
    variables = cms.PSet(
      # these variables are 0
      # AK4PFJets_chargedEmEnergyFraction
      # AK4PFJets_chargedEmEnergy
      # AK4PFJets_electronEnergy
      # AK4PFJets_electronEnergyFraction
      # AK4PFJets_electronMultiplicity
      P4Vars,
      chargedHadronEnergy = Var("chargedHadronEnergy", float, doc = "chargedHadronEnergy"),
      chargedHadronEnergyFraction = Var("chargedHadronEnergyFraction", float, doc = "chargedHadronEnergyFraction"),
      neutralHadronEnergy = Var("neutralHadronEnergy", float, doc = "neutralHadronEnergy"),
      neutralHadronEnergyFraction = Var("neutralHadronEnergyFraction", float, doc = "neutralHadronEnergyFraction"),
      photonEnergy = Var("photonEnergy", float, doc = "photonEnergy"),
      photonEnergyFraction = Var("photonEnergyFraction", float, doc = "photonEnergyFraction"),
      electronEnergy = Var("electronEnergy", float, doc = "electronEnergy"),
      electronEnergyFraction = Var("electronEnergyFraction", float, doc = "electronEnergyFraction"),
      muonEnergy = Var("muonEnergy", float, doc = "muonEnergy"),
      muonEnergyFraction = Var("muonEnergyFraction", float, doc = "muonEnergyFraction"),
      HFHadronEnergy = Var("HFHadronEnergy", float, doc = "HFHadronEnergy"),
      HFHadronEnergyFraction = Var("HFHadronEnergyFraction", float, doc = "HFHadronEnergyFraction"),
      HFEMEnergy = Var("HFEMEnergy", float, doc = "HFEMEnergy"),
      HFEMEnergyFraction = Var("HFEMEnergyFraction", float, doc = "HFEMEnergyFraction"),
      chargedHadronMultiplicity = Var("chargedHadronMultiplicity", float, doc = "chargedHadronMultiplicity"),
      neutralHadronMultiplicity = Var("neutralHadronMultiplicity", float, doc = "neutralHadronMultiplicity"),
      photonMultiplicity = Var("photonMultiplicity", float, doc = "photonMultiplicity"),
      electronMultiplicity = Var("electronMultiplicity", float, doc = "electronMultiplicity"),
      muonMultiplicity = Var("muonMultiplicity", float, doc = "muonMultiplicity"),
      HFHadronMultiplicity = Var("HFHadronMultiplicity", float, doc = "HFHadronMultiplicity"),
      HFEMMultiplicity = Var("HFEMMultiplicity", float, doc = "HFEMMultiplicity"),
      chargedEmEnergy = Var("chargedEmEnergy", float, doc = "chargedEmEnergy"),
      chargedEmEnergyFraction = Var("chargedEmEnergyFraction", float, doc = "chargedEmEnergyFraction"),
      chargedMuEnergy = Var("chargedMuEnergy", float, doc = "chargedMuEnergy"),
      chargedMuEnergyFraction = Var("chargedMuEnergyFraction", float, doc = "chargedMuEnergyFraction"),
      neutralEmEnergy = Var("neutralEmEnergy", float, doc = "neutralEmEnergy"),
      neutralEmEnergyFraction = Var("neutralEmEnergyFraction", float, doc = "neutralEmEnergyFraction"),
      chargedMultiplicity = Var("chargedMultiplicity", float, doc = "chargedMultiplicity"),
      neutralMultiplicity = Var("neutralMultiplicity", float, doc = "neutralMultiplicity"),
      hoEnergy = Var("hoEnergy", float, doc = "hoEnergy"),
      hoEnergyFraction = Var("hoEnergyFraction", float, doc = "hoEnergyFraction"),
      nConstituents = Var("nConstituents", int, doc = "nConstituents"),
      etaetaMoment =  Var("etaetaMoment", float, doc = " eta-eta second moment, ET weighted " ),
      phiphiMoment =  Var("phiphiMoment", float, doc = " phi-phi second moment, ET weighted " ),
      etaphiMoment =  Var("etaphiMoment", float, doc = " eta-phi second moment, ET weighted " ),
      maxDistance =  Var("maxDistance", float, doc = " maximum distance from jet to constituent " ),
      constituentPtDistribution =  Var("constituentPtDistribution", float, doc = " jet structure variables: constituentPtDistribution is the pT distribution among the jet constituents (ptDistribution = 1 if jet made by one constituent carrying all its momentum,  ptDistribution = 0 if jet made by infinite constituents carrying an infinitesimal fraction of pt) "    ),
      constituentEtaPhiSpread =  Var("constituentEtaPhiSpread", float, doc = " the rms of the eta-phi spread of the jet's constituents wrt the jet axis " ),
      jetArea =  Var("jetArea", float, doc = " get jet area " ),
      pileup =  Var("pileup", float, doc = "  pileup energy contribution as calculated by algorithm " ),
    )
  )

  process.L1Table = cms.EDProducer("L1TableProducer",
    egammas = cms.InputTag("hltGtStage2Digis", "EGamma"),
    muons = cms.InputTag("hltGtStage2Digis", "Muon"),
    jets = cms.InputTag("hltGtStage2Digis", "Jet"),
    taus = cms.InputTag("hltGtStage2Digis", "Tau"),
    precision = cms.int32(7)
  )

  process.tauTablesTask = cms.Task(process.tauTable, process.tauExtTable)
  process.pfCandTablesTask = cms.Task(process.pfCandTable)
  process.AK4PFJetsTableTask = cms.Task(process.AK4PFJetsTable)
  process.genParticlesForJetsNoNuTask = cms.Task(process.genParticlesForJetsNoNu, process.selectedHadronsAndPartons)
  process.genJetFlavourInfosTask =  cms.Task(process.genJetFlavourInfos)
  process.recoAllGenJetsNoNuTask=cms.Task(process.ak4GenJetsNoNu)
  process.GenJetTableTask = cms.Task(process.GenJetTable, process.ak4GenJetsNoNuExtTable)
  process.L1TableTask = cms.Task(process.L1Table)
  process.nanoTableTaskFS = cms.Task(process.genParticleTablesTask,
                                     process.genParticleTask,
                                     process.tauTablesTask,
                                     process.pfCandTablesTask,
                                     process.genParticlesForJetsNoNuTask,
                                     process.genJetFlavourInfosTask,
                                     process.AK4PFJetsTableTask,
                                     process.recoAllGenJetsNoNuTask,
                                     process.GenJetTableTask,
                                     process.L1TableTask)
  process.nanoSequenceMC = cms.Sequence(process.nanoTableTaskFS)
  process.finalGenParticles.src = cms.InputTag("genParticles")


  process.MessageLogger.cerr.FwkReport.reportEvery = 100
  process = customiseGenParticles(process)

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
  process.MessageLogger.cerr.enableStatistics = cms.untracked.bool(False)

  process.options.wantSummary = False

  process.schedule.insert(1000000, process.nanoAOD_step)
  process.schedule.insert(1000000, process.NANOAODSIMoutput_step)
  return process