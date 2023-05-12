# How to run: see hlt_configs/README.md

import FWCore.ParameterSet.Config as cms
from PhysicsTools.NanoAOD.common_cff import Var, P4Vars
from RecoJets.JetProducers.ak4GenJets_cfi import ak4GenJets
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
  # process.genParticleTable.variables.mass.expr = cms.string('mass')
  # process.genParticleTable.variables.mass.doc = cms.string('mass')

  return process


def customise(process, output='nano.root', is_data=False):
  process.load('PhysicsTools.NanoAOD.nano_cff')
  process.load('RecoJets.JetProducers.ak4GenJets_cfi')
  from PhysicsTools.NanoAOD.nano_cff import nanoAOD_customizeCommon
  from RecoJets.JetProducers.ak4GenJets_cfi import ak4GenJets
  #call to customisation function nanoAOD_customizeMC imported from PhysicsTools.NanoAOD.nano_cff
  process = nanoAOD_customizeCommon(process)

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
      jetIsValid = Var("jetRef.isNonnull && jetRef.isAvailable", bool, doc = "jet is valid"),
      # variables available in PF jets
      # source: DataFormats/JetReco/interface/PFJet.h
      chargedHadronEnergy = Var("? jetRef.isNonnull && jetRef.isAvailable ? jetRef.chargedHadronEnergy : -999.", float, doc = "chargedHadronEnergy"),
      neutralHadronEnergy = Var("? jetRef.isNonnull && jetRef.isAvailable ? jetRef.neutralHadronEnergy : -999.", float, doc = "neutralHadronEnergy"),
      photonEnergy = Var("? jetRef.isNonnull && jetRef.isAvailable ? jetRef.photonEnergy : -999.", float, doc = "photonEnergy"),
      muonEnergy = Var("? jetRef.isNonnull && jetRef.isAvailable ? jetRef.muonEnergy : -999.", float, doc = "muonEnergy"),
      chargedHadronMultiplicity = Var("? jetRef.isNonnull && jetRef.isAvailable ? jetRef.chargedHadronMultiplicity : -999.", float, doc = "chargedHadronMultiplicity"),
      neutralHadronMultiplicity = Var("? jetRef.isNonnull && jetRef.isAvailable ? jetRef.neutralHadronMultiplicity : -999.", float, doc = "neutralHadronMultiplicity"),
      photonMultiplicity = Var("? jetRef.isNonnull && jetRef.isAvailable ? jetRef.photonMultiplicity : -999.", float, doc = "photonMultiplicity"),
      muonMultiplicity = Var("? jetRef.isNonnull && jetRef.isAvailable ? jetRef.muonMultiplicity : -999.", float, doc = "muonMultiplicity"),
      chargedMuEnergy = Var("? jetRef.isNonnull && jetRef.isAvailable ? jetRef.chargedMuEnergy : -999.", float, doc = "chargedMuEnergy"),
      neutralEmEnergy = Var("? jetRef.isNonnull && jetRef.isAvailable ? jetRef.neutralEmEnergy : -999.", float, doc = "neutralEmEnergy"),
      chargedMultiplicity = Var("? jetRef.isNonnull && jetRef.isAvailable ? jetRef.chargedMultiplicity : -999.", float, doc = "chargedMultiplicity"),
      neutralMultiplicity = Var("? jetRef.isNonnull && jetRef.isAvailable ? jetRef.neutralMultiplicity : -999.", float, doc = "neutralMultiplicity"),
      # # source: DataFormats/TauReco/interface/PFTau.h
      # ## variables available in PF tau
      emFraction = Var("emFraction", float, doc = " Ecal/Hcal Cluster Energy"),
      hcalTotOverPLead = Var("hcalTotOverPLead", float, doc = " total Hcal Cluster E / leadPFChargedHadron P"),
      signalConeSize = Var("signalConeSize", float, doc = "Size of signal cone"),
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
    src = cms.InputTag( "hltParticleFlow" ),
    cut = cms.string(""),
    name= cms.string("PFCand"),
    doc = cms.string("HLT PF candidates"),
    singleton = cms.bool(False), # the number of entries is variable
    extension = cms.bool(False), # this is the main table
    variables = cms.PSet(
      P4Vars,
      charge = Var("charge", int, doc="electric charge"),
      vx = Var("vx", float, doc='x coordinate of vertex position'),
      vy = Var("vy", float, doc='y coordinate of vertex position'),
      vz = Var("vz", float, doc='z coordinate of vertex position'),
      pdgId = Var("pdgId", int, doc='PDG identifier'),
      longLived = Var("longLived", bool, doc='is long lived?'),
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
    src = cms.InputTag( "hltAK4PFJetsCorrected" ),
    cut = cms.string(""),
    name= cms.string("Jet"),
    doc = cms.string("HLT AK4 PF Jets"),
    singleton = cms.bool(False), # the number of entries is variable
    extension = cms.bool(False), # this is the main table
    variables = cms.PSet(
      P4Vars,
      chargedHadronEnergy = Var("chargedHadronEnergy", float, doc = "chargedHadronEnergy"),
      chargedHadronEnergyFraction = Var("chargedHadronEnergyFraction", float, doc = "chargedHadronEnergyFraction"),
      neutralHadronEnergy = Var("neutralHadronEnergy", float, doc = "neutralHadronEnergy"),
      neutralHadronEnergyFraction = Var("neutralHadronEnergyFraction", float, doc = "neutralHadronEnergyFraction"),
      photonEnergy = Var("photonEnergy", float, doc = "photonEnergy"),
      photonEnergyFraction = Var("photonEnergyFraction", float, doc = "photonEnergyFraction"),
      muonEnergy = Var("muonEnergy", float, doc = "muonEnergy"),
      muonEnergyFraction = Var("muonEnergyFraction", float, doc = "muonEnergyFraction"),
      HFHadronEnergy = Var("HFHadronEnergy", float, doc = "HFHadronEnergy"),
      HFHadronEnergyFraction = Var("HFHadronEnergyFraction", float, doc = "HFHadronEnergyFraction"),
      HFEMEnergy = Var("HFEMEnergy", float, doc = "HFEMEnergy"),
      HFEMEnergyFraction = Var("HFEMEnergyFraction", float, doc = "HFEMEnergyFraction"),
      chargedHadronMultiplicity = Var("chargedHadronMultiplicity", float, doc = "chargedHadronMultiplicity"),
      neutralHadronMultiplicity = Var("neutralHadronMultiplicity", float, doc = "neutralHadronMultiplicity"),
      photonMultiplicity = Var("photonMultiplicity", float, doc = "photonMultiplicity"),
      muonMultiplicity = Var("muonMultiplicity", float, doc = "muonMultiplicity"),
      HFHadronMultiplicity = Var("HFHadronMultiplicity", float, doc = "HFHadronMultiplicity"),
      HFEMMultiplicity = Var("HFEMMultiplicity", float, doc = "HFEMMultiplicity"),
      chargedMuEnergy = Var("chargedMuEnergy", float, doc = "chargedMuEnergy"),
      chargedMuEnergyFraction = Var("chargedMuEnergyFraction", float, doc = "chargedMuEnergyFraction"),
      neutralEmEnergy = Var("neutralEmEnergy", float, doc = "neutralEmEnergy"),
      neutralEmEnergyFraction = Var("neutralEmEnergyFraction", float, doc = "neutralEmEnergyFraction"),
      chargedMultiplicity = Var("chargedMultiplicity", float, doc = "chargedMultiplicity"),
      neutralMultiplicity = Var("neutralMultiplicity", float, doc = "neutralMultiplicity"),
      nConstituents = Var("nConstituents", int, doc = "nConstituents"),
      etaetaMoment =  Var("etaetaMoment", float, doc = " eta-eta second moment, ET weighted " ),
      phiphiMoment =  Var("phiphiMoment", float, doc = " phi-phi second moment, ET weighted " ),
      etaphiMoment =  Var("etaphiMoment", float, doc = " eta-phi second moment, ET weighted " ),
      maxDistance =  Var("maxDistance", float, doc = " maximum distance from jet to constituent " ),
      constituentPtDistribution =  Var("constituentPtDistribution", float, doc = " jet structure variables: constituentPtDistribution is the pT distribution among the jet constituents (ptDistribution = 1 if jet made by one constituent carrying all its momentum,  ptDistribution = 0 if jet made by infinite constituents carrying an infinitesimal fraction of pt) "    ),
      constituentEtaPhiSpread =  Var("constituentEtaPhiSpread", float, doc = " the rms of the eta-phi spread of the jet's constituents wrt the jet axis " ),
      jetArea =  Var("jetArea", float, doc = " get jet area " )
    )
  )

  pnetTags = cms.PSet()
  for tag_name in [ "probtauhp", "probtauhm", "probb", "probc", "probuds", "probg", "ptcorr" ]:
    setattr(pnetTags, f"PNet_{tag_name}", cms.InputTag(f"hltParticleNetONNXJetTags:{tag_name}"))

  process.AK4PFJetsExtTable = cms.EDProducer("JetTableProducerHLT",
    jets = cms.InputTag("hltAK4PFJetsCorrected"),
    looseJets = cms.InputTag("hltAK4PFJetsLooseIDCorrected"),
    tightJets = cms.InputTag("hltAK4PFJetsTightIDCorrected"),
    jetTags = pnetTags,
    maxDeltaR = cms.double(0.1),
    precision = cms.int32(7),
  )

  process.L1Table = cms.EDProducer("L1TableProducer",
    egammas = cms.InputTag("hltGtStage2Digis", "EGamma"),
    muons = cms.InputTag("hltGtStage2Digis", "Muon"),
    jets = cms.InputTag("hltGtStage2Digis", "Jet"),
    #taus = cms.InputTag("hltGtStage2Digis", "Tau"),
    taus = cms.InputTag("simCaloStage2Digis"),
    caloTowers = cms.InputTag("simCaloStage2Digis", "MP"),
    l2TauTagNNProducer = cms.string("hltL2TauTagNNProducer"),
    l2Taus = process.hltL2TauTagNNProducer.L1Taus,
    precision = cms.int32(7)
  )

  process.caloTable = cms.EDProducer("CaloTableProducer",
    hbhe = cms.InputTag("hltHbhereco"),
    ho = cms.InputTag("hltHoreco"),
    eb = cms.InputTag("hltEcalRecHit", "EcalRecHitsEB"),
    ee = cms.InputTag("hltEcalRecHit", "EcalRecHitsEE"),
    precision = cms.int32(7)
  )

  process.pixelTrackTable = cms.EDProducer("PixelTrackTableProducer",
    tracks = cms.InputTag("hltPixelTracksSoA"),
    vertices = cms.InputTag("hltPixelVerticesSoA"),
    beamSpot = cms.InputTag("hltOnlineBeamSpot"),
    precision = cms.int32(7)
  )

  process.pfVertexTable = cms.EDProducer("VertexTableProducerHLT",
    src = cms.InputTag("hltVerticesPFFilter"),
    name = cms.string("PFPrimaryVertex")
  )
  process.svCandidateTable.src = 'hltDeepInclusiveSecondaryVerticesPF'
  process.svCandidateTable.name = 'PFSecondaryVertex'
  process.svCandidateTable.extension = False

  process.tauTablesTask = cms.Task(process.tauTable)
  process.tauTablesExtTask = cms.Task(process.tauExtTable)
  process.pfCandTablesTask = cms.Task(process.pfCandTable)
  process.vertexTablesTask = cms.Task(process.pfVertexTable, process.svCandidateTable)
  process.AK4PFJetsTableTask = cms.Task(process.AK4PFJetsTable)
  process.AK4PFJetsExtTableTask = cms.Task(process.AK4PFJetsExtTable)
  process.genParticlesForJetsNoNuTask = cms.Task(process.genParticlesForJetsNoNu, process.selectedHadronsAndPartons)
  process.genJetFlavourInfosTask =  cms.Task(process.genJetFlavourInfos)
  process.recoAllGenJetsNoNuTask=cms.Task(process.ak4GenJetsNoNu)
  process.GenJetTableTask = cms.Task(process.GenJetTable, process.ak4GenJetsNoNuExtTable)
  process.L1TableTask = cms.Task(process.simCaloStage2Digis, process.L1Table)
  process.caloTableTask = cms.Task(process.caloTable)
  process.pixelTrackTableTask = cms.Task(process.pixelTrackTable)
  process.nanoTableTask = cms.Task(
    process.tauTablesTask,
    process.tauTablesExtTask,
    process.pfCandTablesTask,
    process.vertexTablesTask,
    process.AK4PFJetsTableTask,
    process.AK4PFJetsExtTableTask,
    process.recoAllGenJetsNoNuTask,
    process.L1TableTask,
    process.caloTableTask,
    process.pixelTrackTableTask
  )

  if not is_data:
    process.nanoTableTask.add(
      process.genParticleTablesTask,
      process.genParticleTask,
      process.genParticlesForJetsNoNuTask,
      process.genJetFlavourInfosTask,
      process.GenJetTableTask
    )
    process.finalGenParticles.src = cms.InputTag("genParticles")
    process = customiseGenParticles(process)

  process.nanoSequence = cms.Sequence(process.nanoTableTask)
  process.l1bits=cms.EDProducer("L1TriggerResultsConverter",
    src=cms.InputTag("hltGtStage2Digis"),
    legacyL1=cms.bool(False),
    storeUnprefireableBit=cms.bool(True),
    src_ext=cms.InputTag("simGtExtUnprefireable")
  )

  process.HLTTauVertexSequencePF = cms.Sequence(
    process.hltVerticesPF
    + process.hltVerticesPFSelector
    + cms.ignore(process.hltVerticesPFFilter)
    + process.hltDeepInclusiveVertexFinderPF
    + process.hltDeepInclusiveSecondaryVerticesPF
  )

  process.HLTJetFlavourTagParticleNetSequencePF = cms.Sequence(
    process.hltVerticesPF
    + process.hltVerticesPFSelector
    + cms.ignore(process.hltVerticesPFFilter)
    + cms.ignore(process.hltPFJetForBtagSelector)
    + process.hltPFJetForBtag
    + process.hltDeepBLifetimeTagInfosPF
    + process.hltDeepInclusiveVertexFinderPF
    + process.hltDeepInclusiveSecondaryVerticesPF
    + process.hltDeepTrackVertexArbitratorPF
    + process.hltDeepInclusiveMergedVerticesPF
    + process.hltPrimaryVertexAssociation
    + process.hltParticleNetJetTagInfos
    + process.hltParticleNetONNXJetTags
    + process.hltParticleNetDiscriminatorsJetTags
  )

  process.hltHpsPFTauTrack.MinN = 0
  process.hltTauJet5.MinN = 0
  process.hltPFJetForBtagSelector.MinPt = 15
  process.hltPFJetForBtagSelector.MaxEta = 2.7
  process.hltParticleNetJetTagInfos.min_jet_pt = 15
  process.hltParticleNetJetTagInfos.max_jet_eta = 2.7
  process.nanoAOD_step = cms.Path(
    process.SimL1Emulator
    + cms.ignore(process.hltTriggerType)
    + process.HLTL1UnpackerSequence
    + process.HLTBeamSpot
    + process.HLTL2TauTagNNSequence
    + process.HLTGlobalPFTauHPSSequence
    + process.HLTHPSDeepTauPFTauSequenceForVBFIsoTau
    + process.HLTAK4PFJetsSequence
    + process.HLTJetFlavourTagParticleNetSequencePF
    + process.HLTTauVertexSequencePF
    + process.l1bits
    + process.nanoSequence
  )

  process.pickEventsPath = cms.Path(
    process.SimL1Emulator
    + cms.ignore(process.hltTriggerType)
    + process.HLTL1UnpackerSequence
    + process.HLTBeamSpot
    + process.hltStripTrackerHVOn
    + process.hltPixelTrackerHVOn
  )

  process.NANOAODSIMoutput = cms.OutputModule("NanoAODOutputModule",
    compressionAlgorithm = cms.untracked.string('LZMA'),
    compressionLevel = cms.untracked.int32(9),
    dataset = cms.untracked.PSet(
      dataTier = cms.untracked.string('NANOAODSIM'),
      filterName = cms.untracked.string('')
    ),
    fileName = cms.untracked.string(f'file:{output}'),
    outputCommands  = cms.untracked.vstring(
      'drop *',
      'keep nanoaodFlatTable_*Table_*_*',
      'keep edmTriggerResults_*_*_HLTX',
    ),
    SelectEvents = cms.untracked.PSet(
      SelectEvents = cms.vstring('pickEventsPath')
    ),
  )

  process.NANOAODSIMoutput_step = cms.EndPath(process.NANOAODSIMoutput)

  process.options.numberOfThreads = 1
  process.MessageLogger.cerr.FwkReport.reportEvery = 100

  # process.FastTimerService.printEventSummary = False
  # process.FastTimerService.printRunSummary = False
  # process.FastTimerService.printJobSummary = False
  # process.hltL1TGlobalSummary.DumpTrigSummary = False

  # del process.MessageLogger.TriggerSummaryProducerAOD
  # del process.MessageLogger.L1GtTrigReport
  # del process.MessageLogger.L1TGlobalSummary
  # del process.MessageLogger.HLTrigReport
  # del process.MessageLogger.FastReport
  # del process.MessageLogger.ThroughputService
  # process.MessageLogger.cerr.enableStatistics = cms.untracked.bool(False)
  del process.dqmOutput

  # process.options.wantSummary = False

  #process.schedule = cms.Schedule(process.nanoAOD_step, process.pickEventsPath, process.NANOAODSIMoutput_step)
  process.schedule.insert(1000000, process.nanoAOD_step)
  process.schedule.insert(1000000, process.pickEventsPath)
  process.schedule.insert(1000000, process.NANOAODSIMoutput_step)
  #  print(process.dumpPython())

  return process

def customiseMC(process):
  return customise(process, is_data=False)

def customiseData(process):
  return customise(process, is_data=True)