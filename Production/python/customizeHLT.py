import FWCore.ParameterSet.Config as cms
from RecoTauTag.RecoTau.PFRecoTauQualityCuts_cfi import PFTauQualityCuts
from RecoTauTag.RecoTau.PFTauPrimaryVertexProducer_cfi import *
from RecoTauTag.RecoTau.PFTauSecondaryVertexProducer_cfi import *
from RecoTauTag.RecoTau.PFTauTransverseImpactParameters_cfi import *

def update(process, isMC, requireGenMatch, requireGenORRecoTauMatch, applyRecoPtSieve):

    tauJetBuilderSetup = cms.PSet(
        genLepton_genJet_dR     = cms.double(0.4),
        genLepton_tau_dR        = cms.double(0.2),
        genLepton_jet_dR        = cms.double(0.4),
        genLepton_l1Tau_dR      = cms.double(0.4),
        genJet_tau_dR           = cms.double(0.4),
        genJet_jet_dR           = cms.double(0.4),
        genJet_l1Tau_dR         = cms.double(0.4),
        tau_jet_dR              = cms.double(0.4),
        tau_l1Tau_dR            = cms.double(0.4),
        jet_l1Tau_dR            = cms.double(0.4),
        jet_maxAbsEta           = cms.double(3.0),
        genLepton_cone          = cms.double(0.5),
        genJet_cone             = cms.double(0.5),
        tau_cone                = cms.double(0.5),
        jet_cone                = cms.double(0.8),
        l1Tau_cone              = cms.double(0.5),
    )

    PFTauQualityCuts.primaryVertexSrc = cms.InputTag("hltPixelVertices")

    process.hpsPFTauPrimaryVertexProducer = PFTauPrimaryVertexProducer.clone(
        PFTauTag = "hltHpsPFTauProducer",
        ElectronTag = "hltEgammaCandidates",
        MuonTag = "hltMuons",
        PVTag = "hltPixelVertices",
        beamSpot = "hltOnlineBeamSpot",
        discriminators = [
           cms.PSet(
               discriminator = cms.InputTag('hltHpsPFTauDiscriminationByDecayModeFindingNewDMs'),
               selectionCut = cms.double(0.5)
           )
        ],
        cut = "pt > 18.0 & abs(eta) < 2.4",
        qualityCuts = PFTauQualityCuts
    )

    process.hpsPFTauSecondaryVertexProducer = PFTauSecondaryVertexProducer.clone(
        PFTauTag = "hltHpsPFTauProducer",
    )

    process.hpsPFTauTransverseImpactParameters = PFTauTransverseImpactParameters.clone(
        PFTauTag = 'hltHpsPFTauProducer',
        PFTauPVATag = "hpsPFTauPrimaryVertexProducer",
        PFTauSVATag = "hpsPFTauSecondaryVertexProducer",
        useFullCalculation = True
    )

    process.hltFixedGridRhoFastjetAllTau = cms.EDProducer("FixedGridRhoProducerFastjet",
        gridSpacing = cms.double( 0.55 ),
        maxRapidity = cms.double( 5.0 ),
        pfCandidatesTag = cms.InputTag("hltParticleFlowForTaus")
    )

    process.tauTupleProducerHLT = cms.EDAnalyzer("TauTupleProducerHLT",
        isMC                     = cms.bool(isMC),
        requireGenMatch          = cms.bool(requireGenMatch),
        requireGenORRecoTauMatch = cms.bool(requireGenORRecoTauMatch),
        applyRecoPtSieve         = cms.bool(applyRecoPtSieve),
        tauJetBuilderSetup       = tauJetBuilderSetup,

        lheEventProduct     = cms.InputTag('externalLHEProducer'),
        genEvent            = cms.InputTag('generator'),
        genParticles        = cms.InputTag('genParticles'),
        genJets             = cms.InputTag('genJets'),
        genJetFlavourInfos  = cms.InputTag('genJetsFlavourInfos'),
        puInfo              = cms.InputTag('addPileupInfo'),
        beamSpot            = cms.InputTag('hltOnlineBeamSpot'),
        rho                 = cms.InputTag('hltFixedGridRhoFastjetAllTau'),
        hbheRecHits         = cms.InputTag('hltHbhereco'),
        hoRecHits           = cms.InputTag('hltHoreco'),
        ebRecHits           = cms.InputTag('hltEcalRecHit:EcalRecHitsEB'),
        eeRecHits           = cms.InputTag('hltEcalRecHit:EcalRecHitsEE'),
        pataVertices        = cms.InputTag('hltPixelVerticesSoA'),
        pataTracks          = cms.InputTag('hltPixelTracksSoA'),
        taus                = cms.InputTag('hltHpsPFTauProducer'),
        l1Taus              = cms.InputTag('hltGtStage2Digis:Tau'),
        jets                = cms.InputTag('hltAK4PFJetsForTaus'),
        pfCandidates        = cms.InputTag('hltParticleFlowForTaus'),
        tauIP               = cms.InputTag('hpsPFTauTransverseImpactParameters')
    )

    from HLTrigger.Configuration.customizeHLTforPatatrack import customizeHLTforPatatrackTriplets
    process = customizeHLTforPatatrackTriplets(process)

    process.HLT_TauTupleProd = cms.Path(process.HLTBeginSequence + process.HLTGlobalPFTauHPSSequence + process.hltFixedGridRhoFastjetAllTau + process.hpsPFTauPrimaryVertexProducer + process.hpsPFTauSecondaryVertexProducer + process.hpsPFTauTransverseImpactParameters + process.HLTEndSequence, process.HLTDoLocalPixelTask, process.HLTRecoPixelTracksTask, process.HLTRecopixelvertexingTask)

    process.HLTAnalyzerEndpath.insert(1, process.tauTupleProducerHLT)
    process.schedule = cms.Schedule(*[ process.HLTriggerFirstPath, process.HLT_TauTupleProd, process.HLTAnalyzerEndpath, process.HLTriggerFinalPath, process.endjob_step ], tasks=[process.patAlgosToolsTask])

    return process
