import FWCore.ParameterSet.Config as cms
def update(process):

    #extremely big or da definire, e poi tutto diventa *Extended

    process.hltL1sTauExtremelyBigOR = cms.EDFilter( "HLTL1TSeed",
        L1SeedsLogicalExpression = cms.string( "L1_LooseIsoEG22er2p1_IsoTau26er2p1_dR_Min0p3 OR L1_LooseIsoEG24er2p1_IsoTau27er2p1_dR_Min0p3 OR L1_LooseIsoEG22er2p1_Tau70er2p1_dR_Min0p3 OR L1_SingleTau120er2p1 OR L1_SingleTau130er2p1 OR L1_DoubleTau70er2p1 OR L1_DoubleIsoTau28er2p1 OR L1_DoubleIsoTau30er2p1 OR L1_DoubleIsoTau32er2p1 OR L1_DoubleIsoTau34er2p1 OR L1_DoubleIsoTau36er2p1 OR L1_DoubleIsoTau28er2p1_Mass_Max90 OR L1_DoubleIsoTau28er2p1_Mass_Max80 OR L1_DoubleIsoTau30er2p1_Mass_Max90 OR L1_DoubleIsoTau30er2p1_Mass_Max80 OR L1_Mu18er2p1_Tau24er2p1 OR L1_Mu18er2p1_Tau26er2p1 OR L1_Mu22er2p1_IsoTau28er2p1 OR L1_Mu22er2p1_IsoTau30er2p1 OR L1_Mu22er2p1_IsoTau32er2p1 OR L1_Mu22er2p1_IsoTau34er2p1 OR L1_Mu22er2p1_IsoTau36er2p1 OR L1_Mu22er2p1_IsoTau40er2p1 OR L1_Mu22er2p1_Tau70er2p1 OR L1_IsoTau40er2p1_ETMHF80 OR L1_IsoTau40er2p1_ETMHF90 OR L1_IsoTau40er2p1_ETMHF100 OR L1_IsoTau40er2p1_ETMHF110 OR L1_QuadJet36er2p5_IsoTau52er2p1 OR L1_DoubleJet35_Mass_Min450_IsoTau45_RmOvlp OR L1_DoubleJet_80_30_Mass_Min420_IsoTau40_RmOvlp" ),
        L1EGammaInputTag = cms.InputTag( 'hltGtStage2Digis','EGamma' ),
        L1JetInputTag = cms.InputTag( 'hltGtStage2Digis','Jet' ),
        saveTags = cms.bool( True ),
        L1ObjectMapInputTag = cms.InputTag( "hltGtStage2ObjectMap" ),
        L1EtSumInputTag = cms.InputTag( 'hltGtStage2Digis','EtSum' ),
        L1TauInputTag = cms.InputTag( 'hltGtStage2Digis','Tau' ),
        L1MuonInputTag = cms.InputTag( 'hltGtStage2Digis','Muon' ),
        L1GlobalInputTag = cms.InputTag( "hltGtStage2Digis" )
    )
    process.hltCaloTowerL1sTauExtremelyBigORSeededRegional = process.hltCaloTowerL1sTauVeryBigORSeededRegional.clone(
        TauTrigger = cms.InputTag('hltGtStage2Digis','Tau')
    )
    process.hltAkIsoTauL1sTauExtremelyBigORSeededRegional = process.hltAkIsoTauL1sTauVeryBigORSeededRegional.clone(
        src = cms.InputTag( "hltCaloTowerL1sTauExtremelyBigORSeededRegional" )
    )
    process.hltL2TauJetsL1TauSeededExtended = process.hltL2TauJetsL1TauSeeded.clone(
        JetSrc = cms.VInputTag( 'hltAkIsoTauL1sTauExtremelyBigORSeededRegional' )
    )
    process.hltL2TausForPixelIsolationL1TauSeededExtended = process.hltL2TausForPixelIsolationL1TauSeeded.clone(
        src = cms.InputTag( "hltL2TauJetsL1TauSeededExtended" )
    )
    process.hltSiPixelDigisRegL1TauSeededExtended = process.hltSiPixelDigisRegL1TauSeeded.clone()
    process.hltSiPixelDigisRegL1TauSeededExtended.Regions.inputs = cms.VInputTag( 'hltL2TausForPixelIsolationL1TauSeededExtended' )
    process.hltSiPixelClustersRegL1TauSeededExtended = process.hltSiPixelClustersRegL1TauSeeded.clone(
        src = cms.InputTag( "hltSiPixelDigisRegL1TauSeededExtended" )
    )
    process.hltSiPixelClustersRegL1TauSeededCacheExtended = process.hltSiPixelClustersRegL1TauSeededCache.clone(
        src = cms.InputTag( "hltSiPixelClustersRegL1TauSeededExtended" )
    )
    process.hltSiPixelRecHitsRegL1TauSeededExtended = process.hltSiPixelRecHitsRegL1TauSeeded.clone(
        src = cms.InputTag( "hltSiPixelClustersRegL1TauSeededExtended" )
    )
    process.hltPixelTracksTrackingRegionsRegL1TauSeededExtended = process.hltPixelTracksTrackingRegionsRegL1TauSeeded.clone()
    process.hltPixelTracksTrackingRegionsRegL1TauSeededExtended.RegionPSet.input = cms.InputTag( "hltL2TausForPixelIsolationL1TauSeededExtended" )
    process.hltPixelLayerQuadrupletsRegL1TauSeededExtended = process.hltPixelLayerQuadrupletsRegL1TauSeeded.clone()
    process.hltPixelLayerQuadrupletsRegL1TauSeededExtended.FPix.HitProducer = cms.string( "hltSiPixelRecHitsRegL1TauSeededExtended" )
    process.hltPixelLayerQuadrupletsRegL1TauSeededExtended.BPix.HitProducer = cms.string( "hltSiPixelRecHitsRegL1TauSeededExtended" )
    process.hltPixelTracksHitDoubletsRegL1TauSeededExtended = process.hltPixelTracksHitDoubletsRegL1TauSeeded.clone(
        trackingRegions = cms.InputTag( "hltPixelTracksTrackingRegionsRegL1TauSeededExtended" ),
        seedingLayers = cms.InputTag( "hltPixelLayerQuadrupletsRegL1TauSeededExtended" )
    )
    process.hltPixelTracksHitQuadrupletsRegL1TauSeededExtended = process.hltPixelTracksHitQuadrupletsRegL1TauSeeded.clone(
        doublets = cms.InputTag( "hltPixelTracksHitDoubletsRegL1TauSeededExtended" )
    )
    process.hltPixelTracksHitQuadrupletsRegL1TauSeededExtended.SeedComparitorPSet.clusterShapeCacheSrc = cms.InputTag( "hltSiPixelClustersRegL1TauSeededCacheExtended" )
    process.hltPixelTracksFromQuadrupletsRegL1TauSeededExtended = process.hltPixelTracksFromQuadrupletsRegL1TauSeeded.clone(
        SeedingHitSets = cms.InputTag( "hltPixelTracksHitQuadrupletsRegL1TauSeededExtended" )
    )
    process.hltPixelTripletsClustersRefRemovalRegL1TauSeededExtended = process.hltPixelTripletsClustersRefRemovalRegL1TauSeeded.clone(
        trajectories = cms.InputTag( "hltPixelTracksFromQuadrupletsRegL1TauSeededExtended" ),
        pixelClusters = cms.InputTag( "hltSiPixelClustersRegL1TauSeededExtended" )
    )
    process.hltPixelLayerTripletsWithClustersRemovalRegL1TauSeededExtended = process.hltPixelLayerTripletsWithClustersRemovalRegL1TauSeeded.clone()
    process.hltPixelLayerTripletsWithClustersRemovalRegL1TauSeededExtended.FPix.skipClusters = cms.InputTag( "hltPixelTripletsClustersRefRemovalRegL1TauSeededExtended" )
    process.hltPixelLayerTripletsWithClustersRemovalRegL1TauSeededExtended.FPix.HitProducer = cms.string( "hltSiPixelRecHitsRegL1TauSeededExtended" )
    process.hltPixelLayerTripletsWithClustersRemovalRegL1TauSeededExtended.BPix.skipClusters = cms.InputTag( "hltPixelTripletsClustersRefRemovalRegL1TauSeededExtended" )
    process.hltPixelLayerTripletsWithClustersRemovalRegL1TauSeededExtended.BPix.HitProducer = cms.string( "hltSiPixelRecHitsRegL1TauSeededExtended" )
    process.hltPixelTracksHitDoubletsForTripletsRegL1TauSeededExtended = process.hltPixelTracksHitDoubletsForTripletsRegL1TauSeeded.clone(
        trackingRegions = cms.InputTag( "hltPixelTracksTrackingRegionsRegL1TauSeededExtended" ),
        seedingLayers = cms.InputTag( "hltPixelLayerTripletsWithClustersRemovalRegL1TauSeededExtended" )
    )
    process.hltPixelTracksHitTripletsRegL1TauSeededExtended = process.hltPixelTracksHitTripletsRegL1TauSeeded.clone(
        doublets = cms.InputTag( "hltPixelTracksHitDoubletsForTripletsRegL1TauSeededExtended" )
    )
    process.hltPixelTracksHitTripletsRegL1TauSeededExtended.SeedComparitorPSet.clusterShapeCacheSrc = cms.InputTag( "hltSiPixelClustersRegL1TauSeededCacheExtended" )

    process.hltPixelTracksFromTripletsRegL1TauSeededExtended = process.hltPixelTracksFromTripletsRegL1TauSeeded.clone(
        SeedingHitSets = cms.InputTag( "hltPixelTracksHitTripletsRegL1TauSeededExtended" )
    )
    process.hltPixelTracksMergedRegL1TauSeededExtended = process.hltPixelTracksMergedRegL1TauSeeded.clone(
        selectedTrackQuals = cms.VInputTag( 'hltPixelTracksFromQuadrupletsRegL1TauSeededExtended','hltPixelTracksFromTripletsRegL1TauSeededExtended' ),
        TrackProducers = cms.VInputTag( 'hltPixelTracksFromQuadrupletsRegL1TauSeededExtended','hltPixelTracksFromTripletsRegL1TauSeededExtended' )
    )
    process.hltPixelVerticesRegL1TauSeededExtended = process.hltPixelVerticesRegL1TauSeeded.clone(
        TrackCollection = cms.InputTag( "hltPixelTracksMergedRegL1TauSeededExtended" )
    )
    process.hltL2TauPixelIsoTagProducerL1TauSeededExtended = process.hltL2TauPixelIsoTagProducerL1TauSeeded.clone(
        VertexSrc = cms.InputTag( "hltPixelVerticesRegL1TauSeededExtended" ),
        JetSrc = cms.InputTag( "hltL2TausForPixelIsolationL1TauSeededExtended" )
    )

    from HLTrigger.Configuration.customizeHLTforPatatrack import customizeHLTforPatatrack
    process = customizeHLTforPatatrack(process)
    process.TrainTupleProd = cms.EDAnalyzer("TrainTupleProducer",
        isMC = cms.bool(True),
        genEvent = cms.InputTag("generator"),
        #genParticles = cms.InputTag("genParticles"),
        puInfo = cms.InputTag("addPileupInfo"),
        l1taus=cms.InputTag('hltGtStage2Digis','Tau'),
        #l1taus = cms.InputTag("hltL1sTauExtremelyBigOR"), # oggetti di l1, piu basso livello possibile
        caloTowers = cms.InputTag("hltTowerMakerForAll"), # questo e proprio il nome del producer!!, comunque basandosi dove sono l1 taus sono ricostruite queste "calo towers" --> nel calo prendono i segnali in diversi cristalli (nella regione dei taul1, in un certo deltaR) e creano cluster
        caloTaus = cms.InputTag("hltAkIsoTauL1sTauExtremelyBigORSeededRegional"), # in base a questi clusters creano un jet, con algoritmo antiKt --> jet prodotti (noi li definiamo "tau")
        vertices = cms.InputTag("hltPixelVerticesRegL1TauSeededExtended"),
        pataVertices = cms.InputTag("hltTrimmedPixelVertices"), # queste sono le patatracks
        Tracks = cms.InputTag("hltPixelTracksMergedRegL1TauSeededExtended"), # pixel tracks, ricostruiti intorno ai jet (ricostruzione tracce pixel nella regione dove puntano i jets)
        pataTracks = cms.InputTag("hltPixelTracks"), # queste sono le patatracks
        #VeryBigOR = cms.InputTag("hltL1sDoubleTauBigOR"),
        #hltDoubleL2Tau26eta2p2 = cms.InputTag("hltDoubleL2Tau26eta2p2"),
        #hltDoubleL2IsoTau26eta2p2 = cms.InputTag("hltDoubleL2IsoTau26eta2p2")
        # step successivo -> isolamento basato su questi tau, si vedono le tracce attorno con algo semplici
    )


    process.HLTCaloTausCreatorL1TauSeededRegionalSequenceExtended = cms.Sequence( process.HLTDoCaloSequence + cms.ignore(process.hltL1sTauExtremelyBigOR) + process.hltCaloTowerL1sTauExtremelyBigORSeededRegional + process.hltAkIsoTauL1sTauExtremelyBigORSeededRegional )
    process.HLTL2TauJetsL1TauSeededSequenceExtended = cms.Sequence( process.HLTCaloTausCreatorL1TauSeededRegionalSequenceExtended + process.hltL2TauJetsL1TauSeededExtended )
    process.HLTPixelTrackFromQuadAndTriSequenceRegL1TauSeededExtended = cms.Sequence( process.hltPixelTracksFilter + process.hltPixelTracksFitter + process.hltPixelTracksTrackingRegionsRegL1TauSeededExtended + process.hltPixelLayerQuadrupletsRegL1TauSeededExtended + process.hltPixelTracksHitDoubletsRegL1TauSeededExtended + process.hltPixelTracksHitQuadrupletsRegL1TauSeededExtended + process.hltPixelTracksFromQuadrupletsRegL1TauSeededExtended + process.hltPixelTripletsClustersRefRemovalRegL1TauSeededExtended + process.hltPixelLayerTripletsWithClustersRemovalRegL1TauSeededExtended + process.hltPixelTracksHitDoubletsForTripletsRegL1TauSeededExtended + process.hltPixelTracksHitTripletsRegL1TauSeededExtended + process.hltPixelTracksFromTripletsRegL1TauSeededExtended + process.hltPixelTracksMergedRegL1TauSeededExtended )

    process.HLTDoLocalPixelSequenceRegL2TauL1TauSeededExtended = cms.Sequence( process.hltSiPixelDigisRegL1TauSeededExtended + process.hltSiPixelClustersRegL1TauSeededExtended + process.hltSiPixelClustersRegL1TauSeededCacheExtended + process.hltSiPixelRecHitsRegL1TauSeededExtended )

    process.HLTPixelTrackingSequenceRegL2TauL1TauSeededExtended = cms.Sequence( process.HLTDoLocalPixelSequenceRegL2TauL1TauSeededExtended + process.HLTPixelTrackFromQuadAndTriSequenceRegL1TauSeededExtended + process.hltPixelVerticesRegL1TauSeededExtended )

    process.HLTL2TauPixelIsolationSequenceL1TauSeededExtended = cms.Sequence( process.hltL2TausForPixelIsolationL1TauSeededExtended + process.HLTPixelTrackingSequenceRegL2TauL1TauSeededExtended + process.hltL2TauPixelIsoTagProducerL1TauSeededExtended )

    process.HLT_TauTupleProd = cms.Path(process.HLTBeginSequence + process.HLTL2TauJetsL1TauSeededSequenceExtended + process.HLTL2TauPixelIsolationSequenceL1TauSeededExtended, process.HLTDoLocalPixelTask, process.HLTRecoPixelTracksTask, process.HLTRecopixelvertexingTask)

    #process.HLT_TauTupleProd = cms.Path(process.HLTBeginSequence + process.HLTL2TauJetsL1TauSeededSequence +  process.HLTL2TauPixelIsolationSequenceL1TauSeeded + process.hltL1sDoubleTauBigOR + process.hltDoubleL2Tau26eta2p2 + process.hltL2TauIsoFilterL1TauSeeded + process.hltL2TauJetsIsoL1TauSeeded + process.hltDoubleL2IsoTau26eta2p2, process.HLTDoLocalPixelTask, process.HLTRecoPixelTracksTask, process.HLTRecopixelvertexingTask)
    process.endjob_step.insert(0, process.TrainTupleProd)
    # process.HLT_DoubleMediumChargedIsoPFTauHPS35_Trk1_eta2p1_Reg_v4,
    process.schedule = cms.Schedule(*[ process.HLTriggerFirstPath, process.HLT_TauTupleProd, process.HLTriggerFinalPath, process.endjob_step ], tasks=[process.patAlgosToolsTask])
    process.TFileService = cms.Service('TFileService', fileName = cms.string("ntuple_prova_4.root") )

    process.options.wantSummary = cms.untracked.bool(True)
    return process
