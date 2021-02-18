import FWCore.ParameterSet.Config as cms
def update(process):
    from HLTrigger.Configuration.customizeHLTforPatatrack import customizeHLTforPatatrack
    process = customizeHLTforPatatrack(process)

    #extremely big or da definire, e poi tutto diventa *Extended

    process.hltL1sTauExtremelyBigOR = cms.EDFilter( "HLTL1TSeed",
        L1SeedsLogicalExpression = cms.string( "L1_LooseIsoEG22er2p1_IsoTau26er2p1_dR_Min0p3  OR L1_LooseIsoEG24er2p1_IsoTau27er2p1_dR_Min0p3  OR L1_LooseIsoEG22er2p1_Tau70er2p1_dR_Min0p3  OR L1_SingleTau120er2p1  OR L1_SingleTau130er2p1  OR L1_DoubleTau70er2p1 OR L1_DoubleIsoTau28er2p1  OR L1_DoubleIsoTau30er2p1 OR L1_DoubleIsoTau32er2p1 OR L1_DoubleIsoTau34er2p1 OR L1_DoubleIsoTau36er2p1 OR L1_DoubleIsoTau28er2p1_Mass_Max90 OR L1_DoubleIsoTau28er2p1_Mass_Max80 OR L1_DoubleIsoTau30er2p1_Mass_Max90 OR L1_DoubleIsoTau30er2p1_Mass_Max80 OR L1_Mu18er2p1_Tau24er2p1  OR L1_Mu18er2p1_Tau26er2p1  OR L1_Mu22er2p1_IsoTau28er2p1 OR L1_Mu22er2p1_IsoTau30er2p1 OR L1_Mu22er2p1_IsoTau32er2p1 OR L1_Mu22er2p1_IsoTau34er2p1 OR L1_Mu22er2p1_IsoTau36er2p1 OR L1_Mu22er2p1_IsoTau40er2p1  OR L1_Mu22er2p1_Tau70er2p1 OR L1_IsoTau40er2p1_ETMHF80  OR L1_IsoTau40er2p1_ETMHF90 OR L1_IsoTau40er2p1_ETMHF100 OR L1_IsoTau40er2p1_ETMHF110 OR L1_QuadJet36er2p5_IsoTau52er2p1  OR L1_DoubleJet35_Mass_Min450_IsoTau45_RmOvlp  OR L1_DoubleJet_80_30_Mass_Min420_IsoTau40_RmOvlp" ),
        L1EGammaInputTag = cms.InputTag( 'hltGtStage2Digis','EGamma' ),
        L1JetInputTag = cms.InputTag( 'hltGtStage2Digis','Jet' ),
        saveTags = cms.bool( True ),
        L1ObjectMapInputTag = cms.InputTag( "hltGtStage2ObjectMap" ),
        L1EtSumInputTag = cms.InputTag( 'hltGtStage2Digis','EtSum' ),
        L1TauInputTag = cms.InputTag( 'hltGtStage2Digis','Tau' ),
        L1MuonInputTag = cms.InputTag( 'hltGtStage2Digis','Muon' ),
        L1GlobalInputTag = cms.InputTag( "hltGtStage2Digis" )
    )
    process.hltCaloTowerL1sTauExtremelyBigORSeededRegional = cms.EDProducer( "CaloTowerFromL1TSeededCreatorForTauHLT",
        verbose = cms.untracked.int32( 0 ),
        towers = cms.InputTag( "hltTowerMakerForAll" ),
        TauTrigger = cms.InputTag( "hltL1sTauExtremelyBigOR" ),
        minimumE = cms.double( 0.8 ),
        UseTowersInCone = cms.double( 0.8 ),
        minimumEt = cms.double( 0.5 )
    )
    process.hltAkIsoTauL1sTauExtremelyBigORSeededRegional = cms.EDProducer( "FastjetJetProducer",
        Active_Area_Repeats = cms.int32( 5 ),
        useMassDropTagger = cms.bool( False ),
        doAreaFastjet = cms.bool( False ),
        muMin = cms.double( -1.0 ),
        Ghost_EtaMax = cms.double( 6.0 ),
        maxBadHcalCells = cms.uint32( 9999999 ),
        maxRecoveredHcalCells = cms.uint32( 9999999 ),
        applyWeight = cms.bool( False ),
        doAreaDiskApprox = cms.bool( False ),
        subtractorName = cms.string( "" ),
        dRMax = cms.double( -1.0 ),
        useExplicitGhosts = cms.bool( False ),
        puWidth = cms.double( 0.0 ),
        maxRecoveredEcalCells = cms.uint32( 9999999 ),
        R0 = cms.double( -1.0 ),
        jetType = cms.string( "CaloJet" ),
        muCut = cms.double( -1.0 ),
        subjetPtMin = cms.double( -1.0 ),
        csRParam = cms.double( -1.0 ),
        MinVtxNdof = cms.int32( 5 ),
        minSeed = cms.uint32( 0 ),
        voronoiRfact = cms.double( -9.0 ),
        doRhoFastjet = cms.bool( False ),
        jetAlgorithm = cms.string( "AntiKt" ),
        writeCompound = cms.bool( False ),
        muMax = cms.double( -1.0 ),
        nSigmaPU = cms.double( 1.0 ),
        GhostArea = cms.double( 0.01 ),
        Rho_EtaMax = cms.double( 4.4 ),
        restrictInputs = cms.bool( False ),
        nExclude = cms.uint32( 0 ),
        yMin = cms.double( -1.0 ),
        srcWeights = cms.InputTag( "" ),
        maxBadEcalCells = cms.uint32( 9999999 ),
        jetCollInstanceName = cms.string( "" ),
        useFiltering = cms.bool( False ),
        maxInputs = cms.uint32( 1 ),
        rFiltFactor = cms.double( -1.0 ),
        useDeterministicSeed = cms.bool( True ),
        doPVCorrection = cms.bool( False ),
        rFilt = cms.double( -1.0 ),
        yMax = cms.double( -1.0 ),
        zcut = cms.double( -1.0 ),
        useTrimming = cms.bool( False ),
        puCenters = cms.vdouble(  ),
        MaxVtxZ = cms.double( 15.0 ),
        rParam = cms.double( 0.2 ),
        csRho_EtaMax = cms.double( -1.0 ),
        UseOnlyVertexTracks = cms.bool( False ),
        dRMin = cms.double( -1.0 ),
        gridSpacing = cms.double( -1.0 ),
        minimumTowersFraction = cms.double( 0.0 ),
        doFastJetNonUniform = cms.bool( False ),
        usePruning = cms.bool( False ),
        maxDepth = cms.int32( -1 ),
        yCut = cms.double( -1.0 ),
        useSoftDrop = cms.bool( False ),
        DzTrVtxMax = cms.double( 0.0 ),
        UseOnlyOnePV = cms.bool( False ),
        maxProblematicHcalCells = cms.uint32( 9999999 ),
        correctShape = cms.bool( False ),
        rcut_factor = cms.double( -1.0 ),
        src = cms.InputTag( "hltCaloTowerL1sTauExtremelyBigORSeededRegional" ),
        gridMaxRapidity = cms.double( -1.0 ),
        sumRecHits = cms.bool( False ),
        jetPtMin = cms.double( 1.0 ),
        puPtMin = cms.double( 10.0 ),
        useDynamicFiltering = cms.bool( False ),
        verbosity = cms.int32( 0 ),
        inputEtMin = cms.double( 0.3 ),
        useConstituentSubtraction = cms.bool( False ),
        beta = cms.double( -1.0 ),
        trimPtFracMin = cms.double( -1.0 ),
        radiusPU = cms.double( 0.4 ),
        nFilt = cms.int32( -1 ),
        useKtPruning = cms.bool( False ),
        DxyTrVtxMax = cms.double( 0.0 ),
        maxProblematicEcalCells = cms.uint32( 9999999 ),
        srcPVs = cms.InputTag( "NotUsed" ),
        useCMSBoostedTauSeedingAlgorithm = cms.bool( False ),
        doPUOffsetCorr = cms.bool( False ),
        writeJetsWithConst = cms.bool( False ),
        inputEMin = cms.double( 0.0 )
    )
    process.hltL2TauJetsL1ExtremelyBigORTauSeeded = cms.EDProducer( "L2TauJetsMerger",
        EtMin = cms.double( 20.0 ),
        JetSrc = cms.VInputTag( 'hltAkIsoTauL1sTauExtremelyBigORSeededRegional' )
    )
    process.hltL2TausForPixelIsolationL1TauSeededExtended = cms.EDFilter( "CaloJetSelector",
        filter = cms.bool( False ),
        src = cms.InputTag( "hltL2TauJetsL1ExtremelyBigORTauSeeded" ),
        cut = cms.string( "pt > 20 & abs(eta) < 2.5" )
    )
    process.hltSiPixelDigisRegL1TauSeededExtended = cms.EDProducer( "SiPixelRawToDigi",
        UseQualityInfo = cms.bool( False ),
        UsePilotBlade = cms.bool( False ),
        UsePhase1 = cms.bool( True ),
        InputLabel = cms.InputTag( "rawDataCollector" ),
        IncludeErrors = cms.bool( False ),
        ErrorList = cms.vint32(  ),
        Regions = cms.PSet(
          maxZ = cms.vdouble( 24.0 ),
          inputs = cms.VInputTag( 'hltL2TausForPixelIsolationL1TauSeededExtended' ),
          beamSpot = cms.InputTag( "hltOnlineBeamSpot" ),
          deltaPhi = cms.vdouble( 0.5 )
        ),
        Timing = cms.untracked.bool( False ),
        CablingMapLabel = cms.string( "" ),
        UserErrorList = cms.vint32(  )
    )
    process.hltSiPixelClustersRegL1TauSeededExtended = cms.EDProducer( "SiPixelClusterProducer",
        src = cms.InputTag( "hltSiPixelDigisRegL1TauSeededExtended" ),
        ChannelThreshold = cms.int32( 1000 ),
        Phase2DigiBaseline = cms.double( 1200.0 ),
        ElectronPerADCGain = cms.double( 135.0 ),
        Phase2ReadoutMode = cms.int32( -1 ),
        maxNumberOfClusters = cms.int32( 20000 ),
        ClusterThreshold_L1 = cms.int32( 2000 ),
        MissCalibrate = cms.bool( True ),
        VCaltoElectronGain = cms.int32( 1 ),
        VCaltoElectronGain_L1 = cms.int32( 1 ),
        VCaltoElectronOffset = cms.int32( 0 ),
        SplitClusters = cms.bool( False ),
        payloadType = cms.string( "HLT" ),
        Phase2Calibration = cms.bool( False ),
        Phase2KinkADC = cms.int32( 8 ),
        ClusterMode = cms.string( "PixelThresholdClusterizer" ),
        SeedThreshold = cms.int32( 1000 ),
        VCaltoElectronOffset_L1 = cms.int32( 0 ),
        ClusterThreshold = cms.int32( 4000 )
    )
    process.hltSiPixelClustersRegL1TauSeededCacheExtended = cms.EDProducer( "SiPixelClusterShapeCacheProducer",
        src = cms.InputTag( "hltSiPixelClustersRegL1TauSeededExtended" ),
        onDemand = cms.bool( False )
    )
    process.hltSiPixelRecHitsRegL1TauSeededExtended = cms.EDProducer( "SiPixelRecHitConverter",
        VerboseLevel = cms.untracked.int32( 0 ),
        src = cms.InputTag( "hltSiPixelClustersRegL1TauSeededExtended" ),
        CPE = cms.string( "hltESPPixelCPEGeneric" )
    )
    process.hltPixelTracksTrackingRegionsRegL1TauSeededExtended = cms.EDProducer( "CandidateSeededTrackingRegionsEDProducer",
        RegionPSet = cms.PSet(
          vertexCollection = cms.InputTag( "" ),
          zErrorVetex = cms.double( 0.2 ),
          beamSpot = cms.InputTag( "hltOnlineBeamSpot" ),
          zErrorBeamSpot = cms.double( 24.2 ),
          maxNVertices = cms.int32( 1 ),
          maxNRegions = cms.int32( 10 ),
          nSigmaZVertex = cms.double( 3.0 ),
          nSigmaZBeamSpot = cms.double( 4.0 ),
          ptMin = cms.double( 0.9 ),
          mode = cms.string( "BeamSpotSigma" ),
          input = cms.InputTag( "hltL2TausForPixelIsolationL1TauSeededExtended" ),
          searchOpt = cms.bool( False ),
          whereToUseMeasurementTracker = cms.string( "Never" ),
          originRadius = cms.double( 0.2 ),
          measurementTrackerName = cms.InputTag( "" ),
          precise = cms.bool( True ),
          deltaEta = cms.double( 0.5 ),
          deltaPhi = cms.double( 0.5 )
        )
    )
    process.hltPixelLayerQuadrupletsRegL1TauSeededExtended = cms.EDProducer( "SeedingLayersEDProducer",
        layerList = cms.vstring( 'BPix1+BPix2+BPix3+BPix4',
          'BPix1+BPix2+BPix3+FPix1_pos',
          'BPix1+BPix2+BPix3+FPix1_neg',
          'BPix1+BPix2+FPix1_pos+FPix2_pos',
          'BPix1+BPix2+FPix1_neg+FPix2_neg',
          'BPix1+FPix1_pos+FPix2_pos+FPix3_pos',
          'BPix1+FPix1_neg+FPix2_neg+FPix3_neg' ),
        MTOB = cms.PSet(  ),
        TEC = cms.PSet(  ),
        MTID = cms.PSet(  ),
        FPix = cms.PSet(
          hitErrorRPhi = cms.double( 0.0051 ),
          TTRHBuilder = cms.string( "hltESPTTRHBuilderPixelOnly" ),
          useErrorsFromParam = cms.bool( True ),
          hitErrorRZ = cms.double( 0.0036 ),
          HitProducer = cms.string( "hltSiPixelRecHitsRegL1TauSeededExtended" )
        ),
        MTEC = cms.PSet(  ),
        MTIB = cms.PSet(  ),
        TID = cms.PSet(  ),
        TOB = cms.PSet(  ),
        BPix = cms.PSet(
          hitErrorRPhi = cms.double( 0.0027 ),
          TTRHBuilder = cms.string( "hltESPTTRHBuilderPixelOnly" ),
          useErrorsFromParam = cms.bool( True ),
          hitErrorRZ = cms.double( 0.006 ),
          HitProducer = cms.string( "hltSiPixelRecHitsRegL1TauSeededExtended" )
        ),
        TIB = cms.PSet(  )
    )
    process.hltPixelTracksHitDoubletsRegL1TauSeededExtended = cms.EDProducer( "HitPairEDProducer",
        trackingRegions = cms.InputTag( "hltPixelTracksTrackingRegionsRegL1TauSeededExtended" ),
        layerPairs = cms.vuint32( 0, 1, 2 ),
        clusterCheck = cms.InputTag( "" ),
        produceSeedingHitSets = cms.bool( False ),
        produceIntermediateHitDoublets = cms.bool( True ),
        trackingRegionsSeedingLayers = cms.InputTag( "" ),
        maxElementTotal = cms.uint32( 50000000 ),
        maxElement = cms.uint32( 0 ),
        seedingLayers = cms.InputTag( "hltPixelLayerQuadrupletsRegL1TauSeededExtended" )
    )
    process.hltPixelTracksHitQuadrupletsRegL1TauSeededExtended = cms.EDProducer( "CAHitQuadrupletEDProducer",
        CAHardPtCut = cms.double( 0.0 ),
        CAPhiCut_byTriplets = cms.VPSet(
          cms.PSet(  seedingLayers = cms.string( "" ),
            cut = cms.double( -1.0 )
          )
        ),
        extraHitRPhitolerance = cms.double( 0.032 ),
        doublets = cms.InputTag( "hltPixelTracksHitDoubletsRegL1TauSeededExtended" ),
        fitFastCircle = cms.bool( True ),
        maxChi2 = cms.PSet(
          value2 = cms.double( 50.0 ),
          value1 = cms.double( 200.0 ),
          pt1 = cms.double( 0.7 ),
          enabled = cms.bool( True ),
          pt2 = cms.double( 2.0 )
        ),
        CAThetaCut = cms.double( 0.002 ),
        SeedComparitorPSet = cms.PSet(
          clusterShapeHitFilter = cms.string( "ClusterShapeHitFilter" ),
          ComponentName = cms.string( "LowPtClusterShapeSeedComparitor" ),
          clusterShapeCacheSrc = cms.InputTag( "hltSiPixelClustersRegL1TauSeededCacheExtended" )
        ),
        CAThetaCut_byTriplets = cms.VPSet(
          cms.PSet(  seedingLayers = cms.string( "" ),
            cut = cms.double( -1.0 )
          )
        ),
        CAPhiCut = cms.double( 0.2 ),
        useBendingCorrection = cms.bool( True ),
        fitFastCircleChi2Cut = cms.bool( True )
    )
    process.hltPixelTracksFromQuadrupletsRegL1TauSeededExtended = cms.EDProducer( "PixelTrackProducer",
        Cleaner = cms.string( "hltPixelTracksCleanerBySharedHits" ),
        passLabel = cms.string( "" ),
        Fitter = cms.InputTag( "hltPixelTracksFitter" ),
        Filter = cms.InputTag( "hltPixelTracksFilter" ),
        SeedingHitSets = cms.InputTag( "hltPixelTracksHitQuadrupletsRegL1TauSeededExtended" )
    )
    process.hltPixelTripletsClustersRefRemovalRegL1TauSeededExtended = cms.EDProducer( "TrackClusterRemover",
        trackClassifier = cms.InputTag( '','QualityMasks' ),
        minNumberOfLayersWithMeasBeforeFiltering = cms.int32( 0 ),
        maxChi2 = cms.double( 3000.0 ),
        trajectories = cms.InputTag( "hltPixelTracksFromQuadrupletsRegL1TauSeededExtended" ),
        oldClusterRemovalInfo = cms.InputTag( "" ),
        stripClusters = cms.InputTag( "" ),
        overrideTrkQuals = cms.InputTag( "" ),
        pixelClusters = cms.InputTag( "hltSiPixelClustersRegL1TauSeededExtended" ),
        TrackQuality = cms.string( "undefQuality" )
    )
    process.hltPixelLayerTripletsWithClustersRemovalRegL1TauSeededExtended = cms.EDProducer( "SeedingLayersEDProducer",
        layerList = cms.vstring( 'BPix1+BPix2+BPix3',
          'BPix2+BPix3+BPix4',
          'BPix1+BPix3+BPix4',
          'BPix1+BPix2+BPix4',
          'BPix2+BPix3+FPix1_pos',
          'BPix2+BPix3+FPix1_neg',
          'BPix1+BPix2+FPix1_pos',
          'BPix1+BPix2+FPix1_neg',
          'BPix2+FPix1_pos+FPix2_pos',
          'BPix2+FPix1_neg+FPix2_neg',
          'BPix1+FPix1_pos+FPix2_pos',
          'BPix1+FPix1_neg+FPix2_neg',
          'FPix1_pos+FPix2_pos+FPix3_pos',
          'FPix1_neg+FPix2_neg+FPix3_neg',
          'BPix1+BPix3+FPix1_pos',
          'BPix1+BPix2+FPix2_pos',
          'BPix1+BPix3+FPix1_neg',
          'BPix1+BPix2+FPix2_neg',
          'BPix1+FPix2_neg+FPix3_neg',
          'BPix1+FPix1_neg+FPix3_neg',
          'BPix1+FPix2_pos+FPix3_pos',
          'BPix1+FPix1_pos+FPix3_pos' ),
        MTOB = cms.PSet(  ),
        TEC = cms.PSet(  ),
        MTID = cms.PSet(  ),
        FPix = cms.PSet(
          hitErrorRPhi = cms.double( 0.0051 ),
          TTRHBuilder = cms.string( "hltESPTTRHBuilderPixelOnly" ),
          skipClusters = cms.InputTag( "hltPixelTripletsClustersRefRemovalRegL1TauSeededExtended" ),
          useErrorsFromParam = cms.bool( True ),
          hitErrorRZ = cms.double( 0.0036 ),
          HitProducer = cms.string( "hltSiPixelRecHitsRegL1TauSeededExtended" )
        ),
        MTEC = cms.PSet(  ),
        MTIB = cms.PSet(  ),
        TID = cms.PSet(  ),
        TOB = cms.PSet(  ),
        BPix = cms.PSet(
          hitErrorRPhi = cms.double( 0.0027 ),
          TTRHBuilder = cms.string( "hltESPTTRHBuilderPixelOnly" ),
          skipClusters = cms.InputTag( "hltPixelTripletsClustersRefRemovalRegL1TauSeeded" ),
          useErrorsFromParam = cms.bool( True ),
          hitErrorRZ = cms.double( 0.006 ),
          HitProducer = cms.string( "hltSiPixelRecHitsRegL1TauSeededExtended" )
        ),
        TIB = cms.PSet(  )
    )
    process.hltPixelTracksHitDoubletsForTripletsRegL1TauSeededExtended = cms.EDProducer( "HitPairEDProducer",
        trackingRegions = cms.InputTag( "hltPixelTracksTrackingRegionsRegL1TauSeededExtended" ),
        layerPairs = cms.vuint32( 0, 1 ),
        clusterCheck = cms.InputTag( "" ),
        produceSeedingHitSets = cms.bool( False ),
        produceIntermediateHitDoublets = cms.bool( True ),
        trackingRegionsSeedingLayers = cms.InputTag( "" ),
        maxElementTotal = cms.uint32( 50000000 ),
        maxElement = cms.uint32( 0 ),
        seedingLayers = cms.InputTag( "hltPixelLayerTripletsWithClustersRemovalRegL1TauSeededExtended" )
    )
    process.hltPixelTracksHitTripletsRegL1TauSeededExtended = cms.EDProducer( "CAHitTripletEDProducer",
        CAThetaCut = cms.double( 0.002 ),
        CAPhiCut_byTriplets = cms.VPSet(
          cms.PSet(  seedingLayers = cms.string( "" ),
            cut = cms.double( -1.0 )
          )
        ),
        maxChi2 = cms.PSet(
          value2 = cms.double( 50.0 ),
          value1 = cms.double( 200.0 ),
          pt1 = cms.double( 0.7 ),
          enabled = cms.bool( False ),
          pt2 = cms.double( 2.0 )
        ),
        doublets = cms.InputTag( "hltPixelTracksHitDoubletsForTripletsRegL1TauSeededExtended" ),
        CAHardPtCut = cms.double( 0.0 ),
        SeedComparitorPSet = cms.PSet(
          clusterShapeHitFilter = cms.string( "ClusterShapeHitFilter" ),
          ComponentName = cms.string( "LowPtClusterShapeSeedComparitor" ),
          clusterShapeCacheSrc = cms.InputTag( "hltSiPixelClustersRegL1TauSeededCacheExtended" )
        ),
        CAThetaCut_byTriplets = cms.VPSet(
          cms.PSet(  seedingLayers = cms.string( "" ),
            cut = cms.double( -1.0 )
          )
        ),
        CAPhiCut = cms.double( 0.2 ),
        useBendingCorrection = cms.bool( True ),
        extraHitRPhitolerance = cms.double( 0.032 )
    )
    process.hltPixelTracksFromTripletsRegL1TauSeededExtended = cms.EDProducer( "PixelTrackProducer",
        Cleaner = cms.string( "hltPixelTracksCleanerBySharedHits" ),
        passLabel = cms.string( "" ),
        Fitter = cms.InputTag( "hltPixelTracksFitter" ),
        Filter = cms.InputTag( "hltPixelTracksFilter" ),
        SeedingHitSets = cms.InputTag( "hltPixelTracksHitTripletsRegL1TauSeededExtended" )
    )
    process.hltPixelTracksMergedRegL1TauSeededExtended = cms.EDProducer( "TrackListMerger",
        ShareFrac = cms.double( 0.19 ),
        writeOnlyTrkQuals = cms.bool( False ),
        MinPT = cms.double( 0.05 ),
        allowFirstHitShare = cms.bool( True ),
        copyExtras = cms.untracked.bool( True ),
        Epsilon = cms.double( -0.001 ),
        selectedTrackQuals = cms.VInputTag( 'hltPixelTracksFromQuadrupletsRegL1TauSeededExtended','hltPixelTracksFromTripletsRegL1TauSeededExtended' ),
        indivShareFrac = cms.vdouble( 1.0, 1.0 ),
        MaxNormalizedChisq = cms.double( 1000.0 ),
        copyMVA = cms.bool( False ),
        FoundHitBonus = cms.double( 5.0 ),
        LostHitPenalty = cms.double( 20.0 ),
        setsToMerge = cms.VPSet(
          cms.PSet(  pQual = cms.bool( False ),
            tLists = cms.vint32( 0, 1 )
          )
        ),
        MinFound = cms.int32( 3 ),
        hasSelector = cms.vint32( 0, 0 ),
        TrackProducers = cms.VInputTag( 'hltPixelTracksFromQuadrupletsRegL1TauSeededExtended','hltPixelTracksFromTripletsRegL1TauSeededExtended' ),
        trackAlgoPriorityOrder = cms.string( "hltESPTrackAlgoPriorityOrder" ),
        newQuality = cms.string( "confirmed" )
    )
    process.hltPixelVerticesRegL1TauSeededExtended = cms.EDProducer( "PixelVertexProducer",
        WtAverage = cms.bool( True ),
        Method2 = cms.bool( True ),
        beamSpot = cms.InputTag( "hltOnlineBeamSpot" ),
        PVcomparer = cms.PSet(  refToPSet_ = cms.string( "HLTPSetPvClusterComparerForIT" ) ),
        Verbosity = cms.int32( 0 ),
        UseError = cms.bool( True ),
        TrackCollection = cms.InputTag( "hltPixelTracksMergedRegL1TauSeededExtended" ),
        PtMin = cms.double( 1.0 ),
        NTrkMin = cms.int32( 2 ),
        ZOffset = cms.double( 5.0 ),
        Finder = cms.string( "DivisiveVertexFinder" ),
        ZSeparation = cms.double( 0.05 )
    )

    process.TrainTupleProd = cms.EDAnalyzer("TrainTupleProducer",
        isMC = cms.bool(True),
        genEvent = cms.InputTag("generator"),
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

    process.HLTL2TauJetsL1TauSeededSequenceExtended = cms.Sequence( process.HLTCaloTausCreatorL1TauSeededRegionalSequenceExtended + process.hltL2TauJetsL1ExtremelyBigORTauSeeded )

    process.HLT_TauTupleProd = cms.Path(process.HLTBeginSequence + process.HLTL2TauJetsL1TauSeededSequenceExtended, process.HLTDoLocalPixelTask, process.HLTRecoPixelTracksTask, process.HLTRecopixelvertexingTask)

    #process.HLT_TauTupleProd = cms.Path(process.HLTBeginSequence + process.HLTL2TauJetsL1TauSeededSequence +  process.HLTL2TauPixelIsolationSequenceL1TauSeeded + process.hltL1sDoubleTauBigOR + process.hltDoubleL2Tau26eta2p2 + process.hltL2TauIsoFilterL1TauSeeded + process.hltL2TauJetsIsoL1TauSeeded + process.hltDoubleL2IsoTau26eta2p2, process.HLTDoLocalPixelTask, process.HLTRecoPixelTracksTask, process.HLTRecopixelvertexingTask)
    process.endjob_step.insert(0, process.TrainTupleProd)
    process.schedule = cms.Schedule(*[ process.HLTriggerFirstPath, process.HLT_TauTupleProd, process.HLT_DoubleMediumChargedIsoPFTauHPS35_Trk1_eta2p1_Reg_v4, process.HLTriggerFinalPath, process.endjob_step ], tasks=[process.patAlgosToolsTask])
    process.TFileService = cms.Service('TFileService', fileName = cms.string("ntuple_prova_4.root") )

    process.options.wantSummary = cms.untracked.bool(True)
    return process
