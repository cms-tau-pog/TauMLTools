import FWCore.ParameterSet.Config as cms
def update(process):
    from HLTrigger.Configuration.customizeHLTforPatatrack import customizeHLTforPatatrack
    process = customizeHLTforPatatrack(process)
    process.TrainTupleProd = cms.EDAnalyzer("TrainTupleProducer",
        isMC = cms.bool(True),
        genEvent = cms.InputTag("generator"),
        puInfo = cms.InputTag("addPileupInfo"),
        l1taus=cms.InputTag('hltGtStage2Digis','Tau'),
        #l1taus = cms.InputTag("hltL1sTauVeryBigOR"), # oggetti di l1, piu basso livello possibile
        caloTowers = cms.InputTag("hltTowerMakerForAll"), # questo e proprio il nome del producer!!, comunque basandosi dove sono l1 taus sono ricostruite queste "calo towers" --> nel calo prendono i segnali in diversi cristalli (nella regione dei taul1, in un certo deltaR) e creano cluster
        caloTaus = cms.InputTag("hltAkIsoTauL1sTauVeryBigORSeededRegional"), # in base a questi clusters creano un jet, con algoritmo antiKt --> jet prodotti (noi li definiamo "tau")
        vertices = cms.InputTag("hltPixelVerticesRegL1TauSeeded"),
        pataVertices = cms.InputTag("hltTrimmedPixelVertices"), # queste sono le patatracks
        pixelTracks = cms.InputTag("hltPixelTracksMergedRegL1TauSeeded"), # pixel tracks, ricostruiti intorno ai jet (ricostruzione tracce pixel nella regione dove puntano i jets)
        pataTracks = cms.InputTag("hltPixelTracks"), # queste sono le patatracks
        VeryBigOR= cms.InputTag("hltL1sDoubleTauBigOR"),
        hltDoubleL2Tau26eta2p2 = cms.InputTag("hltDoubleL2Tau26eta2p2"),
        hltDoubleL2IsoTau26eta2p2 = cms.InputTag("hltDoubleL2IsoTau26eta2p2")

        # step successivo -> isolamento basato su questi tau, si vedono le tracce attorno con algo semplici
    )

    process.HLTL2p5IsoTauL1TauSeededSequence = cms.Sequence( process.HLTL2TauPixelIsolationSequenceL1TauSeeded +  cms.ignore(process.hltL2TauIsoFilterL1TauSeeded) + process.hltL2TauJetsIsoL1TauSeeded)

    process.HLT_TauTupleProd = cms.Path(process.HLTBeginSequence + cms.ignore(process.hltL1sDoubleTauBigOR) +  process.HLTL2TauJetsL1TauSeededSequence + cms.ignore(process.hltDoubleL2Tau26eta2p2) + process.HLTL2p5IsoTauL1TauSeededSequence + cms.ignore(process.hltDoubleL2IsoTau26eta2p2) + process.TrainTupleProd, process.HLTDoLocalPixelTask, process.HLTRecoPixelTracksTask, process.HLTRecopixelvertexingTask, process.HLTDoFullUnpackingEgammaEcalTask, process.HLTDoLocalHcalTask)

    process.schedule = cms.Schedule(*[ process.HLTriggerFirstPath, process.HLT_TauTupleProd, process.HLTriggerFinalPath, process.endjob_step ], tasks=[process.patAlgosToolsTask])
    process.TFileService = cms.Service('TFileService', fileName = cms.string("ntuple_prova_3.root") )

    process.options.wantSummary = cms.untracked.bool(False)
    return process
