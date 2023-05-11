from HLTrigger.Configuration.common import *
import FWCore.ParameterSet.Config as cms
import os

HCAL_PFclusters_2023=[0.6,0.5,0.5,0.5]
HCAL_PFrechits_2023=[0.4,0.3,0.3,0.3]

HCAL_PFclusters_2022=[0.125, 0.25, 0.35, 0.35]
HCAL_PFrechits_2022=[0.1, 0.2, 0.3, 0.3]

def customizeHCALFor2023(process):
    return customizeHCAL(process, HCAL_PFclusters_2023, HCAL_PFrechits_2023)

def customizeHCALFor2022(process):
    return customizeHCAL(process, HCAL_PFclusters_2022, HCAL_PFrechits_2022)

def customizeHCAL(process, HCAL_PFclusters, HCAL_PFrechits):
    if hasattr(process, "hltParticleFlowClusterHBHE"):
        process.hltParticleFlowClusterHBHE.seedFinder.thresholdsByDetector[0].seedingThreshold=HCAL_PFclusters
        process.hltParticleFlowClusterHBHE.initialClusteringStep.thresholdsByDetector[0].gatheringThreshold=HCAL_PFrechits
        process.hltParticleFlowClusterHBHE.pfClusterBuilder.recHitEnergyNorms[0].recHitEnergyNorm=HCAL_PFrechits
        process.hltParticleFlowClusterHBHE.pfClusterBuilder.positionCalc.logWeightDenominatorByDetector[0].logWeightDenominator=HCAL_PFrechits
        process.hltParticleFlowClusterHBHE.pfClusterBuilder.allCellsPositionCalc.logWeightDenominatorByDetector[0].logWeightDenominator=HCAL_PFrechits
    if hasattr(process, "hltParticleFlowClusterHCAL"):
        process.hltParticleFlowClusterHCAL.pfClusterBuilder.allCellsPositionCalc.logWeightDenominatorByDetector[0].logWeightDenominator=HCAL_PFrechits
    if hasattr(process, "hltParticleFlowRecHitHBHE"):
        process.hltParticleFlowRecHitHBHE.producers[0].qualityTests[0].cuts[0].threshold=HCAL_PFrechits

    if hasattr(process, "hltEgammaHoverE"):
        process.hltEgammaHoverE.eThresHB=HCAL_PFrechits
    if hasattr(process, "hltEgammaHoverEUnseeded"):
        process.hltEgammaHoverEUnseeded.eThresHB=HCAL_PFrechits
    if hasattr(process, "hltEgammaHToverET"):
        process.hltEgammaHToverET.eThresHB=HCAL_PFrechits
    if hasattr(process, "hltFixedGridRhoFastjetECALMFForMuons"):
        process.hltFixedGridRhoFastjetECALMFForMuons.eThresHB=HCAL_PFrechits
    if hasattr(process, "hltFixedGridRhoFastjetAllCaloForMuons"):
        process.hltFixedGridRhoFastjetAllCaloForMuons.eThresHB=HCAL_PFrechits
    if hasattr(process, "hltFixedGridRhoFastjetHCAL"):
        process.hltFixedGridRhoFastjetHCAL.eThresHB=HCAL_PFrechits

    if hasattr(process, "hltParticleFlowClusterHBHECPUOnly"):
        process.hltParticleFlowClusterHBHECPUOnly.seedFinder.thresholdsByDetector[0].seedingThreshold=HCAL_PFclusters
        process.hltParticleFlowClusterHBHECPUOnly.initialClusteringStep.thresholdsByDetector[0].gatheringThreshold=HCAL_PFrechits
        process.hltParticleFlowClusterHBHECPUOnly.pfClusterBuilder.recHitEnergyNorms[0].recHitEnergyNorm=HCAL_PFrechits
        process.hltParticleFlowClusterHBHECPUOnly.pfClusterBuilder.positionCalc.logWeightDenominatorByDetector[0].logWeightDenominator=HCAL_PFrechits
        process.hltParticleFlowClusterHBHECPUOnly.pfClusterBuilder.allCellsPositionCalc.logWeightDenominatorByDetector[0].logWeightDenominator=HCAL_PFrechits
    if hasattr(process, "hltParticleFlowClusterHCALCPUOnly"):
        process.hltParticleFlowClusterHCALCPUOnly.pfClusterBuilder.allCellsPositionCalc.logWeightDenominatorByDetector[0].logWeightDenominator=HCAL_PFrechits
    if hasattr(process, "hltParticleFlowRecHitHBHECPUOnly"):
        process.hltParticleFlowRecHitHBHECPUOnly.producers[0].qualityTests[0].cuts[0].threshold=HCAL_PFrechits

    return process

def customizeHCALinCaloJets(process, HCAL_PFrechits):
    if hasattr(process, "hltTowerMakerForAll"):
        process.hltTowerMakerForAll.HBThreshold1 = HCAL_PFrechits[0]
        process.hltTowerMakerForAll.HBThreshold2 = HCAL_PFrechits[1]
        process.hltTowerMakerForAll.HBThreshold  = HCAL_PFrechits[2]
    return process

def customizeHCALinCaloJets_v4(process, HCAL_PFrechits):
    for mod in producers_by_type(process, 'CaloTowersCreator'):
        mod.HBThreshold1 = HCAL_PFrechits[0]
        mod.HBThreshold2 = HCAL_PFrechits[1]
        mod.HBThreshold  = HCAL_PFrechits[2]
    return process

def customizeHCALinCaloJetsFor2023(process):
    return customizeHCALinCaloJets(process, HCAL_PFrechits_2023)

def customizeHCALinCaloJetsFor2022(process):
    return customizeHCALinCaloJets(process, HCAL_PFrechits_2022)

def customizeHCALinCaloJetsFor2023_v4(process):
    return customizeHCALinCaloJets_v4(process, HCAL_PFrechits_2023)

def customizeHCALinCaloJetsFor2022_v4(process):
    return customizeHCALinCaloJets_v4(process, HCAL_PFrechits_2022)


def findPath(file_name):
    this_dir = os.path.dirname(__file__)
    file_path = os.path.join(this_dir, file_name)
    if not os.path.exists(file_path):
        data_dir = os.path.abspath(os.path.join(this_dir, '..', 'data'))
        file_path = os.path.join(data_dir, file_name)
        if not os.path.exists(file_path):
            raise RuntimeError('File {} not found in {} or {}'.format(file_name, this_dir, data_dir))
    return file_path

def customizePFHadronCalibrationFor2023_v5_fromFile(process):
    if hasattr(process, "GlobalTag") and hasattr(process.GlobalTag, "toGet"):
        process.GlobalTag.toGet.append(
            cms.PSet(
                record = cms.string("PFCalibrationRcd"),
                label = cms.untracked.string('HLT'),
                connect = cms.string("sqlite_file:" + findPath("PFCalibration_v5.db")),
                tag = cms.string('PFCalibration_CMSSW_13_0_0_HLT_126X_fixEE_mcRun3_2023'),
                snapshotTime = cms.string('9999-12-31 23:59:59.000'),
            )
        )
    else:
            raise Exception("Warning process.GlobalTag not found. customizePFHadronCalibration will not be applied.")
    return process

def customizeJECFor2023_v5_fromFile(process):
    if hasattr(process, "GlobalTag") and hasattr(process.GlobalTag, "toGet"):
        connect_path = "sqlite_file:" + findPath("Run3Winter23Digi_v5.db")
        process.GlobalTag.toGet.append(
            cms.PSet(
                record = cms.string("JetCorrectionsRecord"),
                label = cms.untracked.string('AK4CaloHLT'),
                connect = cms.string(connect_path),
                tag = cms.string('JetCorrectorParametersCollection_Run3Winter23Digi_AK4CaloHLT'),
                snapshotTime = cms.string('9999-12-31 23:59:59.000'),
            )
        )
        process.GlobalTag.toGet.append(
            cms.PSet(
                record = cms.string("JetCorrectionsRecord"),
                label = cms.untracked.string('AK4PFHLT'),
                connect = cms.string(connect_path),
                tag = cms.string('JetCorrectorParametersCollection_Run3Winter23Digi_AK4PFHLT'),
                snapshotTime = cms.string('9999-12-31 23:59:59.000'),
            )
        )
        process.GlobalTag.toGet.append(
            cms.PSet(
                record = cms.string("JetCorrectionsRecord"),
                label = cms.untracked.string('AK8CaloHLT'),
                connect = cms.string(connect_path),
                tag = cms.string('JetCorrectorParametersCollection_Run3Winter23Digi_AK8CaloHLT'),
                snapshotTime = cms.string('9999-12-31 23:59:59.000'),
            )
        )
        process.GlobalTag.toGet.append(
            cms.PSet(
                record = cms.string("JetCorrectionsRecord"),
                label = cms.untracked.string('AK8PFHLT'),
                connect = cms.string(connect_path),
                tag = cms.string('JetCorrectorParametersCollection_Run3Winter23Digi_AK8PFHLT'),
                snapshotTime = cms.string('9999-12-31 23:59:59.000'),
            )
        )
    else:
            raise Exception("Warning process.GlobalTag not found. customizePFHadronCalibration will not be applied.")
    return process

def customizePFHadronCalibrationFor2023(process):
    if hasattr(process, "GlobalTag") and hasattr(process.GlobalTag, "toGet"):
        process.GlobalTag.toGet.append(
            cms.PSet(
                record = cms.string("PFCalibrationRcd"),
                label = cms.untracked.string('HLT'),
                connect = cms.string("sqlite_file:/eos/cms/store/group/dpg_trigger/comm_trigger/TriggerStudiesGroup/PF/PFCalibration.db"),
                tag = cms.string('PFCalibration_CMSSW_13_0_0_pre4_HLT_126X_mcRun3_2023'),
                snapshotTime = cms.string('9999-12-31 23:59:59.000'),
            )
        )
    else:
            raise Exception("Warning process.GlobalTag not found. customizePFHadronCalibration will not be applied.")
    return process

def customizePFHadronCalibrationFor2023_fromCondDb(process):
    if hasattr(process, "GlobalTag") and hasattr(process.GlobalTag, "toGet"):
        process.GlobalTag.toGet.append(
            cms.PSet(
                record = cms.string("PFCalibrationRcd"),
                label = cms.untracked.string('HLT'),
                tag = cms.string('PFCalibration_Run3Winter23Digi_mc_v2'),
            )
        )
    else:
            raise Exception("Warning process.GlobalTag not found. customizePFHadronCalibration will not be applied.")
    return process

def customizeJECFor2023_noAK8CaloHLT(process):
    if hasattr(process, "GlobalTag") and hasattr(process.GlobalTag, "toGet"):
        process.GlobalTag.toGet.append(
            cms.PSet(
                record = cms.string("JetCorrectionsRecord"),
                label = cms.untracked.string('AK4CaloHLT'),
                connect = cms.string("sqlite_file:/eos/cms/store/group/dpg_trigger/comm_trigger/TriggerStudiesGroup/JME/Run3Winter23/Run3Winter23Digi_onlyAK4.db"),
                tag = cms.string('JetCorrectorParametersCollection_Run3Winter23Digi_AK4CaloHLT'),
                snapshotTime = cms.string('9999-12-31 23:59:59.000'),
            )
        )
        process.GlobalTag.toGet.append(
            cms.PSet(
                record = cms.string("JetCorrectionsRecord"),
                label = cms.untracked.string('AK4PFHLT'),
                connect = cms.string("sqlite_file:/eos/cms/store/group/dpg_trigger/comm_trigger/TriggerStudiesGroup/JME/Run3Winter23/Run3Winter23Digi_onlyAK4.db"),
                tag = cms.string('JetCorrectorParametersCollection_Run3Winter23Digi_AK4PFHLT'),
                snapshotTime = cms.string('9999-12-31 23:59:59.000'),
            )
        )
        process.GlobalTag.toGet.append(
            cms.PSet(
                record = cms.string("JetCorrectionsRecord"),
                label = cms.untracked.string('AK8PFHLT'),
                connect = cms.string("sqlite_file:/eos/cms/store/group/dpg_trigger/comm_trigger/TriggerStudiesGroup/JME/Run3Winter23/Run3Winter23Digi_onlyAK4.db"),
                tag = cms.string('JetCorrectorParametersCollection_Run3Winter23Digi_AK8PFHLT'),
                snapshotTime = cms.string('9999-12-31 23:59:59.000'),
            )
        )
    else:
            raise Exception("Warning process.GlobalTag not found. customizePFHadronCalibration will not be applied.")
    raise Exception("You are trying to use the old customization function (customizeJECFor2023_noAK8CaloHLT or customizeHLTFor2023WithJEC). Please move the the new customization function customizeHLTFor2023WithJEC_v2 or customizeJECFor2023_v2.")
    return process

def customizeJECFor2023_v2(process):
    if hasattr(process, "GlobalTag") and hasattr(process.GlobalTag, "toGet"):
        process.GlobalTag.toGet.append(
            cms.PSet(
                record = cms.string("JetCorrectionsRecord"),
                label = cms.untracked.string('AK4CaloHLT'),
                connect = cms.string("sqlite_file:/eos/cms/store/group/dpg_trigger/comm_trigger/TriggerStudiesGroup/JME/Run3Winter23/Run3Winter23Digi.db"),
                tag = cms.string('JetCorrectorParametersCollection_Run3Winter23Digi_AK4CaloHLT'),
                snapshotTime = cms.string('9999-12-31 23:59:59.000'),
            )
        )
        process.GlobalTag.toGet.append(
            cms.PSet(
                record = cms.string("JetCorrectionsRecord"),
                label = cms.untracked.string('AK4PFHLT'),
                connect = cms.string("sqlite_file:/eos/cms/store/group/dpg_trigger/comm_trigger/TriggerStudiesGroup/JME/Run3Winter23/Run3Winter23Digi.db"),
                tag = cms.string('JetCorrectorParametersCollection_Run3Winter23Digi_AK4PFHLT'),
                snapshotTime = cms.string('9999-12-31 23:59:59.000'),
            )
        )
        process.GlobalTag.toGet.append(
            cms.PSet(
                record = cms.string("JetCorrectionsRecord"),
                label = cms.untracked.string('AK8CaloHLT'),
                connect = cms.string("sqlite_file:/eos/cms/store/group/dpg_trigger/comm_trigger/TriggerStudiesGroup/JME/Run3Winter23/Run3Winter23Digi.db"),
                tag = cms.string('JetCorrectorParametersCollection_Run3Winter23Digi_AK8CaloHLT'),
                snapshotTime = cms.string('9999-12-31 23:59:59.000'),
            )
        )
        process.GlobalTag.toGet.append(
            cms.PSet(
                record = cms.string("JetCorrectionsRecord"),
                label = cms.untracked.string('AK8PFHLT'),
                connect = cms.string("sqlite_file:/eos/cms/store/group/dpg_trigger/comm_trigger/TriggerStudiesGroup/JME/Run3Winter23/Run3Winter23Digi.db"),
                tag = cms.string('JetCorrectorParametersCollection_Run3Winter23Digi_AK8PFHLT'),
                snapshotTime = cms.string('9999-12-31 23:59:59.000'),
            )
        )
    else:
            raise Exception("Warning process.GlobalTag not found. customizePFHadronCalibration will not be applied.")
    return process


def customizeJECFor2023_v3_noCaloJets(process):
    if hasattr(process, "GlobalTag") and hasattr(process.GlobalTag, "toGet"):
        process.GlobalTag.toGet.append(
            cms.PSet(
                record = cms.string("JetCorrectionsRecord"),
                label = cms.untracked.string('AK4PFHLT'),
                connect = cms.string("sqlite_file:/eos/cms/store/group/dpg_trigger/comm_trigger/TriggerStudiesGroup/JME/Run3Winter23/Run3Winter23Digi.db"),
                tag = cms.string('JetCorrectorParametersCollection_Run3Winter23Digi_AK4PFHLT'),
                snapshotTime = cms.string('9999-12-31 23:59:59.000'),
            )
        )
        process.GlobalTag.toGet.append(
            cms.PSet(
                record = cms.string("JetCorrectionsRecord"),
                label = cms.untracked.string('AK8PFHLT'),
                connect = cms.string("sqlite_file:/eos/cms/store/group/dpg_trigger/comm_trigger/TriggerStudiesGroup/JME/Run3Winter23/Run3Winter23Digi.db"),
                tag = cms.string('JetCorrectorParametersCollection_Run3Winter23Digi_AK8PFHLT'),
                snapshotTime = cms.string('9999-12-31 23:59:59.000'),
            )
        )
    else:
            raise Exception("Warning process.GlobalTag not found. customizePFHadronCalibration will not be applied.")
    return process


def customizeJECFor2023_v3(process):
    if hasattr(process, "GlobalTag") and hasattr(process.GlobalTag, "toGet"):
        process.GlobalTag.toGet.append(
            cms.PSet(
                record = cms.string("JetCorrectionsRecord"),
                label = cms.untracked.string('AK4PFHLT'),
                connect = cms.string("sqlite_file:/eos/cms/store/group/dpg_trigger/comm_trigger/TriggerStudiesGroup/JME/Run3Winter23/Run3Winter23Digi_caloJetFixed.db"),
                tag = cms.string('JetCorrectorParametersCollection_Run3Winter23Digi_AK4PFHLT'),
                snapshotTime = cms.string('9999-12-31 23:59:59.000'),
            )
        )
        process.GlobalTag.toGet.append(
            cms.PSet(
                record = cms.string("JetCorrectionsRecord"),
                label = cms.untracked.string('AK8PFHLT'),
                connect = cms.string("sqlite_file:/eos/cms/store/group/dpg_trigger/comm_trigger/TriggerStudiesGroup/JME/Run3Winter23/Run3Winter23Digi_caloJetFixed.db"),
                tag = cms.string('JetCorrectorParametersCollection_Run3Winter23Digi_AK8PFHLT'),
                snapshotTime = cms.string('9999-12-31 23:59:59.000'),
            )
        )
        process.GlobalTag.toGet.append(
            cms.PSet(
                record = cms.string("JetCorrectionsRecord"),
                label = cms.untracked.string('AK4CaloHLT'),
                connect = cms.string("sqlite_file:/eos/cms/store/group/dpg_trigger/comm_trigger/TriggerStudiesGroup/JME/Run3Winter23/Run3Winter23Digi_caloJetFixed.db"),
                tag = cms.string('JetCorrectorParametersCollection_Run3Winter23Digi_AK4CaloHLT'),
                snapshotTime = cms.string('9999-12-31 23:59:59.000'),
            )
        )
        process.GlobalTag.toGet.append(
            cms.PSet(
                record = cms.string("JetCorrectionsRecord"),
                label = cms.untracked.string('AK8CaloHLT'),
                connect = cms.string("sqlite_file:/eos/cms/store/group/dpg_trigger/comm_trigger/TriggerStudiesGroup/JME/Run3Winter23/Run3Winter23Digi_caloJetFixed.db"),
                tag = cms.string('JetCorrectorParametersCollection_Run3Winter23Digi_AK8CaloHLT'),
                snapshotTime = cms.string('9999-12-31 23:59:59.000'),
            )
        )
    else:
            raise Exception("Warning process.GlobalTag not found. customizePFHadronCalibration will not be applied.")
    return process

def customizeJECFor2023_v3_fromCondDb(process):
    if hasattr(process, "GlobalTag") and hasattr(process.GlobalTag, "toGet"):
        process.GlobalTag.toGet.append(
            cms.PSet(
                record = cms.string("JetCorrectionsRecord"),
                label = cms.untracked.string('AK4PFHLT'),
                tag = cms.string('JetCorrectorParametersCollection_Run3Winter23Digi_AK4PFHLT_mc_v2'),
            )
        )
        process.GlobalTag.toGet.append(
            cms.PSet(
                record = cms.string("JetCorrectionsRecord"),
                label = cms.untracked.string('AK8PFHLT'),
                tag = cms.string('JetCorrectorParametersCollection_Run3Winter23Digi_AK8PFHLT_mc_v2'),
            )
        )
        process.GlobalTag.toGet.append(
            cms.PSet(
                record = cms.string("JetCorrectionsRecord"),
                label = cms.untracked.string('AK4CaloHLT'),
                tag = cms.string('JetCorrectorParametersCollection_Run3Winter23Digi_AK4CaloHLT_mc_v2'),
            )
        )
        process.GlobalTag.toGet.append(
            cms.PSet(
                record = cms.string("JetCorrectionsRecord"),
                label = cms.untracked.string('AK8CaloHLT'),
                tag = cms.string('JetCorrectorParametersCollection_Run3Winter23Digi_AK8CaloHLT_mc_v2'),
            )
        )
    else:
            raise Exception("Warning process.GlobalTag not found. customizePFHadronCalibration will not be applied.")
    return process

def customizePFHadronCalibrationFor2023_v4_fromCondDb(process):
    if hasattr(process, "GlobalTag") and hasattr(process.GlobalTag, "toGet"):
        process.GlobalTag.toGet.append(
            cms.PSet(
                record = cms.string("PFCalibrationRcd"),
                label = cms.untracked.string('HLT'),
                tag = cms.string('PFCalibration_Run3Winter23Digi_mc_v3'),
                snapshotTime = cms.string('9999-12-31 23:59:59.000'),
            )
        )
    else:
            raise Exception("Warning process.GlobalTag not found. customizePFHadronCalibration will not be applied.")
    return process

def customizeJECFor2023_v4_fromCondDb(process):
    if hasattr(process, "GlobalTag") and hasattr(process.GlobalTag, "toGet"):
        process.GlobalTag.toGet.append(
            cms.PSet(
                record = cms.string("JetCorrectionsRecord"),
                label = cms.untracked.string('AK4PFHLT'),
                tag = cms.string('JetCorrectorParametersCollection_Run3Winter23Digi_AK4PFHLT_mc_v3'),
                snapshotTime = cms.string('9999-12-31 23:59:59.000'),
            )
        )
        process.GlobalTag.toGet.append(
            cms.PSet(
                record = cms.string("JetCorrectionsRecord"),
                label = cms.untracked.string('AK8PFHLT'),
                tag = cms.string('JetCorrectorParametersCollection_Run3Winter23Digi_AK8PFHLT_mc_v3'),
                snapshotTime = cms.string('9999-12-31 23:59:59.000'),
            )
        )
        process.GlobalTag.toGet.append(
            cms.PSet(
                record = cms.string("JetCorrectionsRecord"),
                label = cms.untracked.string('AK4CaloHLT'),
                tag = cms.string('JetCorrectorParametersCollection_Run3Winter23Digi_AK4CaloHLT_mc_v3'),
                snapshotTime = cms.string('9999-12-31 23:59:59.000'),
            )
        )
        process.GlobalTag.toGet.append(
            cms.PSet(
                record = cms.string("JetCorrectionsRecord"),
                label = cms.untracked.string('AK8CaloHLT'),
                tag = cms.string('JetCorrectorParametersCollection_Run3Winter23Digi_AK8CaloHLT_mc_v3'),
                snapshotTime = cms.string('9999-12-31 23:59:59.000'),
            )
        )
    else:
            raise Exception("Warning process.GlobalTag not found. customizePFHadronCalibration will not be applied.")
    return process

def customizeHLTFor2023(process):
    process = customizePFHadronCalibrationFor2023(process)
    process = customizeHCALFor2023(process)
    return process

def customizeHLTFor2023WithJEC(process):
    process = customizeJECFor2023_noAK8CaloHLT(process)
    process = customizeHLTFor2023(process)
    return process

def customizeHLTFor2023WithJEC_v2(process):
    process = customizeJECFor2023_v2(process)
    process = customizeHLTFor2023(process)
    return process

def customizeHLTFor2023_v3_NoCaloJEC(process):
    process = customizeJECFor2023_v3_noCaloJets(process)
    process = customizePFHadronCalibrationFor2023(process)
    process = customizeHCALFor2023(process)
    process = customizeHCALinCaloJetsFor2023(process)
    return process

def customizeHLTFor2023_v3(process):
    process = customizeJECFor2023_v3(process)
    process = customizePFHadronCalibrationFor2023(process)
    process = customizeHCALFor2023(process)
    process = customizeHCALinCaloJetsFor2023(process)
    return process

def customizeHLTFor2023_v3_fromCondDb(process):
    process = customizeJECFor2023_v3_fromCondDb(process)
    process = customizePFHadronCalibrationFor2023_fromCondDb(process)
    process = customizeHCALFor2023(process)
    process = customizeHCALinCaloJetsFor2023(process)
    return process

def customizeHLTFor2023_v4(process):
    process = customizeJECFor2023_v4_fromCondDb(process)
    process = customizePFHadronCalibrationFor2023_v4_fromCondDb(process)
    process = customizeHCALFor2023(process)
    process = customizeHCALinCaloJetsFor2023_v4(process)
    return process

def customizeHLTFor2023_onlyHCALrechits(process):
    process = customizeHCALFor2023(process)
    process = customizeHCALinCaloJetsFor2023_v4(process)
    return process

def customizeHLTFor2023_v5_fromFile(process):
    process = customizeHLTFor2023_onlyHCALrechits(process)
    process = customizeJECFor2023_v5_fromFile(process)
    process = customizePFHadronCalibrationFor2023_v5_fromFile(process)
    return process


def _updateL1TSeedModules(process, l1tMenu):

    dictL1TSeeds = {
      # VBF Parking (exclusive seeds)
      'hltL1DiJetVBFdoubleJet': 'L1_DoubleJet_110_35_DoubleJet35_Mass_Min620',
      'hltL1DiJetVBFMET': 'L1_DoubleJet_110_35_DoubleJet35_Mass_Min620',
      'hltL1DoubleJet8030MassMin500Mu3': 'L1_DoubleJet_110_35_DoubleJet35_Mass_Min620',
      'hltL1VBFIsoEG': 'L1_DoubleJet_110_35_DoubleJet35_Mass_Min620',
    }

    if l1tMenu == '2022_v1_4_0':

        dictL1TSeeds.update({
          # BPH (Tau3Mu)
          'hltL1sTripleMuControl': 'L1_TripleMu_5SQ_3SQ_0OQ',
          'hltL1sDoubleMu0er2p0SQOSdEtaMax1p6orTripleMu21p50': 'L1_DoubleMu0er2p0_SQ_OS_dEta_Max1p6 OR L1_DoubleMu0er2p0_SQ_OS_dEta_Max1p5 OR L1_DoubleMu0er1p5_SQ_OS_dR_Max1p4 OR L1_DoubleMu0er1p4_SQ_OS_dR_Max1p4 OR L1_TripleMu_5SQ_3SQ_0_DoubleMu_5_3_SQ_OS_Mass_Max9 OR L1_TripleMu_5SQ_3SQ_0OQ_DoubleMu_5_3_SQ_OS_Mass_Max9 OR L1_TripleMu_2SQ_1p5SQ_0OQ_Mass_Max12 OR L1_DoubleMu4_SQ_OS_dR_Max1p2 OR L1_DoubleMu4p5_SQ_OS_dR_Max1p2',

          # BTV
          'hltL1sDiJet16er2p5Mu3dRMax0p4': 'L1_Mu3_Jet16er2p5_dR_Max0p4',
          'hltL1sDiJet35er2p5Mu3dRMax0p4': 'L1_Mu3_Jet35er2p5_dR_Max0p4',
          'hltL1sDiJet60er2p5Mu3dRMax0p4': 'L1_Mu3_Jet60er2p5_dR_Max0p4',
          'hltL1sDiJet80er2p5Mu3dRMax0p4': 'L1_Mu3_Jet80er2p5_dR_Max0p4',
          'hltL1sDiJet120er2p5Mu3dRMax0p8': 'L1_Mu3_Jet120er2p5_dR_Max0p8',

          # JME
          'hltL1sSingleJet35Fwd': 'L1_SingleJet35_FWD3p0',
          'hltL1sSingleJet35OrZeroBias': 'L1_SingleJet35 OR L1_SingleJet35_FWD3p0 OR L1_ZeroBias',
          'hltL1sSingleJet60Fwd': 'L1_SingleJet60_FWD3p0',
          'hltL1sSingleJet60Or60Fwd': 'L1_SingleJet60 OR L1_SingleJet60_FWD3p0',
          'hltL1sSingleJet90Fwd': 'L1_SingleJet90_FWD3p0',
          'hltL1sV0SingleJet60Or60Fwd': 'L1_SingleJet60 OR L1_SingleJet60_FWD3p0 OR L1_SingleJet90 OR L1_SingleJet90_FWD3p0',
          'hltL1sSingleJet120Fwd': 'L1_SingleJet120_FWD3p0',
          'hltL1sSingleJet120Or120Fwd': 'L1_SingleJet120 OR L1_SingleJet120_FWD3p0',

          # HIG (VBF HTauTau)
          'hltL1VBFDiJetIsoTau': 'L1_DoubleJet35_Mass_Min450_IsoTau45er2p1_RmOvlp_dR0p5',

          # BPH (BsMuMu)
          'hltL1sDoubleMuForBs': 'L1_DoubleMu3er2p0_SQ_OS_dR_Max1p4 OR L1_DoubleMu0er2p0_SQ_OS_dEta_Max1p6 OR L1_DoubleMu0er1p4_OQ_OS_dEta_Max1p6 OR L1_DoubleMu0er2p0_SQ_OS_dEta_Max1p5 OR L1_DoubleMu0er1p4_SQ_OS_dR_Max1p4 OR L1_DoubleMu0er1p5_SQ_OS_dR_Max1p4 OR L1_DoubleMu4p5_SQ_OS_dR_Max1p2 OR L1_DoubleMu4_SQ_OS_dR_Max1p2',
          'hltL1sDoubleMuForBsToMMG': 'L1_DoubleMu3er2p0_SQ_OS_dR_Max1p4 OR L1_DoubleMu0er2p0_SQ_OS_dEta_Max1p6 OR L1_DoubleMu0er2p0_SQ_OS_dEta_Max1p5 OR L1_DoubleMu0er1p4_SQ_OS_dR_Max1p4 OR L1_DoubleMu0er1p5_SQ_OS_dR_Max1p4 OR L1_DoubleMu4p5_SQ_OS_dR_Max1p2 OR L1_DoubleMu4_SQ_OS_dR_Max1p2',
          'hltL1sDoubleMuForLowMassDisplaced': 'L1_DoubleMu3er2p0_SQ_OS_dR_Max1p4 OR L1_DoubleMu0er2p0_SQ_OS_dEta_Max1p6 OR L1_DoubleMu0er1p4_OQ_OS_dEta_Max1p6 OR L1_DoubleMu0er2p0_SQ_OS_dEta_Max1p5 OR L1_DoubleMu0er1p4_SQ_OS_dR_Max1p4 OR L1_DoubleMu0er1p5_SQ_OS_dR_Max1p4 OR L1_DoubleMu4p5_SQ_OS_dR_Max1p2 OR L1_DoubleMu4_SQ_OS_dR_Max1p2',
          'hltL1sDoubleMuForLowMassInclusive': 'L1_DoubleMu3er2p0_SQ_OS_dR_Max1p4 OR L1_DoubleMu0er2p0_SQ_OS_dEta_Max1p6 OR L1_DoubleMu0er1p4_OQ_OS_dEta_Max1p6 OR L1_DoubleMu0er2p0_SQ_OS_dEta_Max1p5 OR L1_DoubleMu0er1p4_SQ_OS_dR_Max1p4 OR L1_DoubleMu0er1p5_SQ_OS_dR_Max1p4 OR L1_DoubleMu4p5_SQ_OS_dR_Max1p2 OR L1_DoubleMu4_SQ_OS_dR_Max1p2',

          # SMP (W->3pi)
          'hltL1sDoubleTauBigORWithLowMass': 'L1_DoubleIsoTau34er2p1 OR L1_DoubleIsoTau36er2p1 OR L1_DoubleTau70er2p1 OR L1_DoubleIsoTau28er2p1_Mass_Max80 OR L1_DoubleIsoTau30er2p1_Mass_Max80',
        })

    elif l1tMenu != '2023_v1_0_0':
        raise RuntimeError(f'ERROR -- invalid value for argument "l1tMenu" (must be "2022_v1_4_0", or "2023_v1_0_0"): "{l1tMenu}"')

    for modName,oldSeed in dictL1TSeeds.items():
        try: getattr(process, modName).L1SeedsLogicalExpression = oldSeed
        except: pass

    return process

def customizeHLTFor2022L1TMenu(process):
    return _updateL1TSeedModules(process, l1tMenu = '2022_v1_4_0')

def customizeHLTFor2023L1TMenu_v1_0_0(process):
    return _updateL1TSeedModules(process, l1tMenu = '2023_v1_0_0')
