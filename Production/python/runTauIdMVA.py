from __future__ import print_function
import FWCore.ParameterSet.Config as cms
from RecoTauTag.RecoTau.TauDiscriminatorTools import noPrediscriminants
from RecoTauTag.RecoTau.PATTauDiscriminationByMVAIsolationRun2_cff import patDiscriminationByIsolationMVArun2v1raw, patDiscriminationByIsolationMVArun2v1VLoose
import os
import re
import six

availableDiscriminators = [ "2017v2", "newDM2017v2", "dR0p32017v2", "deepTau2017v2p1" ]

def runTauID(process, outputTauCollection='slimmedTausNewID', inputTauCollection='slimmedTaus', isBoosted=False,
             toKeep=["deepTau2017v2p1"], debug=False):
    for discr in toKeep:
        if discr not in availableDiscriminators:
            raise RuntimeError('TauIDEmbedder: discriminator "{}" is not supported'.format(discr))

    process.load('RecoTauTag.Configuration.loadRecoTauTagMVAsFromPrepDB_cfi')

    setattr(process, outputTauCollection + 'rerunMvaIsolationTask', cms.Task())
    setattr(process, outputTauCollection + 'rerunMvaIsolationSequence', cms.Sequence())
    tauIDSources = cms.PSet()

    mva_discrs = {
        "2017v2": {
            "mvaName": "RecoTauTag_tauIdMVAIsoDBoldDMwLT2017v2",
            "mvaOpt": "DBoldDMwLTwGJ",
            "mvaOutput_normalization": "RecoTauTag_tauIdMVAIsoDBoldDMwLT2017v2_mvaOutput_normalization",
            "rawName": "byIsolationMVArun2017v2DBoldDMwLTraw2017",
            "wpNamePattern": "by{}IsolationMVArun2017v2DBoldDMwLT2017",
            "reducedCone": False,
            "cut_dict": {
                "VVLoose": "RecoTauTag_tauIdMVAIsoDBoldDMwLT2017v2_WPEff95",
                "VLoose": "RecoTauTag_tauIdMVAIsoDBoldDMwLT2017v2_WPEff90",
                "Loose": "RecoTauTag_tauIdMVAIsoDBoldDMwLT2017v2_WPEff80",
                "Medium": "RecoTauTag_tauIdMVAIsoDBoldDMwLT2017v2_WPEff70",
                "Tight": "RecoTauTag_tauIdMVAIsoDBoldDMwLT2017v2_WPEff60",
                "VTight": "RecoTauTag_tauIdMVAIsoDBoldDMwLT2017v2_WPEff50",
                "VVTight": "RecoTauTag_tauIdMVAIsoDBoldDMwLT2017v2_WPEff40",
            },
        },
        "newDM2017v2": {
            "mvaName": "RecoTauTag_tauIdMVAIsoDBnewDMwLT2017v2",
            "mvaOpt": "DBnewDMwLTwGJ",
            "mvaOutput_normalization": "RecoTauTag_tauIdMVAIsoDBnewDMwLT2017v2_mvaOutput_normalization",
            "rawName": "byIsolationMVArun2017v2DBnewDMwLTraw2017",
            "wpNamePattern": "by{}IsolationMVArun2017v2DBnewDMwLT2017",
            "reducedCone": False,
            "cut_dict": {
                "VVLoose": "RecoTauTag_tauIdMVAIsoDBnewDMwLT2017v2_WPEff95",
                "VLoose": "RecoTauTag_tauIdMVAIsoDBnewDMwLT2017v2_WPEff90",
                "Loose": "RecoTauTag_tauIdMVAIsoDBnewDMwLT2017v2_WPEff80",
                "Medium": "RecoTauTag_tauIdMVAIsoDBnewDMwLT2017v2_WPEff70",
                "Tight": "RecoTauTag_tauIdMVAIsoDBnewDMwLT2017v2_WPEff60",
                "VTight": "RecoTauTag_tauIdMVAIsoDBnewDMwLT2017v2_WPEff50",
                "VVTight": "RecoTauTag_tauIdMVAIsoDBnewDMwLT2017v2_WPEff40",
            },
        },
        "dR0p32017v2": {
            "mvaName": "RecoTauTag_tauIdMVAIsoDBoldDMdR0p3wLT2017v2",
            "mvaOpt": "DBoldDMwLTwGJ",
            "mvaOutput_normalization": "RecoTauTag_tauIdMVAIsoDBoldDMdR0p3wLT2017v2_mvaOutput_normalization",
            "rawName": "byIsolationMVArun2017v2DBoldDMdR0p3wLTraw2017",
            "wpNamePattern": "by{}IsolationMVArun2017v2DBoldDMdR0p3wLT2017",
            "reducedCone": True,
            "cut_dict": {
                "VVLoose": "RecoTauTag_tauIdMVAIsoDBoldDMdR0p3wLT2017v2_WPEff95",
                "VLoose": "RecoTauTag_tauIdMVAIsoDBoldDMdR0p3wLT2017v2_WPEff90",
                "Loose": "RecoTauTag_tauIdMVAIsoDBoldDMdR0p3wLT2017v2_WPEff80",
                "Medium": "RecoTauTag_tauIdMVAIsoDBoldDMdR0p3wLT2017v2_WPEff70",
                "Tight": "RecoTauTag_tauIdMVAIsoDBoldDMdR0p3wLT2017v2_WPEff60",
                "VTight": "RecoTauTag_tauIdMVAIsoDBoldDMdR0p3wLT2017v2_WPEff50",
                "VVTight": "RecoTauTag_tauIdMVAIsoDBoldDMdR0p3wLT2017v2_WPEff40",
            },
        },
    }

    for discr_name, params in mva_discrs.items():
        if discr_name not in toKeep:
            continue

        if params["reducedCone"]:
            srcChargedIsoPtSum = 'chargedIsoPtSumdR03'
            srcNeutralIsoPtSum = 'neutralIsoPtSumdR03'
            srcFootprintCorrection = 'footprintCorrectiondR03'
            srcPhotonPtSumOutsideSignalCone = 'photonPtSumOutsideSignalConedR03'
        elif isBoosted:
            srcChargedIsoPtSum = 'chargedIsoPtSumNoOverLap'
            srcNeutralIsoPtSum = 'neutralIsoPtSumNoOverLap'
            srcFootprintCorrection = 'footprintCorrection'
            srcPhotonPtSumOutsideSignalCone = 'photonPtSumOutsideSignalCone'
        else:
            srcChargedIsoPtSum = 'chargedIsoPtSum'
            srcNeutralIsoPtSum = 'neutralIsoPtSum'
            srcFootprintCorrection = 'footprintCorrection'
            srcPhotonPtSumOutsideSignalCone = 'photonPtSumOutsideSignalCone'

        setattr(process, outputTauCollection + params["rawName"],
            patDiscriminationByIsolationMVArun2v1raw.clone(
                PATTauProducer = cms.InputTag(inputTauCollection),
                srcChargedIsoPtSum = cms.string(srcChargedIsoPtSum),
                srcNeutralIsoPtSum = cms.string(srcNeutralIsoPtSum),
                srcFootprintCorrection = cms.string(srcFootprintCorrection),
                srcPhotonPtSumOutsideSignalCone = cms.string(srcPhotonPtSumOutsideSignalCone),
                Prediscriminants = noPrediscriminants,
                loadMVAfromDB = cms.bool(True),
                mvaName = cms.string(params["mvaName"]),
                mvaOpt = cms.string(params["mvaOpt"]),
                verbosity = cms.int32(0)
            )
        )

        task = cms.Task(getattr(process, outputTauCollection + params["rawName"]))
        setattr(tauIDSources, params["rawName"], cms.InputTag(outputTauCollection + params["rawName"]))

        for wp, cut in params["cut_dict"].items():
            setattr(process, outputTauCollection + params["wpNamePattern"].format(wp),
                patDiscriminationByIsolationMVArun2v1VLoose.clone(
                    PATTauProducer = cms.InputTag(inputTauCollection),
                    Prediscriminants = noPrediscriminants,
                    toMultiplex = cms.InputTag(outputTauCollection + params["rawName"]),
                    key = cms.InputTag(outputTauCollection + params["rawName"] + ':category'),
                    loadMVAfromDB = cms.bool(True),
                    mvaOutput_normalization = cms.string(params["mvaOutput_normalization"]),
                    mapping = cms.VPSet(
                        cms.PSet(
                            category = cms.uint32(0),
                            cut = cms.string(cut),
                            variable = cms.string("pt"),
                        )
                    ),
                    verbosity = cms.int32(0)
                )
            )

            task.add(getattr(process, outputTauCollection + params["wpNamePattern"].format(wp)))

            setattr(tauIDSources, params["wpNamePattern"].format(wp),
                cms.InputTag(outputTauCollection + params["wpNamePattern"].format(wp))
            )

        getattr(process, outputTauCollection + 'rerunMvaIsolationTask').add(task)
        process.__dict__[outputTauCollection + 'rerunMvaIsolationSequence'] += cms.Sequence(task)

    if "deepTau2017v2p1" in toKeep:
        if debug: print ("Adding DeepTau IDs")

        workingPoints_ = {
            "e": {
                "VVVLoose": 0.0630386,
                "VVLoose": 0.1686942,
                "VLoose": 0.3628130,
                "Loose": 0.6815435,
                "Medium": 0.8847544,
                "Tight": 0.9675541,
                "VTight": 0.9859251,
                "VVTight": 0.9928449,
            },
            "mu": {
                "VLoose": 0.1058354,
                "Loose": 0.2158633,
                "Medium": 0.5551894,
                "Tight": 0.8754835,
            },
            "jet": {
                "VVVLoose": 0.2599605,
                "VVLoose": 0.4249705,
                "VLoose": 0.5983682,
                "Loose": 0.7848675,
                "Medium": 0.8834768,
                "Tight": 0.9308689,
                "VTight": 0.9573137,
                "VVTight": 0.9733927,
            },
        }

        file_names = [
            'core:RecoTauTag/TrainingFiles/data/DeepTauId/deepTau_2017v2p6_e6_core.pb',
            'inner:RecoTauTag/TrainingFiles/data/DeepTauId/deepTau_2017v2p6_e6_inner.pb',
            'outer:RecoTauTag/TrainingFiles/data/DeepTauId/deepTau_2017v2p6_e6_outer.pb',
        ]
        setattr(process, outputTauCollection + 'deepTau2017v2p1',
            cms.EDProducer("DeepTauId",
                electrons                = cms.InputTag('slimmedElectrons'),
                muons                    = cms.InputTag('slimmedMuons'),
                taus                     = cms.InputTag(inputTauCollection),
                pfcands                  = cms.InputTag('packedPFCandidates'),
                vertices                 = cms.InputTag('offlineSlimmedPrimaryVertices'),
                rho                      = cms.InputTag('fixedGridRhoAll'),
                graph_file               = cms.vstring(file_names),
                mem_mapped               = cms.bool(True),
                version                  = cms.uint32(getDeepTauVersion(file_names[0])[1]),
                debug_level              = cms.int32(0),
                disable_dxy_pca          = cms.bool(True)
            )
        )

        processDeepProducer(process, 'DeepTau2017v2p1', outputTauCollection + 'deepTau2017v2p1', tauIDSources,
                            workingPoints_)

        getattr(process, outputTauCollection + 'rerunMvaIsolationTask').add(
            getattr(process, outputTauCollection + 'deepTau2017v2p1')
        )
        process.__dict__[outputTauCollection + 'rerunMvaIsolationSequence'] += \
            getattr(process, outputTauCollection + 'deepTau2017v2p1')

    ##
    if debug: print('Embedding new TauIDs into "' + outputTauCollection + '"')
    if not hasattr(process, outputTauCollection):
        embedID = cms.EDProducer("PATTauIDEmbedder",
           src = cms.InputTag(inputTauCollection),
           tauIDSources = tauIDSources
        )
        setattr(process, outputTauCollection, embedID)
    else: #assume same type
        tauIDSources = cms.PSet( getattr(process, outputTauCollection).tauIDSources, tauIDSources)
        getattr(process, outputTauCollection).tauIDSources = tauIDSources


def processDeepProducer(process, id_name, producer_name, tauIDSources, workingPoints_):
    for target, points in six.iteritems(workingPoints_):
        cuts = cms.PSet()
        setattr(tauIDSources, 'by{}VS{}raw'.format(id_name, target),
                    cms.InputTag(producer_name, 'VS{}'.format(target)))
        for point,cut in six.iteritems(points):
            setattr(cuts, point, cms.string(str(cut)))

            setattr(tauIDSources, 'by{}{}VS{}'.format(point, id_name, target),
                    cms.InputTag(producer_name, 'VS{}{}'.format(target, point)))

        setattr(getattr(process, producer_name), 'VS{}WP'.format(target), cuts)

def getDeepTauVersion(file_name):
    """returns the DeepTau year, version, subversion. File name should contain a version label with data takig year \
    (2011-2, 2015-8), version number (vX) and subversion (pX), e.g. 2017v0p6, in general the following format: \
    {year}v{version}p{subversion}"""
    version_search = re.search('(201[125678])v([0-9]+)(p[0-9]+|)[\._]', file_name)
    if not version_search:
        raise RuntimeError('File "{}" has an invalid name pattern, should be in the format "{year}v{version}p{subversion}". \
                            Unable to extract version number.'.format(file_name))
    year = version_search.group(1)
    version = version_search.group(2)
    subversion = version_search.group(3)
    if len(subversion) > 0:
        subversion = subversion[1:]
    else:
        subversion = 0
    return int(year), int(version), int(subversion)
