from __future__ import print_function
import FWCore.ParameterSet.Config as cms
from RecoTauTag.RecoTau.TauDiscriminatorTools import noPrediscriminants
# from RecoTauTag.RecoTau.PATTauDiscriminationByMVAIsolationRun2_cff import patDiscriminationByIsolationMVArun2v1raw, patDiscriminationByIsolationMVArun2v1
import os
import re

class TauIDEmbedder(object):
    """class to rerun the tau seq and acces trainings from the database"""
    availableDiscriminators = [ "deepTau2017v2p5", "deepTau2017v2p1ReRun" ]

    def __init__(self, process, debug = False,
                 originalTauName = "slimmedTaus",
                 updatedTauName = "slimmedTausNewID",
                 postfix = "",
                 toKeep =  ["deepTau2017v2p1"],
                 tauIdDiscrMVA_trainings_run2_2017 = { 'tauIdMVAIsoDBoldDMwLT2017' : "tauIdMVAIsoDBoldDMwLT2017", },
                 tauIdDiscrMVA_WPs_run2_2017 = {
                    'tauIdMVAIsoDBoldDMwLT2017' : {
                        'Eff95' : "DBoldDMwLTEff95",
                        'Eff90' : "DBoldDMwLTEff90",
                        'Eff80' : "DBoldDMwLTEff80",
                        'Eff70' : "DBoldDMwLTEff70",
                        'Eff60' : "DBoldDMwLTEff60",
                        'Eff50' : "DBoldDMwLTEff50",
                        'Eff40' : "DBoldDMwLTEff40"
                    }
                 },
                 tauIdDiscrMVA_2017_version = "v1",
                 conditionDB = "" # preparational DB: 'frontier://FrontierPrep/CMS_CONDITIONS'
                 ):
        super(TauIDEmbedder, self).__init__()
        self.process = process
        self.debug = debug
        self.originalTauName = originalTauName
        self.updatedTauName = updatedTauName
        self.postfix = postfix
        self.process.load('RecoTauTag.Configuration.loadRecoTauTagMVAsFromPrepDB_cfi')
        if len(conditionDB) != 0:
            self.process.CondDBTauConnection.connect = cms.string(conditionDB)
            self.process.loadRecoTauTagMVAsFromPrepDB.connect = cms.string(conditionDB)
            # if debug:
            # 	print self.process.CondDBTauConnection.connect
            # 	print dir(self.process.loadRecoTauTagMVAsFromPrepDB)
            # 	print self.process.loadRecoTauTagMVAsFromPrepDB.parameterNames_

        self.tauIdDiscrMVA_trainings_run2_2017 = tauIdDiscrMVA_trainings_run2_2017
        self.tauIdDiscrMVA_WPs_run2_2017 = tauIdDiscrMVA_WPs_run2_2017
        self.tauIdDiscrMVA_2017_version = tauIdDiscrMVA_2017_version
        for discr in toKeep:
            if discr not in TauIDEmbedder.availableDiscriminators:
                raise RuntimeError('TauIDEmbedder: discriminator "{}" is not supported'.format(discr))
        self.toKeep = toKeep

    def runTauID(self):
        _rerunMvaIsolationTask = cms.Task()
        _rerunMvaIsolationSequence = cms.Sequence()
        tauIDSources = cms.PSet()

        if "deepTau2017v2p1ReRun" in self.toKeep:

            if self.debug: print ("Adding DeepTau IDs")

            _deepTauName = "deepTau2017v2p1ReRun"
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
            setattr(self.process,_deepTauName+self.postfix,cms.EDProducer("DeepTauId",
                electrons                = cms.InputTag('slimmedElectrons'),
                muons                    = cms.InputTag('slimmedMuons'),
                taus                     = cms.InputTag(self.originalTauName),
                pfcands                  = cms.InputTag('packedPFCandidates'),
                vertices                 = cms.InputTag('offlineSlimmedPrimaryVertices'),
                rho                      = cms.InputTag('fixedGridRhoAll'),
                graph_file               = cms.vstring(file_names),
                mem_mapped               = cms.bool(False),
                version                  = cms.uint32(self.getDeepTauVersion(file_names[0])[1]),
                sub_version              = cms.uint32(1),
                debug_level              = cms.int32(0),
                disable_dxy_pca          = cms.bool(True),
                is_online                = cms.bool(False)
            ))

            self.processDeepProducer(_deepTauName, tauIDSources, workingPoints_)

            _deepTauProducer = getattr(self.process,_deepTauName+self.postfix)
            _rerunMvaIsolationTask.add(_deepTauProducer)
            _rerunMvaIsolationSequence += _deepTauProducer

        if "deepTau2017v2p5" in self.toKeep:
            if self.debug: print ("Adding DeepTau IDs")

            _deepTauName = "deepTau2017v2p5"
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
                'core:/nfs/dust/cms/user/mykytaua/softDeepTau/DeepTau_FullTrain2022_cmssw12_4/models/base/deepTau_2022v2p5_base_core.pb',
                'inner:/nfs/dust/cms/user/mykytaua/softDeepTau/DeepTau_FullTrain2022_cmssw12_4/models/base/deepTau_2022v2p5_base_inner.pb',
                'outer:/nfs/dust/cms/user/mykytaua/softDeepTau/DeepTau_FullTrain2022_cmssw12_4/models/base/deepTau_2022v2p5_base_outer.pb',
            ]

            setattr(self.process,_deepTauName+self.postfix,cms.EDProducer("DeepTauId",
                electrons                       = cms.InputTag('slimmedElectrons'),
                muons                           = cms.InputTag('slimmedMuons'),
                taus                            = cms.InputTag(self.originalTauName),
                pfcands                         = cms.InputTag('packedPFCandidates'),
                vertices                        = cms.InputTag('offlineSlimmedPrimaryVertices'),
                rho                             = cms.InputTag('fixedGridRhoAll'),
                graph_file                      = cms.vstring(file_names),
                mem_mapped                      = cms.bool(False),
                version                         = cms.uint32(self.getDeepTauVersion(file_names[0])[1]),
                sub_version                     = cms.uint32(5),
                debug_level                     = cms.int32(0),
                disable_dxy_pca                 = cms.bool(True),
                disable_hcalFraction_workaround = cms.bool(True),
                disable_CellIndex_workaround    = cms.bool(True),
                is_online                       = cms.bool(False),
                save_inputs                     = cms.bool(True)
            ))


            self.processDeepProducer(_deepTauName, tauIDSources, workingPoints_)

            _deepTauProducer = getattr(self.process,_deepTauName+self.postfix)
            _rerunMvaIsolationTask.add(_deepTauProducer)
            _rerunMvaIsolationSequence += _deepTauProducer


        if self.debug: print('Embedding new TauIDs into \"'+self.updatedTauName+'\"')
        if not hasattr(self.process, self.updatedTauName):
            embedID = cms.EDProducer("PATTauIDEmbedder",
               src = cms.InputTag(self.originalTauName),
               tauIDSources = tauIDSources
            )
            setattr(self.process, self.updatedTauName, embedID)
        else: #assume same type
            tauIDSources = cms.PSet(
                getattr(self.process, self.updatedTauName).tauIDSources,
                tauIDSources)
            getattr(self.process, self.updatedTauName).tauIDSources = tauIDSources
        if not hasattr(self.process,"rerunMvaIsolationTask"+self.postfix):
            setattr(self.process,"rerunMvaIsolationTask"+self.postfix,_rerunMvaIsolationTask)
        else:
            _updatedRerunMvaIsolationTask = getattr(self.process,"rerunMvaIsolationTask"+self.postfix)
            _updatedRerunMvaIsolationTask.add(_rerunMvaIsolationTask)
            setattr(self.process,"rerunMvaIsolationTask"+self.postfix,_updatedRerunMvaIsolationTask)
        if not hasattr(self.process,"rerunMvaIsolationSequence"+self.postfix):
            setattr(self.process,"rerunMvaIsolationSequence"+self.postfix,_rerunMvaIsolationSequence)
        else:
            _updatedRerunMvaIsolationSequence = getattr(self.process,"rerunMvaIsolationSequence"+self.postfix)
            _updatedRerunMvaIsolationSequence += _rerunMvaIsolationSequence
            setattr(self.process,"rerunMvaIsolationSequence"+self.postfix,_updatedRerunMvaIsolationSequence)


    def processDeepProducer(self, producer_name, tauIDSources, workingPoints_):
        for target,points in workingPoints_.items():
            setattr(tauIDSources, 'by{}VS{}raw'.format(producer_name[0].upper()+producer_name[1:], target),
                        cms.PSet(inputTag = cms.InputTag(producer_name+self.postfix, 'VS{}'.format(target)), workingPointIndex = cms.int32(-1)))
            
            cut_expressions = []
            for index, (point,cut) in enumerate(points.items()):
                cut_expressions.append(str(cut))

                setattr(tauIDSources, 'by{}{}VS{}'.format(point, producer_name[0].upper()+producer_name[1:], target),
                        cms.PSet(inputTag = cms.InputTag(producer_name+self.postfix, 'VS{}'.format(target)), workingPointIndex = cms.int32(index)))

            setattr(getattr(self.process, producer_name+self.postfix), 'VS{}WP'.format(target), cms.vstring(*cut_expressions))


    def getDpfTauVersion(self, file_name):
        """returns the DNN version. File name should contain a version label with data takig year (2011-2, 2015-8) and \
           version number (vX), e.g. 2017v0, in general the following format: {year}v{version}"""
        version_search = re.search('201[125678]v([0-9]+)[\._]', file_name)
        if not version_search:
            raise RuntimeError('File "{}" has an invalid name pattern, should be in the format "{year}v{version}". \
                                Unable to extract version number.'.format(file_name))
        version = version_search.group(1)
        return int(version)

    def getDeepTauVersion(self, file_name):
        """returns the DeepTau year, version, subversion. File name should contain a version label with data takig year \
        (2011-2, 2015-8), version number (vX) and subversion (pX), e.g. 2017v0p6, in general the following format: \
        {year}v{version}p{subversion}"""
        version_search = re.search('(20[1,2][125678])v([0-9]+)(p[0-9]+|)[\._]', file_name)
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
