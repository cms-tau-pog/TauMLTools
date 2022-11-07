# Configurations dependent on the sample type.

import sys
import FWCore.ParameterSet.Config as cms
import os

if sys.version_info.major < 3:
    from sets import Set as set

mcSampleTypes = set([ 'MC_16', 'MC_UL16', 'MC_UL16APV', 'MC_17', 'MC_UL17', 'MC_18', 'MC_UL18', 'Emb_16', 'Emb_17', 'Emb_18ABC', 'Emb_18D', 'MC_Phase2_113X', 'MC_Phase2_111X', 'MC_Phase2_110X', 'MC_RUN3_122X'])
dataSampleTypes = set([ 'Run2016' , 'Run2017', 'Run2018ABC', 'Run2018D', 'RunUL2018' ])

periodDict = { 'MC_16' : 'Run2016',
               'MC_UL16': 'Run2016',
               'MC_UL16APV': 'Run2016',
               'Run2016' : 'Run2016',
               'Emb_16' : 'Run2016',
               'MC_17' : 'Run2017',
               'MC_UL17': 'Run2017',
               'Run2017' : 'Run2017',
               'Emb_17' : 'Run2017',
               'MC_18' : 'Run2018',
               'MC_UL18' : 'Run2018',
               'Run2018ABC' : 'Run2018',
               'Run2018D' : 'Run2018',
               'RunUL2018' : 'Run2018',
               'Emb_18ABC' : 'Run2018',
               'Emb_18D' : 'Run2018',
               'MC_Phase2_110X' : 'Phase2',
               'MC_Phase2_111X' : 'Phase2',
               'MC_Phase2_113X' : 'Phase2',
               'MC_RUN3_122X': "Run3"
             }

globalTagMap = { 'MC_16' : '102X_mcRun2_asymptotic_v7',
                 'MC_UL16': '106X_mcRun2_asymptotic_v17',
                 'MC_UL16APV': '106X_mcRun2_asymptotic_preVFP_v11',
                 'Run2016' : '102X_dataRun2_v12',
                 'Emb_16' : '102X_dataRun2_v12',
                 #'Emb_16' : '80X_dataRun2_2016SeptRepro_v7',
                 'MC_17' : '102X_mc2017_realistic_v7',
                 'MC_UL17': '106X_mc2017_realistic_v9',
                 'Run2017' : '102X_dataRun2_v12',
                 'Emb_17' : '102X_dataRun2_v12',
                 'MC_18' : '102X_upgrade2018_realistic_v20',
                 'MC_UL18' : '106X_upgrade2018_realistic_v16_L1v1',
                 'Run2018ABC' : '102X_dataRun2_v12',
                 'Run2018D' : '102X_dataRun2_Prompt_v15',
                 'RunUL2018' : '106X_dataRun2_v35',
                 'Emb_18ABC' : '102X_dataRun2_v12',
                 'Emb_18D' : '102X_dataRun2_Prompt_v15',
                 'MC_Phase2_110X' : '110X_mcRun4_realistic_v3',
                 'MC_Phase2_111X' : 'auto:phase2_realistic_T15',
                 'MC_Phase2_113X' : 'auto:phase2_realistic_T15',
                 'MC_RUN3_122X' : '122X_mcRun3_2021_realistic_v9'
               }

def IsEmbedded(sampleType):
    isEmbedded = sampleType in mcSampleTypes and 'Emb' in sampleType
    if not sampleType in mcSampleTypes and not sampleType in dataSampleTypes:
        print ("ERROR: unknown sample type = '{}'".format(sampleType))
        sys.exit(1)
    return isEmbedded

def IsData(sampleType):
    isData = sampleType in dataSampleTypes
    if not isData and not sampleType in mcSampleTypes:
        print ("ERROR: unknown sample type = '{}'".format(sampleType))
        sys.exit(1)
    return isData

def GetPeriod(sampleType):
    if sampleType not in periodDict:
        print ("ERROR: unknown sample type = '{}'".format(sampleType))
        sys.exit(1)
    return periodDict[sampleType]

def GetGlobalTag(sampleType):
    if sampleType not in globalTagMap:
        print ("ERROR: unknown sample type = '{}'".format(sampleType))
        sys.exit(1)
    return globalTagMap[sampleType]

def isRun2UL(sampleType):
    if sampleType not in periodDict:
        print ("ERROR: unknown sample type = '{}'".format(sampleType))
        sys.exit(1)
    return sampleType in ['MC_UL16', 'MC_UL16APV', 'MC_UL17', 'MC_UL18', 'RunUL2018']

def isPhase2(sampleType):
    if sampleType not in periodDict:
        print ("ERROR: unknown sample type = '{}'".format(sampleType))
        sys.exit(1)
    return sampleType in ['MC_Phase2_113X', 'MC_Phase2_111X', 'MC_Phase2_110X']

def isRun2PreUL(sampleType):
    if sampleType not in periodDict:
        print ("ERROR: unknown sample type = '{}'".format(sampleType))
        sys.exit(1)
    return sampleType in ['MC_16', 'MC_17', 'MC_18','Run2018ABC','Run2018D','Emb_18ABC','Emb_18D']

def isRun3(sampleType):
    if sampleType not in periodDict:
        print ("ERROR: unknown sample type = '{}'".format(sampleType))
        sys.exit(1)
    return sampleType in ['MC_RUN3_122X']

def GetPeriodCfg(sampleType):
    period = GetPeriod(sampleType)
    if period == 'Run2016':
        from Configuration.Eras.Era_Run2_2016_cff import Run2_2016
        return Run2_2016
    elif period == 'Run2017':
        from Configuration.Eras.Era_Run2_2017_cff import Run2_2017
        return Run2_2017
    elif period == 'Run2018':
        from Configuration.Eras.Era_Run2_2018_cff import Run2_2018
        return Run2_2018
    elif period == 'Phase2':
        from Configuration.Eras.Era_Phase2C9_cff import Phase2C9
        return Phase2C9
    elif period == 'Run3':
        from Configuration.Eras.Era_Run3_cff import Run3
        return Run3
    else:
        raise RuntimeError('Period = "{}" is not supported.'.format(period))
