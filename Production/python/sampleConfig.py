# Configurations dependent on the sample type.

import FWCore.ParameterSet.Config as cms
from enum import Enum

class Era(Enum):
    Run2_2016_HIPM = 0
    Run2_2016 = 1
    Run2_2017 = 2
    Run2_2018 = 3
    Run3_2022 = 4
    Run3_2024 = 5
    Phase2_110X = 100
    Phase2_111X = 101
    Phase2_113X = 102

class SampleType(Enum):
    MC = 1
    Data = 2
    Embedded = 3

_globalTagDict = {
    (Era.Run2_2016_HIPM, SampleType.MC) : 'auto:run2_mc_pre_vfp',
    (Era.Run2_2016, SampleType.MC) : 'auto:run2_mc',
    (Era.Run2_2017, SampleType.MC) : 'auto:phase1_2017_realistic',
    (Era.Run2_2018, SampleType.MC) : 'auto:phase1_2018_realistic',
    (Era.Run2_2016_HIPM, SampleType.Data) : 'auto:run2_data',
    (Era.Run2_2016, SampleType.Data) : 'auto:run2_data',
    (Era.Run2_2017, SampleType.Data) : 'auto:run2_data',
    (Era.Run2_2018, SampleType.Data) : 'auto:run2_data',
    (Era.Run3_2022, SampleType.MC) : 'auto:phase1_2022_realistic',
    (Era.Run3_2022, SampleType.Data) : 'auto:run3_data',
    (Era.Run3_2024, SampleType.MC) : '150X_mcRun3_2024_realistic_v2',
    (Era.Phase2_110X, SampleType.MC) : '110X_mcRun4_realistic_v3',
    (Era.Phase2_111X, SampleType.MC) : 'auto:phase2_realistic_T15',
    (Era.Phase2_113X, SampleType.MC) : 'auto:phase2_realistic_T15',
}

def getGlobalTag(era, sampleType):
    globalTag = _globalTagDict.get((era, sampleType), None)
    if globalTag is None:
        raise RuntimeError("ERROR: unknown sample type = '{}'".format(sampleType))
    return globalTag

def isRun2(era):
    return era.name.startswith('Run2_')

def isPhase2(era):
    return era.name.startswith('Phase2_')

def isRun3(era):
    return era.name.startswith('Run3_')

def getEraCfg(era):
    if era == Era.Run2_2016_HIPM:
        from Configuration.Eras.Era_Run2_2016_HIPM_cff import Run2_2016_HIPM
        return Run2_2016_HIPM
    elif era == Era.Run2_2016:
        from Configuration.Eras.Era_Run2_2016_cff import Run2_2016
        return Run2_2016
    elif era == Era.Run2_2017:
        from Configuration.Eras.Era_Run2_2017_cff import Run2_2017
        return Run2_2017
    elif era == Era.Run2_2018:
        from Configuration.Eras.Era_Run2_2018_cff import Run2_2018
        return Run2_2018
    elif era == Era.Run3_2022:
        from Configuration.Eras.Era_Run3_cff import Run3
        return Run3
    elif era == Era.Run3_2024:
        from Configuration.Eras.Era_Run3_cff import Run3
        return Run3
    elif isPhase2(era):
        from Configuration.Eras.Era_Phase2C9_cff import Phase2C9
        return Phase2C9
    else:
        raise RuntimeError('era = "{}" is not supported.'.format(era))
