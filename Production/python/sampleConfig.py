# Configurations dependent on the sample type.

import sys
from sets import Set
import FWCore.ParameterSet.Config as cms

mcSampleTypes = Set([ 'Summer16MC', 'Fall17MC' ])
dataSampleTypes = Set([ 'Run2016' , 'Run2017' ])

periodDict = { 'Summer16MC' : 'Run2016', 'Run2016' : 'Run2016',
               'Fall17MC' : 'Run2017', 'Run2017' : 'Run2017'
}

def IsData(sampleType):
    isData = sampleType in dataSampleTypes
    if not isData and not sampleType in mcSampleTypes:
        raise RuntimeError("ERROR: unknown sample type = '{}'".format(sampleType))
    return isData

def GetPeriod(sampleType):
    if sampleType not in periodDict:
        raise RuntimeError("ERROR: unknown sample type = '{}'".format(sampleType))
    return periodDict[sampleType]
