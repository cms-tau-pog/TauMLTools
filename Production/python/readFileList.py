## Definition of the function to read intput file list from a text file.
## This file is part of https://github.com/hh-italian-group/AnalysisTools.

def readFileList(fileList, inputFileName, fileNamePrefix):
    """read intput file list from a text file"""
    with open(inputFileName, 'r') as inputFile:
        for name in inputFile.readlines():
            if len(name) > 0 and name[0] != '#':
                fileList.append(fileNamePrefix + name)

def addFilesToList(fileList, inputFiles, fileNamePrefix):
    """read intput file list from a another list"""
    for name in inputFiles:
        if len(name) > 0 and name[0] != '#':
            fileList.append(fileNamePrefix + name)
