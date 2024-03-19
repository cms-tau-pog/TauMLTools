

from HLTrigger.Configuration.customizeHLTforCMSSW import customizeHLTforCMSSW
process = customizeHLTforCMSSW(process,"GRun")

# Eras-based customisations
from HLTrigger.Configuration.Eras import modifyHLTforEras
modifyHLTforEras(process)

from FWCore.ParameterSet.VarParsing import VarParsing
options = VarParsing('analysis')
options.register('output', 'nano.root', VarParsing.multiplicity.singleton, VarParsing.varType.string, "Output file.")
options.parseArguments()

process.source.fileNames = options.inputFiles
process.maxEvents.input = options.maxEvents

import importlib.util
import os
import sys

def load(module_file, default_path):
  module_path = os.path.join(default_path, module_file)
  if not os.path.exists(module_path):
    module_path = os.path.join(os.path.dirname(__file__), module_file)
    if not os.path.exists(module_path):
      module_path = os.path.join(os.getenv("CMSSW_BASE"), 'src', module_file)
      if not os.path.exists(module_path):
        raise RuntimeError(f"Cannot find path to {module_file}.")

  module_name, module_ext = os.path.splitext(module_file)
  spec = importlib.util.spec_from_file_location(module_name, module_path)
  module = importlib.util.module_from_spec(spec)
  sys.modules[module_name] = module
  spec.loader.exec_module(module)
  return module

base_path = os.path.join(os.getenv("CMSSW_BASE"), 'src/TauMLTools/Production/python')

customiseHLT = load('customiseHLT.py', base_path)
process = customiseHLT.customise(process, output=options.output, is_data=IS_DATA)

# print(process.dumpPython())