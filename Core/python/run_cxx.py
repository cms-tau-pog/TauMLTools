import os
import sys
import ROOT

cxx_file = sys.argv[1]
base_path = os.path.abspath(os.environ['ANALYSIS_PATH'] + '/..')
if ROOT.gROOT.ProcessLine(f'.include {base_path}') != 0:
    raise RuntimeError('Failed to include base path')
if not ROOT.gInterpreter.Declare(f'#include "{cxx_file}"'):
    raise RuntimeError('Failed to include cxx file')

arg_str = 'static std::string _ARGV[] = {'
for arg in sys.argv[2:]:
    arg_str += f'"{arg}",'
arg_str += '}; static char* ARGV[] = {';
for n in range(len(sys.argv[2:])):
    arg_str += f'_ARGV[{n}].data(),'
arg_str += f'}}; static const int ARGC = {len(sys.argv[2:])};'
if not ROOT.gInterpreter.Declare(arg_str):
    raise RuntimeError('Failed to declare arguments')

ROOT.gROOT.ProcessLine('main(ARGC, ARGV);')