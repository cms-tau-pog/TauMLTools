import ROOT
import glob
import os
from makeTree import MakeTupleClass
from array import array
import argparse

parser = argparse.ArgumentParser(description='Drop events randomly to rebalance pT spectrum of an input file')
parser.add_argument('--input_file', required=True, type=str, help="Input file")
parser.add_argument('--target_file', required=True, type=str, help="Target datacard")
parser.add_argument('--target_histo', required=True, type=str, help="Target histogram within datacard")
parser.add_argument('--n_tau', required=True, type=int, help="Total number of taus to process")
parser.add_argument('--save_path', required=True, type=str, help="Save path")
args = parser.parse_args()

ROOT.gROOT.SetBatch(True) # Don't show window
ROOT.ROOT.EnableImplicitMT(4) # multi thread (use 4 threads)


# Define tau tuple class from input files
_rootpath = os.path.abspath(os.path.dirname(__file__)+"../../..")
ROOT.gROOT.ProcessLine(".include "+_rootpath)
class_def = MakeTupleClass('taus', args.input_file, 'input_tuple',
               'Tau', 'TauTuple')
ROOT.gInterpreter.ProcessLine(class_def)

ROOT.gInterpreter.Declare(' # include "TauMLTools/Training/interface/DataMixer.h" ')


# pT binning:
edges = array('d', [30,35,40,45,50,55,60,65,70,80,90,100,120,140,200,400])

# Generate Observed Histo
model = ROOT.RDF.TH1DModel("Input histogram", "h", 15, edges)
df = ROOT.RDataFrame("taus", args.input_file)
observed = df.Histo1D(model, "tau_pt")

# Import Target histogram
target_datacard = ROOT.TFile.Open(args.target_file,"READ")
target = target_datacard.Get(args.target_histo)

# Find observed/target for each bin and scale by max value
ratio = target.Clone("ratio")
ratio.Divide(observed.GetPtr())
maxval = ratio.GetBinContent(ratio.GetMaximumBin()) 
ratio.Scale(1/maxval)
# This gives a value between 0 and 1 representing the probability that an event
# in that bin should be kept, eg if there are more events in a bin than expected
# the value will be low

print("Target ratios defined")

Desc = ROOT.DatasetDesc()
Desc.n_per_batch = args.n_tau
Desc.threshold = ratio
Desc.fileNames.push_back(args.input_file)
print(f"Number of taus to be selected: {args.n_tau}")


allDescs = ROOT.std.vector(ROOT.DatasetDesc)()
allDescs.push_back(Desc)

outputVect = ROOT.std.vector(ROOT.OutputDesc)()

output_train = ROOT.OutputDesc()
output_train.fileName_ = args.save_path
output_train.nBatches_ = 1

outputVect.push_back(output_train)

dataMixer = ROOT.DataMixer(allDescs, outputVect, "Drop")
print("Event dropping initiated")
dataMixer.Run()
print(f"Event dropping finished, file saved at {args.save_path}")