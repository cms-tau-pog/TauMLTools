import ROOT
import glob
import os
from makeTree import MakeTupleClass
from array import array

ROOT.gROOT.SetBatch(True) # Don't show window
ROOT.ROOT.EnableImplicitMT(4) # multi thread (use 4 threads)
_rootpath = os.path.abspath(os.path.dirname(__file__)+"../") 
ROOT.gROOT.ProcessLine(".include "+_rootpath)
# call function 
class_def = MakeTupleClass('taus', '/home/russell/skimmed_tuples/WJetsToLNu/WJetsToLNu_skimmed_pt30.root', 'input_tuple',
               'Tau', 'TauTuple')
ROOT.gInterpreter.ProcessLine(class_def)

ROOT.gInterpreter.Declare(' # include "/home/russell/AdversarialTauML/TauMLTools/EventDrop.h" ')



edges = array('d', [30,35,40,45,50,55,60,65,70,80,90,100,120,140,200,400])


# # Import target Histo
ICL_datacard = ROOT.TFile.Open("/home/russell/histograms/datacard_pt_2_inclusive_mt_2018_0p9VSjet.root","READ")
target = ICL_datacard.Get("mt_inclusive/EMB")

# # Create Observed Histo
model = ROOT.RDF.TH1DModel("QCD", "h", 15, edges)
df = ROOT.RDataFrame("taus", "/home/russell/skimmed_tuples/MuTau_prod2018/DY_taus_skimmed.root")
observed = df.Histo1D(model, "tau_pt")

# Divide Target by Observed
ratio = target.Clone("ratio")
ratio.Divide(observed.GetPtr())
maxval = ratio.GetBinContent(ratio.GetMaximumBin()) 
ratio.Scale(1/maxval)



n_qcd = 126127

print("Mix Characteristics:")

qcdDesc = ROOT.DatasetDesc()
qcdDesc.n_per_batch = n_qcd
qcdDesc.threshold = ratio
qcdDesc.fileNames.push_back("/home/russell/skimmed_tuples/MuTau_prod2018/DY_taus_skimmed.root")
print(f"Number of qcd per batch: {n_qcd}")





print("---------------------------------------------------------")

allDescs = ROOT.std.vector(ROOT.DatasetDesc)()
allDescs.push_back(qcdDesc)
# allDescs.push_back(fifteenDesc)
# allDescs.push_back(thirtyDesc)
# allDescs.push_back(fiftyDesc)
# allDescs.push_back(eightyDesc)
# allDescs.push_back(onetwentyDesc)
# allDescs.push_back(oneseventyDesc)
print("BeforeMixer")

outputVect = ROOT.std.vector(ROOT.OutputDesc)()

output_train = ROOT.OutputDesc()
output_train.fileName_ = 'DY_reweighted.root'
output_train.nBatches_ = 1
# output_train.fileName_ = 'QCD_mixed_preliminary.root'
# output_train.nBatches_ = 94


outputVect.push_back(output_train)



dataMixer = ROOT.DataMixer(allDescs, outputVect)
print("Passed dataMixer")
dataMixer.Run()
print("End of mixing")