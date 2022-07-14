import ROOT
import glob
import os
from makeTree import MakeTupleClass


ROOT.gROOT.SetBatch(True) # Don't show window
ROOT.ROOT.EnableImplicitMT(4) # multi thread (use 4 threads)
_rootpath = os.path.abspath(os.path.dirname(__file__)+"../") 
ROOT.gROOT.ProcessLine(".include "+_rootpath)
# call function 
class_def = MakeTupleClass('taus', '/home/russell/skimmed_tuples/WJetsToLNu/WJetsToLNu_skimmed_pt30.root', 'input_tuple',
               'Tau', 'TauTuple')
ROOT.gInterpreter.ProcessLine(class_def)

ROOT.gInterpreter.Declare(' # include "/home/russell/AdversarialTauML/TauMLTools/DataMixer.h" ')

# copy_to_def = MakeCopyTo('input_tuple::Tau', 'tau_tuple::Tau')
# ROOT.gInterpreter.ProcessLine(copy_to_def)



# f'''void CopyTo(const {class1}& a, {class2}& b) {{
#        '''
# colums = intersection
# for column in columns:
#     copy_def += f'b.{column} = a.{column};'

# dir(ROOT.tau_tuple.Tau)

n_data = 50
n_DY_taus = 36
n_tt_taus = 2
n_W_jets = 5
n_QCD_jets = 5
n_tt_jets = 1
n_DY_muons = 1

print("Mix Characteristics:")


dataDesc = ROOT.DatasetDesc()
dataDesc.n_per_batch = n_data
dataDesc.selection = "data"
dataDesc.fileNames.push_back("/home/russell/skimmed_tuples/MuTau_prod2018/SingleMuon_skimmed.root")
print(f"Number of data per batch: {n_data}")

DY_tausDesc = ROOT.DatasetDesc()
DY_tausDesc.n_per_batch = n_DY_taus
DY_tausDesc.selection = "DY_taus"
DY_tausDesc.fileNames.push_back("/home/russell/AdversarialTauML/TauMLTools/DY_reweighted.root")
print(f"Number of DY taus per batch: {n_DY_taus}")

tt_tausDesc = ROOT.DatasetDesc()
tt_tausDesc.n_per_batch = n_tt_taus 
tt_tausDesc.selection = "tt_taus"
tt_tausDesc.fileNames.push_back("/home/russell/skimmed_tuples/MuTau_prod2018/TTtau_skimmed.root")
print(f"Number of tt taus per batch: {n_tt_taus}")

W_jetsDesc = ROOT.DatasetDesc()
W_jetsDesc.n_per_batch = n_W_jets 
W_jetsDesc.selection = "W_jets"
W_jetsDesc.fileNames.push_back("/home/russell/skimmed_tuples/MuTau_prod2018/Wjets_skimmed.root")
print(f"Number of W jets per batch: {n_W_jets}")

QCD_jetsDesc = ROOT.DatasetDesc()
QCD_jetsDesc.n_per_batch = n_QCD_jets
QCD_jetsDesc.selection = "QCD_jets"
QCD_jetsDesc.fileNames.push_back("/home/russell/skimmed_tuples/QCD_Pt/QCD_mixed_final.root")
print(f"Number of QCD jets per batch: {n_QCD_jets}")

tt_jetsDesc = ROOT.DatasetDesc()
tt_jetsDesc.n_per_batch = n_tt_jets
tt_jetsDesc.selection = "tt_jets"
tt_jetsDesc.fileNames.push_back("/home/russell/skimmed_tuples/MuTau_prod2018/TTjets_skimmed.root")
print(f"Number of tt jets per batch: {n_tt_jets}")

DY_muonsDesc = ROOT.DatasetDesc()
DY_muonsDesc.n_per_batch = n_DY_muons
DY_muonsDesc.selection = "DY_muons"
DY_muonsDesc.fileNames.push_back("/home/russell/skimmed_tuples/DY1JetsToLL_M50/DY1JetsToLL_M50_skimmed_mu_pt30.root")
print(f"Number of DY muons per batch: {n_DY_muons}")

print("---------------------------------------------------------")

allDescs = ROOT.std.vector(ROOT.DatasetDesc)()
allDescs.push_back(dataDesc)
allDescs.push_back(DY_tausDesc)
allDescs.push_back(tt_tausDesc)
allDescs.push_back(W_jetsDesc)
allDescs.push_back(QCD_jetsDesc)
allDescs.push_back(tt_jetsDesc)
allDescs.push_back(DY_muonsDesc)
print("BeforeMixer")

outputVect = ROOT.std.vector(ROOT.OutputDesc)()

output_train = ROOT.OutputDesc()
output_train.fileName_ = 'adv_dataset_DEBUG.root'
output_train.nBatches_ = 1125


output_test = ROOT.OutputDesc()
output_test.fileName_ = 'adv_dataset_DEBUG2.root'
output_test.nBatches_ = 750

outputVect.push_back(output_train)
outputVect.push_back(output_test)


dataMixer = ROOT.DataMixer(allDescs, outputVect)
print("Passed dataMixer")
dataMixer.Run()
print("End of mixing")