import ROOT

ROOT.gROOT.ProcessLine('ROOT::EnableImplicitMT(6)')

chain = ROOT.TChain("taus")

prefix = "TTToSemiLeptonic_TuneCP5_PSweights_13TeV-powheg-pythia8/train/full"

file_list = [ "eventTuple_1.root", "eventTuple_2.root", "eventTuple_3.root", "eventTuple_4.root", "eventTuple_5.root" ]

for f in file_list:
    chain.Add('{}/{}'.format(prefix, f))


print("Copy electrons")
f_out = ROOT.TFile("taus_ele.root", "RECREATE")
f_out.SetCompressionAlgorithm(4)
f_out.SetCompressionLevel(3)
taus_out = chain.CopyTree("pt > 20 && abs(eta) < 2.3 && (gen_match==1 || gen_match == 3)")
f_out.WriteTObject(taus_out, "taus", "Overwrite")
f_out.Close()

print("Copy muons")
f_out = ROOT.TFile("taus_mu.root", "RECREATE")
f_out.SetCompressionAlgorithm(4)
f_out.SetCompressionLevel(3)
taus_out = chain.CopyTree("pt > 20 && abs(eta) < 2.3 && (gen_match==2 || gen_match == 4)")
f_out.WriteTObject(taus_out, "taus", "Overwrite")
f_out.Close()

print("Copy taus")
f_out = ROOT.TFile("taus_tau.root", "RECREATE")
f_out.SetCompressionAlgorithm(4)
f_out.SetCompressionLevel(3)
taus_out = chain.CopyTree("pt > 20 && abs(eta) < 2.3 && gen_match==5")
f_out.WriteTObject(taus_out, "taus", "Overwrite")
f_out.Close()

print("Copy jets")
f_out = ROOT.TFile("taus_jet.root", "RECREATE")
f_out.SetCompressionAlgorithm(4)
f_out.SetCompressionLevel(3)
taus_out = chain.CopyTree("pt > 20 && abs(eta) < 2.3 && gen_match==6")
f_out.WriteTObject(taus_out, "taus", "Overwrite")
f_out.Close()


#taus_out = taus.CloneTree(-1, "SortBasketsByBranch")
#taus_out = taus.CloneTree(500000, "SortBasketsByBranch")
#taus_out = taus.CloneTree(0)
#taus_out.SetBasketSize("*", 100000000)
#taus_out.CopyEntries(taus, 1000000, "SortBasketsByBranch")
#f_out.WriteTObject(taus_out, "taus", "Overwrite")
#f_out.Close()
#f_in.Close()
