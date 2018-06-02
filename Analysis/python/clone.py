import ROOT

ROOT.gROOT.ProcessLine('ROOT::EnableImplicitMT(6)')

f_in = ROOT.TFile("split.root", "READ")

id = 1

taus = f_in.Get("taus_{}".format(id))

f_out = ROOT.TFile("tau_5e5_{}.root".format(id), "RECREATE")
f_out.SetCompressionAlgorithm(4)
f_out.SetCompressionLevel(3)

#taus_out = taus.CloneTree(-1, "SortBasketsByBranch")
#taus_out = taus.CloneTree(500000, "SortBasketsByBranch")
taus_out = taus.CloneTree(0)
#taus_out.SetBasketSize("*", 1024 * 1024 * 1024)
#taus_out.SetMaxVirtualSize(1024 * 1024 * 1024)
#taus_out.SetAutoFlush(1024 * 1024 * 1024)

taus_out.CopyEntries(taus, 500000, "SortBasketsByBranch")
f_out.WriteTObject(taus_out, "taus", "Overwrite")
f_out.Close()
f_in.Close()
