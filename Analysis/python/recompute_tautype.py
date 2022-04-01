import ROOT as R
import os
import numpy as npd

_rootpath = os.path.abspath(os.path.dirname(__file__)+"../../..")
R.gROOT.ProcessLine(".include "+_rootpath)
R.gInterpreter.ProcessLine('''

#include "TauMLTools/Analysis/interface/TauTuple.h"
#include "TROOT.h"
#include "TLorentzVector.h"
#include "TauMLTools/Analysis/interface/TauSelection.h"
#include "TauMLTools/Analysis/interface/AnalysisTypes.h"

''')

df = R.RDataFrame("taus", "~/histograms/eventTuple_15.root")

#print(df.GetColumnNames())
def compute(df):
    df = df.Define("gen_match", """analysis::GetGenLeptonMatch(genLepton_kind, genLepton_index, tau_pt, tau_eta, tau_phi, tau_mass, 
    genLepton_vis_pt, genLepton_vis_eta, genLepton_vis_phi, genLepton_vis_mass, genJet_index)""")
    df = df.Define("sample_type", """static_cast<analysis::SampleType>(sampleType)""")
    df = df.Define("mytauType", "static_cast<int>(analysis::GenMatchToTauType(*gen_match, sample_type))")
    print("Tau Types Recomputed")
    return df

df = compute(df)
print(df.GetColumnNames())
test = df.AsNumpy(columns=["tauType"])
npdf = df.AsNumpy(columns=["gen_match"])
npdf2 = df.AsNumpy(columns=["sample_type"])
npdf3 = df.AsNumpy(columns=["mytauType"])
print(npdf)
print(npdf2)
print(test)
print(npdf3)

    # df = df.Define("mytauType", """static_cast<Int_t>(analysis::GenMatchToTauType(*analysis::GetGenLeptonMatch(genLepton_kind, genLepton_index, tau_pt, tau_eta, tau_phi, tau_mass,
    #     genLepton_vis_pt, genLepton_vis_eta, genLepton_vis_phi, genLepton_vis_mass, genJet_index), static_cast<analysis::SampleType>(sampleType)))""")