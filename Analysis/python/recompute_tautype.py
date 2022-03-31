import ROOT as R
import os

_rootpath = os.path.abspath(os.path.dirname(__file__)+"../../..")
R.gROOT.ProcessLine(".include "+_rootpath)
R.gInterpreter.ProcessLine('''

#include "TauMLTools/Analysis/interface/TauTuple.h"
#include "TROOT.h"
#include "TLorentzVector.h"
#include "TauMLTools/Analysis/interface/TauSelection.h"
#include "TauMLTools/Analysis/interface/AnalysisTypes.h"

''')

def redefine(df):
    df = df.Define("mytauType", """analysis::GenMatchToTauType(*analysis::GetGenLeptonMatch(genLepton_kind, genLepton_index, tau_pt, tau_eta, tau_phi, tau_mass,
        genLepton_vis_pt, genLepton_vis_eta, genLepton_vis_phi, genLepton_vis_mass, genJet_index), analysis::SampleType(sampleType))""")
    print("Tau Types Recomputed")
    return df


