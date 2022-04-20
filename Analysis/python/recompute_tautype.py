import ROOT as R
import os

_rootpath = os.path.abspath(os.path.dirname(__file__)+"../../../..")
R.gROOT.ProcessLine(".include "+_rootpath)
R.gInterpreter.ProcessLine('''

#include "TauMLTools/Analysis/interface/TauSelection.h"
#include "TauMLTools/Analysis/interface/AnalysisTypes.h"

int GetMyTauType(Int_t genLepton_kind,  Int_t genLepton_index,  Float_t tau_pt,  Float_t tau_eta,
     Float_t tau_phi,  Float_t tau_mass,  Float_t genLepton_vis_pt,  Float_t genLepton_vis_eta,  Float_t genLepton_vis_phi,  
     Float_t genLepton_vis_mass, Int_t genJet_index, Int_t sampleType)
     {
        const auto gen_match = analysis::GetGenLeptonMatch(static_cast<reco_tau::gen_truth::GenLepton::Kind>(genLepton_kind), genLepton_index, tau_pt, tau_eta, tau_phi, tau_mass, 
                                                            genLepton_vis_pt, genLepton_vis_eta, genLepton_vis_phi, 
                                                            genLepton_vis_mass, genJet_index);
        const auto sample_type= static_cast<analysis::SampleType> (sampleType);
        int mytauType;
        if (gen_match){
            mytauType = static_cast<int>(analysis::GenMatchToTauType(*gen_match, sample_type));
            return mytauType;
        }
        else{
            mytauType = -1;
            return mytauType;
        }   
}

''')

def compute(df):
    # currently define a new branch in the dataframe while we wait for ROOT v6.26 where we can use Redefine("tauType")
    df = df.Define("mytauType", """GetMyTauType(genLepton_kind, genLepton_index, tau_pt, tau_eta, tau_phi, tau_mass, 
                        genLepton_vis_pt, genLepton_vis_eta, genLepton_vis_phi, genLepton_vis_mass, genJet_index, sampleType)""")
    print("Tau Types Recomputed")
    return df