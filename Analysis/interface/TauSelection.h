#pragma once

#include "TauMLTools/Analysis/interface/GenLepton.h"
#include "TauMLTools/Analysis/interface/AnalysisTypes.h"
#include "TauMLTools/Core/interface/AnalysisMath.h"
#include "TauMLTools/Core/interface/exception.h"

namespace analysis {
    std::optional<GenLeptonMatch> GetGenLeptonMatch(reco_tau::gen_truth::GenLepton::Kind genLepton_kind,  Int_t genLepton_index,  Float_t tau_pt,  Float_t tau_eta,
     Float_t tau_phi,  Float_t tau_mass,  Float_t genLepton_vis_pt,  Float_t genLepton_vis_eta,  Float_t genLepton_vis_phi,  Float_t genLepton_vis_mass,
     Int_t genJet_index)
    {   
        using GTGenLeptonKind = reco_tau::gen_truth::GenLepton::Kind;

        if (genLepton_index >= 0){
            LorentzVectorM genv(tau_pt, tau_eta, tau_phi, tau_mass);
            LorentzVectorM tauv(  genLepton_vis_pt  ,
                                genLepton_vis_eta ,
                                genLepton_vis_phi ,
                                genLepton_vis_mass);

            if (ROOT::Math::VectorUtil::DeltaR(tauv, genv) > 0.2){
                return std::nullopt;
            }

            if (genLepton_kind == GTGenLeptonKind::PromptElectron){
                if (genLepton_vis_pt < 8.0) return std::nullopt;
                return GenLeptonMatch::Electron;
            }
            else if (genLepton_kind == GTGenLeptonKind::PromptMuon){
                if (genLepton_vis_pt < 8.0) return std::nullopt;
                return GenLeptonMatch::Muon;
            }
            else if (genLepton_kind == GTGenLeptonKind::TauDecayedToElectron){
                if (genLepton_vis_pt < 8.0) return std::nullopt;
                return GenLeptonMatch::TauElectron;
            }
            else if (genLepton_kind == GTGenLeptonKind::TauDecayedToMuon){
                if (genLepton_vis_pt < 8.0) return std::nullopt;
                return GenLeptonMatch::TauMuon;
            }
            else if (genLepton_kind == GTGenLeptonKind::TauDecayedToHadrons){
                if (genLepton_vis_pt < 15.0) return std::nullopt;
                return GenLeptonMatch::Tau;
            }
            else {
                throw exception("genLepton_kind = %1% should not happend when genLepton_index is %2%")
                    %static_cast<int>(genLepton_kind) %genLepton_index;
            }
        }
        else if (genJet_index >= 0){
            return GenLeptonMatch::NoMatch;
        }
        else {
            return std::nullopt;
        }
    }
} // namespace analysis
