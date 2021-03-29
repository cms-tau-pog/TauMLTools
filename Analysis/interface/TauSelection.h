#pragma once

#include "TauMLTools/Analysis/interface/GenLepton.h"
#include "TauMLTools/Analysis/interface/AnalysisTypes.h"
#include "TauMLTools/Core/interface/AnalysisMath.h"
#include "TauMLTools/Core/interface/exception.h"

namespace analysis {
    boost::optional<GenLeptonMatch> GetGenLeptonMatch(const tau_tuple::Tau& tau)
    {   
        using GTGenLeptonKind = reco_tau::gen_truth::GenLepton::Kind;
        const GTGenLeptonKind genLepton_kind = static_cast<GTGenLeptonKind> (tau.genLepton_kind);

        if (tau.genLepton_index >= 0){
            LorentzVectorM genv(tau.tau_pt, tau.tau_eta, tau.tau_phi, tau.tau_mass);
            LorentzVectorM tauv(  tau.genLepton_vis_pt  ,
                                tau.genLepton_vis_eta ,
                                tau.genLepton_vis_phi ,
                                tau.genLepton_vis_mass);

            if (ROOT::Math::VectorUtil::DeltaR(tauv, genv) > 0.2){
                return boost::none;
            }

            if (genLepton_kind == GTGenLeptonKind::PromptElectron){
                if (tau.genLepton_vis_pt < 8.0) return boost::none;
                return GenLeptonMatch::Electron;
            }
            else if (genLepton_kind == GTGenLeptonKind::PromptMuon){
                if (tau.genLepton_vis_pt < 8.0) return boost::none;
                return GenLeptonMatch::Muon;
            }
            else if (genLepton_kind == GTGenLeptonKind::TauDecayedToElectron){
                if (tau.genLepton_vis_pt < 8.0) return boost::none;
                return GenLeptonMatch::TauElectron;
            }
            else if (genLepton_kind == GTGenLeptonKind::TauDecayedToMuon){
                if (tau.genLepton_vis_pt < 8.0) return boost::none;
                return GenLeptonMatch::TauMuon;
            }
            else if (genLepton_kind == GTGenLeptonKind::TauDecayedToHadrons){
                if (tau.genLepton_vis_pt < 15.0) return boost::none;
                return GenLeptonMatch::Tau;
            }
            else {
                throw exception("genLepton_kind = %1% should not happend when genLepton_index is %2%")
                    %tau.genLepton_kind %tau.genLepton_index;
            }
        }
        else if (tau.genJet_index >= 0){
            return GenLeptonMatch::NoMatch;
        }
        else {
            return boost::none;
        }
    }
} // namespace analysis
