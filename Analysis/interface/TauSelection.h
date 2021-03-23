#include "TauMLTools/Analysis/interface/TauTuple.h"
#include "TauMLTools/Core/interface/exception.h"

#include "TLorentzVector.h"

namespace analysis {
    bool PassGenLeptonCut(const tau_tuple::Tau& tau)
    {   
        if (tau.genLepton_index >= 0){
            TLorentzVector genv;
            TLorentzVector tauv;

            tauv.SetPtEtaPhiM(tau.tau_pt, tau.tau_eta, tau.tau_phi, tau.tau_mass);
            genv.SetPtEtaPhiM(  tau.genLepton_vis_pt   ,
                                tau.genLepton_vis_eta  ,
                                tau.genLepton_vis_phi  ,
                                tau.genLepton_vis_mass );
            
            if (tauv.DeltaR(genv) > 0.2){
                return false;
            }

            if (tau.genLepton_kind > 0 && tau.genLepton_kind < 5){
                return tau.genLepton_vis_pt >= 8.0;
            } 
            else if (tau.genLepton_kind == 5){
                return tau.genLepton_vis_pt >= 15.0;
            } 
            else{
                throw exception("genLepton_kind = %1% should not happend when genLepton_index is %2%")
                    %tau.genLepton_kind %tau.genLepton_index;
            }
        } 
        else if (tau.genLepton_index < 0 && tau.genJet_index >= 0){
            return true;
        } 
        else {
            return false;
        }
        throw exception("Invalid execution. tau.genLepton_index = %1% tau.genLepton_kind = %2%")
            %tau.genLepton_index %tau.genLepton_kind;
    }
} // namespace analysis