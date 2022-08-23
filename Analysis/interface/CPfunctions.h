/*Interface for CP observable computation at generated level
  Author : Mario Sessini
*/

#pragma once

#include "GenLepton.h"
#include "PolarimetricA1.h"
#include "SCalculator.h"
#include "TLorentzVector.h"
#include <Math/LorentzVector.h>
#include <Math/PtEtaPhiM4D.h>

using LorentzVectorXYZ = ROOT::Math::LorentzVector<ROOT::Math::PxPyPzE4D<double>>;
using LorentzVectorM = ROOT::Math::LorentzVector<ROOT::Math::PtEtaPhiM4D<double>>;
using Point3D = ROOT::Math::PositionVector3D<ROOT::Math::Cartesian3D<double>>;
using Vector3D = ROOT::Math::DisplacementVector3D<ROOT::Math::Cartesian3D<double>>;

namespace tau_cp {

class CPfunctions {
 public:

  static float PhiCP(const reco_tau::gen_truth::GenLepton& genLepton, ULong64_t evt) {

    static reco_tau::gen_truth::GenLepton prev_genLepton;
    static ULong64_t prev_evt = 0;
    //
    float acop;
    if (evt == prev_evt) {
      acop = PhiCP(prev_genLepton, genLepton);
    } else {
      acop = -99;
    }
    prev_genLepton = genLepton;
    prev_evt = evt;
    return acop; 
  } 

  static float PhiCP(const reco_tau::gen_truth::GenLepton& genLepton1, const reco_tau::gen_truth::GenLepton& genLepton2) {

    float PhiCP;
    //
    const std::set<const reco_tau::gen_truth::GenParticle*> genHadDaughters1 = genLepton1.hadrons();
    const std::set<const reco_tau::gen_truth::GenParticle*> genHadDaughters2 = genLepton2.hadrons();
    //
    std::tuple<TLorentzVector,std::vector<TLorentzVector>,std::vector<int>> TauandProd1 = GetTauandProdTLV(genLepton1, genHadDaughters1);
    std::tuple<TLorentzVector,std::vector<TLorentzVector>,std::vector<int>> TauandProd2 = GetTauandProdTLV(genLepton2, genHadDaughters2);
    //
    SCalculator Scalc;
    PhiCP = Scalc.AcopAngle(TauandProd1, TauandProd2);
    //
    return PhiCP;
  }

 private:

  static std::tuple<TLorentzVector,std::vector<TLorentzVector>,std::vector<int>> GetTauandProdTLV(const reco_tau::gen_truth::GenLepton& genLepton, const std::set<const reco_tau::gen_truth::GenParticle*>& genHadDaughters) {

    std::tuple<TLorentzVector,std::vector<TLorentzVector>,std::vector<int>> tauandprod;
    //
    LorentzVectorM TauP4 = genLepton.lastCopy().p4;
    std::get<0>(tauandprod) = TLorentzVector(TauP4.px(), TauP4.py(), TauP4.pz(), TauP4.energy());
    //
    for(const auto& genHadDau : genHadDaughters) {
      std::get<1>(tauandprod).push_back(TLorentzVector(genHadDau->p4.px(), genHadDau->p4.py(), genHadDau->p4.pz(), genHadDau->p4.energy()));
      std::get<2>(tauandprod).push_back(genHadDau->charge);
    }
    return tauandprod;
  }
};

} // namespace tau_cp
