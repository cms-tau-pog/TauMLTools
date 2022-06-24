/*Interface for CP observable computation at generated level, inheriting from SCalculator class
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

using namespace reco_tau::gen_truth;

class CPfunctions : public SCalculator {
 public:

  static float PhiCP(const GenLepton& genLepton, ULong64_t evt) {

    static GenLepton prev_genLepton;
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

  static float PhiCP(const GenLepton& genLepton1, const GenLepton& genLepton2) {

    float PhiCP;
    //
    const std::set<const GenParticle*> genHadDaughters1 = genLepton1.hadrons();
    const std::set<const GenParticle*> genHadDaughters2 = genLepton2.hadrons();
    //
    std::tuple<TLorentzVector,std::vector<TLorentzVector>,std::vector<int>> TauandProd1 = GetTauandProdTLV(genLepton1, genHadDaughters1);
    std::tuple<TLorentzVector,std::vector<TLorentzVector>,std::vector<int>> TauandProd2 = GetTauandProdTLV(genLepton2, genHadDaughters2);
    //
    PhiCP = AcopAngle(TauandProd1, TauandProd2);
    //
    return PhiCP;
  }

 private:

  static std::tuple<TLorentzVector,std::vector<TLorentzVector>,std::vector<int>> GetTauandProdTLV(const GenLepton& genLepton, const std::set<const GenParticle*>& genHadDaughters) {

    std::tuple<TLorentzVector,std::vector<TLorentzVector>,std::vector<int>> tauandprod;
    const std::set<const GenParticle*> genDaughters = genLepton.finalStateFromDecay();
    //
    LorentzVectorXYZ TauP4(0.,0.,0.,0.);
    for(std::set<const GenParticle*>::const_iterator igenDau=genDaughters.begin(); igenDau!=genDaughters.end(); igenDau++) {
      TauP4 += (*igenDau)->p4;
    }
    std::get<0>(tauandprod) = TLorentzVector(TauP4.px(), TauP4.py(), TauP4.pz(), TauP4.energy());
    //
    for(std::set<const GenParticle*>::const_iterator igenDau=genHadDaughters.begin(); igenDau!=genHadDaughters.end(); igenDau++) {
      std::get<1>(tauandprod).push_back(TLorentzVector((*igenDau)->p4.px(), (*igenDau)->p4.py(), (*igenDau)->p4.pz(), (*igenDau)->p4.energy()));
      std::get<2>(tauandprod).push_back((*igenDau)->charge);
    }
    return tauandprod;
  }
};


