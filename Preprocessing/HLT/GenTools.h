#pragma once

#include "AnalysisTools.h"
#include "GenLepton.h"

using Displ3D = reco_tau::gen_truth::Displacement3D;
using RVecDispl3D = ROOT::VecOps::RVec<reco_tau::gen_truth::Displacement3D>;
using GenLepton = reco_tau::gen_truth::GenLepton;
using RVecGenLepton = ROOT::VecOps::RVec<reco_tau::gen_truth::GenLepton>;

inline bool IsLepton(TauType tau_type)
{
  static const std::set<TauType> lepton_types = {
    TauType::e, TauType::mu, TauType::tau, TauType::emb_e, TauType::emb_mu, TauType::emb_tau,
    TauType::displaced_e, TauType::displaced_mu, TauType::displaced_tau
  };
  return lepton_types.count(tau_type);
}

inline RVecLV GetGenP4(const RVecI& tauType, const RVecI& genLepUniqueIdx, const RVecI& genJetUniqueIdx,
                       const RVecGenLepton& genLeptons, const RVecLV& genJet_p4)
{
  RVecLV gen_p4(tauType.size());
  for(size_t tau_idx = 0; tau_idx < tauType.size(); ++tau_idx) {
    const auto tau_type = static_cast<TauType>(tauType[tau_idx]);
    if(IsLepton(tau_type)) {
      const int genLepton_idx = genLepUniqueIdx[tau_idx];
      gen_p4[tau_idx] = genLeptons.at(genLepton_idx).visibleP4();
    } else if(tau_type == TauType::jet) {
      const int genJet_idx = genJetUniqueIdx[tau_idx];
      gen_p4[tau_idx] = genJet_p4.at(genJet_idx);
    }
  }
  return gen_p4;
}

inline RVecI GetGenCharge(const RVecI& tauType, const RVecI& genLepUniqueIdx, const RVecGenLepton& genLeptons)
{
  RVecI gen_charge(tauType.size(), 0);
  for(size_t tau_idx = 0; tau_idx < tauType.size(); ++tau_idx) {
    const auto tau_type = static_cast<TauType>(tauType[tau_idx]);
    if(IsLepton(tau_type)) {
      const int genLepton_idx = genLepUniqueIdx[tau_idx];
      gen_charge[tau_idx] = genLeptons.at(genLepton_idx).charge();
    }
  }
  return gen_charge;
}

inline RVecI GetGenPartonFlavour(const RVecI& tauType, const RVecI& genJetUniqueIdx,
                                 const ROOT::VecOps::RVec<short>& GenJet_partonFlavour)
{
  RVecI gen_pf(tauType.size(), -999);
  for(size_t tau_idx = 0; tau_idx < tauType.size(); ++tau_idx) {
    const auto tau_type = static_cast<TauType>(tauType[tau_idx]);
    if(tau_type == TauType::jet) {
      const int genJet_idx = genJetUniqueIdx[tau_idx];
      gen_pf[tau_idx] = GenJet_partonFlavour.at(genJet_idx);
    }
  }
  return gen_pf;
}

inline RVecDispl3D GetGenFlightLength(const RVecI& tauType, const RVecI& genLepUniqueIdx,
                                      const RVecGenLepton& genLeptons)
{
  RVecDispl3D fl(tauType.size(), Displ3D(0, 0, 0));
  for(size_t tau_idx = 0; tau_idx < tauType.size(); ++tau_idx) {
    const auto tau_type = static_cast<TauType>(tauType[tau_idx]);
    if(IsLepton(tau_type)) {
      const int genLepton_idx = genLepUniqueIdx[tau_idx];
      fl[tau_idx] = genLeptons.at(genLepton_idx).flightLength();
    }
  }
  return fl;
}

inline TauType RefineTauTypeDefinition(TauType tau_type, bool isEmbedded, bool isDisplaced)
{
  if(tau_type == TauType::e) {
    if(isEmbedded) return TauType::emb_e;
    if(isDisplaced) return TauType::displaced_e;
  }
  if(tau_type == TauType::mu) {
    if(isEmbedded) return TauType::emb_mu;
    if(isDisplaced) return TauType::displaced_mu;
  }
  if(tau_type == TauType::tau) {
    if(isEmbedded) return TauType::emb_tau;
    if(isDisplaced) return TauType::displaced_tau;
  }
  return tau_type;
}

inline RVecI GetTauTypes(const RVecGenLepton& genLeptons, const RVecI& genLepUniqueIdx, const RVecSetInt& genLepIndices,
                         const RVecI& genJetUniqueIdx, bool isEmbedded, bool detectDisplaced,
                         double minFlightLenght_rho = 1., double maxFlightLenght_rho = 100.,
                         double minFlightLenght_z = 0., double maxFlightLenght_z = 20.)
{
  using namespace reco_tau::gen_truth;
  RVecI tau_types(genLepUniqueIdx.size(), static_cast<int>(TauType::other));
  for(size_t obj_idx = 0; obj_idx < tau_types.size(); ++obj_idx) {
    if(genLepIndices[obj_idx].empty()) {
      if(genJetUniqueIdx[obj_idx] >= 0)
        tau_types[obj_idx] = static_cast<int>(isEmbedded ? TauType::emb_jet : TauType::jet);
    } else {
      if(genLepUniqueIdx[obj_idx] >= 0 && genLepIndices[obj_idx].size() == 1) {
        const int genLepton_idx = genLepUniqueIdx[obj_idx];
        const auto& genLepton = genLeptons[genLepton_idx];
        bool isDisplaced = false;
        if(detectDisplaced) {
          const double flightLength_rho = genLepton.flightLength().rho();
          const double flightLength_z = std::abs(genLepton.flightLength().z());
          isDisplaced = flightLength_rho >= minFlightLenght_rho && flightLength_rho <= maxFlightLenght_rho
                        && flightLength_z >= minFlightLenght_z && flightLength_z <= maxFlightLenght_z;
        }
        TauType detectedType = TauType::other;
        if(genLepton.kind() == GenLepton::Kind::TauDecayedToHadrons && genLepton.visibleP4().pt() > 15.)
          detectedType = TauType::tau;
        else if((genLepton.kind() == GenLepton::Kind::PromptElectron
            || genLepton.kind() == GenLepton::Kind::TauDecayedToElectron) && genLepton.visibleP4().pt() > 8.)
          detectedType = TauType::e;
        else if((genLepton.kind() == GenLepton::Kind::PromptMuon
            || genLepton.kind() == GenLepton::Kind::TauDecayedToMuon) && genLepton.visibleP4().pt() > 8.)
          detectedType = TauType::mu;
        detectedType = RefineTauTypeDefinition(detectedType, isEmbedded, isDisplaced);
        tau_types[obj_idx] = static_cast<int>(detectedType);
      }
    }
  }
  return tau_types;
}
