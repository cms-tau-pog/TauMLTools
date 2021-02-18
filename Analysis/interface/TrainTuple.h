/*! Definition of a tuple with all event information that is required for the tau analysis.
*/

#pragma once

#include "TauMLTools/Core/interface/SmartTree.h"
#include "TauMLTools/Analysis/interface/TauIdResults.h"
#include "DataFormats/GeometryVector/interface/GlobalPoint.h"
#include <Math/VectorUtil.h>

#define TAU_ID(name, pattern, has_raw, wp_list) VAR(std::vector<uint16_t>, name) VAR(std::vector<Float_t>, name##raw)
#define TAU_VAR(type, name) VAR(std::vector<type>, tau_##name)
#define CALO_TOWER_VAR(type, name) VAR(std::vector<type>, caloTower_##name)
#define CALO_TAU_VAR(type, name) VAR(std::vector<type>, caloTau_##name)
#define TRACK_VAR(type, name) VAR(std::vector<type>, track_##name) VAR(std::vector<type>, patatrack_##name)
#define VERT(type, name) VAR(std::vector<type>, vert_##name) VAR(std::vector<type>, patavert_##name)

#define VAR2(type, name1, name2) VAR(type, name1) VAR(type, name2)
#define VAR3(type, name1, name2, name3) VAR2(type, name1, name2) VAR(type, name3)
#define VAR4(type, name1, name2, name3, name4) VAR3(type, name1, name2, name3) VAR(type, name4)

#define TAU_DATA() \
    /* Event Variables */ \
    VAR(UInt_t, run) /* run number */ \
    VAR(UInt_t, lumi) /* lumi section */ \
    VAR(ULong64_t, evt) /* event number */ \
    VAR(Int_t, npv) /* number of primary vertices */ \
    VAR(Int_t, nppv) /* number of primary patavertices */ \
    VAR(Float_t, genEventWeight) /* gen event weight */ \
    /*VAR(Float_t, trainingWeight)  training weight */ \
    VAR(Int_t, sampleType) /* type of the sample (MC, Embedded or Data) */ \
    VAR(Float_t, npu) /* number of in-time pu interactions added to the event */ \
    /* L1 objects */ \
    VAR(std::vector<int>, l1Tau_pt) /* L1 pt candidate*/ \
    VAR(std::vector<int>, l1Tau_eta) /* L1 eta candidate*/ \
    VAR(std::vector<int>, l1Tau_phi) /* L1 phi candidate*/ \
    VAR(std::vector<int>, l1Tau_mass) /* L1 mass candidate*/ \
    VAR(std::vector<int>, l1Tau_hwIso) /* L1 hwIso candidate*/ \
    VAR(std::vector<int>, l1Tau_hwQual) /* L1 quality candidate*/ \
    VAR(std::vector<int>, l1Tau_towerIEta) /* L1 towerIEta candidate*/ \
    VAR(std::vector<int>, l1Tau_towerIPhi) /* L1 towerIPhi candidate*/ \
    VAR(std::vector<int>, l1Tau_rawEt) /* L1 rawEt candidate*/ \
    VAR(std::vector<int>, l1Tau_isoEt) /* L1 isoEt candidate*/ \
    VAR(std::vector<bool>, l1Tau_hasEM) /* L1 hasEM candidate*/ \
    VAR(std::vector<bool>, l1Tau_isMerged) /* L1 isMerged candidate*/ \
    /* caloTower candidates */ \
    CALO_TOWER_VAR(Float_t, pt) /* caloTower pt candidate*/ \
    CALO_TOWER_VAR(Float_t, eta) /* caloTower eta candidate*/ \
    CALO_TOWER_VAR(Float_t, phi) /* caloTower phi candidate*/ \
    CALO_TOWER_VAR(Float_t, energy) /* caloTower energy candidate*/ \
    CALO_TOWER_VAR(Float_t, emEnergy) /* caloTower emEnergy candidate*/ \
    CALO_TOWER_VAR(Float_t, hadEnergy) /* caloTower hadEnergy candidate*/ \
    CALO_TOWER_VAR(Float_t, outerEnergy) /* caloTower outerEnergy candidate*/ \
    CALO_TOWER_VAR(Float_t, emPosition_x) /* caloTower emPosition candidate*/ \
    CALO_TOWER_VAR(Float_t, emPosition_y) /* caloTower emPosition candidate*/ \
    CALO_TOWER_VAR(Float_t, emPosition_z) /* caloTower emPosition candidate*/ \
    CALO_TOWER_VAR(Float_t, hadPosition_x) /* caloTower hadPosition candidate*/ \
    CALO_TOWER_VAR(Float_t, hadPosition_y) /* caloTower hadPosition candidate*/ \
    CALO_TOWER_VAR(Float_t, hadPosition_z) /* caloTower hadPosition candidate*/ \
    CALO_TOWER_VAR(Float_t, hadEnergyHeOuterLayer) /* caloTower hadEnergyHeOuterLayer candidate*/ \
    CALO_TOWER_VAR(Float_t, hadEnergyHeInnerLayer) /* caloTower hadEnergyHeInnerLayer candidate*/ \
    CALO_TOWER_VAR(Float_t, energyInHB) /* caloTower energyInHB candidate*/ \
    CALO_TOWER_VAR(Float_t, energyInHE) /* caloTower energyInHE candidate*/ \
    CALO_TOWER_VAR(Float_t, energyInHF) /* caloTower energyInHF candidate*/ \
    CALO_TOWER_VAR(Float_t, energyInHO) /* caloTower energyInHO candidate*/ \
    CALO_TOWER_VAR(Int_t, numBadEcalCells) /* caloTower numBadEcalCells candidate*/ \
    CALO_TOWER_VAR(Int_t, numRecoveredEcalCells) /* caloTower numRecoveredEcalCells candidate*/ \
    CALO_TOWER_VAR(Int_t, numProblematicEcalCells) /* caloTower numProblematicEcalCells candidate*/ \
    CALO_TOWER_VAR(Int_t, numBadHcalCells) /* caloTower numBadHcalCells candidate*/ \
    CALO_TOWER_VAR(Int_t, numRecoveredHcalCells) /* caloTower numRecoveredHcalCells candidate*/ \
    CALO_TOWER_VAR(Int_t, numProblematicHcalCells) /* caloTower numProblematicHcalCells candidate*/ \
    CALO_TOWER_VAR(Float_t, ecalTime) /* caloTower ecalTime candidate*/ \
    CALO_TOWER_VAR(Float_t, hcalTime) /* caloTower hcalTime candidate*/ \
    CALO_TOWER_VAR(Float_t, hottestCellE) /* hottest caloTower cell Energy */ \
    CALO_TOWER_VAR(Int_t, emLvl1) /* em energy at level 1 */ \
    CALO_TOWER_VAR(Int_t, hadLv11) /* had energy at level 1 */ \
    CALO_TOWER_VAR(Int_t, numCrystals) /* number of (fired) crystals */ \
    /* Tracks candidates */ \
    TRACK_VAR(Float_t, pt) /* track pt candidate*/ \
    TRACK_VAR(Float_t, eta) /* track eta candidate*/ \
    TRACK_VAR(Float_t, phi) /* track phi candidate*/ \
    TRACK_VAR(Float_t, chi2) /* track chi2 candidate*/ \
    TRACK_VAR(Int_t, ndof) /* track ndof candidate*/ \
    TRACK_VAR(Int_t, charge) /* pixelTrack charge candidate*/ \
    TRACK_VAR(UInt_t, quality) /* pixelTrack qualityMask candidate*/ \
    TRACK_VAR(Float_t, dxy) /* track dxy candidate*/ \
    TRACK_VAR(Float_t, dz) /* track dz candidate*/ \
    TRACK_VAR(Int_t, vertex_id) /* track dz candidate*/ \
    /* VERTICES */ \
    VERT(Float_t, z) /* x positions of vertices */ \
    VERT(Float_t, weight) /* output weight (1/error^2) on the above */ \
    VERT(Float_t, ptv2) /* vertices pt^2 */ \
    VERT(Float_t, chi2) /* chi^2 of the vertices (PV) */ \
    VERT(Int_t, ndof) /* number of degrees of freedom of vertices (PV) */ \
    /* CaloTaus candidates */ \
    CALO_TAU_VAR(Float_t, pt) /* caloTau pt candidate corresp to p4->at(pt)*/ \
    CALO_TAU_VAR(Float_t, eta) /* caloTau eta candidate corresp to p4->at(eta)*/ \
    CALO_TAU_VAR(Float_t, phi) /* caloTau phi candidate corresp to p4->at(phi)*/ \
    CALO_TAU_VAR(Float_t, energy) /* caloTau energy candidate corresp to p4->at(E)*/ \
    CALO_TAU_VAR(Float_t, maxEInEmTowers) /** Returns the maximum energy deposited in ECAL towers*/ \
    CALO_TAU_VAR(Float_t, maxEInHadTowers) /** Returns the maximum energy deposited in HCAL towers*/ \
    CALO_TAU_VAR(Float_t, energyFractionHadronic) /** Returns the jet hadronic energy fraction*/ \
    CALO_TAU_VAR(Float_t, emEnergyFraction) /** Returns the jet electromagnetic energy fraction*/ \
    CALO_TAU_VAR(Float_t, hadEnergyInHB) /** Returns the jet hadronic energy in HB*/ \
    CALO_TAU_VAR(Float_t, hadEnergyInHO) /** Returns the jet hadronic energy in HO*/ \
    CALO_TAU_VAR(Float_t, hadEnergyInHE) /** Returns the jet hadronic energy in HE*/ \
    CALO_TAU_VAR(Float_t, hadEnergyInHF) /** Returns the jet hadronic energy in HF*/ \
    CALO_TAU_VAR(Float_t, emEnergyInEB) /** Returns the jet electromagnetic energy in EB*/ \
    CALO_TAU_VAR(Float_t, emEnergyInEE) /** Returns the jet electromagnetic energy in EE*/ \
    CALO_TAU_VAR(Float_t, emEnergyInHF) /** Returns the jet electromagnetic energy extracted from HF*/ \
    CALO_TAU_VAR(Float_t, towersArea) /** Returns area of contributing towers */ \
    CALO_TAU_VAR(Int_t, n90) /* caloTau number of constituents carrying a 90% of the total Jet energy*/ \
    CALO_TAU_VAR(Int_t, n60) /* caloTau number of constituents carrying a 60% of the total Jet energy*/ \
    VAR(Bool_t, VeryBigOR_result)/**/ \
    VAR(Bool_t, hltDoubleL2Tau26eta2p2_result)/**/ \
    VAR(Bool_t, hltDoubleL2IsoTau26eta2p2_result)/**/ \

#define VAR(type, name) DECLARE_BRANCH_VARIABLE(type, name)
DECLARE_TREE(train_tuple, Tau, TrainTuple, TAU_DATA, "taus")
#undef VAR

#define VAR(type, name) ADD_DATA_TREE_BRANCH(name)
INITIALIZE_TREE(train_tuple, TrainTuple, TAU_DATA)
#undef VAR
#undef VAR2
#undef VAR3
#undef VAR4
#undef TAU_ID
#undef TAU_VAR
#undef CALO_TOWER_VAR
#undef CALO_TAU_VAR
#undef TRACK_VAR
#undef VERT

namespace train_tuple {

template<typename T>
constexpr T DefaultFillValue() { return std::numeric_limits<T>::lowest(); }
template<>
constexpr float DefaultFillValue<float>() { return -999.; }
template<>
constexpr int DefaultFillValue<int>() { return -999; }
template<>
constexpr unsigned DefaultFillValue<unsigned>() { return 0; }

} // namespace train_tuple
