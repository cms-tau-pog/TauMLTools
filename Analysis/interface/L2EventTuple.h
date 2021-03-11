/*! Definition of a tuple with all the l2 level information. */

#pragma once

#include "TauMLTools/Core/interface/SmartTree.h"
#include <Math/VectorUtil.h>

#define CALO_TOWER_VAR(type, name) VAR(std::vector<type>, caloTower_##name)
#define CALO_TAU_VAR(type, name) VAR(std::vector<type>, caloTau_##name)
#define HBHE_VAR(type, name) VAR(std::vector<type>, caloRecHit_hbhe_##name)
#define HO_VAR(type, name) VAR(std::vector<type>, caloRecHit_ho_##name)
#define HF_VAR(type, name) VAR(std::vector<type>, caloRecHit_hf_##name)
#define ECAL_VAR(type, name) VAR(std::vector<type>, caloRecHit_ee_##name) VAR(std::vector<type>, caloRecHit_eb_##name)
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
    VAR(Int_t, defaultDiTauPath_lastModuleIndex) /*  */ \
    VAR(bool, defaultDiTauPath_result) /*  */ \
    VAR(Float_t, genEventWeight) /* gen event weight */ \
    VAR(Int_t, sampleType) /* type of the sample (MC, Embedded or Data) */ \
    VAR(Float_t, npu) /* number of in-time pu interactions added to the event */ \
    /* Gen lepton with the full decay chain */ \
    VAR(std::vector<Int_t>, genLepton_nParticles) /* index of the gen lepton */ \
    VAR(std::vector<Int_t>, genLepton_kind) /* kind of the gen lepton:
                                              Electron = 1, Muon = 2, TauElectron = 3, TauMuon = 4, Tau = 5, Other = 6 */\
    VAR(std::vector<Int_t>, genLepton_charge) /* charge of the gen lepton */ \
    VAR4(std::vector<Float_t>, genLepton_vis_pt, genLepton_vis_eta, genLepton_vis_phi, genLepton_vis_mass) /* visible 4-momentum of
                                                                                                 the gen lepton */ \
    VAR(std::vector<Int_t>, genLepton_lastMotherIndex) /* index of the last mother in genParticle_* vectors:
                                             >= 0 if at least one mother is available, -1 otherwise */ \
    VAR(std::vector<Int_t>, genParticle_pdgId) /* PDG ID */ \
    VAR(std::vector<Long64_t>, genParticle_mother) /* index of the mother */ \
    VAR(std::vector<Int_t>, genParticle_charge) /* charge */ \
    VAR2(std::vector<Int_t>, genParticle_isFirstCopy, genParticle_isLastCopy) /* indicator whatever a gen particle
                                                                                 is the first or the last copy */ \
    VAR4(std::vector<Float_t>, genParticle_pt, genParticle_eta, \
                               genParticle_phi, genParticle_mass) /* 4-momenta */ \
    VAR3(std::vector<Float_t>, genParticle_vtx_x, genParticle_vtx_y, genParticle_vtx_z) /* position of the vertex */ \
    /* L1 Trigger results for each L1 tau */ \
    VAR(std::vector<bool>, L1_LooseIsoEG22er2p1_IsoTau26er2p1_dR_Min0p3) \
    VAR(std::vector<bool>, L1_LooseIsoEG24er2p1_IsoTau27er2p1_dR_Min0p3) \
    VAR(std::vector<bool>, L1_LooseIsoEG22er2p1_Tau70er2p1_dR_Min0p3) \
    VAR(std::vector<bool>, L1_SingleTau120er2p1) \
    VAR(std::vector<bool>, L1_SingleTau130er2p1) \
    VAR(std::vector<bool>, L1_DoubleTau70er2p1) \
    VAR(std::vector<bool>, L1_DoubleIsoTau28er2p1) \
    VAR(std::vector<bool>, L1_DoubleIsoTau30er2p1) \
    VAR(std::vector<bool>, L1_DoubleIsoTau32er2p1) \
    VAR(std::vector<bool>, L1_DoubleIsoTau34er2p1) \
    VAR(std::vector<bool>, L1_DoubleIsoTau36er2p1) \
    VAR(std::vector<bool>, L1_DoubleIsoTau28er2p1_Mass_Max90) \
    VAR(std::vector<bool>, L1_DoubleIsoTau28er2p1_Mass_Max80) \
    VAR(std::vector<bool>, L1_DoubleIsoTau30er2p1_Mass_Max90) \
    VAR(std::vector<bool>, L1_DoubleIsoTau30er2p1_Mass_Max80) \
    VAR(std::vector<bool>, L1_Mu18er2p1_Tau24er2p1) \
    VAR(std::vector<bool>, L1_Mu18er2p1_Tau26er2p1) \
    VAR(std::vector<bool>, L1_Mu22er2p1_IsoTau28er2p1) \
    VAR(std::vector<bool>, L1_Mu22er2p1_IsoTau30er2p1) \
    VAR(std::vector<bool>, L1_Mu22er2p1_IsoTau32er2p1) \
    VAR(std::vector<bool>, L1_Mu22er2p1_IsoTau34er2p1) \
    VAR(std::vector<bool>, L1_Mu22er2p1_IsoTau36er2p1) \
    VAR(std::vector<bool>, L1_Mu22er2p1_IsoTau40er2p1) \
    VAR(std::vector<bool>, L1_Mu22er2p1_Tau70er2p1) \
    VAR(std::vector<bool>, L1_IsoTau40er2p1_ETMHF80) \
    VAR(std::vector<bool>, L1_IsoTau40er2p1_ETMHF90) \
    VAR(std::vector<bool>, L1_IsoTau40er2p1_ETMHF100) \
    VAR(std::vector<bool>, L1_IsoTau40er2p1_ETMHF110) \
    VAR(std::vector<bool>, L1_QuadJet36er2p5_IsoTau52er2p1) \
    VAR(std::vector<bool>, L1_DoubleJet35_Mass_Min450_IsoTau45_RmOvlp) \
    VAR(std::vector<bool>, L1_DoubleJet_80_30_Mass_Min420_IsoTau40_RmOvlp) \
    /* L1 objects */ \
    VAR(std::vector<Float_t>, l1Tau_pt) /* L1 pt candidate*/ \
    VAR(std::vector<Float_t>, l1Tau_eta) /* L1 eta candidate*/ \
    VAR(std::vector<Float_t>, l1Tau_phi) /* L1 phi candidate*/ \
    VAR(std::vector<Float_t>, l1Tau_mass) /* L1 mass candidate*/ \
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
    CALO_TOWER_VAR(Int_t, emLvl1) /* caloTower em energy at level 1 */ \
    CALO_TOWER_VAR(Int_t, hadLv11) /* caloTower had energy at level 1 */ \
    CALO_TOWER_VAR(Int_t, numCrystals) /* caloTower number of (fired) crystals */ \
    /* CaloRecHits candidates */ \
    ECAL_VAR(Float_t, rho) /* */ \
    ECAL_VAR(Float_t, eta) /* */ \
    ECAL_VAR(Float_t, phi) /* */ \
    ECAL_VAR(Float_t, energy) /* */ \
    ECAL_VAR(Float_t, time) /* */ \
    ECAL_VAR(ULong64_t, detId) /* */ \
    ECAL_VAR(Float_t, chi2) /* */ \
    ECAL_VAR(Float_t, energyError) /* */ \
    ECAL_VAR(Float_t, timeError) /* */ \
    ECAL_VAR(uint32_t, flagsBits) /* */ \
    ECAL_VAR(Bool_t, isRecovered) /* */ \
    ECAL_VAR(Bool_t, isTimeValid) /* */ \
    ECAL_VAR(Bool_t, isTimeErrorValid) /* */ \
    HBHE_VAR(Float_t, rho) /* */ \
    HBHE_VAR(Float_t, eta) /* */ \
    HBHE_VAR(Float_t, phi) /* */ \
    HBHE_VAR(Float_t, energy) /* */ \
    HBHE_VAR(Float_t, time) /* */ \
    HBHE_VAR(ULong64_t, detId) /* */ \
    HBHE_VAR(Float_t, chi2) /* */ \
    HBHE_VAR(ULong64_t, flags) /* */ \
    HBHE_VAR(Float_t, eraw) /* */ \
    HBHE_VAR(Float_t, eaux) /* */ \
    HBHE_VAR(Float_t, timeFalling) /* */ \
    HBHE_VAR(ULong64_t, idFront) /* */ \
    HBHE_VAR(Float_t, rho_front) /* */ \
    HBHE_VAR(Float_t, eta_front) /* */ \
    HBHE_VAR(Float_t, phi_front) /* */ \
    HBHE_VAR(UInt_t, auxHBHE) /* */ \
    HBHE_VAR(UInt_t, auxPhase1) /* */ \
    HBHE_VAR(UInt_t, auxTDC) /* */ \
    HBHE_VAR(Bool_t, isMerged) /* */ \
    HO_VAR(Float_t, rho) /* */ \
    HO_VAR(Float_t, eta) /* */ \
    HO_VAR(Float_t, phi) /* */ \
    HO_VAR(Float_t, energy) /* */ \
    HO_VAR(Float_t, time) /* */ \
    HO_VAR(ULong64_t, detId) /* */ \
    HO_VAR(ULong64_t, aux) /* */ \
    HO_VAR(ULong64_t, flags) /* */ \
    HF_VAR(Float_t, rho) /* */ \
    HF_VAR(Float_t, eta) /* */ \
    HF_VAR(Float_t, phi) /* */ \
    HF_VAR(Float_t, energy) /* */ \
    HF_VAR(Float_t, time) /* */ \
    HF_VAR(ULong64_t, detId) /* */ \
    HF_VAR(ULong64_t, flags) /* */ \
    HF_VAR(Float_t, timeFalling) /* */ \
    HF_VAR(uint32_t, auxHF) /* */ \
    HF_VAR(ULong64_t, aux) /* */ \
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
    TRACK_VAR(Int_t, vertex_id) /* track associated vertex id candidate*/ \
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


#define SUMMARY_L2_DATA() \
    /* Run statistics */ \
    VAR(UInt_t, exeTime) \
    VAR(ULong64_t, numberOfProcessedEvents) \
    VAR(std::vector<std::string>, module_names) \
    /**/

#define VAR(type, name) DECLARE_BRANCH_VARIABLE(type, name)
DECLARE_TREE(train_tuple, L2Summary, L2SummaryTuple, SUMMARY_L2_DATA, "L2Summary")
#undef VAR

#define VAR(type, name) ADD_DATA_TREE_BRANCH(name)
INITIALIZE_TREE(train_tuple, L2SummaryTuple, SUMMARY_L2_DATA)
#undef VAR
#undef SUMMARY_L2_DATA
