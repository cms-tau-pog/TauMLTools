/*! Definition of a tuple with all event information that is required for the tau analysis.
*/

#pragma once

#include "AnalysisTools/Core/include/SmartTree.h"
#include "TauML/Analysis/include/TauIdResults.h"
#include <Math/VectorUtil.h>

#define TAU_ID(name, pattern, has_raw, wp_list) VAR(uint16_t, name) VAR(Float_t, name##raw)
#define CAND_VAR(type, name) VAR(std::vector<type>, pfCand_##name)

#define VAR2(type, name1, name2) VAR(type, name1) VAR(type, name2)
#define VAR3(type, name1, name2, name3) VAR2(type, name1, name2) VAR(type, name3)
#define VAR4(type, name1, name2, name3, name4) VAR3(type, name1, name2, name3) VAR(type, name4)

#define TAU_DATA() \
    /* Event Variables */ \
    VAR(UInt_t, run) /* run number */ \
    VAR(UInt_t, lumi) /* lumi section */ \
    VAR(ULong64_t, evt) /* event number */ \
    VAR(UInt_t, npv) /* number of primary vertices */ \
    VAR(Float_t, rho) /* fixed grid energy density */ \
    VAR(Float_t, genEventWeight) /* gen event weight */ \
    VAR(Float_t, npu) /* number of in-time pu interactions added to the event */ \
    VAR3(Float_t, pv_x, pv_y, pv_z) /* position of the primary vertex (PV) */ \
    VAR(Float_t, pv_chi2) /* chi^2 of the primary vertex (PV) */ \
    VAR(Float_t, pv_ndof) /* Number of degrees of freedom of the primary vertex (PV) */ \
    /* Jet variables */ \
    VAR(Int_t, jet_index) /* index of the jet */ \
    VAR4(Float_t, jet_pt, jet_eta, jet_phi, jet_mass) /* 4-momentum of the jet */ \
    VAR(Float_t, jet_neutralHadronEnergyFraction) /* jet neutral hadron energy fraction
                                                     (relative to uncorrected jet energy) */ \
    VAR(Float_t, jet_neutralEmEnergyFraction) /* jet neutral EM energy fraction
                                                 (relative to uncorrected jet energy) */ \
    VAR(Int_t, jet_nConstituents) /* number of jet constituents */ \
    VAR(Int_t, jet_chargedMultiplicity) /* jet charged multiplicity */ \
    VAR(Int_t, jet_neutralMultiplicity) /* jet neutral multiplicity */ \
    VAR(Int_t, jet_partonFlavour) /* parton-based flavour of the jet */ \
    VAR(Int_t, jet_hadronFlavour) /* hadron-based flavour of the jet */ \
    VAR(Int_t, jet_has_gen_match) /* jet has a matched gen-jet */ \
    VAR4(Float_t, jet_gen_pt, jet_gen_eta, jet_gen_phi, jet_gen_mass) /* 4-momentum of the gen jet */ \
    VAR(Int_t, jet_gen_n_b) /* number of b hadrons clustered inside the jet */ \
    VAR(Int_t, jet_gen_n_c) /* number of c hadrons clustered inside the jet */ \
    /* Basic tau variables */ \
    VAR(Int_t, jetTauMatch) /* match between jet and tau */ \
    VAR(Int_t, tau_index) /* index of the tau */ \
    VAR4(Float_t, tau_pt, tau_eta, tau_phi, tau_mass) /* 4-momentum of the tau */ \
    VAR(Int_t, tau_charge) /* tau charge */ \
    VAR(Int_t, lepton_gen_match) /* matching with leptons on the generator level, see Htautau Twiki for details */\
    VAR(Int_t, lepton_gen_charge) /* charge of the matched gen lepton */\
    VAR4(Float_t, lepton_gen_pt, lepton_gen_eta, \
                  lepton_gen_phi, lepton_gen_mass) /* 4-momentum of the matched gen lepton */ \
    VAR(std::vector<Int_t>, lepton_gen_vis_pdg) /* PDG of the matched lepton */\
    VAR4(std::vector<Float_t>, lepton_gen_vis_pt, lepton_gen_vis_eta, \
                               lepton_gen_vis_phi, lepton_gen_vis_mass) /* 4-momenta of the visible products
                                                                           of the matched gen lepton */ \
    VAR(Int_t, qcd_gen_match) /* matching with QCD particles on the generator level */\
    VAR(Int_t, qcd_gen_charge) /* charge of the matched gen QCD particle */\
    VAR4(Float_t, qcd_gen_pt, qcd_gen_eta, qcd_gen_phi, qcd_gen_mass) /* 4-momentum of the matched gen QCD particle */ \
    /* Tau ID variables */ \
    VAR(Int_t, tau_decayMode) /* tau decay mode */ \
    VAR(Int_t, tau_decayModeFinding) /* tau passed the old decay mode finding requirements */ \
    VAR(Int_t, tau_decayModeFindingNewDMs) /* tau passed the new decay mode finding requirements */ \
    VAR(Float_t, chargedIsoPtSum) /* sum of the transverse momentums of charged pf candidates inside
                                     the tau isolation cone with dR < 0.5 */ \
    VAR(Float_t, chargedIsoPtSumdR03) /* sum of the transverse momentums of charged pf candidates inside
                                         the tau isolation cone with dR < 0.3 */ \
    VAR(Float_t, footprintCorrection) /* tau footprint correction inside the tau isolation cone with dR < 0.5 */ \
    VAR(Float_t, footprintCorrectiondR03) /* tau footprint correction inside the tau isolation cone with dR < 0.3 */ \
    VAR(Float_t, neutralIsoPtSum) /* sum of the transverse momentums of neutral pf candidates inside
                                     the tau isolation cone with dR < 0.5 */ \
    VAR(Float_t, neutralIsoPtSumWeight) /* weighted sum of the transverse momentums of neutral pf candidates inside
                                           the tau isolation cone with dR < 0.5 */ \
    VAR(Float_t, neutralIsoPtSumWeightdR03) /* weighted sum of the transverse momentums of neutral pf candidates inside
                                               the tau isolation cone with dR < 0.3 */ \
    VAR(Float_t, neutralIsoPtSumdR03) /* sum of the transverse momentums of neutral pf candidates inside
                                         the tau isolation cone with dR < 0.3 */ \
    VAR(Float_t, photonPtSumOutsideSignalCone) /* sum of the transverse momentums of photons
                                                  inside the tau isolation cone with dR < 0.5 */ \
    VAR(Float_t, photonPtSumOutsideSignalConedR03) /* sum of the transverse momentums of photons inside
                                                      the tau isolation cone with dR < 0.3 */ \
    VAR(Float_t, puCorrPtSum) /* pile-up correction for the sum of the transverse momentums */ \
    TAU_IDS() \
    /* Tau transverse impact paramters.
       See cmssw/RecoTauTag/RecoTau/plugins/PFTauTransverseImpactParameters.cc for details */ \
    VAR3(Float_t, tau_dxy_pca_x, tau_dxy_pca_y, tau_dxy_pca_z) /* The point of closest approach (PCA) of
                                                                  the leadPFChargedHadrCand to the primary vertex */ \
    VAR(Float_t, tau_dxy) /* tau signed transverse impact parameter wrt to the primary vertex */ \
    VAR(Float_t, tau_dxy_sig) /* significance of the transverse impact parameter measurement */ \
    VAR(Float_t, tau_ip3d) /* tau signed 3D impact parameter wrt to the primary vertex */ \
    VAR(Float_t, tau_ip3d_sig) /* significance of the 3D impact parameter measurement */ \
    VAR(Float_t, tau_dz) /* tau dz of the leadChargedHadrCand wrt to the primary vertex */ \
    VAR(Int_t, tau_hasSecondaryVertex) /* tau has the secondary vertex */ \
    VAR3(Float_t, tau_sv_x, tau_sv_y, tau_sv_z) /* position of the secondary vertex */ \
    VAR3(Float_t, tau_flightLength_x, tau_flightLength_y, tau_flightLength_z) /* flight length of the tau */ \
    VAR(Float_t, tau_flightLength_sig) /* significance of the flight length measurement */ \
    /* Extended tau variables */ \
    VAR(Float_t, tau_pt_weighted_deta_strip) /* */ \
    VAR(Float_t, tau_pt_weighted_dphi_strip) /* */ \
    VAR(Float_t, tau_pt_weighted_dr_signal) /* */ \
    VAR(Float_t, tau_pt_weighted_dr_iso) /* */ \
    VAR(Float_t, tau_leadingTrackNormChi2) /* */ \
    VAR(Float_t, tau_e_ratio) /* */ \
    VAR(Float_t, tau_gj_angle_diff) /* */ \
    VAR(UInt_t, tau_n_photons) /* */ \
    VAR(Float_t, tau_emFraction) /* */ \
    VAR(Int_t, tau_inside_ecal_crack) /* */ \
    /* PF candidates */ \
    CAND_VAR(Int_t, jetDaughter) \
    CAND_VAR(Int_t, tauSignal) \
    CAND_VAR(Int_t, leadChargedHadrCand) \
    CAND_VAR(Int_t, tauIso) \
    CAND_VAR(Float_t, pt) \
    CAND_VAR(Float_t, eta) \
    CAND_VAR(Float_t, phi) \
    CAND_VAR(Float_t, mass) \
    /* Against electron */ \
    CAND_VAR(Float_t, etaAtEcalEntrance) /* */ \
    CAND_VAR(Float_t, emFraction) /* */ \
    CAND_VAR(Int_t, has_gsf_track) /* */ \
    CAND_VAR(Int_t, inside_ecal_crack) /* */ \
    CAND_VAR(Int_t, gsf_ele_matched) /* */ \
    CAND_VAR(Float_t, gsf_ele_pt) /* */ \
    CAND_VAR(Float_t, gsf_ele_dEta) /* */ \
    CAND_VAR(Float_t, gsf_ele_dPhi) /* */ \
    CAND_VAR(Float_t, gsf_ele_energy) /* */ \
    CAND_VAR(Float_t, gsf_ele_Ee) /* */ \
    CAND_VAR(Float_t, gsf_ele_Egamma) /* */ \
    CAND_VAR(Float_t, gsf_ele_Pin) /* */ \
    CAND_VAR(Float_t, gsf_ele_Pout) /* */ \
    CAND_VAR(Float_t, gsf_ele_Eecal) /* */ \
    CAND_VAR(Float_t, gsf_ele_dEta_SeedClusterTrackAtCalo) /* */ \
    CAND_VAR(Float_t, gsf_ele_dPhi_SeedClusterTrackAtCalo) /* */ \
    CAND_VAR(Float_t, gsf_ele_mvaIn_sigmaEtaEta) /* */ \
    CAND_VAR(Float_t, gsf_ele_mvaIn_hadEnergy) /* */ \
    CAND_VAR(Float_t, gsf_ele_mvaIn_deltaEta) /* */ \
    CAND_VAR(Float_t, gsf_ele_Chi2NormGSF) /* */ \
    CAND_VAR(Float_t, gsf_ele_GSFNumHits) /* */ \
    CAND_VAR(Float_t, gsf_ele_GSFTrackResol) /* */ \
    CAND_VAR(Float_t, gsf_ele_GSFTracklnPt) /* */ \
    CAND_VAR(Float_t, gsf_ele_Chi2NormKF) /* */ \
    CAND_VAR(Float_t, gsf_ele_KFNumHits) /* */ \
    /* Against muon */ \
    CAND_VAR(UInt_t, n_matched_muons) /* */ \
    CAND_VAR(Float_t, muon_pt) /* */ \
    CAND_VAR(Float_t, muon_dEta) /* */ \
    CAND_VAR(Float_t, muon_dPhi) /* */ \
    CAND_VAR(UInt_t, muon_n_matches_DT_1) /* */ \
    CAND_VAR(UInt_t, muon_n_matches_DT_2) /* */ \
    CAND_VAR(UInt_t, muon_n_matches_DT_3) /* */ \
    CAND_VAR(UInt_t, muon_n_matches_DT_4) /* */ \
    CAND_VAR(UInt_t, muon_n_matches_CSC_1) /* */ \
    CAND_VAR(UInt_t, muon_n_matches_CSC_2) /* */ \
    CAND_VAR(UInt_t, muon_n_matches_CSC_3) /* */ \
    CAND_VAR(UInt_t, muon_n_matches_CSC_4) /* */ \
    CAND_VAR(UInt_t, muon_n_matches_RPC_1) /* */ \
    CAND_VAR(UInt_t, muon_n_matches_RPC_2) /* */ \
    CAND_VAR(UInt_t, muon_n_matches_RPC_3) /* */ \
    CAND_VAR(UInt_t, muon_n_matches_RPC_4) /* */ \
    CAND_VAR(UInt_t, muon_n_hits_DT_1) /* */ \
    CAND_VAR(UInt_t, muon_n_hits_DT_2) /* */ \
    CAND_VAR(UInt_t, muon_n_hits_DT_3) /* */ \
    CAND_VAR(UInt_t, muon_n_hits_DT_4) /* */ \
    CAND_VAR(UInt_t, muon_n_hits_CSC_1) /* */ \
    CAND_VAR(UInt_t, muon_n_hits_CSC_2) /* */ \
    CAND_VAR(UInt_t, muon_n_hits_CSC_3) /* */ \
    CAND_VAR(UInt_t, muon_n_hits_CSC_4) /* */ \
    CAND_VAR(UInt_t, muon_n_hits_RPC_1) /* */ \
    CAND_VAR(UInt_t, muon_n_hits_RPC_2) /* */ \
    CAND_VAR(UInt_t, muon_n_hits_RPC_3) /* */ \
    CAND_VAR(UInt_t, muon_n_hits_RPC_4) /* */ \
    /**/

#define VAR(type, name) DECLARE_BRANCH_VARIABLE(type, name)
DECLARE_TREE(tau_tuple, Tau, TauTuple, TAU_DATA, "taus")
#undef VAR

#define VAR(type, name) ADD_DATA_TREE_BRANCH(name)
INITIALIZE_TREE(tau_tuple, TauTuple, TAU_DATA)
#undef VAR
#undef VAR2
#undef VAR3
#undef VAR4
#undef TAU_DATA
#undef CAND_VAR
#undef TAU_ID

namespace tau_tuple {

template<typename T>
constexpr T DefaultFillValue() { return std::numeric_limits<T>::lowest(); }
template<>
constexpr float DefaultFillValue<float>() { return -999.; }

enum class ComponenetType { Gamma = 0, ChargedHadronCandidate = 1, NeutralHadronCandidate = 2};

} // namespace tau_tuple
