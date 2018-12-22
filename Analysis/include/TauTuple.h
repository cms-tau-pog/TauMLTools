/*! Definition of a tuple with all event information that is required for the tau analysis.
*/

#pragma once

#include "AnalysisTools/Core/include/SmartTree.h"
#include <Math/VectorUtil.h>

#define RAW_TAU_IDS() \
    VAR(Float_t, againstElectronMVA6Raw) /* */ \
    VAR(Float_t, againstElectronMVA6category) /* */ \
    VAR(Float_t, byCombinedIsolationDeltaBetaCorrRaw3Hits) /* */ \
    VAR(Float_t, byIsolationMVArun2v1DBoldDMwLTraw) /* */ \
    VAR(Float_t, byIsolationMVArun2v1DBdR03oldDMwLTraw) /* */ \
    VAR(Float_t, byIsolationMVArun2v1DBoldDMwLTraw2016) /* */ \
    VAR(Float_t, byIsolationMVArun2017v2DBoldDMwLTraw2017) /* */ \
    VAR(Float_t, byIsolationMVArun2017v2DBoldDMdR0p3wLTraw2017) /* */ \
    VAR(Float_t, chargedIsoPtSum) /* */ \
    VAR(Float_t, chargedIsoPtSumdR03) /* */ \
    VAR(Float_t, footprintCorrection) /* */ \
    VAR(Float_t, footprintCorrectiondR03) /* */ \
    VAR(Float_t, neutralIsoPtSum) /* */ \
    VAR(Float_t, neutralIsoPtSumWeight) /* */ \
    VAR(Float_t, neutralIsoPtSumWeightdR03) /* */ \
    VAR(Float_t, neutralIsoPtSumdR03) /* */ \
    VAR(Float_t, photonPtSumOutsideSignalCone) /* */ \
    VAR(Float_t, photonPtSumOutsideSignalConedR03) /* */ \
    VAR(Float_t, puCorrPtSum) /* */ \
    /**/

#define CAND_VAR(type, name) VAR(std::vector<type>, pfCand_##name)

#define TAU_DATA() \
    /* Event Variables */ \
    VAR(UInt_t, run) /* run number */ \
    VAR(UInt_t, lumi) /* lumi section */ \
    VAR(ULong64_t, evt) /* event number */ \
    VAR(UInt_t, npv) /* number of primary vertices */ \
    VAR(Float_t, rho) /* fixed grid energy density */ \
    VAR(Float_t, genEventWeight) /* gen event weight */ \
    VAR(Float_t, npu) /* number of in-time pu interactions added to the event */ \
    /* Jet variables */ \
    VAR(Int_t, jet_index) /* index of the jet */ \
    VAR(Float_t, jet_pt) /* jet pt */ \
    VAR(Float_t, jet_eta) /* jet eta */ \
    VAR(Float_t, jet_phi) /* jet phi */ \
    VAR(Float_t, jet_mass) /* jet mass */ \
    VAR(Float_t, jet_neutralHadronEnergyFraction) /* jet neutral hadron energy fraction (relative to uncorrected jet energy) */ \
    VAR(Float_t, jet_neutralEmEnergyFraction) /* jet neutral EM energy fraction (relative to uncorrected jet energy) */ \
    VAR(Int_t, jet_nConstituents) /* number of jet constituents */ \
    VAR(Int_t, jet_chargedMultiplicity) /* jet charged multiplicity */ \
    VAR(Int_t, jet_neutralMultiplicity) /* jet neutral multiplicity */ \
    VAR(Int_t, jet_partonFlavour) /* parton-based flavour of the jet */ \
    VAR(Int_t, jet_hadronFlavour) /* hadron-based flavour of the jet */ \
    VAR(Int_t, jet_has_gen_match) /* jet has a matched gen-jet */ \
    VAR(Float_t, jet_gen_pt) /* gen jet pt */ \
    VAR(Float_t, jet_gen_eta) /* gen jet eta */ \
    VAR(Float_t, jet_gen_phi) /* gen jet phi */ \
    VAR(Float_t, jet_gen_mass) /* gen jet mass */ \
    VAR(Int_t, jet_gen_n_b) /* number of b hadrons clustered inside the jet */ \
    VAR(Int_t, jet_gen_n_c) /* number of c hadrons clustered inside the jet */ \
    /* Basic tau variables */ \
    VAR(Int_t, jetTauMatch) /* match between jet and tau */ \
    VAR(Int_t, tau_index) /* index of the tau */ \
    VAR(Float_t, tau_pt) /* tau pt */ \
    VAR(Float_t, tau_eta) /* tau eta */ \
    VAR(Float_t, tau_phi) /* tau phi */ \
    VAR(Float_t, tau_mass) /* tau mass */ \
    VAR(Int_t, tau_charge) /* tau charge */ \
    VAR(Int_t, lepton_gen_match) /* generator matching, see Htautau Twiki*/\
    VAR(Int_t, lepton_gen_charge) /* generator matching, see Htautau Twiki*/\
    VAR(Float_t, lepton_gen_pt) /* pt of the matched gen particle */ \
    VAR(Float_t, lepton_gen_eta) /* eta of the matched gen particle */ \
    VAR(Float_t, lepton_gen_phi) /* phi of the matched gen particle */ \
    VAR(Float_t, lepton_gen_mass) /* mass of the matched gen particle */ \
    VAR(std::vector<Int_t>, lepton_gen_vis_pdg) /* generator matching, see Htautau Twiki*/\
    VAR(std::vector<Float_t>, lepton_gen_vis_pt) \
    VAR(std::vector<Float_t>, lepton_gen_vis_eta) \
    VAR(std::vector<Float_t>, lepton_gen_vis_phi) \
    VAR(std::vector<Float_t>, lepton_gen_vis_mass) \
    VAR(Int_t, qcd_gen_match) /* generator matching, see Htautau Twiki*/\
    VAR(Int_t, qcd_gen_charge) /* generator matching, see Htautau Twiki*/\
    VAR(Float_t, qcd_gen_pt) /* pt of the matched gen particle */ \
    VAR(Float_t, qcd_gen_eta) /* eta of the matched gen particle */ \
    VAR(Float_t, qcd_gen_phi) /* phi of the matched gen particle */ \
    VAR(Float_t, qcd_gen_mass) /* mass of the matched gen particle */ \
    /* Tau ID variables */ \
    VAR(Int_t, tau_decayMode) /* tau decay mode */ \
    VAR(ULong64_t, tau_id_flags) /* boolean tau id variables */ \
    RAW_TAU_IDS() \
    /* Extended tau variables */ \
    VAR(Float_t, tau_dxy) /* tau dxy with respect to primary vertex */ \
    VAR(Float_t, tau_dxy_sig) /* significance of dxy */ \
    VAR(Float_t, tau_dz) /* tau dz with respect to primary vertex */ \
    VAR(Float_t, tau_ip3d) /* */ \
    VAR(Float_t, tau_ip3d_sig) /* */ \
    VAR(Int_t, tau_hasSecondaryVertex) /* */ \
    VAR(Float_t, tau_flightLength_r) /* */ \
    VAR(Float_t, tau_flightLength_dEta) /* */ \
    VAR(Float_t, tau_flightLength_dPhi) /* */ \
    VAR(Float_t, tau_flightLength_sig) /* */ \
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
#undef TAU_DATA
#undef CAND_VAR

namespace tau_tuple {

template<typename T>
constexpr T DefaultFillValue() { return std::numeric_limits<T>::lowest(); }
template<>
constexpr float DefaultFillValue<float>() { return -999.; }

enum class ComponenetType { Gamma = 0, ChargedHadronCandidate = 1, NeutralHadronCandidate = 2};

} // namespace tau_tuple
