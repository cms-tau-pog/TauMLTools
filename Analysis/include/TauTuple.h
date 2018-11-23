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

#define CSVAR(type, name, col) VAR(type, col##_##name)

#define TAU_COMP(col) \
    CSVAR(Float_t, sum_pt, col) /* */ \
    CSVAR(Float_t, sum_ht, col) /* */ \
    CSVAR(Float_t, sum_dEta, col) /* */ \
    CSVAR(Float_t, sum_dPhi, col) /* */ \
    CSVAR(Float_t, sum_energy, col) /* */ \
    CSVAR(Float_t, sum_mass, col) /* */ \
    CSVAR(UInt_t, nTotal, col) /* */ \
    /**/

#define TAU_COMP_SET(col) \
    TAU_COMP(col##_ChargedHadrCands) \
    TAU_COMP(col##_NeutrHadrCands) \
    TAU_COMP(col##_GammaCands) \
    /**/

#define TAU_GEN_MATCH(col) \
    VAR(Bool_t, has_gen_##col##_match) /* */\
    VAR(Float_t, gen_##col##_match_dR) /* */\
    VAR(Float_t, gen_##col##_pdg) /* */\
    VAR(Float_t, gen_##col##_pt) /* */\
    VAR(Float_t, gen_##col##_eta) /* */\
    VAR(Float_t, gen_##col##_phi) /* */\
    VAR(Float_t, gen_##col##_mass) /* */\
    /**/

#define TAU_DATA() \
    /* Event Variables */ \
    VAR(UInt_t, run) /* run */ \
    VAR(UInt_t, lumi) /* lumi section */ \
    VAR(ULong64_t, evt) /* event number */ \
    VAR(Int_t, genEventType) /* gen event type */ \
    VAR(Float_t, genEventWeight) /* gen event weight */ \
    VAR(Int_t, npv) /* number of primary vertices */ \
    VAR(Float_t, rho) /* rho */ \
    VAR(Float_t, npu) /* number of in-time pu interactions added to the event */ \
    /* Jet variables */ \
    VAR(UInt_t, jet_index) /* index of the jet */ \
    VAR(Float_t, jet_pt) /* jet pt */ \
    VAR(Float_t, jet_eta) /* jet eta */ \
    VAR(Float_t, jet_phi) /* jet phi */ \
    VAR(Float_t, jet_mass) /* jet mass */ \
    /* Basic tau variables */ \
    VAR(Int_t, tau_index) /* index of the tau */ \
    VAR(Float_t, tau_pt) /* tau pt */ \
    VAR(Float_t, tau_eta) /* tau eta */ \
    VAR(Float_t, tau_phi) /* tau phi */ \
    VAR(Float_t, tau_mass) /* tau mass */ \
    VAR(Int_t, charge) /* tau charge */ \
    VAR(Int_t, gen_match) /* generator matching, see Htautau Twiki*/\
    VAR(Float_t, gen_pt) /* pt of the matched gen particle */ \
    VAR(Float_t, gen_eta) /* eta of the matched gen particle */ \
    VAR(Float_t, gen_phi) /* phi of the matched gen particle */ \
    VAR(Float_t, gen_mass) /* mass of the matched gen particle */ \
    TAU_GEN_MATCH(ele) /* */ \
    TAU_GEN_MATCH(muon) /* */ \
    TAU_GEN_MATCH(qg) /* */ \
    /* Tau ID variables */ \
    VAR(Int_t, decayMode) /* tau decay mode */ \
    VAR(ULong64_t, id_flags) /* boolean tau id variables */ \
    RAW_TAU_IDS() \
    /* Extended tau variables */ \
    VAR(Float_t, dxy) /* tau dxy with respect to primary vertex */ \
    VAR(Float_t, dxy_sig) /* significance of dxy */ \
    VAR(Float_t, dz) /* tau dz with respect to primary vertex */ \
    VAR(Float_t, ip3d) /* */ \
    VAR(Float_t, ip3d_sig) /* */ \
    VAR(Bool_t, hasSecondaryVertex) /* */ \
    VAR(Float_t, flightLength_r) /* */ \
    VAR(Float_t, flightLength_dEta) /* */ \
    VAR(Float_t, flightLength_dPhi) /* */ \
    VAR(Float_t, flightLength_sig) /* */ \
    VAR(Float_t, pt_weighted_deta_strip) /* */ \
    VAR(Float_t, pt_weighted_dphi_strip) /* */ \
    VAR(Float_t, pt_weighted_dr_signal) /* */ \
    VAR(Float_t, pt_weighted_dr_iso) /* */ \
    VAR(Float_t, leadingTrackNormChi2) /* */ \
    VAR(Float_t, e_ratio) /* */ \
    VAR(Float_t, gj_angle_diff) /* */ \
    VAR(UInt_t, n_photons) /* */ \
    /* Against electron */ \
    VAR(Float_t, emFraction) /* */ \
    VAR(Bool_t, has_gsf_track) /* */ \
    VAR(Bool_t, inside_ecal_crack) /* */ \
    VAR(Bool_t, gsf_ele_matched) /* */ \
    VAR(Float_t, gsf_ele_pt) /* */ \
    VAR(Float_t, gsf_ele_dEta) /* */ \
    VAR(Float_t, gsf_ele_dPhi) /* */ \
    VAR(Float_t, gsf_ele_energy) /* */ \
    VAR(Float_t, gsf_ele_Ee) /* */ \
    VAR(Float_t, gsf_ele_Egamma) /* */ \
    VAR(Float_t, gsf_ele_Pin) /* */ \
    VAR(Float_t, gsf_ele_Pout) /* */ \
    VAR(Float_t, gsf_ele_Eecal) /* */ \
    VAR(Float_t, gsf_ele_dEta_SeedClusterTrackAtCalo) /* */ \
    VAR(Float_t, gsf_ele_dPhi_SeedClusterTrackAtCalo) /* */ \
    VAR(Float_t, gsf_ele_mvaIn_sigmaEtaEta) /* */ \
    VAR(Float_t, gsf_ele_mvaIn_hadEnergy) /* */ \
    VAR(Float_t, gsf_ele_mvaIn_deltaEta) /* */ \
    VAR(Float_t, gsf_ele_Chi2NormGSF) /* */ \
    VAR(Float_t, gsf_ele_GSFNumHits) /* */ \
    VAR(Float_t, gsf_ele_GSFTrackResol) /* */ \
    VAR(Float_t, gsf_ele_GSFTracklnPt) /* */ \
    VAR(Float_t, gsf_ele_Chi2NormKF) /* */ \
    VAR(Float_t, gsf_ele_KFNumHits) /* */ \
    /* Against muon */ \
    VAR(UInt_t, n_matched_muons) /* */ \
    VAR(Float_t, muon_pt) /* */ \
    VAR(Float_t, muon_dEta) /* */ \
    VAR(Float_t, muon_dPhi) /* */ \
    VAR(UInt_t, muon_n_matches_DT_1) /* */ \
    VAR(UInt_t, muon_n_matches_DT_2) /* */ \
    VAR(UInt_t, muon_n_matches_DT_3) /* */ \
    VAR(UInt_t, muon_n_matches_DT_4) /* */ \
    VAR(UInt_t, muon_n_matches_CSC_1) /* */ \
    VAR(UInt_t, muon_n_matches_CSC_2) /* */ \
    VAR(UInt_t, muon_n_matches_CSC_3) /* */ \
    VAR(UInt_t, muon_n_matches_CSC_4) /* */ \
    VAR(UInt_t, muon_n_matches_RPC_1) /* */ \
    VAR(UInt_t, muon_n_matches_RPC_2) /* */ \
    VAR(UInt_t, muon_n_matches_RPC_3) /* */ \
    VAR(UInt_t, muon_n_matches_RPC_4) /* */ \
    VAR(UInt_t, muon_n_hits_DT_1) /* */ \
    VAR(UInt_t, muon_n_hits_DT_2) /* */ \
    VAR(UInt_t, muon_n_hits_DT_3) /* */ \
    VAR(UInt_t, muon_n_hits_DT_4) /* */ \
    VAR(UInt_t, muon_n_hits_CSC_1) /* */ \
    VAR(UInt_t, muon_n_hits_CSC_2) /* */ \
    VAR(UInt_t, muon_n_hits_CSC_3) /* */ \
    VAR(UInt_t, muon_n_hits_CSC_4) /* */ \
    VAR(UInt_t, muon_n_hits_RPC_1) /* */ \
    VAR(UInt_t, muon_n_hits_RPC_2) /* */ \
    VAR(UInt_t, muon_n_hits_RPC_3) /* */ \
    VAR(UInt_t, muon_n_hits_RPC_4) /* */ \
    /* Components */ \
    VAR(Float_t, leadChargedHadrCand_pt) /* pt of the leading charged hadron candidate */ \
    VAR(Float_t, leadChargedHadrCand_dEta) /* eta of the leading charged hadron candidate */ \
    VAR(Float_t, leadChargedHadrCand_dPhi) /* phi of the leading charged hadron candidate */ \
    VAR(Float_t, leadChargedHadrCand_energy) /* energy of the leading charged hadron candidate */ \
    VAR(Float_t, leadChargedHadrCand_mass) /* mass of the leading charged hadron candidate */ \
    VAR(Float_t, leadChargedCand_etaAtEcalEntrance) /* */ \
    TAU_COMP_SET(innerSigCone) /* */ \
    TAU_COMP_SET(outerSigCone) /* */ \
    TAU_COMP_SET(isoRing02) /* */ \
    TAU_COMP_SET(isoRing03) /* */ \
    TAU_COMP_SET(isoRing04) /* */ \
    TAU_COMP_SET(isoRing05) /* */ \
    /**/

#define VAR(type, name) DECLARE_BRANCH_VARIABLE(type, name)
DECLARE_TREE(tau_tuple, Tau, TauTuple, TAU_DATA, "taus")
#undef VAR

#define VAR(type, name) ADD_DATA_TREE_BRANCH(name)
INITIALIZE_TREE(tau_tuple, TauTuple, TAU_DATA)
#undef VAR
#undef TAU_DATA
#undef CVAR
#undef CSVAR
#undef TAU_SIG_COMP
#undef TAU_ISO_COMP
#undef TAU_GEN_MATCH

namespace tau_tuple {

template<typename T>
constexpr T DefaultFillValue() { return std::numeric_limits<T>::lowest(); }
template<>
constexpr float DefaultFillValue<float>() { return -999.; }

enum class ComponenetType { Gamma = 0, ChargedHadronCandidate = 1, NeutralHadronCandidate = 2};

} // namespace tau_tuple
