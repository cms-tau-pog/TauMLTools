/*! Definition of a tuple with all event information that is required for the tau analysis.
*/

#pragma once

#include "AnalysisTools/Core/include/SmartTree.h"

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

#define CVAR(type, name, col) VAR(std::vector<type>, col##_##name)
#define CSVAR(type, name, col) VAR(type, col##_##name)

#define TAU_SIG_COMP(col) \
    CVAR(Bool_t, isInside_innerSigCone, col) /* */ \
    CVAR(Float_t, pt, col) /* */ \
    CVAR(Float_t, eta, col) /* */ \
    CVAR(Float_t, phi, col) /* */ \
    CVAR(Float_t, mass, col) /* */ \
    CSVAR(Float_t, sum_innerSigCone_pt, col) /* */ \
    CSVAR(Float_t, sum_innerSigCone_eta, col) /* */ \
    CSVAR(Float_t, sum_innerSigCone_phi, col) /* */ \
    CSVAR(Float_t, sum_innerSigCone_mass, col) /* */ \
    CSVAR(Float_t, sum_outerSigCone_pt, col) /* */ \
    CSVAR(Float_t, sum_outerSigCone_eta, col) /* */ \
    CSVAR(Float_t, sum_outerSigCone_phi, col) /* */ \
    CSVAR(Float_t, sum_outerSigCone_mass, col) /* */ \
    CSVAR(UInt_t, nTotal_innerSigCone, col) /* */ \
    CSVAR(UInt_t, nTotal_outerSigCone, col) /* */ \
    /**/

#define TAU_ISO_COMP(col) \
    CVAR(Float_t, pt, col) /* */ \
    CVAR(Float_t, eta, col) /* */ \
    CVAR(Float_t, phi, col) /* */ \
    CVAR(Float_t, mass, col) /* */ \
    CSVAR(Float_t, sum_pt, col) /* */ \
    CSVAR(Float_t, sum_eta, col) /* */ \
    CSVAR(Float_t, sum_phi, col) /* */ \
    CSVAR(Float_t, sum_mass, col) /* */ \
    CSVAR(UInt_t, nTotal, col) /* */ \
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
    /* Basic tau variables */ \
    VAR(Float_t, pt) /* tau pt */ \
    VAR(Float_t, eta) /* tau eta */ \
    VAR(Float_t, phi) /* tau phi */ \
    VAR(Float_t, mass) /* tau mass */ \
    VAR(Int_t, charge) /* tau charge */ \
    VAR(Int_t, gen_match) /* generator matching, see Htautau Twiki*/\
    VAR(Float_t, gen_pt) /* pt of the matched gen particle */ \
    VAR(Float_t, gen_eta) /* eta of the matched gen particle */ \
    VAR(Float_t, gen_phi) /* phi of the matched gen particle */ \
    VAR(Float_t, gen_mass) /* mass of the matched gen particle */ \
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
    VAR(Float_t, flightLength_eta) /* */ \
    VAR(Float_t, flightLength_phi) /* */ \
    VAR(Float_t, flightLength_sig) /* */ \
    VAR(Float_t, leadChargedHadrCand_pt) /* pt of the leading charged hadron candidate */ \
    VAR(Float_t, leadChargedHadrCand_eta) /* eta of the leading charged hadron candidate */ \
    VAR(Float_t, leadChargedHadrCand_phi) /* phi of the leading charged hadron candidate */ \
    VAR(Float_t, leadChargedHadrCand_mass) /* mass of the leading charged hadron candidate */ \
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
    VAR(Float_t, etaAtEcalEntrance) /* */ \
    VAR(Float_t, phiAtEcalEntrance) /* */ \
    VAR(Bool_t, has_gsf_track) /* */ \
    VAR(Bool_t, inside_ecal_crack) /* */ \
    VAR(Float_t, ecal_crack_dPhi) /* */ \
    VAR(Float_t, ecal_crack_dEta) /* */ \
    VAR(Bool_t, gsf_ele_matched) /* */ \
    VAR(Float_t, gsf_ele_pt) /* */ \
    VAR(Float_t, gsf_ele_eta) /* */ \
    VAR(Float_t, gsf_ele_phi) /* */ \
    VAR(Float_t, gsf_ele_mass) /* */ \
    VAR(Float_t, gsf_ele_Ee) /* */ \
    VAR(Float_t, gsf_ele_Egamma) /* */ \
    VAR(Float_t, gsf_ele_Pin) /* */ \
    VAR(Float_t, gsf_ele_Pout) /* */ \
    VAR(Float_t, gsf_ele_EtotOverPin) /* */ \
    VAR(Float_t, gsf_ele_Eecal) /* */ \
    VAR(Float_t, gsf_ele_deta) /* */ \
    VAR(Float_t, gsf_ele_dphi) /* */ \
    VAR(Float_t, gsf_ele_mvaIn_sigmaEtaEta) /* */ \
    VAR(Float_t, gsf_ele_mvaIn_hadEnergy) /* */ \
    VAR(Float_t, gsf_ele_mvaIn_deltaEta) /* */ \
    VAR(Float_t, gsf_ele_Chi2NormGSF) /* */ \
    VAR(Float_t, gsf_ele_GSFNumHits) /* */ \
    VAR(Float_t, gsf_ele_GSFTrackResol) /* */ \
    VAR(Float_t, gsf_ele_GSFTracklnPt) /* */ \
    VAR(Float_t, gsf_ele_Chi2NormKF) /* */ \
    VAR(Float_t, gsf_ele_KFNumHits) /* */ \
    VAR(Float_t, leadChargedCand_etaAtEcalEntrance) /* */ \
    VAR(Float_t, leadChargedCand_pt) /* */ \
    VAR(Float_t, leadChargedHadrCand_HoP) /* */ \
    VAR(Float_t, leadChargedHadrCand_EoP) /* */ \
    VAR(Float_t, tau_visMass_innerSigCone) /* */ \
    /* Against muon */ \
    VAR(Bool_t, has_matched_muon) /* */ \
    VAR(std::vector<Float_t>, muon_pt) /* */ \
    VAR(std::vector<Float_t>, muon_eta) /* */ \
    VAR(std::vector<Float_t>, muon_phi) /* */ \
    VAR(std::vector<UInt_t>, muon_n_matches_DT) /* */ \
    VAR(std::vector<UInt_t>, muon_n_matches_CSC) /* */ \
    VAR(std::vector<UInt_t>, muon_n_matches_RPC) /* */ \
    VAR(std::vector<UInt_t>, muon_n_hits_DT) /* */ \
    VAR(std::vector<UInt_t>, muon_n_hits_CSC) /* */ \
    VAR(std::vector<UInt_t>, muon_n_hits_RPC) /* */ \
    VAR(UInt_t, muon_n_stations_with_matches_03) /* */ \
    VAR(UInt_t, muon_n_stations_with_hits_23) /* */ \
    VAR(Float_t, energy_ECAL) /* */ \
    VAR(Float_t, energy_HCAL) /* */ \
    VAR(Float_t, leadTrack_p) /* */ \
    /* Components */ \
    TAU_SIG_COMP(signalChargedHadrCands) /* */ \
    TAU_SIG_COMP(signalNeutrHadrCands) /* */ \
    TAU_SIG_COMP(signalGammaCands) /* */ \
    TAU_ISO_COMP(isolationChargedHadrCands) /* */ \
    TAU_ISO_COMP(isolationNeutrHadrCands) /* */ \
    TAU_ISO_COMP(isolationGammaCands) /* */ \
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

namespace tau_tuple {
template<typename T>
constexpr T DefaultFillValue() { return std::numeric_limits<T>::lowest(); }

enum class ComponenetType { Gamma = 0, ChargedHadronCandidate = 1, NeutralHadronCandidate = 2};
} // namespace tau_tuple
