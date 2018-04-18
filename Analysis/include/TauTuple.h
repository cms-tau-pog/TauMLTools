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
    /* Tau variables */ \
    VAR(Float_t, pt) /* tau pt */ \
    VAR(Float_t, eta) /* tau eta */ \
    VAR(Float_t, phi) /* tau phi */ \
    VAR(Float_t, mass) /* tau mass */ \
    VAR(Int_t, charge) /* tau charge */ \
    VAR(Float_t, dxy) /* tau dxy with respect to primary vertex */ \
    VAR(Float_t, dz) /* tau dz with respect to primary vertex */ \
    VAR(Int_t, gen_match) /* generator matching, see Htautau Twiki*/\
    VAR(Float_t, gen_pt) /* pt of the matched gen particle */ \
    VAR(Float_t, gen_eta) /* eta of the matched gen particle */ \
    VAR(Float_t, gen_phi) /* phi of the matched gen particle */ \
    VAR(Float_t, gen_mass) /* tau mass of the matched gen particle */ \
    VAR(Int_t, decayMode) /* tau decay mode */ \
    VAR(ULong64_t, id_flags) /* boolean tau id variables */ \
    RAW_TAU_IDS() \
    /**/

#define VAR(type, name) DECLARE_BRANCH_VARIABLE(type, name)
DECLARE_TREE(tau_tuple, Tau, TauTuple, TAU_DATA, "taus")
#undef VAR

#define VAR(type, name) ADD_DATA_TREE_BRANCH(name)
INITIALIZE_TREE(tau_tuple, TauTuple, TAU_DATA)
#undef VAR
#undef TAU_DATA

namespace tau_tuple {
template<typename T>
constexpr T DefaultFillValue() { return std::numeric_limits<T>::lowest(); }
} // namespace tau_tuple
