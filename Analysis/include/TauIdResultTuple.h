/*! Definition of a tuple with tau id results.
*/

#pragma once

#include "AnalysisTools/Core/include/SmartTree.h"

#define TAU_ID_DATA() \
    VAR(UInt_t, run) /* run */ \
    VAR(UInt_t, lumi) /* lumi section */ \
    VAR(ULong64_t, evt) /* event number */ \
    VAR(Float_t, genEventWeight) /* gen event weight */ \
    VAR(UInt_t, tau_index) /* index of the tau */ \
    VAR(Float_t, pt) /* tau pt */ \
    VAR(Float_t, eta) /* tau eta */ \
    VAR(Float_t, phi) /* tau phi */ \
    VAR(Float_t, dxy) /* tau dxy */ \
    VAR(Float_t, dz) /* tau dz */ \
    VAR(Bool_t, decayModeFinding) /* tau old decay mode finding result */ \
    VAR(Int_t, decayMode) /* tau decay mode */ \
    VAR(Bool_t, gen_e) /* matches e at gen level */ \
    VAR(Bool_t, gen_mu) /* matches mu at gen level */ \
    VAR(Bool_t, gen_tau) /* matches hadronic tau at gen level */ \
    VAR(Bool_t, gen_jet) /* matches jet at gen level */ \
    VAR(Float_t, deepId_tau_vs_e) /* DeepTau tau vs e id value */ \
    VAR(Float_t, deepId_tau_vs_mu) /* DeepTau tau vs mu id value */ \
    VAR(Float_t, deepId_tau_vs_jet) /* DeepTau tau vs jet id value */ \
    VAR(Float_t, deepId_tau_vs_all) /* DeepTau tau vs all id value */ \
    VAR(Float_t, refId_e) /* Reference e->tau id value */ \
    VAR(Float_t, refId_mu_loose) /* Reference mu->tau id value (loose WP) */ \
    VAR(Float_t, refId_mu_tight) /* Reference mu->tau id value (Tight WP) */ \
    VAR(Float_t, refId_jet) /* Reference jet->tau id value */ \
    VAR(Float_t, otherId_tau_vs_all) /* Alternative tau vs all id value */ \
    VAR(Float_t, refId_jet_dR0p32017v2) /* */ \
    VAR(Float_t, refId_jet_newDM2017v2) /* */ \
    VAR(Float_t, byDPFTau2016v0VSallraw) /* */ \
    VAR(Float_t, byDPFTau2016v1VSallraw) /* */ \
    VAR(Float_t, byDeepTau2017v1VSeraw) /* */ \
    VAR(Float_t, byDeepTau2017v1VSmuraw) /* */ \
    VAR(Float_t, byDeepTau2017v1VSjetraw) /* */ \
    /**/

#define VAR(type, name) DECLARE_BRANCH_VARIABLE(type, name)
DECLARE_TREE(tau_tuple, TauIdResult, TauIdResultTuple, TAU_ID_DATA, "taus")
#undef VAR

#define VAR(type, name) ADD_DATA_TREE_BRANCH(name)
INITIALIZE_TREE(tau_tuple, TauIdResultTuple, TAU_ID_DATA)
#undef VAR
#undef TAU_ID_DATA
