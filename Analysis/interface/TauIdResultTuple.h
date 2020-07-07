/*! Definition of a tuple with tau id results.
*/

#pragma once

#include "TauMLTools/Core/interface/SmartTree.h"

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
    VAR(Float_t, dxy_pca_x) /* */ \
    VAR(Float_t, dxy_pca_y) /* */ \
    VAR(Float_t, dxy_pca_z) /* */ \
    VAR(Bool_t, decayModeFinding) /* tau old decay mode finding result */ \
    VAR(Int_t, decayMode) /* tau decay mode */ \
    VAR(Bool_t, gen_e) /* matches e at gen level */ \
    VAR(Bool_t, gen_mu) /* matches mu at gen level */ \
    VAR(Bool_t, gen_tau) /* matches hadronic tau at gen level */ \
    VAR(Bool_t, gen_jet) /* matches jet at gen level */ \
    VAR(Float_t, byDeepTau2017v2VSeraw) /* DeepTau tau vs e id value */ \
    VAR(Float_t, byDeepTau2017v2VSmuraw) /* DeepTau tau vs mu id value */ \
    VAR(Float_t, byDeepTau2017v2VSjetraw) /* DeepTau tau vs jet id value */ \
    /**/

#define VAR(type, name) DECLARE_BRANCH_VARIABLE(type, name)
DECLARE_TREE(tau_tuple, TauIdResult, TauIdResultTuple, TAU_ID_DATA, "taus")
#undef VAR

#define VAR(type, name) ADD_DATA_TREE_BRANCH(name)
INITIALIZE_TREE(tau_tuple, TauIdResultTuple, TAU_ID_DATA)
#undef VAR
#undef TAU_ID_DATA
