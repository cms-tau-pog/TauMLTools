/* Common methods used in tau analysis.
*/

#pragma once

#include <Math/VectorUtil.h>
#include "DataFormats/PatCandidates/interface/Muon.h"
#include "DataFormats/PatCandidates/interface/Electron.h"
#include "DataFormats/PatCandidates/interface/Photon.h"
#include "DataFormats/PatCandidates/interface/Tau.h"

namespace tau_analysis {

template<typename LVector1, typename LVector2>
float dEta(const LVector1& p4, const LVector2& tau_p4)
{
    return static_cast<float>(p4.eta() - tau_p4.eta());
}

template<typename LVector1, typename LVector2>
float dPhi(const LVector1& p4, const LVector2& tau_p4)
{
    return static_cast<float>(ROOT::Math::VectorUtil::DeltaPhi(p4, tau_p4));
}

template<typename Float>
Float AbsMin(Float a, Float b)
{
    return std::abs(b) < std::abs(a) ? b : a;
}

inline double GetInnerSignalConeRadius(double pt)
{
    static constexpr double min_pt = 30., min_radius = 0.05, cone_opening_coef = 3.;
    // This is equivalent of the original formula std::max(std::min(0.1, 3.0/pt), 0.05).
    return std::max(cone_opening_coef / std::max(pt, min_pt), min_radius);
}

inline bool IsInEcalCrack(double eta)
{
    const double abs_eta = std::abs(eta);
    return abs_eta > 1.46 && abs_eta < 1.558;
}

// Copied from https://github.com/cms-sw/cmssw/blob/CMSSW_9_4_X/RecoTauTag/RecoTau/src/AntiElectronIDMVA6.cc#L1317
// Compute the (unsigned) distance to the closest eta-crack in the ECAL barrel
double CalculateDeltaEtaCrack(double eta);

// Based on https://github.com/cms-sw/cmssw/blob/CMSSW_9_4_X/RecoTauTag/RecoTau/plugins/PATTauDiscriminationByMVAIsolationRun2.cc#L218
bool CalculateGottfriedJacksonAngleDifference(const pat::Tau& tau, double& gj_diff);

double PFRelIsolation(const pat::Muon& muon);

double PFRelIsolation(const pat::Electron& electron, float rho);
} // namespace tau_analysis
