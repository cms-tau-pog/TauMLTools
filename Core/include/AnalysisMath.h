/*! Common math functions and definitions suitable for analysis purposes.
This file is part of https://github.com/hh-italian-group/AnalysisTools. */

#pragma once

#include <iostream>
#include <cmath>

#include <TLorentzVector.h>
#include <TH1.h>

#include "PhysicalValue.h"

namespace analysis {

//see AN-13-178
inline double Calculate_MT(const TLorentzVector& lepton_momentum, double met_pt, double met_phi)
{
    const double delta_phi = TVector2::Phi_mpi_pi(lepton_momentum.Phi() - met_phi);
    return std::sqrt( 2.0 * lepton_momentum.Pt() * met_pt * ( 1.0 - std::cos(delta_phi) ) );
}

// from DataFormats/TrackReco/interface/TrackBase.h
inline double Calculate_dxy(const TVector3& legV, const TVector3& PV, const TLorentzVector& leg)
{
    return ( - (legV.x()-PV.x()) * leg.Py() + (legV.y()-PV.y()) * leg.Px() ) / leg.Pt();
}

// from DataFormats/TrackReco/interface/TrackBase.h
inline double Calculate_dz(const TVector3& trkV, const TVector3& PV, const TVector3& trkP)
{
  return (trkV.z() - PV.z()) - ( (trkV.x() - PV.x()) * trkP.x() + (trkV.y() - PV.y()) * trkP.y() ) / trkP.Pt()
                               * trkP.z() / trkP.Pt();
}

inline TLorentzVector MakeLorentzVectorPtEtaPhiM(Double_t pt, Double_t eta, Double_t phi, Double_t m)
{
    TLorentzVector v;
    v.SetPtEtaPhiM(pt, eta, phi, m);
    return v;
}

inline TLorentzVector MakeLorentzVectorPtEtaPhiE(Double_t pt, Double_t eta, Double_t phi, Double_t e)
{
    TLorentzVector v;
    v.SetPtEtaPhiE(pt, eta, phi, e);
    return v;
}

inline PhysicalValue Integral(const TH1D& histogram, bool include_overflows = true)
{
    using limit_pair = std::pair<Int_t, Int_t>;
    const limit_pair limits = include_overflows ? limit_pair(0, histogram.GetNbinsX() + 1)
                                                : limit_pair(1, histogram.GetNbinsX());

    double error = 0;
    const double integral = histogram.IntegralAndError(limits.first, limits.second, error);
    return PhysicalValue(integral, error);
}

inline void RenormalizeHistogram(TH1D& histogram, const PhysicalValue& norm, bool include_overflows = true)
{
    histogram.Scale(norm.GetValue() / Integral(histogram,include_overflows).GetValue());
}

} // namespace analysis
