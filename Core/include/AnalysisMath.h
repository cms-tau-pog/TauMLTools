/*! Common math functions and definitions suitable for analysis purposes.
This file is part of https://github.com/hh-italian-group/AnalysisTools. */

#pragma once

#include <iostream>
#include <cmath>
#include <Math/PtEtaPhiE4D.h>
#include <Math/PtEtaPhiM4D.h>
#include <Math/PxPyPzE4D.h>
#include <Math/LorentzVector.h>
#include <Math/SMatrix.h>
#include <Math/VectorUtil.h>
#include <TVector3.h>
#include <TH1.h>
#include <TMatrixD.h>
#include <TLorentzVector.h>


#include "PhysicalValue.h"

namespace analysis {

using LorentzVectorXYZ = ROOT::Math::LorentzVector<ROOT::Math::PxPyPzE4D<double>>;
using LorentzVectorM = ROOT::Math::LorentzVector<ROOT::Math::PtEtaPhiM4D<double>>;
using LorentzVectorE = ROOT::Math::LorentzVector<ROOT::Math::PtEtaPhiE4D<double>>;
using LorentzVector = LorentzVectorE;

using LorentzVectorXYZ_Float = ROOT::Math::LorentzVector<ROOT::Math::PxPyPzE4D<float>>;
using LorentzVectorM_Float = ROOT::Math::LorentzVector<ROOT::Math::PtEtaPhiM4D<float>>;
using LorentzVectorE_Float = ROOT::Math::LorentzVector<ROOT::Math::PtEtaPhiE4D<float>>;

template<unsigned n>
using SquareMatrix = ROOT::Math::SMatrix<double, n, n, ROOT::Math::MatRepStd<double, n>>;

template<unsigned n>
TMatrixD ConvertMatrix(const SquareMatrix<n>& m)
{
    TMatrixD result(n, n);
    for(unsigned k = 0; k < n; ++k) {
        for(unsigned l = 0; l < n; ++l)
            result[k][l] = m[k][l];
    }
    return result;
}

template<typename LVector>
TLorentzVector ConvertVector(const LVector& v)
{
    return TLorentzVector(v.Px(), v.Py(), v.Pz(), v.E());
}

//see AN-13-178
template<typename LVector1, typename LVector2>
double Calculate_MT(const LVector1& lepton_p4, const LVector2& met_p4)
{
    const double delta_phi = TVector2::Phi_mpi_pi(lepton_p4.Phi() - met_p4.Phi());
    return std::sqrt( 2.0 * lepton_p4.Pt() * met_p4.Pt() * ( 1.0 - std::cos(delta_phi) ) );
}

template<typename LVector1, typename LVector2, typename LVector3>
double Calculate_TotalMT(const LVector1& lepton1_p4, const LVector2& lepton2_p4, const LVector3& met_p4)
{
    const double mt_1 = Calculate_MT(lepton1_p4, met_p4);
    const double mt_2 = Calculate_MT(lepton2_p4, met_p4);
    const double mt_ll = Calculate_MT(lepton1_p4, lepton2_p4);
    return std::sqrt(std::pow(mt_1, 2) + std::pow(mt_2, 2) + std::pow(mt_ll, 2));
}

// from DataFormats/TrackReco/interface/TrackBase.h
template<typename Point>
double Calculate_dxy(const Point& legV, const Point& PV, const LorentzVector& leg)
{
    return ( - (legV.x()-PV.x()) * leg.Py() + (legV.y()-PV.y()) * leg.Px() ) / leg.Pt();
}

// from DataFormats/TrackReco/interface/TrackBase.h
inline double Calculate_dz(const TVector3& trkV, const TVector3& PV, const TVector3& trkP)
{
  return (trkV.z() - PV.z()) - ( (trkV.x() - PV.x()) * trkP.x() + (trkV.y() - PV.y()) * trkP.y() ) / trkP.Pt()
                               * trkP.z() / trkP.Pt();
}

template<typename LVector1, typename LVector2, typename LVector3>
double Calculate_Pzeta(const LVector1& l1_p4, const LVector2& l2_p4, const LVector3& met_p4)
{
    const auto ll_p4 = l1_p4 + l2_p4;
    const TVector2 ll_p2(ll_p4.Px(), ll_p4.Py());
    const TVector2 met_p2(met_p4.Px(), met_p4.Py());
    const TVector2 ll_s = ll_p2 + met_p2;
    const TVector2 l1_u(std::cos(l1_p4.Phi()), std::sin(l1_p4.Phi()));
    const TVector2 l2_u(std::cos(l2_p4.Phi()), std::sin(l2_p4.Phi()));
    const TVector2 ll_u = l1_u + l2_u;
    const double ll_u_met = ll_s * ll_u;
    const double ll_mod = ll_u.Mod();
    return ll_u_met / ll_mod;
}

template<typename LVector1, typename LVector2>
double Calculate_visiblePzeta(const LVector1& l1_p4, const LVector2& l2_p4)
{
    const auto ll_p4 = l1_p4 + l2_p4;
    const TVector2 ll_p2(ll_p4.Px(), ll_p4.Py());
    const TVector2 l1_u(std::cos(l1_p4.Phi()), std::sin(l1_p4.Phi()));
    const TVector2 l2_u(std::cos(l2_p4.Phi()), std::sin(l2_p4.Phi()));
    const TVector2 ll_u = l1_u + l2_u;
    const double ll_p2u = ll_p2 * ll_u;
    const double ll_mod = ll_u.Mod();
    return ll_p2u / ll_mod;
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
