/*! Common math functions and definitions suitable for analysis purposes.
This file is part of https://github.com/cms-tau-pog/TauMLTools. */

#pragma once

#include <iostream>
#include <cmath>
#include <Math/PtEtaPhiE4D.h>
#include <Math/PtEtaPhiM4D.h>
#include <Math/PxPyPzE4D.h>
#include <Math/LorentzVector.h>
#include <Math/SMatrix.h>
#include <Math/VectorUtil.h>
#include <Math/Point3D.h>
#include <TVector3.h>
#include <TH1.h>
#include <TH2.h>
#include <TMatrixD.h>
#include <TLorentzVector.h>
#include "Math/GenVector/Cartesian3D.h"

#include "exception.h"

extern template class TMatrixT<double>;

namespace analysis {

using LorentzVectorXYZ = ROOT::Math::LorentzVector<ROOT::Math::PxPyPzE4D<double>>;
using LorentzVectorM = ROOT::Math::LorentzVector<ROOT::Math::PtEtaPhiM4D<double>>;
using LorentzVectorE = ROOT::Math::LorentzVector<ROOT::Math::PtEtaPhiE4D<double>>;
using LorentzVector = LorentzVectorE;

using LorentzVectorXYZ_Float = ROOT::Math::LorentzVector<ROOT::Math::PxPyPzE4D<float>>;
using LorentzVectorM_Float = ROOT::Math::LorentzVector<ROOT::Math::PtEtaPhiM4D<float>>;
using LorentzVectorE_Float = ROOT::Math::LorentzVector<ROOT::Math::PtEtaPhiE4D<float>>;
using Point3D = ROOT::Math::PositionVector3D<ROOT::Math::Cartesian3D<double>>;
using Point3D_Float = ROOT::Math::PositionVector3D<ROOT::Math::Cartesian3D<float>>;

template<unsigned n>
using SquareMatrix = ROOT::Math::SMatrix<double, n, n, ROOT::Math::MatRepStd<double, n>>;

template<unsigned n>
TMatrixD ConvertMatrix(const SquareMatrix<n>& m)
{
  TMatrixD result(n,n);
  for(unsigned k = 0; k < n; ++k) {
    for(unsigned l = 0; l < n; ++l) {
      int kk = static_cast<int>(k);
      int ll = static_cast<int>(l);
      result(kk, ll) = m(k,l);
    }
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
  const double dx = trkV.x() - PV.x();
  const double dy = trkV.y() - PV.y();
  const double dz = trkV.z() - PV.z();
  return dz - ( dx * trkP.x() + dy * trkP.y() ) / trkP.Pt() * trkP.z() / trkP.Pt();
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

template<typename Iterator>
double Calculate_HT(Iterator begin, const Iterator& end){
  double sum = 0;
  for(; begin != end; ++begin)
    sum += begin->pt();
  return sum;
}

struct PhysicalValue {
  PhysicalValue(double value, double error) : value(value), error(error) {}
  PhysicalValue()
    : value(std::numeric_limits<double>::quiet_NaN()), error(std::numeric_limits<double>::quiet_NaN()) {}
  double value, error;
};

inline PhysicalValue Integral(const TH1& histogram, Int_t first_bin, Int_t last_bin)
{
  double error = 0;
  const double integral = histogram.IntegralAndError(first_bin, last_bin, error);
  return PhysicalValue(integral, error);
}

inline PhysicalValue Integral(const TH1& histogram, bool include_overflows = true)
{
  const Int_t first_bin = include_overflows ? 0 : 1;
  const Int_t last_bin = include_overflows ? histogram.GetNbinsX() + 1 : histogram.GetNbinsX();
  return Integral(histogram, first_bin, last_bin);
}

inline PhysicalValue Integral(const TH2& histogram, Int_t first_x_bin, Int_t last_x_bin,
                              Int_t first_y_bin, Int_t last_y_bin)
{
  double error = 0;
  const double integral = histogram.IntegralAndError(first_x_bin, last_x_bin, first_y_bin, last_y_bin, error);
  return PhysicalValue(integral, error);
}

inline PhysicalValue Integral(const TH2& histogram, bool include_overflows = true)
{
  const Int_t first_bin = include_overflows ? 0 : 1;
  const Int_t last_x_bin = include_overflows ? histogram.GetNbinsX() + 1 : histogram.GetNbinsX();
  const Int_t last_y_bin = include_overflows ? histogram.GetNbinsY() + 1 : histogram.GetNbinsY();
  return Integral(histogram, first_bin, last_x_bin, first_bin, last_y_bin);
}

template<typename Histogram>
void RenormalizeHistogram(Histogram& histogram, double norm, bool include_overflows = true)
{
  const double integral = Integral(histogram,include_overflows).value;
  if (integral == 0)
    throw analysis::exception("Integral is zero.");
  histogram.Scale(norm / integral);
}

inline double crystalball(double m, double m0, double sigma, double alpha, double n, double norm)
{
  if(m<1. || 1000.<m)
    throw analysis::exception("The parameter m is out of range");

  static const double sqrtPiOver2 = std::sqrt(std::atan(1) * 2);
  static const double sqrt2 = std::sqrt(2);
  double sig = std::abs(sigma);
  double t = (m - m0)/sig;
  if(alpha < 0) t = -t;
  double absAlpha =  std::abs(alpha/sig);
  double a = std::pow(n/absAlpha,n)*std::exp(-0.5*absAlpha*absAlpha);
  double b = absAlpha - n/absAlpha;
  double ApproxErf;
  double arg = absAlpha / sqrt2;
  if (arg > 5.) ApproxErf = 1;
  else if (arg < -5.) ApproxErf = -1;
  else ApproxErf = std::erf(arg);
  double leftArea = (1 + ApproxErf) * sqrtPiOver2;
  double rightArea = ( a * 1/std::pow(absAlpha - b,n-1)) / (n - 1);
  double area = leftArea + rightArea;
  if( t <= absAlpha ) {
    arg = t / sqrt2;
    if(arg > 5.) ApproxErf = 1;
    else if (arg < -5.) ApproxErf = -1;
    else ApproxErf = std::erf(arg);
    return norm * (1 + ApproxErf) * sqrtPiOver2 / area;
  }
  else {
    return norm * (leftArea + a * (1/std::pow(t-b,n-1) -  1/std::pow(absAlpha - b,n-1)) / (1 - n)) / area;
  }
}

} // namespace analysis
