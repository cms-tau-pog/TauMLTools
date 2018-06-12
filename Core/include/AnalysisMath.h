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
#include <TH2.h>
#include <TMatrixD.h>
#include <TLorentzVector.h>
#include <boost/math/constants/constants.hpp>
#include <boost/math/special_functions/erf.hpp>


#include "PhysicalValue.h"

extern template TMatrixT<double>::TMatrixT(int, int);

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


template<typename Iterator>
double Calculate_HT(Iterator begin, const Iterator& end){
    double sum = 0;
    for(; begin != end; ++begin)
        sum += begin->pt();
    return sum;
}


namespace four_bodies{

template<typename LVector1, typename LVector2, typename LVector3, typename LVector4, typename LVector5 >
std::pair<double, double> Calculate_topPairMasses(const LVector1& lepton1_p4, const LVector2& lepton2_p4, const LVector3& bjet_1, const LVector4& bjet_2, const LVector5& met_p4){
    static constexpr double mass_top = 172.5;
    std::vector<std::pair<double, double>> vector_mass_top = {
        { (lepton1_p4 + bjet_1 + met_p4).mass(), (lepton2_p4 + bjet_2).mass() },
        { (lepton1_p4 + bjet_1).mass(), (lepton2_p4 + bjet_2 + met_p4).mass() },
        { (lepton1_p4 + bjet_2 + met_p4).mass(), (lepton2_p4 + bjet_1).mass() },
        { (lepton1_p4 + bjet_2).mass(), (lepton2_p4 + bjet_1 + met_p4).mass() }
    };
    std::vector<std::pair<size_t, double>> distance;
    for (size_t i = 0; i < vector_mass_top.size(); ++i) {
        distance.emplace_back(i, pow(vector_mass_top[i].first - mass_top,2)
                              + pow(vector_mass_top[i].second - mass_top,2));
    }
    std::sort(distance.begin(), distance.end(), [](const std::pair<size_t, double>& el1,const std::pair<size_t, double>& el2){
        return el1.second < el2.second;
    });
    return vector_mass_top.at(distance.front().first);
}


// dR between the two final state particle in the h rest frame
template<typename LVector1, typename LVector2, typename LVector3 >
double Calculate_dR_boosted(const LVector1& particle_1, const LVector2& particle_2, const LVector3& h){
    const auto boosted_1 = ROOT::Math::VectorUtil::boost(particle_1, h.BoostToCM());
    const auto boosted_2 = ROOT::Math::VectorUtil::boost(particle_2, h.BoostToCM());
    return ROOT::Math::VectorUtil::DeltaR(boosted_1, boosted_2);
}

//angle between the decay planes of the four final state elements expressed in the hh rest frame
template<typename LVector1, typename LVector2, typename LVector3, typename LVector4, typename LVector5,  typename LVector6>
double Calculate_phi(const LVector1& lepton1, const LVector2& lepton2, const LVector3& bjet1, const LVector4& bjet2, const LVector5& ll, const LVector6& bb){
    const auto H = bb + ll;
    const auto boosted_l1 = ROOT::Math::VectorUtil::boost(lepton1, H.BoostToCM());
    const auto boosted_l2 = ROOT::Math::VectorUtil::boost(lepton2, H.BoostToCM());
    const auto boosted_j1 = ROOT::Math::VectorUtil::boost(bjet1, H.BoostToCM());
    const auto boosted_j2 = ROOT::Math::VectorUtil::boost(bjet2, H.BoostToCM());
    const auto n1 = boosted_l1.Vect().Cross(boosted_l2.Vect());
    const auto n2 = boosted_j1.Vect().Cross(boosted_j2.Vect());
    return ROOT::Math::VectorUtil::Angle(n1, n2);
}


// Cosin of the production angle between the h  and the parton axis defined in the hh rest frame
template<typename LVector1, typename LVector2>
double  Calculate_cosThetaStar(const LVector1& h1, const LVector2& h2){
    const auto H = h2 + h1;
    const auto boosted_h1 = ROOT::Math::VectorUtil::boost(h1, H.BoostToCM());
    return ROOT::Math::VectorUtil::CosTheta(boosted_h1, ROOT::Math::Cartesian3D<>(0, 0, 1));
}

template<typename LVector1, typename LVector2, typename LVector3, typename LVector4>
double Calculate_phi1(const LVector1& object1, const LVector2& object2, const LVector3& ll, const LVector4& bb){
    const auto H = bb + ll;
    const auto boosted_1 = ROOT::Math::VectorUtil::boost(object1, H.BoostToCM());
    const auto boosted_2 = ROOT::Math::VectorUtil::boost(object2, H.BoostToCM());
    const auto boosted_h = ROOT::Math::VectorUtil::boost(ll, H.BoostToCM());
    ROOT::Math::Cartesian3D<> z_axis(0, 0, 1);
    const auto n1 = boosted_1.Vect().Cross(boosted_2.Vect());
    const auto n3 = boosted_h.Vect().Cross(z_axis);
    return ROOT::Math::VectorUtil::Angle(n1,n3);
}


//Cosin of theta angle between the first final state particle and the direction of flight of h in the h rest frame
template<typename LVector1, typename LVector2>
double Calculate_cosTheta_2bodies(const LVector1& object1, const LVector2&  hh){
    const auto boosted_object1 = ROOT::Math::VectorUtil::boost(object1, hh.BoostToCM());
    return  ROOT::Math::VectorUtil::CosTheta(boosted_object1, hh);
}

template<typename LVector1, typename LVector2, typename LVector3, typename LVector4, typename LVector5>
double Calculate_MX(const LVector1& lepton1, const LVector2& lepton2, const LVector3& bjet1, const LVector4& bjet2, const LVector5& met){

    static constexpr double shift = 250.;
    auto mass_4 = (lepton1 + lepton2 + met + bjet1 + bjet2).M();
    auto mass_ll =  (lepton1 + lepton2 +  met).M();
    auto mass_bb =  (bjet1 +  bjet2).M();
    return mass_4 - mass_ll - mass_bb + shift;
}

}


struct EllipseParameters {

    double x0{0.0};
    double r_x{0.0};
    double y0{0.0};
    double r_y{0.0};

    bool IsInside(double x, double y) const
    {
        const double ellipse_cut = std::pow(x-x0, 2)/std::pow(r_x, 2)
                                 + std::pow(y-y0, 2)/std::pow(r_y, 2);
        return ellipse_cut<1;
    }
};


inline std::ostream& operator<<(std::ostream& os, const EllipseParameters& ellipseParams)
{
    os << ellipseParams.x0 << ellipseParams.r_x << ellipseParams.y0 << ellipseParams.r_y;
    return os;
}

inline std::istream& operator>>(std::istream& is, EllipseParameters& ellipseParams)
{
    is >> ellipseParams.x0 >> ellipseParams.r_x >> ellipseParams.y0 >> ellipseParams.r_y;
    return is;
}

inline PhysicalValue Integral(const TH1& histogram, bool include_overflows = true)
{
    using limit_pair = std::pair<Int_t, Int_t>;
    const limit_pair limits = include_overflows ? limit_pair(0, histogram.GetNbinsX() + 1)
                                                : limit_pair(1, histogram.GetNbinsX());

    double error = 0;
    const double integral = histogram.IntegralAndError(limits.first, limits.second, error);
    return PhysicalValue(integral, error);
}

inline PhysicalValue Integral(const TH1& histogram, Int_t first_bin, Int_t last_bin)
{
    double error = 0;
    const double integral = histogram.IntegralAndError(first_bin, last_bin, error);
    return PhysicalValue(integral, error);
}


inline PhysicalValue Integral(const TH2& histogram, bool include_overflows = true)
{
    using limit_pair = std::pair<Int_t, Int_t>;
    const limit_pair limits_x = include_overflows ? limit_pair(0, histogram.GetNbinsX() + 1)
                                                : limit_pair(1, histogram.GetNbinsX());
    const limit_pair limits_y = include_overflows ? limit_pair(0, histogram.GetNbinsY() + 1)
                                                : limit_pair(1, histogram.GetNbinsY());

    double error = 0;
    const double integral = histogram.IntegralAndError(limits_x.first, limits_x.second, limits_y.first, limits_y.second,
                                                       error);
    return PhysicalValue(integral, error);
}

inline PhysicalValue Integral(const TH2& histogram, Int_t first_x_bin, Int_t last_x_bin,
                              Int_t first_y_bin, Int_t last_y_bin)
{
    double error = 0;
    const double integral = histogram.IntegralAndError(first_x_bin, last_x_bin, first_y_bin, last_y_bin, error);
    return PhysicalValue(integral, error);
}

template<typename Histogram>
inline void RenormalizeHistogram(Histogram& histogram, double norm, bool include_overflows = true)
{
    const double integral = Integral(histogram,include_overflows).GetValue();
    if (integral == 0)
        throw analysis::exception("Integral is zero.");
    histogram.Scale(norm / integral);
}

inline double crystalball(double m, double m0, double sigma, double alpha, double n, double norm)
{
    if(m<1. || 1000.<m)
        throw analysis::exception("The parameter m is out of range");

    static constexpr double sqrtPiOver2 = boost::math::constants::root_half_pi<double>();
    static constexpr double sqrt2 = boost::math::constants::root_two<double>();
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
    else ApproxErf = boost::math::erf(arg);
    double leftArea = (1 + ApproxErf) * sqrtPiOver2;
    double rightArea = ( a * 1/std::pow(absAlpha - b,n-1)) / (n - 1);
    double area = leftArea + rightArea;
    if( t <= absAlpha ) {
        arg = t / sqrt2;
        if(arg > 5.) ApproxErf = 1;
        else if (arg < -5.) ApproxErf = -1;
        else ApproxErf = boost::math::erf(arg);
        return norm * (1 + ApproxErf) * sqrtPiOver2 / area;
    }
    else {
        return norm * (leftArea + a * (1/std::pow(t-b,n-1) -  1/std::pow(absAlpha - b,n-1)) / (1 - n)) / area;
    }
}


} // namespace analysis
