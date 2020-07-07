/*! Common math functions and definitions suitable for analysis purposes.
This file is part of https://github.com/hh-italian-group/TauMLTools. */

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
#include "PhysicalValue.h"

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

struct StVariable {
    using ValueType = double;
    static constexpr int max_precision = -std::numeric_limits<ValueType>::digits10;
    static constexpr int number_of_significant_digits_in_error = 2;

    ValueType value, error_up, error_low;

    StVariable();
    StVariable(double _value, double _error_up);
    StVariable(double _value, double _error_up, double _error_low);

    double error(int scale = 1) const;
    int precision_up() const;
    int precision_low() const;
    int precision() const;

    int decimals_to_print_low() const;
    int decimals_to_print_up() const;
    int decimals_to_print() const;

    std::string ToLatexString() const;
};

struct Cut1D {
    using ValueType = double;
    Cut1D() = default;
    Cut1D(Cut1D&&) = default;
    Cut1D(const Cut1D&) = default;
    virtual bool operator() (ValueType x) const = 0;
    virtual ~Cut1D(){}
};

struct Cut1D_Bound : Cut1D {
    ValueType value{std::numeric_limits<ValueType>::quiet_NaN()};
    bool abs{false}, is_lower_bound{false}, equals_pass{false};
    bool operator() (ValueType x) const override;
    static Cut1D_Bound L(ValueType lower, bool equals_pass = false);
    static Cut1D_Bound U(ValueType upper, bool equals_pass = false);
    static Cut1D_Bound AbsL(ValueType lower, bool equals_pass = false);
    static Cut1D_Bound AbsU(ValueType upper, bool equals_pass = false);
};

struct Cut1D_Interval  : Cut1D {
    Cut1D_Bound lower, upper;
    bool inverse;
    Cut1D_Interval(const Cut1D_Bound& _lower, const Cut1D_Bound& _upper, bool _inverse = false);
    bool operator() (ValueType x) const;
};

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

template<typename LVector>
double Calculate_min_dR_lj(const LVector& t1, const LVector& t2, const LVector& b1, const LVector& b2)
{
    const std::vector<LVector> taus = {t1, t2};
    const std::vector<LVector> jets = {b1, b2};
    std::vector<double> dR;

    for(size_t jet_index = 0; jet_index < taus.size(); ++jet_index) {
        for(size_t lep_index = 0; lep_index < jets.size(); ++lep_index)
            dR.push_back(ROOT::Math::VectorUtil::DeltaR(taus.at(lep_index), jets.at(jet_index)));
    }
    auto min_dR = *std::min_element(dR.begin(), dR.end());
    return min_dR;
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

    bool IsInside(double x, double y) const;
};

std::ostream& operator<<(std::ostream& os, const EllipseParameters& ellipseParams);
std::istream& operator>>(std::istream& is, EllipseParameters& ellipseParams);

PhysicalValue Integral(const TH1& histogram, bool include_overflows = true);
PhysicalValue Integral(const TH1& histogram, Int_t first_bin, Int_t last_bin);
PhysicalValue Integral(const TH2& histogram, bool include_overflows = true);
PhysicalValue Integral(const TH2& histogram, Int_t first_x_bin, Int_t last_x_bin,
                       Int_t first_y_bin, Int_t last_y_bin);

template<typename Histogram>
inline void RenormalizeHistogram(Histogram& histogram, double norm, bool include_overflows = true)
{
    const double integral = Integral(histogram,include_overflows).GetValue();
    if (integral == 0)
        throw analysis::exception("Integral is zero.");
    histogram.Scale(norm / integral);
}

double crystalball(double m, double m0, double sigma, double alpha, double n, double norm);

} // namespace analysis
