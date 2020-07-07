/*! Common math functions and definitions suitable for analysis purposes.
This file is part of https://github.com/hh-italian-group/TauMLTools. */

#include "TauMLTools/Core/interface/AnalysisMath.h"

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

namespace analysis {

StVariable::StVariable() : value(0), error_up(0), error_low(0) {}
StVariable::StVariable(double _value, double _error_up) :
    value(_value), error_up(_error_up), error_low(_error_up) {}
StVariable::StVariable(double _value, double _error_up, double _error_low) :
    value(_value), error_up(_error_up), error_low(_error_low) {}

double StVariable::error(int scale) const
{
    const double err = scale > 0 ? error_up : error_low;
    return err * std::abs(scale);
}

int StVariable::precision_up() const
{
    return error_up != 0.
            ? static_cast<int>(std::floor(std::log10(error_up)) - number_of_significant_digits_in_error + 1)
            : max_precision;
}

int StVariable::precision_low() const
{
    return error_low != 0.
            ? static_cast<int>(std::floor(std::log10(error_low)) - number_of_significant_digits_in_error + 1)
            : max_precision;
}

int StVariable::precision() const { return std::max(precision_up(), precision_low()); }

int StVariable::decimals_to_print_low() const { return std::max(0, -precision_low()); }
int StVariable::decimals_to_print_up() const { return std::max(0, -precision_up()); }
int StVariable::decimals_to_print() const { return std::min(decimals_to_print_low(), decimals_to_print_up()); }

std::string StVariable::ToLatexString() const
{
    const ValueType ten_pow_p = std::pow(10.0, precision());
    const ValueType value_rounded = std::round(value / ten_pow_p) * ten_pow_p;
    const ValueType error_up_rounded = std::ceil(error_up / ten_pow_p) * ten_pow_p;
    const ValueType error_low_rounded = std::ceil(error_low / ten_pow_p) * ten_pow_p;

    std::ostringstream ss;
    ss << std::setprecision(decimals_to_print()) << std::fixed;
    if(error_up == 0 && error_low == 0)
        ss << value_rounded<< "^{+0}_{-0}";
    else if(!std::isnan(error_low))
        ss << value_rounded<< "^{+" << error_up_rounded << "}_{-" << error_low_rounded << "}";
    else if(std::isnan(error_low)) {
        ss << value_rounded << " \\pm ";
        if(error_up == 0)
            ss << "0";
        else
            ss << error_up_rounded;
   }

   return ss.str();
}

bool Cut1D_Bound::operator() (double x) const
{
    if(std::isnan(value))
        throw exception("Cut1D: cut value is not set.");
    const double y = abs ? std::abs(x) : x;
    if(is_lower_bound)
        return equals_pass ? y >= value : y > value;
    return equals_pass ? y <= value : y < value;
}

Cut1D_Bound Cut1D_Bound::L(double lower, bool equals_pass)
{
    Cut1D_Bound cut;
    cut.value = lower;
    cut.abs = false;
    cut.is_lower_bound = true;
    cut.equals_pass = equals_pass;
    return cut;
}

Cut1D_Bound Cut1D_Bound::U(double upper, bool equals_pass)
{
    Cut1D_Bound cut;
    cut.value = upper;
    cut.abs = false;
    cut.is_lower_bound = false;
    cut.equals_pass = equals_pass;
    return cut;
}

Cut1D_Bound Cut1D_Bound::AbsL(double lower, bool equals_pass)
{
    Cut1D_Bound cut = L(lower, equals_pass);
    cut.abs = true;
    return cut;
}

Cut1D_Bound Cut1D_Bound::AbsU(double upper, bool equals_pass)
{
    Cut1D_Bound cut = U(upper, equals_pass);
    cut.abs = true;
    return cut;
}

Cut1D_Interval::Cut1D_Interval(const Cut1D_Bound& _lower, const Cut1D_Bound& _upper, bool _inverse)
        : lower(_lower), upper(_upper), inverse(_inverse)
{
}

bool Cut1D_Interval::operator() (ValueType x) const
{
    bool result = lower(x) && upper(x);
    if(inverse)
        result = !result;
    return result;
}

bool EllipseParameters::IsInside(double x, double y) const
{
    const double ellipse_cut = std::pow(x-x0, 2)/std::pow(r_x, 2)
                             + std::pow(y-y0, 2)/std::pow(r_y, 2);
    return ellipse_cut<1;
}


std::ostream& operator<<(std::ostream& os, const EllipseParameters& ellipseParams)
{
    os << ellipseParams.x0 << ellipseParams.r_x << ellipseParams.y0 << ellipseParams.r_y;
    return os;
}

std::istream& operator>>(std::istream& is, EllipseParameters& ellipseParams)
{
    is >> ellipseParams.x0 >> ellipseParams.r_x >> ellipseParams.y0 >> ellipseParams.r_y;
    return is;
}

PhysicalValue Integral(const TH1& histogram, bool include_overflows)
{
    using limit_pair = std::pair<Int_t, Int_t>;
    const limit_pair limits = include_overflows ? limit_pair(0, histogram.GetNbinsX() + 1)
                                                : limit_pair(1, histogram.GetNbinsX());

    double error = 0;
    const double integral = histogram.IntegralAndError(limits.first, limits.second, error);
    return PhysicalValue(integral, error);
}

PhysicalValue Integral(const TH1& histogram, Int_t first_bin, Int_t last_bin)
{
    double error = 0;
    const double integral = histogram.IntegralAndError(first_bin, last_bin, error);
    return PhysicalValue(integral, error);
}


PhysicalValue Integral(const TH2& histogram, bool include_overflows)
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

PhysicalValue Integral(const TH2& histogram, Int_t first_x_bin, Int_t last_x_bin,
                       Int_t first_y_bin, Int_t last_y_bin)
{
    double error = 0;
    const double integral = histogram.IntegralAndError(first_x_bin, last_x_bin, first_y_bin, last_y_bin, error);
    return PhysicalValue(integral, error);
}

double crystalball(double m, double m0, double sigma, double alpha, double n, double norm)
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
