/*! Definition of data driven statistical estimators.
This file is part of https://github.com/hh-italian-group/TauMLTools. */

#pragma once

#include <stdexcept>
#include <iomanip>
#include <sstream>
#include <cmath>
#include <limits>
#include <numeric>
#include <random>
#include <vector>
#include <boost/math/constants/constants.hpp>
#include <boost/math/special_functions/hermite.hpp>
#include <boost/math/tools/roots.hpp>
#include <boost/math/distributions.hpp>

namespace analysis {
namespace stat_estimators {

struct EstimatedQuantity {
    double value, unc_down, unc_up;
    explicit EstimatedQuantity(double _value = std::numeric_limits<double>::quiet_NaN(),
                               double _unc_down = std::numeric_limits<double>::quiet_NaN(),
                               double _unc_up = std::numeric_limits<double>::quiet_NaN()) :
        value(_value), unc_down(_unc_down), unc_up(_unc_up)
    {
        if(unc_up < 0 || unc_down < 0)
            throw std::runtime_error("Uncertainty of an estimated quantity should be a non-negative number.");
    }

    bool IsCompatible(const EstimatedQuantity& other) const
    {
        if(value < other.value)
            return other.value - value <= other.unc_down + unc_up;
        if(value > other.value)
            return value - other.value <= unc_down + other.unc_up;
        return value == other.value;
    }
};

inline std::ostream& operator<<(std::ostream& s, const EstimatedQuantity& q)
{
    static const int n_digits_in_unc = 2;
    const int precision_up = q.unc_up != 0
                           ? static_cast<int>(std::floor(std::log10(q.unc_up)) - n_digits_in_unc + 1) : -15;
    const int precision_down = q.unc_down != 0
                             ? static_cast<int>(std::floor(std::log10(q.unc_down)) - n_digits_in_unc + 1) : -15;
    const int precision = std::min(precision_up, precision_down);
    const double ten_pow_p = std::pow(10.0, precision);
    const double unc_up_rounded = std::ceil(q.unc_up / ten_pow_p) * ten_pow_p;
    const double unc_down_rounded = std::ceil(q.unc_down / ten_pow_p) * ten_pow_p;
    const double value_rounded = std::round(q.value / ten_pow_p) * ten_pow_p;
    const int decimals_to_print = std::max(0, -precision);
    std::ostringstream ss;
    ss << std::setprecision(decimals_to_print) << std::fixed << value_rounded;
    ss << " +/- " << unc_up_rounded;
    if(std::abs(unc_up_rounded - unc_down_rounded) >= ten_pow_p)
        ss << "/" << unc_down_rounded;
    s << ss.str();
    return s;
}

// Calculate minimal central interval that contains at least 1-quantile fraction of the observations.
template<typename Container, typename Value = typename Container::value_type>
std::pair<Value, Value> GetCentralConfidenceInterval(const Value& center, const Container& values,
                                                     double quantile = 0.31731)
{
    if(quantile <= 0 || quantile >= 1)
        throw std::runtime_error("Quantile value should be withhin (0, 1) interval.");
    const double q_min = std::min(quantile, 1 - quantile);
    const size_t n = values.size();
    if(std::floor(q_min * n) < 1)
        throw std::runtime_error("Number of obserations is too small to calculate CI with the given quantile.");
    std::vector<Value> v(values.begin(), values.end());
    std::sort(v.begin(), v.end());
    auto best_limits = std::make_pair(std::numeric_limits<Value>::lowest(), std::numeric_limits<Value>::max());
    size_t n_loss = std::floor(n * quantile);
    for(size_t left_loss = 0; left_loss <= n_loss; ++left_loss) {
        const size_t right_loss = n_loss - left_loss;
        const auto current_limits = std::make_pair(v.at(left_loss), v.at(n - right_loss - 1));
        if(center >= current_limits.first && center <= current_limits.second &&
                current_limits.second - current_limits.first < best_limits.second - best_limits.first)
            best_limits = current_limits;
    }
    return best_limits;
}


// Resample input, keeping sample size constant.
template<typename Value, typename RandomSource>
std::vector<std::vector<Value>> Resample(RandomSource& gen, const std::vector<const std::vector<Value>*>& values,
                                         bool allow_duplicates)
{
    if(!values.size() || values.front()->size() < 2)
        throw std::runtime_error("Can't resample a sample with less than 2 observations.");
    const size_t n = values.front()->size();
    for(const auto& var_sample : values) {
        if(!var_sample || var_sample->size() != n)
            throw std::runtime_error("Inconsistent number of observations in samples.");
    }

    size_t N = values.size();
    if(!allow_duplicates) {
        std::uniform_int_distribution<size_t> N_pdf(2, n);
        N = N_pdf(gen);
    }
    std::vector<std::vector<Value>> result(values.size());
    for(auto& var_result : result)
        var_result.reserve(N);

    std::vector<size_t> indexes(n);
    std::iota(indexes.begin(), indexes.end(), 0);
    for(size_t k = 0; k < N; ++k) {
        std::uniform_int_distribution<size_t> pdf(0, indexes.size() - 1);
        const size_t pos = pdf(gen);
        const size_t index = indexes.at(pos);
        if(!allow_duplicates)
            indexes.erase(indexes.begin() + pos);
        for(size_t v = 0; v < values.size(); ++v)
            result.at(v).push_back(values.at(v)->at(index));
    }
    return result;
}


// Evaluate the estimator and assess its errors by evaluating it N times on resampled intputs.
template<typename Value, typename Estimator = Value(const std::vector<Value>&, const std::vector<Value>&)>
EstimatedQuantity EstimateWithErrorsByResampling(Estimator estimator,
        const std::vector<Value>& x, const std::vector<Value>& y, bool simultaneous_resample, bool allow_duplicates,
        size_t n_trials = 1000, double quantile = 0.31731, unsigned seed = 123456)
{
    const Value central = estimator(x, y);
    std::vector<Value> trials;
    trials.reserve(n_trials);
    std::mt19937 gen(seed);
    const std::vector<const std::vector<Value>*> simult_inputs = { &x, &y }, x_input = { &x }, y_input = { &y };
    for(size_t n = 0; n < n_trials; ++n) {
        std::vector<std::vector<Value>> resampled;
        if(simultaneous_resample) {
            resampled = Resample(gen, simult_inputs, allow_duplicates);
        } else {
            resampled.push_back(Resample(gen, x_input, allow_duplicates).at(0));
            resampled.push_back(Resample(gen, y_input, allow_duplicates).at(0));
        }
        const auto resampled_estimate = estimator(resampled.at(0), resampled.at(1));
        trials.push_back(resampled_estimate);
    }

    const auto confidence_interval = GetCentralConfidenceInterval(central, trials, quantile);
    if(central < confidence_interval.first || central > confidence_interval.second)
        throw std::runtime_error("Central value of an estimator is outside of the confidence interval.");
    return EstimatedQuantity(central, central - confidence_interval.first, confidence_interval.second - central);
}

// Estimate var(X) based on sample of n independent observations.
template<typename Value>
double Variance(const std::vector<Value>& x)
{
    if(x.size() <= 1)
        throw std::runtime_error("Can't estimate variance using a sample with less than 2 observations.");
    const size_t n = x.size();
    const double x_mean = double(std::accumulate(x.begin(), x.end(), Value(0))) / n;
    double var = 0;
    for(size_t k = 0; k < n; ++k)
        var += std::pow(x.at(k) - x_mean, 2);
    var /= n - 1;
    return var;
}

// Estimate cov(X, Y) based on sample of n independent observations.
template<typename Value>
double Covariance(const std::vector<Value>& x, const std::vector<Value>& y)
{
    if(x.size() <= 1)
        throw std::runtime_error("Can't estimate covariance using a sample with less than 2 observations.");
    const size_t n = x.size();
    if(y.size() != n)
        throw std::runtime_error("Inconsistent number of observations in x and y samples.");
    const double x_mean = double(std::accumulate(x.begin(), x.end(), 0)) / n;
    const double y_mean = double(std::accumulate(y.begin(), y.end(), 0)) / n;
    double cov = 0;
    for(size_t k = 0; k < n; ++k)
        cov += (x.at(k) - x_mean) * (y.at(k) - y_mean);
    cov /= n - 1;
    return cov;
}

// Estimate corr(X, Y) based on sample of n independent observations.
template<typename Value>
double Correlation(const std::vector<Value>& x, const std::vector<Value>& y, double corr_limit = 1.)
{
    const double sigma_x = std::sqrt(Variance(x));
    const double sigma_y = std::sqrt(Variance(y));
    const double cov_xy = Covariance(x, y);
    const double corr_xy = cov_xy / (sigma_x * sigma_y);
    if(corr_xy <= -corr_limit) return -corr_limit;
    if(corr_xy >= corr_limit) return corr_limit;
    return corr_xy;
}

// Get interquartile range of the sample.
template<typename Container, typename Value = typename Container::value_type>
double InterquartileRange(const Container& sample, Value* min = nullptr, Value* max = nullptr)
{
    if(sample.size() <= 1)
        throw std::runtime_error("Can't compute interquartile range for a sample with less than 2 observations.");
    std::vector<Value> x(sample.begin(), sample.end());
    std::sort(x.begin(), x.end());
    if(min) *min = x.front();
    if(max) *max = x.back();
    const size_t n = x.size();
    const size_t r = n % 2, n2 = (n - r) / 2;
    const size_t r2 = n2 % 2, n4 = (n2 - r2) / 2;
    const double q1 = r2 ? x.at(n4) : (x.at(n4 - 1) + x.at(n4)) / 2.;
    const double q3 = r2 ? x.at(n2 + n4 + r) : (x.at(n2 + n4 - 1 + r) + x.at(n2 + n4 + r)) / 2.;
    return q3 - q1;
}

// Get optimal histogram bin size using Freedman-Diaconis rule (doi:10.1007/BF01025868).
template<typename Container, typename Value = typename Container::value_type>
double FreedmanDiaconisBinSize(const Container& sample, Value* min = nullptr, Value* max = nullptr)
{
    static constexpr double one_third = 1./3.;
    const double iqr = InterquartileRange(sample, min, max);
    return 2. * iqr / std::pow(sample.size(), one_third);
}

// Compute the probabilists' Hermite polynomials.
inline double HermitePolynomial(unsigned n, double x)
{
    return boost::math::hermite(n, x / std::sqrt(2.)) * std::pow(2., -(n/2.));
}

// Estimate optimal bandwith using the soleve-the-equation plug-in method using the algorithm defined in
// CS-TR-4774/UMIACS-TR-2005-73 (http://www.umiacs.umd.edu/labs/cvl/pirl/vikas/publications/CS-TR-4774.pdf).
template<typename Value>
double OptimalBandwith(const std::vector<Value>& x, double relative_tolerance = 0.01)
{
    static constexpr double sqrt_pi = boost::math::constants::root_pi<double>();
    static constexpr double sqrt_2pi = boost::math::constants::root_two_pi<double>();

    const size_t N = x.size();
    const double sigma = std::sqrt(Variance(x));
    const double Phi_6 = -15./(16. * sqrt_pi) * std::pow(sigma, -7);
    const double Phi_8 = 105. /(32. * sqrt_pi) * std::pow(sigma, -9);
    const double g1 = std::pow(-6. / (sqrt_2pi * Phi_6 * N), 1./7.);
    const double g2 = std::pow(30. / (sqrt_2pi * Phi_8 * N), 1./9.);

    const auto Phi_fn = [&](unsigned n, double y) {
        double result = 0;
        for(size_t i = 0; i < N; ++i) {
            for(size_t j = 0; j < N; ++j) {
                const double delta = (x.at(i) - x.at(j))/y;
                result += HermitePolynomial(n, delta) * std::exp(-std::pow(delta, 2) / 2.);
            }
        }
        result /= N * (N - 1) * sqrt_2pi * std::pow(y, n + 1);
        return result;
    };
    const double Phi4_g1 = Phi_fn(4, g1);
    const double Phi6_g2 = Phi_fn(6, g2);
    const double gamma_factor = std::pow(-6. * std::sqrt(2.) * Phi4_g1 / Phi6_g2, 1./7.);
    const auto gamma_fn = [&](double h) { return gamma_factor * std::pow(h, 5./7.); };
    const auto bandwith_fn = [&](double h) { return h - std::pow(2. * sqrt_pi * Phi_fn(4, gamma_fn(h)) * N, -1./5.); };
    const auto tolerance = [&](double h1, double h2) {
        return 2. * std::abs(h2 - h1) / (h2 + h1) < relative_tolerance;
    };
    static constexpr boost::uintmax_t max_iter = 100;
    boost::uintmax_t n_iter = max_iter;
    const auto optimal_h_interval =
            boost::math::tools::bracket_and_solve_root(bandwith_fn, sigma / 4., 2., true, tolerance, n_iter);
    if(n_iter >= max_iter)
        throw std::runtime_error("Unable to find optimal bandwith.");
    return (optimal_h_interval.first + optimal_h_interval.second) / 2.;
}

// Estimate the PDF value at the given point based on sample of n independent observations using the kernel density
// estimator (KDE), also known as the Parzen window estimator.
template<typename Container, typename Value = typename Container::value_type>
double pdf_kde(const Container& sample, const Value& point, double window_bandwidth)
{
    if(!sample.size())
        throw std::runtime_error("Can't compute PDF based on an empty sample.");
    if(window_bandwidth <= 0)
        throw std::runtime_error("Window bandwith should be a positive number.");
    const boost::math::normal_distribution<> G(0, window_bandwidth);
    double p = 0;
    for(const auto& x : sample)
        p += boost::math::pdf(G, point - x);
    p /= sample.size();
    return p;
}

// Estimate the PDF value at the given point based on sample of n independent observations using the KDE with the full
// Gaussian window.
template<typename Value>
double pdf_kde_2d(const std::pair<const std::vector<Value>*, const std::vector<Value>*>& sample,
                  const std::pair<Value, Value>& point, const std::pair<double, double>& window_bandwidth,
                  double correlation)
{
    if(!sample.first || !sample.second || !sample.first->size())
        throw std::runtime_error("Can't compute PDF based on an empty sample.");
    const size_t N = sample.first->size();
    if(sample.second->size() != N)
        throw std::runtime_error("Inconsistent number of observations in the sample.");
    if(window_bandwidth.first <= 0 || window_bandwidth.second <= 0)
        throw std::runtime_error("Window bandwith should be a positive number.");
    if(correlation <= -1 || correlation >= 1)
        throw std::runtime_error("Correlation should be between -1 and 1.");

    static constexpr double two_pi = boost::math::constants::two_pi<double>();
    const double G_norm = 1. / (two_pi * window_bandwidth.first * window_bandwidth.second
                                * std::sqrt(1. - std::pow(correlation, 2)));
    const double G_exp_norm = 1. / (2. * (1. - std::pow(correlation, 2)));
    const auto G_pdf = [&](const Value& x, const Value& y) {
        const double sum = std::pow(x / window_bandwidth.first, 2) + std::pow(y / window_bandwidth.second, 2)
                         - 2. * correlation * x * y / (window_bandwidth.first * window_bandwidth.second);
        return G_norm * std::exp(-G_exp_norm * sum);
    };

    double p = 0;
    for(size_t i = 0; i < N; ++i)
        p += G_pdf(point.first - sample.first->at(i), point.second - sample.second->at(i));
    p /= N;
    return p;
}

// Estimate Kullback–Leibler divergence for two samples of independent observations.
// Using estimator defined in doi:10.3390/e13071229.
// KDE is used for the PDF estimation.
template<typename Value>
double KullbackLeiblerDivergence(const std::vector<Value>& x, const std::vector<Value>& y,
                                 double window_bandwidth_x, double window_bandwidth_y)
{
    if(!x.size() || !y.size())
        throw std::runtime_error("Can't compute Kullback–Leibler divergence for empty samples.");

    double div = 0;
    for(const auto& x_i : x)
        div += std::log2(pdf_kde(x, x_i, window_bandwidth_x)) - std::log2(pdf_kde(y, x_i, window_bandwidth_y));
    div /= x.size();
    return div;
}

// Estimate Jeffrey’s divergence for two samples of independent observations.
// Using estimator defined in doi:10.3390/e13071229.
// KDE is used for the PDF estimation.
template<typename Value>
double JeffreyDivergence(const std::vector<Value>& x, const std::vector<Value>& y,
                         double window_bandwidth_x, double window_bandwidth_y)
{
    return KullbackLeiblerDivergence(x, y, window_bandwidth_x, window_bandwidth_y)
            + KullbackLeiblerDivergence(y, x, window_bandwidth_y, window_bandwidth_x);
}

// Estimate Jensen-Shannon divergence for two samples of independent observations.
// Using estimator defined in doi:10.3390/e13071229.
// KDE is used for the PDF estimation.
template<typename Value>
double JensenShannonDivergence(const std::vector<Value>& x, const std::vector<Value>& y,
                               double window_bandwidth_x, double window_bandwidth_y)
{
    if(!x.size() || !y.size())
        throw std::runtime_error("Can't compute Jensen-Shannon divergence for empty samples.");

    const auto kl_mod = [&](const std::vector<Value>& a, const std::vector<Value>& b, double w_a, double w_b) {
        double div = 0;
        for(const auto& a_i : a) {
            const double p_i = pdf_kde(a, a_i, w_a);
            const double q_i = pdf_kde(b, a_i, w_b);
            const double m_i = (p_i + q_i) / 2.;
            div += std::log2(p_i) - std::log2(m_i);
        }
        div /= a.size();
        return div;
    };

    const double div_xm = kl_mod(x, y, window_bandwidth_x, window_bandwidth_y);
    const double div_ym = kl_mod(y, x, window_bandwidth_y, window_bandwidth_x);
    return (div_xm + div_ym) / 2.;
}

// Estimate Jensen-Shannon divergence for 2D case.
// KDE is used for the PDF estimation.
template<typename Value>
double JensenShannonDivergence_2D(const std::pair<const std::vector<Value>*, const std::vector<Value>*>& x,
                                  const std::pair<const std::vector<Value>*, const std::vector<Value>*>& y,
                                  const std::pair<double, double>& window_bandwidth_x,
                                  const std::pair<double, double>& window_bandwidth_y)
{
    static constexpr double corr_limit = 0.99;

    if(!x.first || !x.second || !x.first->size() || !y.first || !y.second)
        throw std::runtime_error("Can't compute Jensen-Shannon divergence for empty samples.");
    if(x.second->size() != x.first->size() || y.second->size() != y.first->size())
        throw std::runtime_error("Inconsistent number of observations in samples.");

    const auto kl_mod = [&](const std::pair<const std::vector<Value>*, const std::vector<Value>*>& a,
                            const std::pair<const std::vector<Value>*, const std::vector<Value>*>& b,
                            const std::pair<double, double>& w_a, const std::pair<double, double>& w_b,
                            double corr_a, double corr_b) {
        double div = 0;
        for(size_t n = 0; n < a.first->size(); ++n) {
            const auto a_i = std::make_pair(a.first->at(n), a.second->at(n));
            const double p_i = pdf_kde_2d(a, a_i, w_a, corr_a);
            const double q_i = pdf_kde_2d(b, a_i, w_b, corr_b);
            const double m_i = (p_i + q_i) / 2.;
            div += std::log2(p_i) - std::log2(m_i);
        }
        div /= a.first->size();
        return div;
    };

    const double corr_x = Correlation(*x.first, *x.second, corr_limit);
    const double corr_y = Correlation(*y.first, *y.second, corr_limit);

    const double div_xm = kl_mod(x, y, window_bandwidth_x, window_bandwidth_y, corr_x, corr_y);
    const double div_ym = kl_mod(y, x, window_bandwidth_y, window_bandwidth_x, corr_y, corr_x);
    return (div_xm + div_ym) / 2.;
}

// Estimate Jensen-Shannon divergence for N-dimension case.
// KDE is used for the PDF estimation.
template<typename Value>
double JensenShannonDivergence_ND(std::vector<const std::vector<Value>*> x,
                                  std::vector<const std::vector<Value>*> y,
                                  std::vector<double> window_bandwidth_x,
                                  std::vector<double> window_bandwidth_y)
{
    const size_t dim = x.size();
    if(dim < 1 || dim > 2)
        throw std::runtime_error("Supported number of dimensions is 1 or 2.");

    if(y.size() != dim || window_bandwidth_x.size() != dim || window_bandwidth_y.size() != dim)
        throw std::runtime_error("Inconsistent number of dimensions.");
    for(size_t d = 0; d < dim; ++d) {
        if(!x.at(d) || !y.at(d))
            throw std::runtime_error("Can't compute Jensen-Shannon divergence for empty samples.");
    }

    if(dim == 1)
        return JensenShannonDivergence(*x.front(), *y.front(), window_bandwidth_x.front(), window_bandwidth_y.front());
    return JensenShannonDivergence_2D(std::make_pair(x.at(0), x.at(1)), std::make_pair(y.at(0), y.at(1)),
                                      std::make_pair(window_bandwidth_x.at(0), window_bandwidth_x.at(1)),
                                      std::make_pair(window_bandwidth_y.at(0), window_bandwidth_y.at(1)));
}

// Estimate the differential entropy (extention of Shannon entropy for continuous random variables).
// KDE is used for the PDF estimation.
template<typename Container, typename Value = typename Container::value_type>
double Entropy(const Container& x, double window_bandwidth)
{
    if(!x.size())
        throw std::runtime_error("Can't compute entropy for an empty sample.");

    double entropy = 0;
    for(const auto& x_i : x)
        entropy -= std::log2(pdf_kde(x, x_i, window_bandwidth));
    entropy /= x.size();
    return entropy;
}

// Estimate mutual information.
// KDE is used for the PDF estimation.
template<typename Value>
double MutualInformation(const std::vector<Value>& x, const std::vector<Value>& y,
                         double window_bandwidth_x, double window_bandwidth_y)
{
    if(x.size() <= 1)
        throw std::runtime_error("Can't estimate mutual information using a sample with less than 2 observations.");
    const size_t N = x.size();
    if(y.size() != N)
        throw std::runtime_error("Inconsistent number of observations in x and y samples.");

    const double corr_xy = Correlation(x, y);
    if(corr_xy <= -1 || corr_xy >= 1) return std::numeric_limits<Value>::infinity();

    const auto xy = std::make_pair(&x, &y);
    const auto w = std::make_pair(window_bandwidth_x, window_bandwidth_y);
    double I = 0;
    for(size_t i = 0; i < N; ++i) {
        const auto xy_i = std::make_pair(x.at(i), y.at(i));
        const double p_xy = pdf_kde_2d(xy, xy_i, w, corr_xy);
        const double p_x = pdf_kde(x, x.at(i), window_bandwidth_x);
        const double p_y = pdf_kde(y, y.at(i), window_bandwidth_y);
        I += std::log2(p_xy / (p_x * p_y));
    }
    I /= N;
    return I;
}

// Estimate 1 - I(X,Y) / max(H(X), H(Y)).
// KDE is used for the PDF estimation.
template<typename Value>
double ScaledMutualInformation(const std::vector<Value>& x, const std::vector<Value>& y,
                               double window_bandwidth_x, double window_bandwidth_y)
{
    const double H_X = Entropy(x, window_bandwidth_x);
    const double H_Y = Entropy(y, window_bandwidth_y);
    const double I_XY = MutualInformation(x, y, window_bandwidth_x, window_bandwidth_y);
    return 1 - I_XY / std::max(H_X, H_Y);
}

} // namespace analysis
} // namespace stat_estimators
