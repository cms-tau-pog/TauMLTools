/*! Definition of the class that represent measured physical value.
This file is part of https://github.com/hh-italian-group/TauMLTools. */

#pragma once

#include <set>
#include <map>
#include <cmath>
#include <iomanip>

#include "exception.h"

namespace analysis {
namespace detail {

template<typename _ValueType>
class PhysicalValue {
public:
    using ValueType = _ValueType;
    using ValueTypeCR =
        typename std::conditional<std::is_fundamental<ValueType>::value, ValueType, const ValueType&>::type;
    using SystematicNameSet = std::set<std::string>;
    using SystematicMap = std::map<std::string, ValueType>;

    static const PhysicalValue<ValueType> Zero;
    static const PhysicalValue<ValueType> One;
    static const PhysicalValue<ValueType> Two;

    template<typename Collection = std::vector<PhysicalValue<ValueType>>>
    static PhysicalValue<ValueType> WeightedAverage(const Collection& values)
    {
        if(!values.size())
            throw exception("Can't calculate weighted average for an empty collection.");
        ValueType total_weight = 0;
        if (values.size() == 1)
            return *values.begin();
        PhysicalValue<ValueType> weighted_sum;
        for(const auto& value : values) {
            const ValueType full_error = value.GetFullError();
            if(!full_error)
                throw exception("Can't calculate weighted average. One of the errors equals zero.");
            const double weight =  1. / std::pow(full_error, 2);
            total_weight += weight;
            weighted_sum += value * PhysicalValue<ValueType>(weight);
        }
        return weighted_sum / PhysicalValue(total_weight);
    }

    PhysicalValue() : value(0), stat_error(0) {}
    explicit PhysicalValue(ValueTypeCR _value, ValueTypeCR _stat_error = 0)
        : value(_value), stat_error(_stat_error)
    {
        if(stat_error < 0)
            throw exception("Negative statistical error = %1%.") % stat_error;
    }

    PhysicalValue(ValueTypeCR _value, ValueTypeCR _stat_error, const SystematicMap& _systematic_uncertainties)
        : value(_value), stat_error(_stat_error), systematic_uncertainties(_systematic_uncertainties)
    {
        if(stat_error < 0)
            throw exception("Negative statistical error = %1%.") % stat_error;

        for(const auto& syst : systematic_uncertainties)
            systematic_names.insert(syst.first);
    }

    void AddSystematicUncertainty(const std::string& unc_name, ValueTypeCR unc_value, bool is_relative = true)
    {
        if(systematic_uncertainties.count(unc_name))
            throw exception("Uncertainty '%1%' is already defined for the current physical value.") % unc_name;
        systematic_names.insert(unc_name);
        const ValueType final_unc_value = is_relative ? value * unc_value : unc_value;
        systematic_uncertainties[unc_name] = final_unc_value;
    }

    ValueTypeCR GetValue() const { return value; }
    ValueTypeCR GetStatisticalError() const { return stat_error; }

    ValueType GetSystematicUncertainty(const std::string& unc_name) const
    {
        const auto iter = systematic_uncertainties.find(unc_name);
        if(iter == systematic_uncertainties.end())
            return 0;
        return iter->second;
    }
    const SystematicMap& GetSystematicUncertainties() const { return systematic_uncertainties; }
    ValueType GetFullSystematicUncertainty() const
    {
        ValueType unc = 0;
        for(const auto& unc_iter : systematic_uncertainties)
            unc += std::pow(unc_iter.second, 2);
        return std::sqrt(unc);
    }

    ValueType GetFullError() const
    {
        return std::hypot(stat_error, GetFullSystematicUncertainty());
    }

    ValueType GetRelativeStatisticalError() const { return stat_error / value; }

    ValueType GetRelativeSystematicUncertainty(const std::string& unc_name) const
    {
        return GetSystematicUncertainty(unc_name) / value;
    }

    ValueType GetRelativeFullSystematicUncertainty() const
    {
        return GetFullSystematicUncertainty() / value;
    }

    ValueType GetRelativeFullError() const { return GetFullError() / value; }

    ValueType Covariance(const PhysicalValue<ValueType>& other) const
    {
        ValueType cov = 0;
        SystematicNameSet all_systematic_names = systematic_names;
        all_systematic_names.insert(other.systematic_names.begin(), other.systematic_names.end());
        for(const std::string& unc_name : all_systematic_names) {
            cov += std::abs(GetSystematicUncertainty(unc_name) * other.GetSystematicUncertainty(unc_name));
        }
        return cov;
    }

    PhysicalValue<ValueType> operator+(const PhysicalValue<ValueType>& other) const
    {
        return ApplyBinaryOperation(other, value + other.value, 1, 1);
    }

    PhysicalValue<ValueType>& operator+=(const PhysicalValue<ValueType>& other)
    {
        *this = (*this) + other;
        return *this;
    }

    PhysicalValue<ValueType> operator-(const PhysicalValue<ValueType>& other) const
    {
        return ApplyBinaryOperation(other, value - other.value, 1, -1);
    }

    PhysicalValue<ValueType>& operator-=(const PhysicalValue<ValueType>& other)
    {
        *this = (*this) - other;
        return *this;
    }

    PhysicalValue<ValueType> operator*(const PhysicalValue<ValueType>& other) const
    {
        return ApplyBinaryOperation(other, value * other.value, other.value, value);
    }

    PhysicalValue<ValueType>& operator*=(const PhysicalValue<ValueType>& other)
    {
        *this = (*this) * other;
        return *this;
    }

    PhysicalValue<ValueType> operator/(const PhysicalValue<ValueType>& other) const
    {
        return ApplyBinaryOperation(other, value / other.value, 1 / other.value, - value / std::pow(other.value, 2));
    }

    PhysicalValue<ValueType>& operator/=(const PhysicalValue<ValueType>& other)
    {
        *this = (*this) / other;
        return *this;
    }

    bool operator<(const PhysicalValue<ValueType>& other) const { return value < other.value; }
    bool operator<=(const PhysicalValue<ValueType>& other) const { return value <= other.value; }
    bool operator>(const PhysicalValue<ValueType>& other) const { return value > other.value; }
    bool operator>=(const PhysicalValue<ValueType>& other) const { return value >= other.value; }

    bool IsCompatible(const PhysicalValue<ValueType>& other) const
    {
        const PhysicalValue<ValueType> delta = (*this) - other;
        return std::abs(delta.GetValue()) < delta.GetFullError();
    }

    template<typename char_type>
    std::basic_string<char_type> ToString(bool print_stat_error, bool print_syst_uncs) const
    {
        static const int number_of_significant_digits_in_error = 2;
        static const std::string stat_suffix_str = " (stat.)";
        static const std::string syst_prefix_str = " (syst: ";
        static const std::string syst_equal_str = "=";
        static const std::string syst_separator_str = ", ";
        static const std::string syst_suffix_str = ")";

        static const auto transform = [](const std::string& str) -> std::basic_string<char_type> {
            return std::basic_string<char_type>(str.begin(), str.end());
        };

        static const auto errorSeparators = std::make_tuple(std::string(" +/- "),  std::wstring(L" \u00B1 "));

        const int precision = stat_error != 0.
                ? static_cast<int>(std::floor(std::log10(stat_error)) - number_of_significant_digits_in_error + 1)
                : -15;
        const ValueType ten_pow_p = std::pow(10.0, precision);
        const ValueType stat_error_rounded = std::ceil(stat_error / ten_pow_p) * ten_pow_p;
        const ValueType value_rounded = std::round(value / ten_pow_p) * ten_pow_p;
        const int decimals_to_print = std::max(0, -precision);
        std::basic_ostringstream<char_type> ss;
        ss << std::setprecision(decimals_to_print) << std::fixed << value_rounded;
        if(print_stat_error)
            ss << std::get<std::basic_string<char_type>>(errorSeparators) << stat_error_rounded;
        if(print_syst_uncs && systematic_uncertainties.size()) {
            const ValueType full_syst_rounded = std::round(GetFullSystematicUncertainty() / ten_pow_p) * ten_pow_p;
            ss << transform(stat_suffix_str) << std::get<std::basic_string<char_type>>(errorSeparators)
               << full_syst_rounded << transform(syst_prefix_str);
            bool is_first = true;
            for(const auto& unc_iter : systematic_uncertainties) {
                const ValueType unc_rounded = std::round(unc_iter.second / ten_pow_p) * ten_pow_p;
                if(!is_first)
                    ss << transform(syst_separator_str);
                ss << transform(unc_iter.first) << transform(syst_equal_str) << unc_rounded;
                is_first = false;
            }
            ss << transform(syst_suffix_str);
        }
        return ss.str();
    }

    PhysicalValue ApplyUnaryOperation(ValueTypeCR op_result, ValueTypeCR derivate) const
    {
        PhysicalValue new_value(op_result);
        new_value.stat_error = Propagate(derivate, stat_error, false);
        new_value.systematic_names = systematic_names;
        for(const std::string& unc_name : new_value.systematic_names) {
            const ValueType new_unc = Propagate(derivate, GetSystematicUncertainty(unc_name), true);
            new_value.systematic_uncertainties[unc_name] = new_unc;
        }
        return new_value;
    }

    PhysicalValue ApplyBinaryOperation(const PhysicalValue<ValueType>& other, ValueTypeCR op_result,
                                       ValueTypeCR first_derivate, ValueTypeCR second_derivate) const
    {
        PhysicalValue new_value(op_result);
        new_value.stat_error = Propagate(first_derivate, stat_error, second_derivate, other.stat_error, false);
        new_value.systematic_names = systematic_names;
        new_value.systematic_names.insert(other.systematic_names.begin(), other.systematic_names.end());
        for(const std::string& unc_name : new_value.systematic_names) {
            const ValueType new_unc = Propagate(first_derivate, GetSystematicUncertainty(unc_name),
                                                second_derivate, other.GetSystematicUncertainty(unc_name), true);
            new_value.systematic_uncertainties[unc_name] = new_unc;
        }
        return new_value;
    }

private:
    static ValueType Propagate(ValueTypeCR derivate, ValueTypeCR error, bool correlated)
    {
        return correlated ? derivate * error : std::abs(derivate) * error;
    }

    static ValueType Propagate(ValueTypeCR first_derivate, ValueTypeCR first_error,
                               ValueTypeCR second_derivate, ValueTypeCR second_error, bool correlated)
    {
        const ValueType first_contribution = first_derivate * first_error;
        const ValueType second_contribution = second_derivate * second_error;
        return correlated ? first_contribution + second_contribution
                          : std::hypot(first_contribution, second_contribution);
    }

private:
    double value;
    double stat_error;
    SystematicNameSet systematic_names;
    SystematicMap systematic_uncertainties;
};

template<typename ValueType>
const PhysicalValue<ValueType> PhysicalValue<ValueType>::Zero(0);

template<typename ValueType>
const PhysicalValue<ValueType> PhysicalValue<ValueType>::One(1);

template<typename ValueType>
const PhysicalValue<ValueType> PhysicalValue<ValueType>::Two(2);

template<typename ValueType>
std::ostream& operator<<(std::ostream& s, const PhysicalValue<ValueType>& v)
{
    s << v.template ToString<char>(true, true);
    return s;
}

template<typename ValueType>
std::wostream& operator<<(std::wostream& s, const PhysicalValue<ValueType>& v)
{
    s << v.template ToString<wchar_t>(true, true);
    return s;
}

template<typename T>
std::istream& operator>>(std::istream& s, PhysicalValue<T>& r)
{
    T value, error;
    s >> value >> error;
    if(s.fail())
        throw exception("Invalid Physical Value.");
    r = PhysicalValue<T>(value, error);
    return s;
}

} // namespace detail

using PhysicalValue = detail::PhysicalValue<double>;

} // namespace analysis

namespace std {

template<typename ValueType>
::analysis::detail::PhysicalValue<ValueType> abs(const ::analysis::detail::PhysicalValue<ValueType>& v)
{
    return analysis::detail::PhysicalValue<ValueType>(
                std::abs(v.GetValue()), v.GetStatisticalError(), v.GetSystematicUncertainties());
}

template<typename ValueType>
::analysis::detail::PhysicalValue<ValueType> sqrt(const ::analysis::detail::PhysicalValue<ValueType>& v)
{
    const auto sqrt = std::sqrt(v.GetValue());
    return v.ApplyUnaryOperation(sqrt, 0.5 / sqrt);
}

template<typename ValueType>
::analysis::detail::PhysicalValue<ValueType> exp(const ::analysis::detail::PhysicalValue<ValueType>& v)
{
    const auto exp = std::exp(v.GetValue());
    return v.ApplyUnaryOperation(exp, exp);
}

template<typename ValueType>
::analysis::detail::PhysicalValue<ValueType> log(const ::analysis::detail::PhysicalValue<ValueType>& v)
{
    return v.ApplyUnaryOperation(std::log(v.GetValue()), 1. / v.GetValue());
}

template<typename ValueType, typename ExpType>
::analysis::detail::PhysicalValue<ValueType> pow(const ::analysis::detail::PhysicalValue<ValueType>& v, ExpType&& exp)
{
    const auto derivate = exp != 0 ? exp * std::pow(v.GetValue(), exp - 1) : ValueType(0);
    return v.ApplyUnaryOperation(std::pow(v.GetValue(), exp), derivate);
}

}
