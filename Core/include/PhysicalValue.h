/*! Definition of the class that represent measured physical value.
This file is part of https://github.com/hh-italian-group/AnalysisTools. */

#pragma once

#include <map>
#include <cmath>
#include <iomanip>

#include "exception.h"
#include "Tools.h"

namespace analysis {
namespace detail {

template<typename char_type>
struct PhysicalValueErrorSeparator;

template<>
struct PhysicalValueErrorSeparator<char> {
    static const std::string& Get() { static const std::string sep = " +/- "; return sep; }
};

template<>
struct PhysicalValueErrorSeparator<wchar_t> {
    static const std::wstring& Get() { static const std::wstring sep = L" \u00B1 "; return sep; }
};

template<typename _ValueType>
class PhysicalValue {
public:
    using ValueType = _ValueType;
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
            const double weight =  1. / sqr(full_error);
            total_weight += weight;
            weighted_sum += value * PhysicalValue<ValueType>(weight);
        }
        return weighted_sum / PhysicalValue(total_weight);
    }

    PhysicalValue() : value(0), stat_error(0) {}

    explicit PhysicalValue(const ValueType& _value, const ValueType& _stat_error = 0)
        : value(_value), stat_error(_stat_error)
    {
        if(stat_error < 0)
            throw exception("Negative statistical error = %1%.") % stat_error;
    }

    PhysicalValue(const ValueType& _value, const ValueType& _stat_error, const SystematicMap& _systematic_uncertainties)
        : value(_value), stat_error(_stat_error), systematic_uncertainties(_systematic_uncertainties)
    {
        if(stat_error < 0)
            throw exception("Negative statistical error = %1%.") % stat_error;
        systematic_names = tools::collect_map_keys(systematic_uncertainties);
    }

    void AddSystematicUncertainty(const std::string& unc_name, const ValueType& unc_value, bool is_relative = true)
    {
        if(systematic_uncertainties.count(unc_name))
            throw exception("Uncertainty '%1%' is already defined for the current physical value.") % unc_name;
        systematic_names.insert(unc_name);
        const ValueType final_unc_value = is_relative ? value * unc_value : unc_value;
        systematic_uncertainties[unc_name] = final_unc_value;
    }

    const ValueType& GetValue() const { return value; }
    const ValueType& GetStatisticalError() const { return stat_error; }

    ValueType GetSystematicUncertainty(const std::string& unc_name) const
    {
        const auto iter = systematic_uncertainties.find(unc_name);
        if(iter == systematic_uncertainties.end())
            return 0;
        return iter->second;
    }

    ValueType GetFullSystematicUncertainty() const
    {
        ValueType unc = 0;
        for(const auto& unc_iter : systematic_uncertainties)
            unc += sqr(unc_iter.second);
        return std::sqrt(unc);
    }

    ValueType GetFullError() const
    {
        return std::sqrt( sqr(stat_error) + sqr(GetFullSystematicUncertainty()) );
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
        static const auto operation = [](const ValueType& my_value, const ValueType& other_value) -> ValueType {
            return my_value + other_value;
        };
        static const auto derivator = [](const ValueType& my_value, const ValueType& other_value) -> ValueType {
            return 1;
        };
        return ApplyBinaryOperation(other, operation, derivator, derivator);
    }

    PhysicalValue<ValueType>& operator+=(const PhysicalValue<ValueType>& other)
    {
        *this = (*this) + other;
        return *this;
    }

    PhysicalValue<ValueType> operator-(const PhysicalValue<ValueType>& other) const
    {
        static const auto operation = [](const ValueType& my_value, const ValueType& other_value) -> ValueType {
            return my_value - other_value;
        };
        static const auto first_derivator = [](const ValueType& my_value, const ValueType& other_value) -> ValueType {
            return 1;
        };
        static const auto second_derivator = [](const ValueType& my_value, const ValueType& other_value) -> ValueType {
            return -1;
        };
        return ApplyBinaryOperation(other, operation, first_derivator, second_derivator);
    }

    PhysicalValue<ValueType>& operator-=(const PhysicalValue<ValueType>& other)
    {
        *this = (*this) - other;
        return *this;
    }

    PhysicalValue<ValueType> operator*(const PhysicalValue<ValueType>& other) const
    {
        static const auto operation = [](const ValueType& my_value, const ValueType& other_value) -> ValueType {
            return my_value * other_value;
        };
        static const auto first_derivator = [](const ValueType& my_value, const ValueType& other_value) -> ValueType {
            return other_value;
        };
        static const auto second_derivator = [](const ValueType& my_value, const ValueType& other_value) -> ValueType {
            return my_value;
        };
        return ApplyBinaryOperation(other, operation, first_derivator, second_derivator);
    }

    PhysicalValue<ValueType>& operator*=(const PhysicalValue<ValueType>& other)
    {
        *this = (*this) * other;
        return *this;
    }

    PhysicalValue<ValueType> operator/(const PhysicalValue<ValueType>& other) const
    {
        static const auto operation = [](const ValueType& my_value, const ValueType& other_value) -> ValueType {
            return my_value / other_value;
        };
        static const auto first_derivator = [](const ValueType& my_value, const ValueType& other_value) -> ValueType {
            return 1 / other_value;
        };
        static const auto second_derivator = [](const ValueType& my_value, const ValueType& other_value) -> ValueType {
            return - my_value / sqr(other_value);
        };
        return ApplyBinaryOperation(other, operation, first_derivator, second_derivator);
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

        const int precision = stat_error
                ? std::floor(std::log10(stat_error)) - number_of_significant_digits_in_error + 1 : -15;
        const ValueType ten_pow_p = std::pow(10.0, precision);
        const ValueType stat_error_rounded = std::ceil(stat_error / ten_pow_p) * ten_pow_p;
        const ValueType value_rounded = std::round(value / ten_pow_p) * ten_pow_p;
        const int decimals_to_print = std::max(0, -precision);
        std::basic_ostringstream<char_type> ss;
        ss << std::setprecision(decimals_to_print) << std::fixed << value_rounded;
        if(print_stat_error)
            ss << detail::PhysicalValueErrorSeparator<char_type>::Get() << stat_error_rounded;
        if(print_syst_uncs && systematic_uncertainties.size()) {
            const ValueType full_syst_rounded = std::round(GetFullSystematicUncertainty() / ten_pow_p) * ten_pow_p;
            ss << transform(stat_suffix_str) << detail::PhysicalValueErrorSeparator<char_type>::Get()
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

    template<typename Operation, typename Derivator>
    PhysicalValue ApplyUnaryOperation(const Operation& operation, const Derivator& derivator) const
    {
        PhysicalValue new_value;
        new_value.value = operation(value);
        new_value.stat_error = Propagate(value, stat_error, false, derivator);
        new_value.systematic_names = systematic_names;
        for(const auto& unc_iter : systematic_uncertainties) {
            const ValueType new_unc = Propagate(value, unc_iter->second, true, derivator);
            new_value.systematic_uncertainties[unc_iter->first] = new_unc;
        }
        return new_value;
    }

    template<typename Operation, typename FirstDerivator, typename SecondDerivator>
    PhysicalValue ApplyBinaryOperation(const PhysicalValue<ValueType>& other, const Operation& operation,
                                       const FirstDerivator& first_derivator,
                                       const SecondDerivator& second_derivator) const
    {
        PhysicalValue new_value;
        new_value.value = operation(value, other.value);
        new_value.stat_error = Propagate(value, stat_error, other.value, other.stat_error, false,
                                         first_derivator, second_derivator);
        new_value.systematic_names = systematic_names;
        new_value.systematic_names.insert(other.systematic_names.begin(), other.systematic_names.end());
        for(const std::string& unc_name : new_value.systematic_names) {
            const ValueType new_unc = Propagate(value, GetSystematicUncertainty(unc_name), other.value,
                                                other.GetSystematicUncertainty(unc_name), true, first_derivator,
                                                second_derivator);
            new_value.systematic_uncertainties[unc_name] = new_unc;
        }
        return new_value;
    }

private:
    template<typename Derivator>
    static ValueType Propagate(const ValueType& value, const ValueType& error, bool correlated,
                               const Derivator& derivator)
    {
        const ValueType derivate = derivator(value);
        if(correlated)
            return derivate * error;
        return std::abs(derivate) * error;
    }

    template<typename FirstDerivator, typename SecondDerivator>
    static ValueType Propagate(const ValueType& first_value, const ValueType& first_error,
                               const ValueType& second_value, const ValueType& second_error,
                               bool correlated,
                               const FirstDerivator& first_derivator, const SecondDerivator& second_derivator)
    {
        const ValueType first_derivate = first_derivator(first_value, second_value);
        const ValueType second_derivate = second_derivator(first_value, second_value);
        const ValueType first_contribution = first_derivate * first_error;
        const ValueType second_contribution = second_derivate * second_error;
        if(correlated)
            return first_contribution + second_contribution;
        return std::sqrt(sqr(first_contribution) + sqr(second_contribution));
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
