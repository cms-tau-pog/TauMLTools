/*! Definition of primitives for a text based input/output.
This file is part of https://github.com/hh-italian-group/AnalysisTools. */

#pragma once

#include "exception.h"
#include <iomanip>
#include <unordered_set>
#include <boost/algorithm/string.hpp>

namespace analysis {

namespace detail {
template<typename T, typename CharT>
struct StringIOImpl {
    static std::basic_string<CharT> ToString(const T& t)
    {
        std::basic_ostringstream<CharT> ss;
        ss << std::boolalpha << t;
        return ss.str();
    }

    static bool TryParse(const std::basic_string<CharT>& str, T& t)
    {
        try {
            std::basic_istringstream<CharT> ss(str);
            ss >> std::boolalpha >> t;
            return !ss.fail();
        } catch(std::exception&) {}
        return false;
    }
};

template<typename CharT>
struct StringIOImpl<std::basic_string<CharT>, CharT> {
    static std::basic_string<CharT> ToString(const std::basic_string<CharT>& str) { return str; }
    static bool TryParse(const std::basic_string<CharT>& str, std::basic_string<CharT>& t) { t = str; return true; }
};

template<typename CharT, int N>
struct StringIOImpl<const CharT[N], CharT> {
    static std::basic_string<CharT> ToString(const CharT str[N]) { return str; }
};

} // namespace detail

template<typename T, typename CharT = char>
std::basic_string<CharT> ToString(T&& t)
{
    return detail::StringIOImpl<typename std::remove_reference<T>::type, CharT>::ToString(std::forward<T>(t));
}

template<typename T, typename CharT>
bool TryParse(const std::basic_string<CharT>& str, T& t)
{
    return detail::StringIOImpl<typename std::remove_reference<T>::type, CharT>::TryParse(str, t);
}

template<typename T, typename CharT>
T Parse(const std::basic_string<CharT>& str)
{
    T t;
    if(!TryParse(str, t))
        throw exception("Parse of string '%1%' to %2% is failed.") % str % typeid(T).name();
    return t;
}

template<typename T>
std::vector<std::string> ToStringVector(const std::vector<T>& v)
{
    std::vector<std::string> result;
    std::transform(v.begin(), v.end(), std::back_inserter(result),
                   [](const T& x) { std::ostringstream ss; ss << x; return ss.str(); });
    return result;
}
template<typename Collection>
static std::string CollectionToString(const Collection& col, const std::string& separator = ", ")
{
    std::ostringstream ss;
    auto iter = col.begin();
    if(iter != col.end())
        ss << *iter++;
    for(; iter != col.end(); ++iter)
        ss << separator << *iter;
    return ss.str();
}

inline std::vector<std::string> SplitValueList(std::string values_str, bool allow_duplicates = true,
                                               const std::string& separators = " \t",
                                               bool enable_token_compress = true)
{
    std::vector<std::string> result;
    if(enable_token_compress)
        boost::trim_if(values_str, boost::is_any_of(separators));
    if(!values_str.size()) return result;
    const auto token_compress = enable_token_compress ? boost::algorithm::token_compress_on
                                                      : boost::algorithm::token_compress_off;
    boost::split(result, values_str, boost::is_any_of(separators), token_compress);
    if(!allow_duplicates) {
        std::unordered_set<std::string> set_result;
        for(const std::string& value : result) {
            if(set_result.count(value))
                throw exception("Value '%1%' listed more than once in the value list '%2%'.") % value % values_str;
            set_result.insert(value);
        }
    }
    return result;
}

inline std::vector<std::string> ReadValueList(std::istream& stream, size_t number_of_items,
                                              bool allow_duplicates = true,
                                              const std::string& separators = " \t",
                                              bool enable_token_compress = true)
{
    const auto stream_exceptions = stream.exceptions();
    stream.exceptions(std::istream::goodbit);
    try {
        std::vector<std::string> result;
        std::unordered_set<std::string> set_result;
        const auto predicate = boost::is_any_of(separators);
        size_t n = 0;
        for(; n < number_of_items; ++n) {
            std::string value;
            while(true) {
                const auto c = stream.get();
                if(!stream.good()) {
                    if(stream.eof()) break;
                    throw exception("Failed to read values from stream.");
                }
                if(predicate(c)) {
                    if(!value.size() && enable_token_compress) continue;
                    break;
                }
                value.push_back(static_cast<char>(c));
            }
            if(!allow_duplicates && set_result.count(value))
                throw exception("Value '%1%' listed more than once in the input stream.") % value;
            result.push_back(value);
            set_result.insert(value);
        }
        if(n != number_of_items)
            throw exception("Expected %1% items, while read only %2%.") % number_of_items % n;

        stream.clear();
        stream.exceptions(stream_exceptions);
        return result;
    } catch(exception&) {
        stream.clear();
        stream.exceptions(stream_exceptions);
        throw;
    }
}

struct StVariable {
    using ValueType = double;
    static constexpr int max_precision = -std::numeric_limits<ValueType>::digits10;
    static constexpr int number_of_significant_digits_in_error = 2;

        ValueType value, error_up, error_low;

        StVariable() : value(0), error_up(0), error_low(0) {}
        StVariable(double _value, double _error_up, double _error_low = std::numeric_limits<double>::quiet_NaN()) :
            value(_value), error_up(_error_up), error_low(_error_low) {}

    int precision_up() const
    {
        return error_up != 0.
                ? static_cast<int>(std::floor(std::log10(error_up)) - number_of_significant_digits_in_error + 1)
                : max_precision;
    }

    int precision_low() const
    {
        return error_low != 0.
                ? static_cast<int>(std::floor(std::log10(error_low)) - number_of_significant_digits_in_error + 1)
                : max_precision;
    }

    int precision() const { return std::max(precision_up(), precision_low()); }

    int decimals_to_print_low() const { return std::max(0, -precision_low()); }
    int decimals_to_print_up() const { return std::max(0, -precision_up()); }
    int decimals_to_print() const { return std::min(decimals_to_print_low(), decimals_to_print_up()); }

    std::string ToLatexString() const
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
};

} // namespace analysis
