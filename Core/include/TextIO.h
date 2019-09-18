/*! Definition of primitives for a text based input/output.
This file is part of https://github.com/hh-italian-group/AnalysisTools. */

#pragma once

#include "exception.h"
#include <iomanip>
#include <unordered_set>
#include <boost/algorithm/string.hpp>
#include <boost/filesystem/convenience.hpp>

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
std::string CollectionToString(const Collection& col, const std::string& separator = ", ")
{
    std::ostringstream ss;
    auto iter = col.begin();
    if(iter != col.end())
        ss << *iter++;
    for(; iter != col.end(); ++iter)
        ss << separator << *iter;
    return ss.str();
}

std::string RemoveFileExtension(const std::string& file_name);
std::string GetFileNameWithoutPath(const std::string& file_name);

std::vector<std::string> SplitValueList(const std::string& values_str, bool allow_duplicates = true,
                                        const std::string& separators = " \t",
                                        bool enable_token_compress = true);

template<typename T, typename Collection=std::vector<T>>
Collection SplitValueListT(const std::string& values_str, bool allow_duplicates = true,
                                        const std::string& separators = " \t",
                                        bool enable_token_compress = true)
{
    std::vector<std::string> list = SplitValueList(values_str,allow_duplicates,separators,enable_token_compress);
    Collection collection;
    std::transform(list.begin(), list.end(), std::inserter(collection, collection.end()), [](const std::string& str) { return Parse<T>(str);});
    return collection;
}

std::vector<std::string> ReadValueList(std::istream& stream, size_t number_of_items,
                                       bool allow_duplicates = true,
                                       const std::string& separators = " \t",
                                       bool enable_token_compress = true);

struct StVariable {
    using ValueType = double;
    static constexpr int max_precision = -std::numeric_limits<ValueType>::digits10;
    static constexpr int number_of_significant_digits_in_error = 2;

    ValueType value, error_up, error_low;

    StVariable();
    StVariable(double _value, double _error_up, double _error_low = std::numeric_limits<double>::quiet_NaN());

    int precision_up() const;
    int precision_low() const;
    int precision() const;

    int decimals_to_print_low() const;
    int decimals_to_print_up() const;
    int decimals_to_print() const;

    std::string ToLatexString() const;
};

} // namespace analysis
