/*! Definition of primitives for a text based input/output.
This file is part of https://github.com/hh-italian-group/AnalysisTools. */

#pragma once

#include "exception.h"
#include <unordered_set>
#include <boost/algorithm/string.hpp>

namespace analysis {

template<typename T, typename CharT = char>
std::basic_string<CharT> ToString(const T& t)
{
    std::basic_ostringstream<CharT> ss;
    ss << t;
    return ss.str();
}

template<typename T, typename CharT>
bool TryParse(const std::basic_string<CharT>& str, T& t)
{
    try {
        std::basic_istringstream<CharT> ss(str);
        ss >> t;
        return !ss.fail();
    } catch(std::exception&) {}
    return false;
}

template<typename T, typename CharT>
T Parse(const std::basic_string<CharT>& str)
{
    T t;
    std::basic_istringstream<CharT> ss(str);
    ss >> t;
    if(ss.fail())
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
                value.push_back(c);
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

} // namespace analysis
