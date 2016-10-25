/*! Definition of primitives for a text based input/output.
This file is part of https://github.com/hh-italian-group/AnalysisTools. */

#pragma once

#include "exception.h"

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

} // namespace analysis
