/*! Definition of primitives for a text based input/output.
This file is part of https://github.com/hh-italian-group/TauMLTools. */

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

    static bool TryParse(const std::basic_string<CharT>& str, T& t, std::string& error_msg)
    {
        try {
            std::basic_istringstream<CharT> ss(str);
            ss >> std::boolalpha >> t;
            return !ss.fail();
        } catch(std::exception& e){
            error_msg = e.what();
        }
        return false;
    }
};

template<typename CharT>
struct StringIOImpl<std::basic_string<CharT>, CharT> {
    static std::basic_string<CharT> ToString(const std::basic_string<CharT>& str) { return str; }
    static bool TryParse(const std::basic_string<CharT>& str, std::basic_string<CharT>& t, std::string&)
    {
        t = str;
        return true;
    }
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
bool TryParse(const std::basic_string<CharT>& str, T& t, std::string& error_msg)
{
    return detail::StringIOImpl<typename std::remove_reference<T>::type, CharT>::TryParse(str, t, error_msg);
}

template<typename T, typename CharT>
bool TryParse(const std::basic_string<CharT>& str, T& t)
{
    std::string error_msg;
    return TryParse<T, CharT>(str, t, error_msg);
}

template<typename T, typename CharT>
T Parse(const std::basic_string<CharT>& str)
{
    T t;
    std::string error_msg;
    if(!TryParse(str, t, error_msg)) {
        std::ostringstream ss;
        ss << "Parse of string \"" << str << "\" to " << typeid(T).name() << " is failed.";
        if(!error_msg.empty())
            ss << " " << error_msg;
        throw exception(ss.str());
    }

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

std::string GetPathWithoutFileName(const std::string& file_name);
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

enum class LVectorRepr { PtEtaPhiM, PtEtaPhiE, PtEtaPhiME, PxPyPzE, PtPhi, PxPyPtPhi };

namespace detail {
    std::string LorentzVectorToString(double px, double py, double pz, double E, double mass, double pt, double eta,
                                      double phi, LVectorRepr repr, bool print_prefix);
}

template<typename LVector>
std::string LorentzVectorToString(const LVector& p4, LVectorRepr repr = LVectorRepr::PtEtaPhiME,
                                  bool print_prefix = true)
{
    return detail::LorentzVectorToString(p4.Px(), p4.Py(), p4.Pz(), p4.E(), p4.M(), p4.Pt(), p4.Eta(), p4.Phi(),
                                         repr, print_prefix);
}

} // namespace analysis
