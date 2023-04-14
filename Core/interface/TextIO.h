/*! Definition of primitives for a text based input/output.
This file is part of https://github.com/cms-tau-pog/TauMLTools. */

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

inline std::string GetPathWithoutFileName(const std::string& file_name)
{
  const size_t lastindex = file_name.find_last_of("/");

  if(lastindex == std::string::npos)
    return "./";
  else
    return file_name.substr(0,lastindex);
}

inline std::string RemoveFileExtension(const std::string& file_name)
{
  return boost::filesystem::change_extension(file_name, "").string();
}

inline std::string GetFileNameWithoutPath(const std::string& file_name)
{
  const size_t lastindex = file_name.find_last_of("/");
  if(lastindex == std::string::npos)
    return file_name;
  else
    return file_name.substr(lastindex+1);
}

inline std::vector<std::string> SplitValueList(const std::string& _values_str, bool allow_duplicates,
                                               const std::string& separators, bool enable_token_compress=true)
{
  std::string values_str = _values_str;
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

template<typename T, typename Collection=std::vector<T>>
Collection SplitValueListT(const std::string& values_str, bool allow_duplicates = true,
                           const std::string& separators = " \t",
                           bool enable_token_compress = true)
{
  std::vector<std::string> list = SplitValueList(values_str,allow_duplicates,separators,enable_token_compress);
  Collection collection;
  std::transform(list.begin(), list.end(), std::inserter(collection, collection.end()),
                 [](const std::string& str) { return Parse<T>(str);});
  return collection;
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

enum class LVectorRepr { PtEtaPhiM, PtEtaPhiE, PtEtaPhiME, PxPyPzE, PtPhi, PxPyPtPhi };

namespace detail {
  inline std::string LorentzVectorToString(double px, double py, double pz, double E, double mass,
                                           double pt, double eta, double phi, LVectorRepr repr, bool print_prefix)
  {
    static const std::map<LVectorRepr, std::string> prefix = {
        { LVectorRepr::PtEtaPhiM, "pt, eta, phi, m" },
        { LVectorRepr::PtEtaPhiE, "pt, eta, phi, E" },
        { LVectorRepr::PtEtaPhiME, "pt, eta, phi, m, E" },
        { LVectorRepr::PxPyPzE, "px, py, pz, E" },
        { LVectorRepr::PtPhi, "pt, phi" },
        { LVectorRepr::PxPyPtPhi, "px, py, pt, phi" },
    };
    const auto iter = prefix.find(repr);
    if(iter == prefix.end())
      throw exception("LorentzVectorToString: representation is not supported.");
    std::ostringstream ss;
    if(print_prefix)
      ss << "(" << iter->second << ") = ";
    ss << "(";
    if(repr == LVectorRepr::PtEtaPhiM || repr == LVectorRepr::PtEtaPhiE || repr == LVectorRepr::PtEtaPhiME) {
      ss << pt << ", " << eta << ", " << phi;
      if(repr == LVectorRepr::PtEtaPhiM || repr == LVectorRepr::PtEtaPhiME)
        ss << ", " << mass;
      if(repr == LVectorRepr::PtEtaPhiE || repr == LVectorRepr::PtEtaPhiME)
        ss << ", " << E;
    } else if(repr == LVectorRepr::PxPyPzE) {
      ss << px << ", " << py << ", " << pz << ", " << E;
    } else if(repr == LVectorRepr::PtPhi || repr == LVectorRepr::PxPyPtPhi) {
      if(repr == LVectorRepr::PxPyPtPhi)
        ss << px << ", " << py << ", ";
      ss << pt << ", " << phi;
    }
    ss << ")";
    return ss.str();
  }
}

template<typename LVector>
std::string LorentzVectorToString(const LVector& p4, LVectorRepr repr = LVectorRepr::PtEtaPhiME,
                                  bool print_prefix = true)
{
    return detail::LorentzVectorToString(p4.Px(), p4.Py(), p4.Pz(), p4.E(), p4.M(), p4.Pt(), p4.Eta(), p4.Phi(),
                                         repr, print_prefix);
}

} // namespace analysis
