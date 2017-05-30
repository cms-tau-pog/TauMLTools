/*! Base class to parse configuration file.
This file is part of https://github.com/hh-italian-group/AnalysisTools. */

#pragma once

#include <fstream>
#include <vector>
#include <set>
#include <map>
#include <unordered_map>
#include <unordered_set>
#include <functional>
#include <boost/algorithm/string.hpp>

#include "exception.h"
#include "EnumNameMap.h"
#include "TextIO.h"

namespace analysis {

namespace detail {

template<typename T>
struct ConfigParameterParser {
    using Value = T;
    static Value Parse(T& t, const std::string& value, std::istream& s) { s >> t; return t; }
};

template<>
struct ConfigParameterParser<std::string> {
    using Value = std::string;
    static Value Parse(std::string& t, const std::string& value, std::istream& s) { t = value; return t; }
};

template<typename T>
struct ConfigParameterParser<std::vector<T>> {
    using Value = T;
    static Value Parse(std::vector<T>& param_vec, const std::string& value, std::istream& s)
    {
        T t;
        ConfigParameterParser<T>::Parse(t, value, s);
        param_vec.push_back(t);
        return t;
    }
};

template<typename T>
struct ConfigParameterParser<std::set<T>> {
    using Value = T;
    static Value Parse(std::set<T>& param_set, const std::string& value, std::istream& s)
    {
        T t;
        ConfigParameterParser<T>::Parse(t, value, s);
        if(param_set.count(t))
            throw exception("Duplicated parameter value.");
        param_set.insert(t);
        return t;
    }
};

template<typename Key, typename _Value>
struct ConfigParameterParser<std::map<Key, _Value>> {
    using Value = _Value;
    static Value Parse(std::map<Key, Value>& param_map, const std::string& value, std::istream& s)
    {
        const size_t pos = value.find_first_of(' ');
        const std::string k_str = value.substr(0, pos);
        const std::string v_str = value.substr(pos + 1);
        Key k;
        Value v;
        ConfigParameterParser<Key>::Parse(k, k_str, s);
        ConfigParameterParser<Value>::Parse(v, v_str, s);
        if(param_map.count(k))
            throw exception("Duplicated parameter value.");
        param_map[k] = v;
        return v;
    }
};

template<typename T, typename Wrapper, bool same = std::is_same<T, Wrapper>::value>
struct ConfigParameterParserEx {
    using Value = T;
    static Value Parse(T& t, const std::string& value, std::istream& s)
    {
        Wrapper w;
        ConfigParameterParser<Wrapper>::Parse(w, value, s);
        t = w;
        return t;
    }
};

template<typename T, typename Wrapper>
struct ConfigParameterParserEx<std::vector<T>, Wrapper, false> {
    using Value = T;
    static Value Parse(std::vector<T>& param_vec, const std::string& value, std::istream& s)
    {
        Wrapper w;
        ConfigParameterParser<Wrapper>::Parse(w, value, s);
        const T t = w;
        param_vec.push_back(t);
        return t;
    }
};

template<typename T>
struct ConfigParameterParserEx<T, T, true> {
    using Value = typename ConfigParameterParser<T>::Value;
    static Value Parse(T& t, const std::string& value, std::istream& s)
    {
        return ConfigParameterParser<T>::Parse(t, value, s);
    }
};

} // namespace detail

class ConfigEntryReader {
protected:
    using ReadCountMap = std::unordered_map<std::string, size_t>;
    enum class Condition { equal_to, greater_equal, greater, less_equal, less };

    class param_parsed_exception {};

public:
    ConfigEntryReader()
    {
        current_stream.exceptions(std::istringstream::failbit);
        current_stream >> std::boolalpha;
    }

    virtual ~ConfigEntryReader() {}
    virtual void StartEntry(const std::string& name, const std::string& reference_name)
    {
        read_params_counts.clear();
    }
    virtual void EndEntry() {}
    virtual void ReadParameter(const std::string& param_name, const std::string& param_value)
    {
        current_param_name = param_name;
        current_param_value = param_value;
        current_stream.str(param_value);
        current_stream.seekg(0);
        try {
            ReadParameter(param_name, param_value, current_stream);
            throw analysis::exception("Unsupported parameter '%1%'.") % param_name;
        } catch(param_parsed_exception&) {
            ++read_params_counts[param_name];
        }
    }

    static std::vector<std::string> ParseOrderedParameterList(std::string param_list,
                                                              bool allow_duplicates = false,
                                                              const std::string& separators = " \t")
    {
        return SplitValueList(param_list, allow_duplicates, separators, true);
    }

    static std::unordered_set<std::string> ParseParameterList(const std::string& param_list,
                                                              const std::string& separators = " \t")
    {
        const std::vector<std::string> ordered_result = ParseOrderedParameterList(param_list, false, separators);
        return std::unordered_set<std::string>(ordered_result.begin(), ordered_result.end());
    }

protected:
    virtual void ReadParameter(const std::string& param_name, const std::string& param_value,
                               std::istringstream& ss) = 0;

    template<typename T, typename Wrapper = T,
             typename ValidityCheck = std::function<bool(const typename detail::ConfigParameterParser<T>::Value&)>>
    void ParseEntry(const std::string& name, T& result, const ValidityCheck& validity_check)
    {
        using Parser = detail::ConfigParameterParserEx<T, Wrapper>;
        if(name != current_param_name) return;
        const auto& v = Parser::Parse(result, current_param_value, current_stream);
        if(!validity_check(v))
            throw exception("Parameter '%1%' = '%2%' is outside of its validity range.")
                % current_param_name % current_param_value;
        throw param_parsed_exception();
    }

    template<typename T, typename Wrapper = T>
    void ParseEntry(const std::string& name, T& result)
    {
        const auto validity_check = [](const typename detail::ConfigParameterParser<T>::Value&) { return true; };
        ParseEntry<T, Wrapper>(name, result, validity_check);
    }

    template<typename T>
    void ParseEntryList(const std::string& name, std::vector<T>& result, bool allow_duplicates,
                        const std::string& separators,
                        std::function<bool(const typename detail::ConfigParameterParser<T>::Value&)> validity_check)
    {
        if(name != current_param_name) return;
        const auto param_list = ParseOrderedParameterList(current_param_value, allow_duplicates, separators);
        for(const std::string& param_value : param_list) {
            std::istringstream ss(param_value);
            ss.exceptions(std::istringstream::failbit);
            ss >> std::boolalpha;
            T value;
            const auto& v = detail::ConfigParameterParser<T>::Parse(value, param_value, ss);
            if(!validity_check(v))
                throw exception("One of parameters in '%1%' equals '%2%', which is outside of its validity range.")
                    % current_param_name % param_value;
            result.push_back(value);
        }

        throw param_parsed_exception();
    }

    template<typename T>
    void ParseEntryList(const std::string& name, std::vector<T>& result, bool allow_duplicates = false,
                        const std::string& separators = " \t")
    {
        const auto validity_check = [](const typename detail::ConfigParameterParser<T>::Value&) { return true; };
        ParseEntryList<T>(name, result, allow_duplicates, separators, validity_check);
    }

    void CheckReadParamCounts(const std::string& param_name, size_t expected, Condition condition) const
    {
        static const std::map<Condition, std::pair<std::string, std::function<bool(size_t, size_t)>>> conditions {
            { Condition::equal_to, { "exactly", std::equal_to<size_t>() } },
            { Condition::greater_equal, { "at least", std::greater_equal<size_t>() } },
            { Condition::greater, { "more than", std::greater<size_t>() } },
            { Condition::less_equal, { "less than", std::less_equal<size_t>() } },
            { Condition::less, { "no more than", std::less<size_t>() } }
        };

        const size_t count = read_params_counts.count(param_name) ? read_params_counts.at(param_name) : 0;
        const std::string cmp_string = conditions.at(condition).first;
        const auto& cond_operator = conditions.at(condition).second;
        if(!cond_operator(count, expected))
            throw analysis::exception("The number of occurrences of the parameter '%1%' is %2%,"
                                      " while expected %3% %4%.") % param_name % count % cmp_string % expected;
    }

private:
    ReadCountMap read_params_counts;
    std::string current_param_name, current_param_value;
    std::istringstream current_stream;
};

class ConfigReader {
private:
    using EntryReaderMap = std::unordered_map<std::string, ConfigEntryReader*>;
    using EntryNameMap = std::unordered_map<std::string, std::unordered_set<std::string>>;

public:
    ConfigReader() : defaultEntryReader(entryReaderMap.end()) {}

    void AddEntryReader(const std::string& name, ConfigEntryReader& entryReader, bool isDefault = true)
    {
        if(entryReaderMap.count(name))
            throw analysis::exception("Entry reader with name '%1%' is already defined.") % name;
        entryReaderMap[name] = &entryReader;
        if(isDefault) {
            if(defaultEntryReader != entryReaderMap.end())
                throw analysis::exception("Default entry reader is already set.");
            defaultEntryReader = entryReaderMap.find(name);
        }
    }

    void ReadConfig(const std::string& _configName)
    {
        configName = _configName;
        if(defaultEntryReader == entryReaderMap.end())
            throw analysis::exception("Default entry reader is not set.");
        readEntryNames.clear();
        std::ifstream cfg(configName);
        if(cfg.fail())
            throw analysis::exception("Failed to open config file '%1%'.") % configName;
        size_t line_number = 0;
        EntryReaderMap::const_iterator entryReader = entryReaderMap.end();
        while(ReadNextEntry(cfg, line_number, entryReader)) {
            try {
                entryReader->second->EndEntry();
            } catch(std::exception& e) {
                BadSyntax(line_number, e.what());
            }
        }
    }

private:
    bool ReadNextEntry(std::istream& cfg, size_t& line_number, EntryReaderMap::const_iterator& entryReader)
    {
        bool entry_started = false;
        while (cfg.good()) {
            std::string cfgLine;
            std::getline(cfg, cfgLine);
            ++line_number;
            if ((cfgLine.size() && cfgLine.at(0) == '#') || (!cfgLine.size() && !entry_started)) continue;
            if(!cfgLine.size())
                return true;
            if(!entry_started && cfgLine.at(0) == '[') {
                const size_t end_pos = cfgLine.find(']');
                CheckSyntax(end_pos != std::string::npos, line_number);
                const std::string entry_definition = cfgLine.substr(1, end_pos - 1);
                std::vector<std::string> entry_parameters;
                boost::split(entry_parameters, entry_definition, boost::is_any_of(" "), boost::token_compress_on);
                CheckSyntax(entry_parameters.size() >= 1 && entry_parameters.size() <= 4, line_number);
                CheckSyntax(entry_parameters.size() < 3
                            || entry_parameters.at(entry_parameters.size() - 2) == ":", line_number);

                const std::string name = entry_parameters.at((entry_parameters.size() - 1) % 2);
                const std::string reader_name = entry_parameters.size() % 2 ?
                            defaultEntryReader->first : entry_parameters.at(0);
                const std::string reference_name = entry_parameters.size() > 2 ? entry_parameters.back() : "";

                CheckSyntax(entryReaderMap.count(reader_name), line_number, "Unknown entry type");
                entryReader = entryReaderMap.find(reader_name);


                CheckSyntax(!readEntryNames[entryReader->first].count(name), line_number,
                            boost::str(boost::format("Redifinition of the entry with a name '%1%'.") % name));
                CheckSyntax(!reference_name.size() || readEntryNames[entryReader->first].count(reference_name),
                        line_number, boost::str(boost::format("Reference entry '%1%' not found.") % reference_name));
                readEntryNames.at(entryReader->first).insert(name);
                entryReader->second->StartEntry(name, reference_name);
                entry_started = true;
            } else if(entry_started) {
                ReadParameterLine(cfgLine, line_number, *entryReader->second);
            } else
                BadSyntax(line_number);
        }
        return entry_started;
    }

    void ReadParameterLine(const std::string& cfgLine, size_t line_number, ConfigEntryReader& entryReader) const
    {
        static const char separator = ':';

        const size_t pos = cfgLine.find(separator);
        CheckSyntax(pos != std::string::npos && pos + 2 < cfgLine.size(), line_number);
        const std::string param_name = cfgLine.substr(0, pos);
        const std::string param_value = cfgLine.substr(pos + 2);
        try {
            entryReader.ReadParameter(param_name, param_value);
        } catch(std::exception& e) {
            BadSyntax(line_number, e.what());
        }
    }

    void CheckSyntax(bool condition, size_t line_number, const std::string& message = "") const
    {
        if(!condition)
            BadSyntax(line_number, message);
    }

    void BadSyntax(size_t line_number, const std::string& message = "") const
    {
        throw analysis::exception("Bad config syntax: file '%1%' line %2%.\n%3%") % configName % line_number % message;
    }

private:
    std::string configName;
    EntryReaderMap entryReaderMap;
    EntryReaderMap::const_iterator defaultEntryReader;
    EntryNameMap readEntryNames;
};

} // namespace analysis
