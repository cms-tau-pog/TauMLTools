/*! Base class to parse configuration file.
This file is part of https://github.com/hh-italian-group/TauMLTools. */

#include "TauMLTools/Core/interface/ConfigReader.h"

#include <fstream>
#include <vector>
#include <set>
#include <map>
#include <unordered_map>
#include <unordered_set>
#include <functional>
#include <boost/algorithm/string.hpp>

#include "TauMLTools/Core/interface/exception.h"
#include "TauMLTools/Core/interface/EnumNameMap.h"
#include "TauMLTools/Core/interface/TextIO.h"

namespace analysis {


ConfigEntryReader::ConfigEntryReader()
{
    current_stream.exceptions(std::istringstream::failbit);
    current_stream >> std::boolalpha;
}

void ConfigEntryReader::StartEntry(const std::string& /*name*/, const std::string& /*reference_name*/)
{
    read_params_counts.clear();
}

void ConfigEntryReader::ReadParameter(const std::string& param_name, const std::string& param_value)
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

std::vector<std::string> ConfigEntryReader::ParseOrderedParameterList(std::string param_list,
                                                                      bool allow_duplicates,
                                                                      const std::string& separators)
{
    return SplitValueList(param_list, allow_duplicates, separators, true);
}

std::unordered_set<std::string> ConfigEntryReader::ParseParameterList(const std::string& param_list,
                                                                      const std::string& separators)
{
    const std::vector<std::string> ordered_result = ParseOrderedParameterList(param_list, false, separators);
    return std::unordered_set<std::string>(ordered_result.begin(), ordered_result.end());
}

void ConfigEntryReader::CheckReadParamCounts(const std::string& param_name, size_t expected, Condition condition) const
{
    static const std::map<Condition, std::pair<std::string, std::function<bool(size_t, size_t)>>> conditions {
        { Condition::equal_to, { "exactly", std::equal_to<size_t>() } },
        { Condition::greater_equal, { "at least", std::greater_equal<size_t>() } },
        { Condition::greater, { "more than", std::greater<size_t>() } },
        { Condition::less_equal, { "less than", std::less_equal<size_t>() } },
        { Condition::less, { "no more than", std::less<size_t>() } }
    };

    const size_t count = GetReadParamCounts(param_name);
    const std::string cmp_string = conditions.at(condition).first;
    const auto& cond_operator = conditions.at(condition).second;
    if(!cond_operator(count, expected))
        throw analysis::exception("The number of occurrences of the parameter '%1%' is %2%,"
                                  " while expected %3% %4%.") % param_name % count % cmp_string % expected;
}

size_t ConfigEntryReader::GetReadParamCounts(const std::string& param_name) const
{
    return read_params_counts.count(param_name) ? read_params_counts.at(param_name) : 0;
}

ConfigReader::ConfigReader() : defaultEntryReader(entryReaderMap.end()) {}

void ConfigReader::AddEntryReader(const std::string& name, ConfigEntryReader& entryReader, bool isDefault)
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

void ConfigReader::ReadConfig(const std::string& _configName)
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

bool ConfigReader::ReadNextEntry(std::istream& cfg, size_t& line_number, EntryReaderMap::const_iterator& entryReader)
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

void ConfigReader::ReadParameterLine(const std::string& cfgLine, size_t line_number,
                                     ConfigEntryReader& entryReader) const
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

void ConfigReader::CheckSyntax(bool condition, size_t line_number, const std::string& message) const
{
    if(!condition)
        BadSyntax(line_number, message);
}

[[noreturn]] void ConfigReader::BadSyntax(size_t line_number, const std::string& message) const
{
    throw analysis::exception("Bad config syntax: file '%1%' line %2%.\n%3%") % configName % line_number % message;
}

} // namespace analysis
