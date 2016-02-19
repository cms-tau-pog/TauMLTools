/*! Base class to parse configuration file.
This file is part of https://github.com/hh-italian-group/AnalysisTools. */

#pragma once

#include <fstream>
#include <vector>
#include <map>
#include <set>

#include "exception.h"

namespace analysis {

class ConfigEntryReader {
public:
    virtual ~ConfigEntryReader() {}
    virtual void StartEntry(const std::string& name) = 0;
    virtual void EndEntry() = 0;
    virtual void ReadParameter(const std::string& param_name, const std::string& param_value) = 0;
};

class ConfigReader {
private:
    typedef std::map<std::string, ConfigEntryReader*> EntryReaderMap;
public:
    ConfigReader(const std::string& _configName)
        : configName(_configName), defaultEntryReader(entryReaderMap.end())
    {}

    void AddEntryReader(const std::string& name, ConfigEntryReader& entryReader, bool isDefault = true)
    {
        if(entryReaderMap.count(name))
            throw exception("Entry reader with name '") << name << "' is already defined.";
        entryReaderMap[name] = &entryReader;
        if(isDefault) {
            if(defaultEntryReader != entryReaderMap.end())
                throw exception("Default entry reader is already set.");
            defaultEntryReader = entryReaderMap.find(name);
        }
    }

    void ReadConfig() const
    {
        if(defaultEntryReader == entryReaderMap.end())
            throw exception("Default entry reader is not set.");
        std::ifstream cfg(configName);
        if(cfg.fail())
            throw exception("Failed to open config file '") << configName << "'.";
        size_t line_number = 0;
        EntryReaderMap::const_iterator entryReader = entryReaderMap.end();
        while(ReadNextEntry(cfg, line_number, entryReader))
            entryReader->second->EndEntry();
    }

    static std::vector<std::string> ParseOrderedParameterList(const std::string& param_list,
                                                              bool allow_duplicates = false, char separator = ',')
    {
        std::set<std::string> set_result;
        std::vector<std::string> result;
        size_t prev_pos = 0;
        for(bool next = true; next;) {
            const size_t pos = param_list.find(separator, prev_pos);
            next = pos != std::string::npos;
            const std::string param_name = param_list.substr(prev_pos, pos - prev_pos);
            if(set_result.count(param_name) && !allow_duplicates)
                throw exception("Parameter '") << param_name << "' listed more than once in the following parameter"
                                                  " list '" << param_list << "'.";
            result.push_back(param_name);
            prev_pos = pos + 1;
        }
        return result;
    }

    static std::set<std::string> ParseParameterList(const std::string& param_list, char separator = ',')
    {
        const std::vector<std::string> ordered_result = ParseOrderedParameterList(param_list, false, separator);
        return std::set<std::string>(ordered_result.begin(), ordered_result.end());
    }

private:
    bool ReadNextEntry(std::istream& cfg, size_t& line_number, EntryReaderMap::const_iterator& entryReader) const
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
                const size_t split_pos = cfgLine.find(' ');
                size_t name_pos = 1;
                if(split_pos != std::string::npos) {
                    const std::string readerName = cfgLine.substr(1, split_pos - 1);
                    CheckSyntax(entryReaderMap.count(readerName), line_number, "Unknown entry type");
                    entryReader = entryReaderMap.find(readerName);
                    name_pos = split_pos + 1;
                } else
                    entryReader = defaultEntryReader;

                const std::string name = cfgLine.substr(name_pos, end_pos - name_pos);
                entryReader->second->StartEntry(name);
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
        throw exception("Bad config syntax: file '") << configName << "' line " << line_number << ".\n" << message;
    }

private:
    std::string configName;
    EntryReaderMap entryReaderMap;
    EntryReaderMap::const_iterator defaultEntryReader;
};

} // namespace analysis
