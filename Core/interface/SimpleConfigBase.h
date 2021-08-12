/*! Definition of BaseConfig class.
This file is part of https://github.com/hh-italian-group/TauMLTools. */

#pragma once

#include <map>
#include <iostream>

#include "exception.h"

#define ANA_CONFIG_PARAMETER(type, name, default_value) \
    type name() const { \
        type result = default_value; \
        Get(#name, result); \
        return result; \
    } \
    void set##name(const type& value) { Set(#name, value); }

namespace analysis {

namespace BaseConfigInternals {

template<typename Value>
struct ConfigValue {
    static bool Read(const std::string& str, Value& value)
    {
        std::istringstream s(str);
        s >> value;
        return !s.fail();
    }
    static std::string Write(const Value& value)
    {
        std::ostringstream s;
        s << value;
        return s.str();
    }
};

template<>
struct ConfigValue<bool> {
    static bool Read(const std::string& str, bool& value)
    {
        std::istringstream s(str);
        s >> std::boolalpha >> value;
        return !s.fail();
    }

    static std::string Write(const bool& value)
    {
        std::ostringstream s;
        s << std::boolalpha << value;
        return s.str();
    }
};

}

class BaseConfig {
private:
    using Map = std::map<std::string, std::string>;
public:
    virtual ~BaseConfig() {}
    virtual void Read(const std::string& fileName)
    {
        std::ifstream f(fileName.c_str());
        if(!f.is_open())
            throw exception("Unable to read the configuration file '") << fileName << "'.";
        while(f.good()) {
            std::string line;
            std::getline(f, line);
            if(!line.length() || line[0] == '#' || line[0] == '-')
                continue;

            std::istringstream istring(line);
            std::string name;
            std::string value;

            istring >> name >> value;

            if(istring.fail() || !name.length())
                continue;
            parameters[name] = value;
        }
    }
    virtual void Write(const std::string& fileName) const
    {
        std::ofstream f(fileName.c_str());
        if(!f.is_open())
            throw exception("Unable to write the configuration into the file '") << fileName << "'.";
        for(Map::const_iterator iter = parameters.begin(); iter != parameters.end(); ++iter) {
            f << iter->first << " " << iter->second << std::endl;
        }
    }

protected:
    template<typename Value>
    void Get(const std::string& name, Value& value) const
    {
        const Map::const_iterator iter = parameters.find(name);
        if(iter == parameters.end()) {
            const std::string value_str = BaseConfigInternals::ConfigValue<Value>::Write(value);
            std::cerr << "Warning: Parameter '" << name << "' is not set. Using default value = '" << value_str
                      << "'." << std::endl;
            parameters[name] = value_str;
        }
        else if(!BaseConfigInternals::ConfigValue<Value>::Read(iter->second, value))
            throw exception("Invalid configuration for parameter '") << name << "'.";
    }

    template<typename Value>
    void Set(const std::string& name, const Value& value)
    {
        parameters[name] = BaseConfigInternals::ConfigValue<Value>::Write(value);
    }

private:
    mutable Map parameters;
};

} // analysis
