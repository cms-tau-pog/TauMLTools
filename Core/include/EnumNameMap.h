/*! Define I/O operators for enum <-> string conversion.
This file is part of https://github.com/hh-italian-group/AnalysisTools. */

#pragma once

#include<istream>
#include<ostream>
#include<unordered_map>
#include<initializer_list>
#include<typeinfo>

#include "exception.h"

#define ENUM_NAMES(enum_type) \
    template<> \
    const analysis::EnumNameMap<enum_type> analysis::EnumNameMap<enum_type>::data

namespace analysis {

template<typename Enum>
class EnumNameMap {
public:
    typedef std::pair<Enum, std::string> EnumStringPair;

    EnumNameMap(const std::initializer_list<EnumStringPair>& pairs)
    {
        for(const EnumStringPair& entry : pairs) {
            if(enum_to_string_map.count(entry.first) || string_to_enum_map.count(entry.second))
                throw analysis::exception("Duplicated enum entry for the enum '") << typeid(Enum).name() << "'.";
            enum_to_string_map[entry.first] = entry.second;
            string_to_enum_map[entry.second] = entry.first;
        }
    }

    static const std::string& ToString(Enum e)
    {
        if(!data.enum_to_string_map.count(e))
            throw analysis::exception("The corresponding string is not found for an element of the enum '")
                << typeid(Enum).name() << "'.";
        return data.enum_to_string_map.at(e);
    }

    static Enum FromString(const std::string& str)
    {
        if(!data.string_to_enum_map.count(str))
            throw analysis::exception("An element of the enum '") << typeid(Enum).name()
                << "' that corresponds to the string '" << str << "' is not found.";
        return data.string_to_enum_map.at(str);
    }

private:
    static const EnumNameMap data;
    std::unordered_map<Enum, std::string> enum_to_string_map;
    std::unordered_map<std::string, Enum> string_to_enum_map;
};

} // namespace analysis

template<typename Enum, typename = typename std::enable_if<std::is_enum<Enum>::value>::type>
std::ostream& operator<<(std::ostream& os, Enum e)
{
    os << analysis::EnumNameMap<Enum>::ToString(e);
    return os;
}

template<typename Enum, typename = typename std::enable_if<std::is_enum<Enum>::value>::type>
std::istream& operator>>(std::istream& is, Enum& e)
{
    std::string str;
    is >> str;
    e = analysis::EnumNameMap<Enum>::FromString(str);
    return is;
}

