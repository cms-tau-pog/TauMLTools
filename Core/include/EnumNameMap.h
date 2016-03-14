/*! Definition of the I/O operators for enum <-> string conversion.
This file is part of https://github.com/hh-italian-group/AnalysisTools. */

#pragma once

#include<istream>
#include<ostream>
#include<unordered_map>
#include<initializer_list>
#include<typeinfo>

#include "exception.h"

#define ENUM_NAMES(enum_type) \
    const analysis::EnumNameMap<enum_type> __##enum_type##_names

namespace analysis {
template<typename Enum>
class EnumNameMap {
public:
    typedef std::pair<Enum, std::string> EnumStringPair;
    struct EnumHash {
        size_t operator()(const Enum& e) const { std::hash<size_t> h; return h(static_cast<size_t>(e)); }
    };

    EnumNameMap(const std::string& _enum_name, const std::initializer_list<EnumStringPair>& pairs)
        : enum_name(_enum_name)
    {
        Initialize(pairs);
    }

    EnumNameMap(const std::initializer_list<EnumStringPair>& pairs)
        : enum_name(typeid(Enum).name())
    {
        if(GetDefault(false))
            throw exception("Redefinition of enum names for the enum '%1%'") % enum_name;
        GetDefault(false) = this;
        Initialize(pairs);
    }

    bool HasEnum(const Enum& e) const { return enum_to_string_map.count(e); }
    bool HasString(const std::string& str) const { return string_to_enum_map.count(str); }

    const std::string& EnumToString(const Enum& e) const
    {
        if(!HasEnum(e))
            throw exception("The corresponding string is not found for an element of the enum '%1%'.") % enum_name;
        return enum_to_string_map.at(e);
    }

    bool TryParse(const std::string& str, Enum& e) const
    {
        if(!HasString(str))
            return false;
        e = string_to_enum_map.at(str);
        return true;
    }

    Enum Parse(const std::string& str) const
    {
        Enum e;
        if(!TryParse(str, e))
            throw exception("An element of the enum '%1%' that corresponds to the string '%2%' is not found.")
                % enum_name % str;
        return e;
    }

    static const EnumNameMap<Enum>& GetDefault() { return *GetDefault(true); }

private:
    void Initialize(const std::initializer_list<EnumStringPair>& pairs)
    {
        for(const EnumStringPair& entry : pairs) {
            if(enum_to_string_map.count(entry.first) || string_to_enum_map.count(entry.second))
                throw exception("Duplicated enum entry for the enum '%1%'.") % enum_name;
            enum_to_string_map[entry.first] = entry.second;
            string_to_enum_map[entry.second] = entry.first;
        }
    }

    static EnumNameMap<Enum>*& GetDefault(bool null_check)
    {
        static EnumNameMap<Enum>* m = nullptr;
        if(null_check && !m)
            throw exception("Names for the enum '%1%' are not defined.") % typeid(Enum).name();
        return m;
    }

private:
    std::string enum_name;
    std::unordered_map<Enum, std::string, EnumHash> enum_to_string_map;
    std::unordered_map<std::string, Enum> string_to_enum_map;
};

} // namespace analysis

template<typename Enum, typename>
std::ostream& operator<<(std::ostream& os, Enum e)
{
    os << analysis::EnumNameMap<Enum>::GetDefault().EnumToString(e);
    return os;
}

template<typename Enum, typename = typename std::enable_if<std::is_enum<Enum>::value>::type>
std::istream& operator>>(std::istream& is, Enum& e)
{
    std::string str;
    is >> str;
    e = analysis::EnumNameMap<Enum>::GetDefault().Parse(str);
    return is;
}
