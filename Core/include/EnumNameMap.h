/*! Definition of the I/O operators for enum <-> string conversion.
This file is part of https://github.com/hh-italian-group/AnalysisTools. */

#pragma once

#include<istream>
#include<ostream>
#include<unordered_set>
#include<unordered_map>
#include<initializer_list>
#include<typeinfo>

#include "exception.h"

#define ENUM_NAMES(enum_type) \
    template<typename T=void> \
    struct __##enum_type##_names { static const ::analysis::EnumNameMap<enum_type> names; }; \
	struct __##enum_type##_names_impl : __##enum_type##_names<> { const ::analysis::EnumNameMap<enum_type>* names_ptr = &names;  }; \
    template<typename T> \
    const ::analysis::EnumNameMap<enum_type> __##enum_type##_names<T>::names

#define ENUM_OSTREAM_OPERATORS() \
    template<typename Enum, typename = typename std::enable_if<std::is_enum<Enum>::value>::type> \
    inline std::ostream& operator<<(std::ostream& os, Enum e) { return ::analysis::operator <<(os, e); } \
    template<typename Enum, typename = typename std::enable_if<std::is_enum<Enum>::value>::type> \
    inline std::wostream& operator<<(std::wostream& os, Enum e) { return ::analysis::operator <<(os, e); } \
    /**/

#define ENUM_ISTREAM_OPERATORS() \
    template<typename Enum, typename = typename std::enable_if<std::is_enum<Enum>::value>::type> \
    inline std::istream& operator>>(std::istream& is, Enum& e) { return ::analysis::operator >>(is, e); } \
    template<typename Enum, typename = typename std::enable_if<std::is_enum<Enum>::value>::type> \
    inline std::wistream& operator>>(std::wistream& is, Enum& e) { return ::analysis::operator >>(is, e); } \
    /**/

namespace analysis {
template<typename Enum>
class EnumNameMap {
public:
    using EnumStringPair = std::pair<Enum, std::string>;
    struct EnumHash {
        size_t operator()(const Enum& e) const { std::hash<size_t> h; return h(static_cast<size_t>(e)); }
    };
    using EnumEntrySet = std::unordered_set<Enum, EnumHash>;
    using StringEntrySet = std::unordered_set<std::string>;

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

    const EnumEntrySet& GetEnumEntries() const { return enum_entries; }
    const StringEntrySet& GetStringEntries() const { return string_entries; }

    static const EnumNameMap<Enum>& GetDefault() { return *GetDefault(true); }

private:
    void Initialize(const std::initializer_list<EnumStringPair>& pairs)
    {
        for(const EnumStringPair& entry : pairs) {
            if(enum_to_string_map.count(entry.first) || string_to_enum_map.count(entry.second))
                throw exception("Duplicated enum entry for the enum '%1%'.") % enum_name;
            enum_to_string_map[entry.first] = entry.second;
            string_to_enum_map[entry.second] = entry.first;
            enum_entries.insert(entry.first);
            string_entries.insert(entry.second);
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
    EnumEntrySet enum_entries;
    StringEntrySet string_entries;
};

template<typename Enum, typename = typename std::enable_if<std::is_enum<Enum>::value>::type>
std::ostream& operator<<(std::ostream& os, Enum e)
{
    os << analysis::EnumNameMap<Enum>::GetDefault().EnumToString(e);
    return os;
}

template<typename Enum, typename = typename std::enable_if<std::is_enum<Enum>::value>::type>
std::wostream& operator<<(std::wostream& os, Enum e)
{
    const std::string str = analysis::EnumNameMap<Enum>::GetDefault().EnumToString(e);
    os << std::wstring(str.begin(), str.end());
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

template<typename Enum, typename = typename std::enable_if<std::is_enum<Enum>::value>::type>
std::wistream& operator>>(std::wistream& is, Enum& e)
{
    std::wstring wstr;
    is >> wstr;
    const std::string str(wstr.begin(), wstr.end());
    e = analysis::EnumNameMap<Enum>::GetDefault().Parse(str);
    return is;
}

} // namespace analysis
