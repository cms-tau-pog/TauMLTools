/*! Base class to parse configuration file.
This file is part of https://github.com/hh-italian-group/TauMLTools. */

#pragma once

#include <fstream>
#include <vector>
#include <set>
#include <map>
#include <unordered_map>
#include <unordered_set>
#include <functional>

#include "exception.h"
#include "EnumNameMap.h"
#include "TextIO.h"

namespace analysis {

namespace detail {

template<typename T>
struct ConfigParameterParser {
    using Value = T;
    static Value Parse(T& t, const std::string& /*value*/, std::istream& s) { s >> t; const T& tt = t; return tt; }
};

template<>
struct ConfigParameterParser<std::string> {
    using Value = std::string;
    static Value Parse(std::string& t, const std::string& value, std::istream& /*s*/) { t = value; return t; }
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
    static Value Parse(std::map<Key, Value>& param_map, const std::string& value, std::istream&)
    {
        const size_t pos = value.find_first_of(' ');
        const std::string k_str = value.substr(0, pos);
        const std::string v_str = value.substr(pos + 1);
        std::istringstream k_is(k_str), v_is(v_str);
        Key k;
        Value v;
        ConfigParameterParser<Key>::Parse(k, k_str, k_is);
        ConfigParameterParser<Value>::Parse(v, v_str, v_is);
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
    ConfigEntryReader();
    virtual ~ConfigEntryReader() {}
    virtual void StartEntry(const std::string& /*name*/, const std::string& /*reference_name*/);
    virtual void EndEntry() {}
    virtual void ReadParameter(const std::string& param_name, const std::string& param_value);

    static std::vector<std::string> ParseOrderedParameterList(std::string param_list,
                                                              bool allow_duplicates = false,
                                                              const std::string& separators = " \t");
    static std::unordered_set<std::string> ParseParameterList(const std::string& param_list,
                                                              const std::string& separators = " \t");

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

    template<typename Container, typename T = typename Container::value_type>
    void ParseEntryList(const std::string& name, Container& result, bool allow_duplicates,
                        const std::string& separators,
                        std::function<bool(const typename detail::ConfigParameterParser<T>::Value&)> validity_check)
    {
        if(name != current_param_name) return;
        const auto param_list = ParseOrderedParameterList(current_param_value, allow_duplicates, separators);
        auto inserter = std::inserter(result, result.end());
        for(const std::string& param_value : param_list) {
            const T value = Parse<T>(param_value);
            if(!validity_check(value))
                throw exception("One of parameters in '%1%' equals '%2%', which is outside of its validity range.")
                    % current_param_name % param_value;
            inserter = value;
        }

        throw param_parsed_exception();
    }

    template<typename Container, typename T = typename Container::value_type>
    void ParseEntryList(const std::string& name, Container& result, bool allow_duplicates = false,
                        const std::string& separators = " \t")
    {
        const auto validity_check = [](const typename detail::ConfigParameterParser<T>::Value&) { return true; };
        ParseEntryList<Container, T>(name, result, allow_duplicates, separators, validity_check);
    }

    template<typename Container, typename T = typename Container::value_type>
    void ParseEnumList(const std::string& name, Container& result)
    {
        if(name != current_param_name) return;
        auto inserter = std::inserter(result, result.end());
        const auto param_list = ParseOrderedParameterList(current_param_value, false, " \t");
        if(param_list.size() == 1 && param_list.front() == "all") {
            for(const auto& entry : EnumNameMap<T>::GetDefault().GetEnumEntries())
                inserter = entry;
        } else {
            for(const auto& param_value : param_list) {
                const T value = Parse<T>(param_value);
                inserter = value;
            }
        }

        throw param_parsed_exception();
    }

    template<typename Container, typename KeyType = typename Container::key_type,
             typename MappedType = typename Container::mapped_type,
             typename ItemType = typename MappedType::value_type>
    void ParseMappedEntryList(const std::string& name, Container& result, bool allow_duplicates,
            const std::string& separators,
            std::function<bool(const ItemType&)> validity_check)
    {
        if(name != current_param_name) return;
        const auto param_list = ParseOrderedParameterList(current_param_value, true, separators);
        if(param_list.size() < 2)
            throw exception("Invalid mapped entry '%1%'.") % name;
        auto list_iter = param_list.begin();
        const auto key = Parse<KeyType>(*list_iter++);
        if(result.count(key))
            throw exception("Item with name '%1%' already present in map '%2%'") % key % name;
        auto inserter = std::inserter(result[key], result[key].end());
        std::set<std::string> processed_items;
        for(; list_iter != param_list.end(); ++list_iter) {
            if(!allow_duplicates && processed_items.count(*list_iter))
                throw exception("Duplicated list entry = '%1%'.") % *list_iter;
            processed_items.insert(*list_iter);
            const ItemType value = Parse<ItemType>(*list_iter);
            if(!validity_check(value))
                throw exception("One of parameters in '%1%' equals '%2%', which is outside of its validity range.")
                    % current_param_name % *list_iter;
            inserter = value;
        }

        throw param_parsed_exception();
    }

    template<typename Container, typename KeyType = typename Container::key_type,
             typename MappedType = typename Container::mapped_type,
             typename ItemType = typename MappedType::value_type>
    void ParseMappedEntryList(const std::string& name, Container& result, bool allow_duplicates,
            const std::string& separators = " \t")
    {
        const auto validity_check = [](const ItemType&) { return true; };
        ParseMappedEntryList<Container, KeyType, MappedType, ItemType>(
                    name, result, allow_duplicates, separators, validity_check);
    }

    void CheckReadParamCounts(const std::string& param_name, size_t expected, Condition condition) const;
    size_t GetReadParamCounts(const std::string& param_name) const;

private:
    ReadCountMap read_params_counts;
    std::string current_param_name, current_param_value;
    std::istringstream current_stream;
};

template<typename T, typename Collection = std::unordered_map<std::string, T>>
class ConfigEntryReaderT : public virtual ConfigEntryReader {
public:
    ConfigEntryReaderT(Collection& _items) : items(&_items) {}

    virtual void StartEntry(const std::string& name, const std::string& reference_name) override
    {
        ConfigEntryReader::StartEntry(name, reference_name);
        current = reference_name.size() ? items->at(reference_name) : T();
        current.name = name;
    }

    virtual void EndEntry() override
    {
        (*items)[current.name] = current;
    }

protected:
    T current;
    Collection* items;
};

class ConfigReader {
private:
    using EntryReaderMap = std::unordered_map<std::string, ConfigEntryReader*>;
    using EntryNameMap = std::unordered_map<std::string, std::unordered_set<std::string>>;

public:
    ConfigReader();

    void AddEntryReader(const std::string& name, ConfigEntryReader& entryReader, bool isDefault = true);
    void ReadConfig(const std::string& _configName);

private:
    bool ReadNextEntry(std::istream& cfg, size_t& line_number, EntryReaderMap::const_iterator& entryReader);
    void ReadParameterLine(const std::string& cfgLine, size_t line_number, ConfigEntryReader& entryReader) const;
    void CheckSyntax(bool condition, size_t line_number, const std::string& message = "") const;
    [[noreturn]] void BadSyntax(size_t line_number, const std::string& message = "") const;

private:
    std::string configName;
    EntryReaderMap entryReaderMap;
    EntryReaderMap::const_iterator defaultEntryReader;
    EntryNameMap readEntryNames;
};

} // namespace analysis
