/*! Parse configuration file that contains list of properties.
This file is part of https://github.com/hh-italian-group/TauMLTools. */

#pragma once

#include <fstream>
#include <vector>
#include <map>

#include "exception.h"
#include "TextIO.h"

namespace analysis {

class PropertyList {
public:
    static constexpr char prop_sep = '=', quotes = '"', space = ' ';
    static const std::string& whitespaces();

    using PMap = std::map<std::string, std::string>;
    using const_iterator = PMap::const_iterator;

    const_iterator begin() const;
    const_iterator end() const;
    std::string& operator[](const std::string& p_name);
    bool Has(const std::string& p_name) const;

    template<typename T = std::string>
    T Get(const std::string& p_name, const std::string& item_name = "") const
    {
        if(!properties.count(p_name)) {
            std::ostringstream ss;
            ss << "Property '" << p_name << "' not found";
            if(item_name.empty())
                ss << ".";
            else
                ss << " in item '" << item_name << "'.";
            throw exception(ss.str());
        }
        T result;
        const std::string& p_str = properties.at(p_name);
        if(!::analysis::TryParse(p_str, result)) {
            std::ostringstream ss;
            ss << "Unable to parse property '" << p_name << "' = '" << p_str << "' as '" << typeid(T).name() << "'";
            if(item_name.empty())
                ss << ".";
            else
                ss << " for item '" << item_name << "'.";
            throw exception(ss.str());
        }
        return result;
    }

    template<typename T>
    bool Read(const std::string& p_name, T& result, const std::string& item_name = "") const
    {
        if(!Has(p_name)) return false;
        result = Get<T>(p_name, item_name);
        return true;
    }

    std::string ToString() const;
    static bool TryParse(const std::string& line, PropertyList& p_list, std::string& msg);
    static PropertyList Parse(const std::string& line);

    template<typename T = std::string>
    std::vector<T> GetList(const std::string& name, bool allow_duplicates,
                           const std::string& separators = " \t") const
    {
        std::vector<T> list;
        auto split_list = SplitValueList(Get<std::string>(name), allow_duplicates, separators);
        for (const auto& split_element : split_list){
            const T element = analysis::Parse<T>(split_element);
            list.push_back(element);
        }
        return list;
    }

private:
    std::map<std::string, std::string> properties;
};

std::ostream& operator<<(std::ostream& s, const PropertyList& p_list);
std::istream& operator>>(std::istream& s, PropertyList& p_list);

class PropertyConfigReader {
public:
    struct Item {
        std::string name;
        PropertyList properties;

        Item() {}
        bool Has(const std::string& p_name) const;
        template<typename T = std::string>
        T Get(const std::string& p_name) const { return properties.Get<T>(p_name, name); }
        template<typename T>
        bool Read(const std::string& p_name, T& result) const { return properties.Read(p_name, result, name); }
        std::string& operator[](const std::string& p_name);

        static bool TryParse(const std::string& line, Item& item, std::string& msg);
    };

    using ItemCollection = std::map<std::string, Item>;

public:
    void Parse(const std::string& cfg_file_name);
    const ItemCollection& GetItems() const;

private:
    static std::vector<std::string> PreprocessLines(const std::vector<std::string> orig_lines,
                                                    const std::string& whitespaces);

private:
    ItemCollection items;
};

} // namespace analysis
