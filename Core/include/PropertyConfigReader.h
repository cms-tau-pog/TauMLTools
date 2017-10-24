/*! Parse configuration file that contains list of properties.
This file is part of https://github.com/hh-italian-group/AnalysisTools. */

#pragma once

#include <fstream>
#include <vector>
#include <map>
#include <boost/algorithm/string.hpp>

#include "exception.h"
#include "TextIO.h"

namespace analysis {

class PropertyConfigReader {
public:
    struct Item {
        std::string name;
        std::map<std::string, std::string> properties;

        bool Has(const std::string& p_name) const { return properties.count(p_name); }

        template<typename T>
        T Get(const std::string& property_name) const
        {
            if(!properties.count(property_name))
                throw exception("Property '%1% not found in item '%2'.") % property_name % name;
            T result;
            const std::string& property_str = properties.at(property_name);
            if(!TryParse(property_str, result))
                throw exception("Unable to parse property '%1%' = '%2%' as '%3%' for item '%4%'.")
                    % property_name % property_str % typeid(T).name() % name;
            return result;
        }
    };

    using ItemCollection = std::map<std::string, Item>;

public:
    PropertyConfigReader()
    {
    }

    void Parse(const std::string& cfg_file_name)
    {
        static const std::string whitespaces = " \t";

        std::ifstream cfg(cfg_file_name);
        if(cfg.fail())
            throw exception("Failed to open config file '%1%'.") % cfg_file_name;

        std::vector<std::string> lines;
        while(cfg.good()) {
            std::string line;
            std::getline(cfg, line);
            lines.push_back(line);
        }

        lines = PreprocessLines(lines, whitespaces);
        for(const std::string& line : lines) {
            Item item;
            std::string msg;
            if(!TryParseItem(line, item, msg, whitespaces))
                throw exception("Invalid config item '%1%' in the config file '%2%'. %3%") % line % cfg_file_name % msg;
            if(items.count(item.name))
                throw exception("Multiple definition of item with name '%1%' in the config file '%2%'.")
                    % item.name % cfg_file_name;
            items[item.name] = item;
        }
    }

    const ItemCollection& GetItems() const { return items; }

private:
    static std::vector<std::string> PreprocessLines(const std::vector<std::string> orig_lines,
                                                    const std::string& whitespaces)
    {
        static constexpr char comment = '#', line_ext = '\\';

        std::vector<std::string> lines;
        std::string current_line;
        for(auto l : orig_lines) {
            boost::trim_if(l, boost::is_any_of(whitespaces));
            if(!l.size() || l.at(0) == comment) {
                if(current_line.size()) {
                    lines.push_back(current_line);
                    current_line.clear();
                }
                continue;
            }
            current_line += l;
            if(current_line.back() == line_ext) {
                current_line.erase(current_line.size() - 1);
            } else {
                lines.push_back(current_line);
                current_line.clear();
            }
        }
        if(current_line.size())
            throw exception("Unfinished line '%1%' in the config.") % current_line;
        return lines;
    }

    static bool TryParseItem(const std::string& line, Item& item, std::string& msg, const std::string& whitespaces)
    {
        static constexpr char name_sep = ':', prop_sep = '=', quotes = '"';

        const std::string prop_sep_and_ws = whitespaces + prop_sep;
        const auto ws_predicate = boost::is_any_of(whitespaces);
        const size_t name_sep_pos = line.find(name_sep);
        if(name_sep_pos == std::string::npos) {
            msg = "Item name not found.";
            return false;
        }
        item.name = line.substr(0, name_sep_pos);
        for(size_t n = name_sep_pos + 1; n < line.size();) {
            n = line.find_first_not_of(whitespaces, n);
            if(n == std::string::npos) break;
            const size_t name_end_pos = line.find_first_of(prop_sep_and_ws, n);
            if(name_end_pos == std::string::npos) {
                msg = "Invalid property name.";
                return false;
            }
            const std::string p_name = line.substr(n, name_end_pos - n);
            n = name_end_pos;
            bool prop_sep_found = false;
            for(; n < line.size() && !prop_sep_found; ++n) {
                const char c = line.at(n);
                if(c == prop_sep) {
                    prop_sep_found = true;
                } else if(!ws_predicate(line.at(n))) {
                    break;
                }
            }
            if(!prop_sep_found) {
                msg = boost::str(boost::format("Invalid format for property '%1%'.") % p_name);
                return false;
            }
            const size_t prop_value_pos = line.find_first_not_of(whitespaces, n);
            if(prop_value_pos == std::string::npos) {
                msg = boost::str(boost::format("Value is missing for property '%1%'.") % p_name);
                return false;
            }
            std::string p_value;
            if(line.at(prop_value_pos) == quotes) {
                const size_t closing_quotes = line.find(quotes, prop_value_pos + 1);
                if(closing_quotes == std::string::npos) {
                    msg = boost::str(boost::format("Mismatched quotes for property '%1%'.") % p_name);
                    return false;
                }
                p_value = line.substr(prop_value_pos + 1, closing_quotes - prop_value_pos - 1);
                n = closing_quotes + 1;
            } else {
                const size_t p_value_end = line.find_first_of(whitespaces, prop_value_pos);
                p_value = line.substr(prop_value_pos, p_value_end - prop_value_pos);
                n = p_value_end;
            }

            if(n < line.size() && !ws_predicate(line.at(n))) {
                msg = boost::str(boost::format("Invalid termination for property '%1%'.") % p_name);
                return false;
            }

            item.properties[p_name] = p_value;
        }

        return true;
    }

private:
    ItemCollection items;
};

} // namespace analysis
