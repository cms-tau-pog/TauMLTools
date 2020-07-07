/*! Common tools and definitions suitable for general purposes.
This file is part of https://github.com/hh-italian-group/TauMLTools. */

#include "TauMLTools/Core/interface/Tools.h"
#include "TauMLTools/Core/interface/TextIO.h"
#include <boost/regex.hpp>
#include <boost/crc.hpp>
#include <boost/filesystem.hpp>

namespace analysis {

namespace tools {

std::string FullPath(std::initializer_list<std::string> paths)
{
    if(!paths.size())
        return "";

    const auto add_path = [](std::ostringstream& full_path, std::string path, bool add_sep) {
        if(path.size() && path.at(path.size() - 1) == '/')
            path = path.substr(0, path.size() - 1);
        if(add_sep)
            full_path << '/';
        full_path << path;
    };

    std::ostringstream full_path;
    auto iter = paths.begin();
    add_path(full_path, *iter++, false);
    for(; iter != paths.end(); ++iter)
        add_path(full_path, *iter, true);
    return full_path.str();
}

uint32_t hash(const std::string& str)
{
    boost::crc_32_type crc;
    crc.process_bytes(str.data(), str.size());
    return crc.checksum();
}

std::vector<std::string> FindFiles(const std::string& path, const std::string& file_name_pattern)
{
    using directory_iterator = boost::filesystem::directory_iterator;

    std::vector<std::string> all_files;
    for (const auto& dir_entry : directory_iterator(path)){
        std::string n_path = dir_entry.path().string();
        std::string file_name = GetFileNameWithoutPath(n_path);
        all_files.push_back(file_name);
    }
    boost::regex pattern (file_name_pattern, boost::regex::extended);
    std::vector<std::string> names_matched;
    for(size_t n = 0; n < all_files.size(); n++){
        if(regex_match(all_files.at(n), pattern))
            names_matched.push_back(all_files.at(n));
    }
    return names_matched;
}

} // namespace tools
} // namespace analysis
