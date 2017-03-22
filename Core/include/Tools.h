/*! Common tools and definitions suitable for general purposes.
This file is part of https://github.com/hh-italian-group/AnalysisTools. */

#pragma once

#include <vector>
#include <set>
#include <algorithm>
#include <initializer_list>
#include <boost/crc.hpp>

namespace analysis {

template<typename T>
constexpr T sqr(const T& x) { return x * x; }

namespace tools {

template<typename Type>
std::vector<Type> join_vectors(const std::vector< const std::vector<Type>* >& inputVectors)
{
    size_t totalSize = 0;
    for(auto inputVector : inputVectors) {
        if(!inputVector)
            throw std::runtime_error("input vector is nullptr");
        totalSize += inputVector->size();
    }

    std::vector<Type> result;
    result.reserve(totalSize);
    for(auto inputVector : inputVectors)
        result.insert(result.end(), inputVector->begin(), inputVector->end());

    return result;
}

template<typename Type>
std::set<Type> union_sets(std::initializer_list<std::set<Type>> sets)
{
    std::set<Type> result;
    for(const auto& set : sets)
        result.insert(set.begin(), set.end());
    return result;
}

template<typename Container, typename T>
size_t find_index(const Container& container, const T& value)
{
    const auto iter = std::find(container.begin(), container.end(), value);
    return std::distance(container.begin(), iter);
}

template<typename Map, typename Set = std::set<typename Map::key_type>>
Set collect_map_keys(const Map& map)
{
    Set result;
    std::transform(map.begin(), map.end(), std::inserter(result, result.end()),
                   [](const typename Map::value_type& pair) { return pair.first; } );
    return result;
}

template<typename Map, typename Set = std::set<typename Map::mapped_type>>
Set collect_map_values(const Map& map)
{
    Set result;
    std::transform(map.begin(), map.end(), std::inserter(result, result.end()),
                   [](const typename Map::value_type& pair) { return pair.second; } );
    return result;
}

inline uint32_t hash(const std::string& str)
{
    boost::crc_32_type crc;
    crc.process_bytes(str.data(), str.size());
    return crc.checksum();
}

inline std::string FullPath(std::initializer_list<std::string> paths)
{
    if(!paths.size())
        return "";

    std::ostringstream full_path;
    auto iter = paths.begin();
    full_path << *iter++;
    for(; iter != paths.end(); ++iter)
        full_path << "/" << *iter;
    return full_path.str();
}

} // namespace tools
} // namespace analysis
