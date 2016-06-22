/*! Common tools and definitions suitable for general purposes.
This file is part of https://github.com/hh-italian-group/AnalysisTools. */

#pragma once

#include <vector>
#include <set>
#include <algorithm>

namespace analysis {

template<typename T>
T sqr(const T& x) { return x * x; }

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

template<typename Map>
std::set< typename Map::mapped_type > collect_map_values(const Map& map)
{
    std::set< typename Map::mapped_type > result;
    std::transform(map.begin(), map.end(), std::inserter(result, result.end()),
                   [](const typename Map::value_type& pair) { return pair.second; } );
    return result;
}

} // namespace tools
} // namespace analysis
