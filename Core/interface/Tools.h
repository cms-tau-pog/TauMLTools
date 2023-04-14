/*! Common tools and definitions suitable for general purposes.
This file is part of https://github.com/cms-tau-pog/TauMLTools. */

#pragma once

#include <vector>
#include <set>
#include <algorithm>
#include <initializer_list>
#include <sstream>

namespace analysis {

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
void put_back(std::vector<Type>& /*v*/) { }

template<typename Type, typename T2, typename ...Args>
void put_back(std::vector<Type>& v, const T2& t, const Args&... args);

template<typename Type, typename T2, typename ...Args>
void put_back(std::vector<Type>& v, const std::vector<T2>& v2, const Args&... args)
{
    v.insert(v.end(), v2.begin(), v2.end());
    put_back(v, args...);
}

template<typename Type, typename T2, typename ...Args>
void put_back(std::vector<Type>& v, const T2& t, const Args&... args)
{
    v.push_back(t);
    put_back(v, args...);
}

template<typename Type, typename ...Args>
std::vector<Type> join(const Type& t, const Args&... args)
{
    std::vector<Type> result;
    put_back(result, t, args...);
    return result;
}

template<typename Type, typename ...Args>
std::vector<Type> join(const std::vector<Type>& v, const Args&... args)
{
    std::vector<Type> result;
    put_back(result, v, args...);
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

std::string FullPath(std::initializer_list<std::string> paths);
uint32_t hash(const std::string& str);
std::vector<std::string> FindFiles(const std::string& path, const std::string& file_name_pattern);

} // namespace tools
} // namespace analysis
