/*! Definition of the map class that keeps a trace of the insertion order.
This file is part of https://github.com/hh-italian-group/TauMLTools. */

#pragma once

#include <map>

namespace analysis {

template<typename _Key, class _Tp, class _Compare = std::less<_Key>>
class map_vec {
public:
    using std_map = std::map<_Key, _Tp, _Compare>;
    using key_type = typename std_map::key_type;
    using mapped_type = typename std_map::mapped_type;
    using value_type = typename std_map::value_type;
    using key_compare = typename std_map::key_compare;
    using reference =  typename std_map::reference;
    using const_reference = typename std_map::const_reference;
    using iterator = typename std_map::iterator;
    using const_iterator = typename std_map::const_iterator;
    using size_type = typename std_map::size_type;
    using std_vec = std::vector<std::pair<key_type, const mapped_type*>>;

    iterator begin() { return map.begin(); }
    const_iterator begin() const { return map.begin(); }
    iterator end() { return map.end(); }
    const_iterator end() const { return map.end(); }
    mapped_type& at(const key_type& key) { return map.at(key); }
    const mapped_type& at(const key_type& key) const { return map.at(key); }

    bool empty() const { return map.empty(); }
    size_type size() const { return map.size(); }
    size_type count(const key_type& key) const { return map.count(key); }
    iterator find(const key_type& key) { return map.find(key); }
    const_iterator find(const key_type& key) const { return map.find(key); }

    mapped_type& operator[](const key_type& key)
    {
        if(!map.count(key)) {
            mapped_type& v = map[key];
            vec.push_back({key, &v});
        }
        return map[key];
    }

    std::pair<iterator, bool> insert(const value_type& value)
    {
        auto result = map.insert(value);
        if(result.second) {
            mapped_type& v = *result.first;
            vec.push_back({value.first, &v});
        }
        return result;
    }

    void clear()
    {
        map.clear();
        vec.clear();
    }

    const std_vec& get_ordered_by_insertion() const { return vec; }

private:
    std_map map;
    std_vec vec;
};

} // namespace analysis
