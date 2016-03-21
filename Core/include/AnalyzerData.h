/*! Base class for Analyzer data containers.
This file is part of https://github.com/hh-italian-group/AnalysisTools. */

#pragma once

#include <vector>
#include <map>
#include <set>
#include <stdexcept>
#include <sstream>
#include <typeindex>

#include <TH1D.h>
#include <TH2D.h>

#include "RootExt.h"
#include "SmartHistogram.h"

#define ANA_DATA_ENTRY(type, name, ...) \
    template<typename Key> \
    root_ext::SmartHistogram< type >& name(const Key& key) { \
        return Get((type*)nullptr, #name, key, ##__VA_ARGS__); \
    } \
    root_ext::SmartHistogram< type >& name() { \
        static const size_t index = GetUniqueIndex(#name); \
        return GetFast((type*)nullptr, #name, index, ##__VA_ARGS__); \
    } \
    static std::string name##_Name() { return #name; } \
    static std::type_index name##_TypeIndex() { return std::type_index(typeid(type)); } \
    /**/

#define TH1D_ENTRY(name, nbinsx, xlow, xup) ANA_DATA_ENTRY(TH1D, name, nbinsx, xlow, xup)
#define TH1D_ENTRY_FIX(name, binsizex, nbinsx, xlow) TH1D_ENTRY(name, nbinsx, xlow, (xlow+binsizex*nbinsx))
#define TH1D_ENTRY_CUSTOM(name, bins) ANA_DATA_ENTRY(TH1D, name, bins)

#define TH1D_ENTRY_EX(name, nbinsx, xlow, xup, x_axis_title, y_axis_title, use_log_y, max_y_sf, divide, store) \
    ANA_DATA_ENTRY(TH1D, name, nbinsx, xlow, xup, x_axis_title, y_axis_title, use_log_y, max_y_sf, divide, store)
#define TH1D_ENTRY_FIX_EX(name, binsizex, nbinsx, xlow, x_axis_title, y_axis_title, use_log_y, max_y_sf, divide, store) \
    TH1D_ENTRY_EX(name, nbinsx, xlow, (xlow+binsizex*nbinsx), x_axis_title, y_axis_title, use_log_y, max_y_sf, divide, store)
#define TH1D_ENTRY_CUSTOM_EX(name, bins, x_axis_title, y_axis_title, use_log_y, max_y_sf, divide, store) \
    ANA_DATA_ENTRY(TH1D, name, bins, x_axis_title, y_axis_title, use_log_y, max_y_sf, divide, store)

#define TH2D_ENTRY(name, nbinsx, xlow, xup, nbinsy, ylow, yup) \
    ANA_DATA_ENTRY(TH2D, name, nbinsx, xlow, xup, nbinsy, ylow, yup)
#define TH2D_ENTRY_FIX(name, binsizex, nbinsx, xlow, binsizey, nbinsy, ylow) \
    TH2D_ENTRY(name, nbinsx, xlow, (xlow+binsizex*nbinsx), nbinsy, ylow, (ylow+binsizey*nbinsy))

#define TH2D_ENTRY_EX(name, nbinsx, xlow, xup, nbinsy, ylow, yup, x_axis_title, y_axis_title, use_log_y, max_y_sf, \
                      store) \
    ANA_DATA_ENTRY(TH2D, name, nbinsx, xlow, xup, nbinsy, ylow, yup, x_axis_title, y_axis_title, use_log_y, max_y_sf, \
                   store)
#define TH2D_ENTRY_FIX_EX(name, binsizex, nbinsx, xlow, binsizey, nbinsy, ylow, x_axis_title, y_axis_title, \
                          use_log_y, max_y_sf, store) \
    TH2D_ENTRY_EX(name, nbinsx, xlow, (xlow+binsizex*nbinsx), nbinsy, ylow, (ylow+binsizey*nbinsy), x_axis_title, \
                  y_axis_title, use_log_y, max_y_sf, store)

namespace root_ext {
class AnalyzerData {
private:
    using DataVector = std::vector<AbstractHistogram*>;
    using DataMap = std::map<std::string, AbstractHistogram*>;

    template<typename ValueType>
    static std::set<std::string>& HistogramNames()
    {
        static std::set<std::string> names;
        return names;
    }

    template<typename ValueType>
    static std::set<std::string>& OriginalHistogramNames()
    {
        static std::set<std::string> names;
        return names;
    }

    static std::map<std::string, size_t>& IndexMap()
    {
        static std::map<std::string, size_t> index_map;
        return index_map;
    }

    static constexpr size_t MaxIndex = 1000;

public:
    template<typename ValueType>
    static const std::set<std::string>& GetAllHistogramNames() { return HistogramNames<ValueType>(); }

    template<typename ValueType>
    static const std::set<std::string>& GetOriginalHistogramNames() { return OriginalHistogramNames<ValueType>(); }

    static size_t GetUniqueIndex(const std::string& name)
    {
        const auto iter = IndexMap().find(name);
        if(iter != IndexMap().end())
            return iter->second;
        const size_t index = IndexMap().size();
        IndexMap()[name] = index;
        return index;
    }

public:
    AnalyzerData() : directory(nullptr)
    {
        data_vector.assign(MaxIndex, nullptr);
    }

    explicit AnalyzerData(const std::string& outputFileName)
        : outputFile(CreateRootFile(outputFileName))
    {
        data_vector.assign(MaxIndex, nullptr);
        directory = outputFile.get();
    }

    explicit AnalyzerData(std::shared_ptr<TFile> _outputFile, const std::string& directoryName = "")
        : outputFile(_outputFile)
    {
        if(!outputFile)
            throw analysis::exception("Output file is nullptr.");
        data_vector.assign(MaxIndex, nullptr);
        if (directoryName.size()){
            outputFile->mkdir(directoryName.c_str());
            directory = outputFile->GetDirectory(directoryName.c_str());
            if(!directory)
                throw analysis::exception("Unable to create analyzer data directory.");
        } else
            directory = outputFile.get();
    }

    virtual ~AnalyzerData()
    {
        for(const auto& iter : data) {
            if(directory)
                iter.second->WriteRootObject();
            delete iter.second;
        }
    }

    std::shared_ptr<TFile> getOutputFile() { return outputFile; }
    bool Contains(const std::string& name) const { return data.find(name) != data.end(); }

    void Erase(const std::string& name)
    {
        auto iter = data.find(name);
        if(iter != data.end()) {
            delete iter->second;
            data.erase(iter);
            auto index_iter = IndexMap().find(name);
            if(index_iter != IndexMap().end() && index_iter->second < MaxIndex)
                data_vector.at(index_iter->second) = nullptr;
        }
    }

    template<typename ValueType>
    bool CheckType(const std::string& name) const
    {
        const auto iter = data.find(name);
        if(iter == data.end())
            analysis::exception("Histogram '%1%' not found.") % name;
        SmartHistogram<ValueType>* result = dynamic_cast< SmartHistogram<ValueType>* >(iter->second);
        return result;
    }

    std::vector<std::string> KeysCollection() const
    {
        std::vector<std::string> keys;
        for(const auto& iter : data)
            keys.push_back(iter.first);
        return keys;
    }

    template<typename ValueType, typename KeySuffix, typename ...Args>
    SmartHistogram<ValueType>& Get(const ValueType* ptr, const std::string& name, const KeySuffix& suffix, Args... args)
    {

        std::ostringstream ss_suffix;
        ss_suffix << suffix;
        const std::string s_suffix = ss_suffix.str();
        const std::string full_name = s_suffix.size() ? name + "_" + s_suffix : name;
        return GetByFullName(ptr, name, full_name, args...);
    }

    template<typename ValueType>
    SmartHistogram<ValueType>& Get(const ValueType* null_value, const std::string& name)
    {
        return Get(null_value, name, "");
    }

    template<typename ValueType>
    SmartHistogram<ValueType>& Get(const std::string& name)
    {
        return Get((ValueType*)nullptr, name, "");
    }

    template<typename ValueType>
    SmartHistogram<ValueType>* GetPtr(const std::string& name) const
    {
        if(!Contains(name) || !CheckType<ValueType>(name)) return nullptr;
        return &GetAt<ValueType>(data.find(name));
    }

    template<typename ValueType>
    SmartHistogram<ValueType>& Clone(const SmartHistogram<ValueType>& original)
    {
        if(data.count(original.Name()))
            throw analysis::exception("histogram already exists");
        SmartHistogram<ValueType>* h = new SmartHistogram<ValueType>(original);
        data[h->Name()] = h;
        HistogramNames<ValueType>().insert(h->Name());
        h->SetOutputDirectory(directory);
        auto index_iter = IndexMap().find(h->Name());
        if(index_iter != IndexMap().end() && index_iter->second < MaxIndex)
            data_vector.at(index_iter->second) = h;
        return *h;
    }

protected:
    template<typename ValueType, typename ...Args>
    SmartHistogram<ValueType>& GetFast(const ValueType* ptr, const std::string& name, size_t index, Args... args)
    {
        if(index < MaxIndex && data_vector[index] != nullptr)
            return *static_cast< SmartHistogram<ValueType>* >(data_vector[index]);
        return GetByFullName(ptr, name, name, args...);
    }

private:
    template<typename ValueType, typename ...Args>
    SmartHistogram<ValueType>& GetByFullName(const ValueType*, const std::string& name, const std::string& full_name,
                                             Args... args)
    {
        auto iter = data.find(full_name);
        if(iter == data.end()) {
            AbstractHistogram* h = HistogramFactory<ValueType>::Make(full_name, args...);
            data[full_name] = h;
            HistogramNames<ValueType>().insert(h->Name());
            OriginalHistogramNames<ValueType>().insert(name);
            h->SetOutputDirectory(directory);
            iter = data.find(full_name);
            auto index_iter = IndexMap().find(full_name);
            if(index_iter != IndexMap().end() && index_iter->second < MaxIndex)
                data_vector.at(index_iter->second) = h;
        }
        return GetAt<ValueType>(iter);
    }

    template<typename ValueType>
    SmartHistogram<ValueType>& GetAt(const DataMap::const_iterator& iter) const
    {
        if(iter == data.end())
            throw analysis::exception("Invalid iterator to of a histogram collection.");

        SmartHistogram<ValueType>* result = dynamic_cast< SmartHistogram<ValueType>* >(iter->second);
        if(!result)
            throw analysis::exception("Wrong type for histogram '%1%'.") % iter->first;
        return *result;
    }

private:
    std::shared_ptr<TFile> outputFile;
    TDirectory* directory;

    DataMap data;
    DataVector data_vector;
};
} // root_ext
