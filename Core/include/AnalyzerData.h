/*! Base class for Analyzer data containers.
This file is part of https://github.com/hh-italian-group/AnalysisTools. */

#pragma once

#include <vector>
#include <unordered_map>
#include <utility>

#include "RootExt.h"
#include "TextIO.h"
#include "SmartHistogram.h"

#define ANA_DATA_ENTRY(type, name, ...) \
    root_ext::AnalyzerDataEntry<type> name{#name, this, ##__VA_ARGS__};
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
#define TH2D_ENTRY_CUSTOM(name, binsx, binsy) ANA_DATA_ENTRY(TH2D, name, binsx, binsy)

#define GRAPH_ENTRY(name) ANA_DATA_ENTRY(TGraph, name)

namespace root_ext {

class AnalyzerData;

struct AnalyzerDataEntryBase {
    inline AnalyzerDataEntryBase(const std::string& _name, AnalyzerData* _data);
    virtual ~AnalyzerDataEntryBase() {}
    const std::string& Name() const { return name; }
private:
    std::string name;
protected:
    AnalyzerData* data;
};

template<typename _ValueType>
struct AnalyzerDataEntry;

class AnalyzerData {
public:
    using Hist = AbstractHistogram;
    using HistPtr = std::shared_ptr<Hist>;
    using HistContainer = std::unordered_map<std::string, HistPtr>;
    using Entry = AnalyzerDataEntryBase;
    using EntryContainer = std::unordered_map<std::string, Entry*>;

    AnalyzerData() : directory(nullptr), readMode(false) {}

    explicit AnalyzerData(const std::string& outputFileName) :
        outputFile(CreateRootFile(outputFileName)), directory(outputFile.get()), readMode(false) {}

    explicit AnalyzerData(std::shared_ptr<TFile> _outputFile, const std::string& directoryName = "",
                          bool _readMode = false) :
        outputFile(_outputFile), readMode(_readMode)
    {
        if(!outputFile)
            throw analysis::exception("Output file is nullptr.");
        directory = directoryName.size() ? GetDirectory(*outputFile, directoryName, true) : outputFile.get();
    }

    explicit AnalyzerData(TDirectory* _directory, const std::string& subDirectoryName = "", bool _readMode = false) :
        readMode(_readMode)
    {
        if(!_directory)
            throw analysis::exception("Output directory is nullptr.");
        directory = subDirectoryName.size() ? GetDirectory(*_directory, subDirectoryName, true) : _directory;
    }

    virtual ~AnalyzerData()
    {
        if(directory && !readMode) {
            for(const auto& hist : histograms)
                hist.second->WriteRootObject();
        }
    }

    TDirectory* GetOutputDirectory() const { return directory; }
    std::shared_ptr<TFile> GetOutputFile() const { return outputFile; }

    void AddHistogram(HistPtr hist)
    {
        if(!hist)
            throw analysis::exception("Can't add nullptr histogram into AnalyzerData");
        if(histograms.count(hist->Name()))
            throw analysis::exception("Histogram '%1%' already exists in this AnalyzerData.") % hist->Name();
        hist->SetOutputDirectory(directory);
        histograms[hist->Name()] = hist;
    }
    const HistContainer& GetHistograms() const { return histograms; }

    template<typename Histogram>
    std::map<std::string, std::shared_ptr<SmartHistogram<Histogram>>> GetHistogramsEx() const
    {
        std::map<std::string, std::shared_ptr<SmartHistogram<Histogram>>> result;
        for(const auto& hist_entry : histograms) {
            auto smart_hist = std::dynamic_pointer_cast<SmartHistogram<Histogram>>(hist_entry.second);
            if(smart_hist)
                result[hist_entry.first] = smart_hist;
        }
        return result;
    }

    template<typename Histogram>
    std::shared_ptr<SmartHistogram<Histogram>> TryGetHistogramEx(const std::string& name) const
    {
        if(!histograms.count(name))
            return std::shared_ptr<SmartHistogram<Histogram>>();
        const auto& hist = histograms.at(name);
        return std::dynamic_pointer_cast<SmartHistogram<Histogram>>(hist);
    }

    void AddEntry(Entry& entry)
    {
        if(entries.count(entry.Name()))
            throw analysis::exception("Entry '%1%' already exists in this AnalyzerData.") % entry.Name();
        entries[entry.Name()] = &entry;
    }
    const EntryContainer& GetEntries() const { return entries; }

    template<typename Histogram>
    std::map<std::string, AnalyzerDataEntry<Histogram>*> GetEntriesEx() const;
    template<typename Histogram>
    AnalyzerDataEntry<Histogram>& GetEntryEx(const std::string& name) const;

private:
    std::shared_ptr<TFile> outputFile;
    TDirectory* directory;
    bool readMode;
    EntryContainer entries;
    HistContainer histograms;
};

AnalyzerDataEntryBase::AnalyzerDataEntryBase(const std::string& _name, AnalyzerData* _data)
    : name(_name), data(_data)
{
    data->AddEntry(*this);
}

template<typename _ValueType>
struct AnalyzerDataEntry : AnalyzerDataEntryBase  {
    using ValueType = _ValueType;
    using Hist = SmartHistogram<ValueType>;
    using HistPtr = std::shared_ptr<Hist>;
    using HistPtrMap = std::unordered_map<std::string, HistPtr>;
    using RootContainer = typename Hist::RootContainer;

    AnalyzerDataEntry(const std::string& _name, AnalyzerData* data) :
        AnalyzerDataEntryBase(_name, data)
    {
    }

    template<typename... Args>
    AnalyzerDataEntry(const std::string& _name, AnalyzerData* data, Args&&... args) :
        AnalyzerDataEntryBase(_name, data)
    {
        SetMasterHist(std::forward<Args>(args)...);
    }

    Hist& operator()()
    {
        if(!default_hist) {
            default_hist = std::make_shared<Hist>(GetMasterHist());
            histograms[""] = default_hist;
            data->AddHistogram(default_hist);
        }
        return *default_hist;
    }

    template<typename ...KeySuffix>
    Hist& operator()(KeySuffix&&... suffix)
    {
        const auto key = SuffixToKey(std::forward<KeySuffix>(suffix)...);
        if(key == "")
            return (*this)();
        auto iter = histograms.find(key);
        if(iter != histograms.end())
            return *iter->second;
        auto hist = std::make_shared<Hist>(GetMasterHist());
        hist->SetName(FullName(key));
        data->AddHistogram(hist);
        histograms[key] = hist;
        return *hist;
    }

    template<typename KeySuffix, typename... Args>
    void Emplace(KeySuffix&& suffix, Args&&... args)
    {
        const auto key = SuffixToKey(std::forward<KeySuffix>(suffix));
        auto iter = histograms.find(key);
        if(iter != histograms.end())
            throw analysis::exception("Histogram with suffix '%1%' already exists in '%2%'.") % key % Name();
        auto hist = std::make_shared<Hist>(FullName(key), std::forward<Args>(args)...);
        data->AddHistogram(hist);
        histograms[key] = hist;
    }

    template<typename KeySuffix>
    void Set(KeySuffix&& suffix, HistPtr hist)
    {
        const auto key = SuffixToKey(std::forward<KeySuffix>(suffix));
        auto iter = histograms.find(key);
        if(iter != histograms.end())
            throw analysis::exception("Histogram with suffix '%1%' already exists in '%2%'.") % key % Name();
        data->AddHistogram(hist);
        histograms[key] = hist;
    }

    const HistPtrMap& GetHistograms() const { return histograms; }

    const Hist& GetMasterHist()
    {
        if(!master_hist)
            throw analysis::exception("Master histogram for '%1%' is not initialized.") % Name();
        return *master_hist;
    }

    template<typename... Args>
    void SetMasterHist(Args&&... args)
    {
        master_hist = std::make_shared<Hist>(Name(), std::forward<Args>(args)...);
        master_hist->SetOutputDirectory(nullptr);
    }

    std::string FullName(const std::string& key) const { return Name() + "_" + key; }

    Hist& Read() { return ReadFromDirectory((*this)()); }
    template<typename KeySuffix>
    Hist& Read(KeySuffix&& suffix) { return ReadFromDirectory((*this)(std::forward<KeySuffix>(suffix))); }

    static std::string SuffixToKey() { return ""; }

    template<typename T, typename ...KeySuffix>
    static std::string SuffixToKey(T&& first_suffix, KeySuffix&&... suffix)
    {
        std::ostringstream ss_suffix;
        ss_suffix << first_suffix;
        const auto other_suffix = SuffixToKey(std::forward<KeySuffix>(suffix)...);
        if(other_suffix.size())
            ss_suffix << "_" << other_suffix;
        return ss_suffix.str();
    }

private:
    Hist& ReadFromDirectory(Hist& hist)
    {
        auto dir = hist.GetOutputDirectory();
        auto original_hist = ReadObject<RootContainer>(dir, hist.Name());
        hist.CopyContent(original_hist);
        delete original_hist;
        return hist;
    }

private:
    HistPtr master_hist, default_hist;
    HistPtrMap histograms;
};

template<typename Histogram>
std::map<std::string, AnalyzerDataEntry<Histogram>*> AnalyzerData::GetEntriesEx() const
{
    std::map<std::string, AnalyzerDataEntry<Histogram>*> result;
    for(const auto& entry : entries) {
        auto entry_ptr = dynamic_cast<AnalyzerDataEntry<Histogram>*>(entry.second);
        if(entry_ptr)
            result[entry.first] = entry_ptr;
    }
    return result;
}

template<typename Histogram>
AnalyzerDataEntry<Histogram>& AnalyzerData::GetEntryEx(const std::string& name) const
{
    auto iter = entries.find(name);
    if(iter == entries.end())
        throw analysis::exception("Entry with name '%1%' not found.") % name;
    auto entry_ptr = dynamic_cast<AnalyzerDataEntry<Histogram>*>(iter->second);
    if(!entry_ptr)
        throw analysis::exception("Unexpected entry type for the entry with name '%1%'.") % name;
    return *entry_ptr;
}

} // root_ext
