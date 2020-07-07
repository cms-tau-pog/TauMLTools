/*! Base class for Analyzer data containers.
This file is part of https://github.com/hh-italian-group/TauMLTools. */

#include "TauMLTools/Core/interface/AnalyzerData.h"

#include <vector>
#include <unordered_map>
#include <utility>
#include "TauMLTools/Core/interface/RootExt.h"
#include "TauMLTools/Core/interface/TextIO.h"
#include "TauMLTools/Core/interface/SmartHistogram.h"

namespace root_ext {

AnalyzerDataEntryBase::AnalyzerDataEntryBase(const std::string& _name, AnalyzerData* _data)
    : name(_name), data(_data)
{
    data->AddEntry(*this);
}
const std::string& AnalyzerDataEntryBase::Name() const { return name; }
AnalyzerDataEntryBase::Mutex& AnalyzerDataEntryBase::GetMutex() { return mutex; }


AnalyzerData::AnalyzerData() : directory(nullptr), readMode(false), mutex(std::make_unique<Mutex>()) {}

AnalyzerData::AnalyzerData(const std::string& outputFileName) :
    outputFile(CreateRootFile(outputFileName)), directory(outputFile.get()), readMode(false),
    mutex(std::make_unique<Mutex>()) {}

AnalyzerData::AnalyzerData(std::shared_ptr<TFile> _outputFile, const std::string& directoryName,
                           bool _readMode) :
    outputFile(_outputFile), readMode(_readMode), mutex(std::make_unique<Mutex>())
{
    if(!outputFile)
        throw analysis::exception("Output file is nullptr.");
    directory = directoryName.size() ? GetDirectory(*outputFile, directoryName, true) : outputFile.get();
}

AnalyzerData::AnalyzerData(TDirectory* _directory, const std::string& subDirectoryName, bool _readMode) :
    readMode(_readMode), mutex(std::make_unique<Mutex>())
{
    if(!_directory)
        throw analysis::exception("Output directory is nullptr.");
    directory = subDirectoryName.size() ? GetDirectory(*_directory, subDirectoryName, true) : _directory;
}

AnalyzerData::~AnalyzerData()
{
    if(directory && !readMode) {
        for(const auto& hist : histograms)
            hist.second->WriteRootObject();
    }
}

TDirectory* AnalyzerData::GetOutputDirectory() const { return directory; }
std::shared_ptr<TFile> AnalyzerData::GetOutputFile() const { return outputFile; }
bool AnalyzerData::ReadMode() const { return readMode; }
AnalyzerData::Mutex& AnalyzerData::GetMutex() const { return *mutex; }

void AnalyzerData::AddHistogram(HistPtr hist)
{
    std::lock_guard<Mutex> lock(*mutex);
    if(!hist)
        throw analysis::exception("Can't add nullptr histogram into AnalyzerData");
    if(histograms.count(hist->Name()))
        throw analysis::exception("Histogram '%1%' already exists in this AnalyzerData.") % hist->Name();
    TDirectory* hist_dir = readMode ? nullptr : directory;
    hist->SetOutputDirectory(hist_dir);
    histograms[hist->Name()] = hist;
}
const AnalyzerData::HistContainer& AnalyzerData::GetHistograms() const { return histograms; }

void AnalyzerData::AddEntry(Entry& entry)
{
    std::lock_guard<Mutex> lock(*mutex);
    if(entries.count(entry.Name()))
        throw analysis::exception("Entry '%1%' already exists in this AnalyzerData.") % entry.Name();
    entries[entry.Name()] = &entry;
}
const AnalyzerData::EntryContainer& AnalyzerData::GetEntries() const { return entries; }

} // root_ext
