/*! Base class to merge multiple root files into a single file.
This file is part of https://github.com/hh-italian-group/TauMLTools. */

#pragma once

#include <iostream>
#include <unordered_map>
#include <TROOT.h>
#include <TKey.h>
#include <TSystem.h>
#include <TTree.h>
#include <TChain.h>
#include <TH1.h>
#include <memory>
#include "TauMLTools/Core/interface/RootExt.h"
#include "TauMLTools/Core/interface/TextIO.h"

namespace analysis {

class RootFilesMerger {
public:
    struct HistDescriptor {
        static constexpr size_t MergeThreshold = 20;
        using HistPtr = std::unique_ptr<TH1>;
        std::vector<HistPtr> hists;
        HistDescriptor();
        void AddHistogram(HistPtr&& new_hist);
        const HistPtr& GetMergedHisto() const;
        void Merge();
    };

    struct TreeDescriptor {
        using ChainPtr = std::unique_ptr<TChain>;
        static std::atomic<size_t>& NumberOfFiles();
        std::vector<std::string> file_names;
        TreeDescriptor();
        void AddFile(const std::string& file_name);
        ChainPtr CreateChain(const std::string& full_name) const;
    };

    struct Key {
        std::string dir_name, name, full_name;
        Key() {}
        Key(const std::string& _dir_name, const std::string& _name) :
            dir_name(_dir_name), name(_name), full_name(dir_name + name) {}
        bool operator==(const Key& other) const { return full_name == other.full_name; }
        bool operator<(const Key& other) const { return full_name < other.full_name; }
    };

    struct KeyHash {
        size_t operator()(const Key& key) const { return h(key.full_name); }
    private:
        std::hash<std::string> h;
    };

    using HistCollection = std::unordered_map<Key, HistDescriptor, KeyHash>;
    using TreeCollection = std::unordered_map<Key, TreeDescriptor, KeyHash>;

    struct ObjectCollection {
        HistCollection hists;
        TreeCollection trees;
    };

    using HistPtr = HistDescriptor::HistPtr;
    using ChainPtr = TreeDescriptor::ChainPtr;

    RootFilesMerger(const std::string& output, const std::vector<std::string>& input_dirs,
                    const std::string& file_name_pattern, const std::string& exclude_list,
                    const std::string& exclude_dir_list, unsigned n_threads, ROOT::ECompressionAlgorithm compression,
                    int compression_level);

    virtual ~RootFilesMerger() {}

    void Process(bool process_histograms, bool process_trees);
    static std::vector<std::string> FindInputFiles(const std::vector<std::string>& dirs,
                                                   const std::string& file_name_pattern,
                                                   const std::string& exclude_list,
                                                   const std::string& exclude_dir_list);

private:
    virtual void ProcessFile(const std::string& /*file_name*/, const std::shared_ptr<TFile>& /*file*/) {}

    static void ProcessDirectory(const std::string& file_name, const std::string& dir_name, TDirectory* dir,
                                 ObjectCollection& objects, bool process_histograms, bool process_trees);

protected:
    const std::vector<std::string> input_files;
    std::shared_ptr<TFile> output_file;
    ObjectCollection objects;
};

} // namespace analysis
