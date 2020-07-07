/*! Base class to merge multiple root files into a single file.
This file is part of https://github.com/hh-italian-group/TauMLTools. */

#include "TauMLTools/Core/interface/RootFilesMerger.h"

#include <iostream>
#include <unordered_map>
#include <boost/filesystem.hpp>
#include <boost/regex.hpp>
#include <TROOT.h>
#include <TKey.h>
#include <TSystem.h>
#include <TTree.h>
#include <TChain.h>
#include <TH1.h>
#include <memory>
#include "TauMLTools/Core/interface/RootExt.h"
#include "TauMLTools/Core/interface/TextIO.h"

namespace {
void CollectInputFiles(const boost::filesystem::path& dir, std::vector<std::string>& files,
                       const boost::regex& pattern, const std::set<std::string>& exclude,
                       const std::set<std::string>& exclude_dirs)
{
    for(const auto& entry : boost::make_iterator_range(boost::filesystem::directory_iterator(dir), {})) {
        if(boost::filesystem::is_directory(entry)
                && !exclude_dirs.count(entry.path().filename().string()))
            CollectInputFiles(entry.path(), files, pattern, exclude, exclude_dirs);
        else if(boost::regex_match(entry.path().string(), pattern)
                && !exclude.count(entry.path().filename().string()))
            files.push_back(entry.path().string());
    }
}
}

namespace analysis {

RootFilesMerger::HistDescriptor::HistDescriptor() { hists.reserve(MergeThreshold + 1); }

void RootFilesMerger::HistDescriptor::AddHistogram(HistPtr&& new_hist)
{
    hists.push_back(std::move(new_hist));
    if(hists.size() > MergeThreshold)
        Merge();
}

const RootFilesMerger::HistPtr& RootFilesMerger::HistDescriptor::GetMergedHisto() const
{
    if(hists.size() != 1)
        throw analysis::exception("Merged histogram is not ready");
    return hists[0];
}

void RootFilesMerger::HistDescriptor::Merge()
{
    if(hists.size() <= 1) return;
    {
        TList list;
        for(size_t n = 1; n < hists.size(); ++n)
            list.Add(hists[n].get());
        hists[0]->Merge(&list);
    }
    hists.resize(1);
}

std::atomic<size_t>& RootFilesMerger::TreeDescriptor::NumberOfFiles()
{
    static std::atomic<size_t> n_files = 1;
    return n_files;
}

RootFilesMerger::TreeDescriptor::TreeDescriptor() { file_names.reserve(NumberOfFiles()); }
void RootFilesMerger::TreeDescriptor::AddFile(const std::string& file_name) { file_names.push_back(file_name); }

RootFilesMerger::TreeDescriptor::ChainPtr RootFilesMerger::TreeDescriptor::CreateChain(
    const std::string& full_name) const
{
    auto chain = std::make_unique<TChain>(full_name.c_str());
    for(const auto& file_name : file_names)
        chain->AddFile(file_name.c_str());
    return chain;
}

RootFilesMerger::RootFilesMerger(const std::string& output, const std::vector<std::string>& input_dirs,
                const std::string& file_name_pattern, const std::string& exclude_list,
                const std::string& exclude_dir_list, unsigned n_threads, ROOT::ECompressionAlgorithm compression,
                int compression_level) :
    input_files(FindInputFiles(input_dirs, file_name_pattern, exclude_list, exclude_dir_list)),
    output_file(root_ext::CreateRootFile(output, compression, compression_level))
{
    TreeDescriptor::NumberOfFiles() = input_files.size();
    if(n_threads > 1)
        ROOT::EnableImplicitMT(n_threads);
}

void RootFilesMerger::Process(bool process_histograms, bool process_trees)
{
    for(const auto& file_name : input_files) {
        std::cout << "file: " << file_name << std::endl;
        auto file = root_ext::OpenRootFile(file_name);
        ProcessDirectory(file_name, "", file.get(), objects, process_histograms, process_trees);
        ProcessFile(file_name, file);
    }

    if(process_histograms) {
        std::cout << "Writing histograms..." << std::endl;
        {
            std::map<Key, const HistDescriptor*> ordered_histograms;
            for(auto& hist_entry : objects.hists) {
                hist_entry.second.Merge();
                ordered_histograms[hist_entry.first] = &hist_entry.second;
            }

            for(auto& hist_entry : ordered_histograms) {
                auto dir = root_ext::GetDirectory(*output_file, hist_entry.first.dir_name);
                root_ext::WriteObject(*hist_entry.second->GetMergedHisto(), dir);
            }
            objects.hists.clear();
        }
    }

    if(process_trees) {
        std::map<Key, const TreeDescriptor*> ordered_trees;
        for(auto& tree_entry : objects.trees)
            ordered_trees[tree_entry.first] = &tree_entry.second;
        for(auto& tree : ordered_trees) {
            std::cout << "tree: " << tree.first.full_name << std::endl;
            auto dir = root_ext::GetDirectory(*output_file, tree.first.dir_name);
            Long64_t n_entries = -1;
            dir->cd();
            {
                auto chain = tree.second->CreateChain(tree.first.full_name);
                n_entries = chain->GetEntries();
                chain->Merge(output_file.get(), 0, "C keep");
            }
            std::unique_ptr<TTree> merged_tree(root_ext::ReadObject<TTree>(*output_file, tree.first.full_name));
            if(merged_tree->GetEntries() != n_entries)
                throw analysis::exception("Not all files were merged for '%1%' tree.") % tree.first.full_name;
        }
    }
}

std::vector<std::string> RootFilesMerger::FindInputFiles(const std::vector<std::string>& dirs,
                                               const std::string& file_name_pattern,
                                               const std::string& exclude_list,
                                               const std::string& exclude_dir_list)
{
    auto exclude_vector = analysis::SplitValueList(exclude_list, true, ",");
    std::set<std::string> exclude(exclude_vector.begin(), exclude_vector.end());

    auto exclude_dir_vector = analysis::SplitValueList(exclude_dir_list, true, ",");
    std::set<std::string> exclude_dirs(exclude_dir_vector.begin(), exclude_dir_vector.end());

    const boost::regex pattern(file_name_pattern);
    std::vector<std::string> files;
    for(const auto& dir : dirs) {
        boost::filesystem::path path(dir);
        CollectInputFiles(path, files, pattern, exclude, exclude_dirs);
    }
    return files;
}

void RootFilesMerger::ProcessDirectory(const std::string& file_name, const std::string& dir_name, TDirectory* dir,
                             ObjectCollection& objects, bool process_histograms, bool process_trees)
{
    using ClassInheritance = root_ext::ClassInheritance;
    TIter nextkey(dir->GetListOfKeys());
    for(TKey* t_key; (t_key = dynamic_cast<TKey*>(nextkey()));) {
        const ClassInheritance inheritance = root_ext::FindClassInheritance(t_key->GetClassName());
        const Key key(dir_name, t_key->GetName());

        switch (inheritance) {
            case ClassInheritance::TH1: {
                if(process_histograms) {
                    auto hist = HistPtr(root_ext::ReadObject<TH1>(*dir, key.name));
                    hist->SetDirectory(nullptr);
                    objects.hists[key].AddHistogram(std::move(hist));
                }
                break;
            }
            case ClassInheritance::TTree: {
                if(process_trees) {
                    objects.trees[key].AddFile(file_name);
                }
                break;
            } case ClassInheritance::TDirectory: {
                auto subdir = root_ext::ReadObject<TDirectory>(*dir, key.name);
                ProcessDirectory(file_name, key.full_name + "/", subdir, objects, process_histograms,
                                 process_trees);
                break;
            }
        }
    }
}

} // namespace analysis
