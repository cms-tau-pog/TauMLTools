/*! Merge multiple root files into a single file.
Some parts of the code are taken from copyFile.C written by Rene Brun.
This file is part of https://github.com/hh-italian-group/AnalysisTools. */

#include <iostream>
#include <boost/filesystem.hpp>
#include <boost/regex.hpp>
#include <TROOT.h>
#include <TKey.h>
#include <TSystem.h>
#include <TTree.h>
#include <TChain.h>
#include <memory>
#include "RootExt.h"
#include "AnalysisTools/Run/include/program_main.h"

struct Arguments {
    run::Argument<std::string> output{"output", "output root file"};
    run::Argument<size_t> target_size{"max-size", "max size of the output file in MB", 7500};
    run::Argument<std::string> priority_trees{"priority-trees", "list of trees that should be saved first", ""};
    run::Argument<std::vector<std::string>> input_dirs{"input-dir", "input directory"};

};

struct HistDescriptor {
    using HistPtr = std::shared_ptr<TH1>;
    HistPtr hist;
    TDirectory* dir;
};

struct TreeDescriptor {
    using ChainPtr = std::shared_ptr<TChain>;
    ChainPtr chain;
    TDirectory* dir;
};

namespace fs = boost::filesystem;
class MergeRootFiles {
public:
    using HistPtr = HistDescriptor::HistPtr;
    using ChainPtr = TreeDescriptor::ChainPtr;
    using HistCollection = std::map<std::string, HistDescriptor>;
    using TreeCollection = std::map<std::string, TreeDescriptor>;

    MergeRootFiles(const Arguments& _args) :
        args(_args), input_files(FindInputFiles(args.input_dirs())), output(root_ext::CreateRootFile(args.output()))
    {}

    void Run()
    {
        for(const auto& file_name : input_files) {
            std::cout << "file: " << file_name << std::endl;
            auto file = root_ext::OpenRootFile(file_name);
            ProcessDirectory(file_name, "", file.get());
        }
        std::cout << "Writing histograms..." << std::endl;
        for(auto& hist_entry : histograms) {
            root_ext::WriteObject(*hist_entry.second.hist, hist_entry.second.dir);
        }
        for(auto& tree : trees) {
            std::cout << "tree: " << tree.first << std::endl;
            tree.second.dir->cd();
            tree.second.chain->Merge(output.get(), 0, "C keep");
        }
    }

private:
    static std::vector<std::string> FindInputFiles(const std::vector<std::string>& dirs)
    {
        std::vector<std::string> files;
        for(const auto& dir : dirs) {
            fs::path path(dir);
            CollectInputFiles(path, files);
        }
        return files;
    }

    static void CollectInputFiles(const fs::path& dir, std::vector<std::string>& files)
    {
        static const boost::regex root_file_pattern("^.*\\.root$");
        for(const auto& entry : boost::make_iterator_range(fs::directory_iterator(dir), {})) {
            if(fs::is_directory(entry))
                CollectInputFiles(entry.path(), files);
            else if(boost::regex_match(entry.path().string(), root_file_pattern))
                files.push_back(entry.path().string());
        }
    }

    void ProcessDirectory(const std::string& file_name, const std::string& dir_name, TDirectory* dir)
    {
        TIter nextkey(dir->GetListOfKeys());
        for(TKey* key; (key = dynamic_cast<TKey*>(nextkey()));) {
            const char *classname = key->GetClassName();
            TClass *cl = gROOT->GetClass(classname);
            if (!cl) continue;
            std::string name = key->GetName();
            std::string full_name = dir_name + name;
            if (cl->InheritsFrom("TDirectory")) {
                auto subdir = root_ext::ReadObject<TDirectory>(*dir, name);
                ProcessDirectory(file_name, dir_name + subdir->GetName() + "/", subdir);
            } else if(cl->InheritsFrom("TH1")) {
                auto hist = HistPtr(root_ext::ReadObject<TH1>(*dir, name));
                if(histograms.count(full_name)) {
                    TList list;
                    list.Add(hist.get());
                    histograms.at(full_name).hist->Merge(&list);
                } else {
                    histograms[full_name].hist = HistPtr(root_ext::CloneObject(*hist, "", true));
                    histograms[full_name].dir = root_ext::GetDirectory(*output, dir_name);
                }

            } else if(cl->InheritsFrom("TTree")) {
                if(!trees.count(full_name)) {
                    trees[full_name].chain = std::make_shared<TChain>(full_name.c_str());
                    trees[full_name].dir = root_ext::GetDirectory(*output, dir_name);
                }
                trees.at(full_name).chain->AddFile(file_name.c_str());
            } else {
                throw analysis::exception("Unknown objecttype");
            }
        }
    }

private:
    Arguments args;
    std::vector<std::string> input_files;
    std::shared_ptr<TFile> output;
    HistCollection histograms;
    TreeCollection trees;
};

PROGRAM_MAIN(MergeRootFiles, Arguments)
