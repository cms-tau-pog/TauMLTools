/*! Merge multiple root files into a single file.
This file is part of https://github.com/hh-italian-group/AnalysisTools. */

#include <iostream>
#include <unordered_map>
#include <boost/filesystem.hpp>
#include <boost/regex.hpp>
#include <TROOT.h>
#include <TKey.h>
#include <TSystem.h>
#include <TTree.h>
#include <TChain.h>
#include <memory>
#include "AnalysisTools/Core/include/RootExt.h"
#include "AnalysisTools/Core/include/TextIO.h"
#include "AnalysisTools/Run/include/program_main.h"

struct Arguments {
    run::Argument<std::string> output{"output", "output root file"};
    run::Argument<std::vector<std::string>> input_dirs{"input-dir", "input directory"};
    run::Argument<std::string> file_name_pattern{"file-name-pattern", "regex expression to match file names",
                                                 "^.*\\.root$"};
    run::Argument<std::string> exclude_list{"exclude-list", "comma separated list of files to exclude", ""};
    run::Argument<std::string> exclude_dir_list{"exclude-dir-list",
                                                "comma separated list of directories to exclude", ""};
    run::Argument<size_t> n_threads{"n-threads", "number of threads", 1};
};

struct HistDescriptor {
    static constexpr size_t MergeThreshold = 20;
    using HistPtr = std::unique_ptr<TH1>;
    std::vector<HistPtr> hists;

    HistDescriptor() { hists.reserve(MergeThreshold + 1); }

    void AddHistogram(HistPtr&& new_hist)
    {
        hists.push_back(std::move(new_hist));
        if(hists.size() > MergeThreshold)
            Merge();
    }

    const HistPtr& GetMergedHisto() const
    {
        if(hists.size() != 1)
            throw analysis::exception("Merged histogram is not ready");
        return hists[0];
    }

    void Merge()
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
};

struct TreeDescriptor {
    using ChainPtr = std::unique_ptr<TChain>;
    static size_t NumberOfFiles;
    std::vector<std::string> file_names;

    TreeDescriptor() { file_names.reserve(NumberOfFiles); }
    void AddFile(const std::string& file_name) { file_names.push_back(file_name); }

    ChainPtr CreateChain(const std::string& full_name) const
    {
        auto chain = std::make_unique<TChain>(full_name.c_str());
        for(const auto& file_name : file_names)
            chain->AddFile(file_name.c_str());
        return chain;
    }
};
size_t TreeDescriptor::NumberOfFiles = 1;

enum class ClassInheritance { TH1, TTree, TDirectory };

struct Key {
    std::string dir_name, name, full_name;

    Key() {}
    Key(const std::string& _dir_name, const std::string& _name) :
        dir_name(_dir_name), name(_name), full_name(dir_name + name) {}
    bool operator==(const Key& other) const { return full_name == other.full_name; }
    bool operator<(const Key& other) const { return full_name < other.full_name; }
};

namespace std {
template<>
struct hash<Key> {
    size_t operator()(const Key& key) const { return h(key.full_name); }
private:
    std::hash<std::string> h;
};
}

using HistCollection = std::unordered_map<Key, HistDescriptor>;
using TreeCollection = std::unordered_map<Key, TreeDescriptor>;

struct ObjectCollection {
    HistCollection hists;
    TreeCollection trees;
};

namespace fs = boost::filesystem;
class MergeRootFiles {
public:
    using HistPtr = HistDescriptor::HistPtr;
    using ChainPtr = TreeDescriptor::ChainPtr;
    using ClassCache = std::map<std::string, ClassInheritance>;


    MergeRootFiles(const Arguments& _args) :
        args(_args), input_files(FindInputFiles(args.input_dirs(), args.file_name_pattern(), args.exclude_list(),
                                                args.exclude_dir_list())),
        output(root_ext::CreateRootFile(args.output()))
    {
        TreeDescriptor::NumberOfFiles = input_files.size();
        if(args.n_threads() > 1) {
            std::ostringstream ss;
            ss << "ROOT::EnableImplicitMT(" << args.n_threads() << ");";
            gROOT->ProcessLine(ss.str().c_str());
        }
    }

    void Run()
    {
        for(const auto& file_name : input_files)
            ProcessFile(file_name, objects);

        std::cout << "Writing histograms..." << std::endl;
        {
            std::map<Key, const HistDescriptor*> ordered_histograms;
            for(auto& hist_entry : objects.hists) {
                hist_entry.second.Merge();
                ordered_histograms[hist_entry.first] = &hist_entry.second;
            }

            for(auto& hist_entry : ordered_histograms) {
                auto dir = root_ext::GetDirectory(*output, hist_entry.first.dir_name);
                root_ext::WriteObject(*hist_entry.second->GetMergedHisto(), dir);
            }
            objects.hists.clear();
        }

        std::map<Key, const TreeDescriptor*> ordered_trees;
        for(auto& tree_entry : objects.trees)
            ordered_trees[tree_entry.first] = &tree_entry.second;
        for(auto& tree : ordered_trees) {
            std::cout << "tree: " << tree.first.full_name << std::endl;
            auto dir = root_ext::GetDirectory(*output, tree.first.dir_name);
            dir->cd();
            auto chain = tree.second->CreateChain(tree.first.full_name);
            chain->Merge(output.get(), 0, "C keep");
        }
    }

private:
    static std::vector<std::string> FindInputFiles(const std::vector<std::string>& dirs,
                                                   const std::string& file_name_pattern,
                                                   const std::string& exclude_list, const std::string& exclude_dir_list)
    {
        auto exclude_vector = analysis::SplitValueList(exclude_list, true, ",");
        std::set<std::string> exclude(exclude_vector.begin(), exclude_vector.end());

        auto exclude_dir_vector = analysis::SplitValueList(exclude_dir_list, true, ",");
        std::set<std::string> exclude_dirs(exclude_dir_vector.begin(), exclude_dir_vector.end());

        const boost::regex pattern(file_name_pattern);
        std::vector<std::string> files;
        for(const auto& dir : dirs) {
            fs::path path(dir);
            CollectInputFiles(path, files, pattern, exclude, exclude_dirs);
        }
        return files;
    }

    static void CollectInputFiles(const fs::path& dir, std::vector<std::string>& files, const boost::regex& pattern,
                                  const std::set<std::string>& exclude, const std::set<std::string>& exclude_dirs)
    {
        for(const auto& entry : boost::make_iterator_range(fs::directory_iterator(dir), {})) {
            if(fs::is_directory(entry)
                    && !exclude_dirs.count(entry.path().filename().string()))
                CollectInputFiles(entry.path(), files, pattern, exclude, exclude_dirs);
            else if(boost::regex_match(entry.path().string(), pattern)
                    && !exclude.count(entry.path().filename().string()))
                files.push_back(entry.path().string());
        }
    }

    static ClassInheritance FindClassInheritance(const std::string& class_name)
    {
        static ClassCache classes;
        auto iter = classes.find(class_name);
        if(iter != classes.end())
            return iter->second;
        TClass *cl = gROOT->GetClass(class_name.c_str());
        if(!cl)
            throw analysis::exception("Unable to get TClass for class named '%1%'.") % class_name;

        ClassInheritance inheritance;
        if(cl->InheritsFrom("TH1"))
            inheritance = ClassInheritance::TH1;
        else if(cl->InheritsFrom("TTree"))
            inheritance = ClassInheritance::TTree;
        else if(cl->InheritsFrom("TDirectory"))
            inheritance = ClassInheritance::TDirectory;
        else
            throw analysis::exception("Unknown class inheritance for class named '%1%'.") % class_name;
        classes[class_name] = inheritance;
        return inheritance;
    }

    static void ProcessFile(const std::string& file_name, ObjectCollection& objects)
    {
        std::cout << "file: " << file_name << std::endl;
        auto file = root_ext::OpenRootFile(file_name);
        ProcessDirectory(file_name, "", file.get(), objects);
    }

    static void ProcessDirectory(const std::string& file_name, const std::string& dir_name, TDirectory* dir,
                                 ObjectCollection& objects)
    {
        TIter nextkey(dir->GetListOfKeys());
        for(TKey* t_key; (t_key = dynamic_cast<TKey*>(nextkey()));) {
            const ClassInheritance inheritance = FindClassInheritance(t_key->GetClassName());
            const Key key(dir_name, t_key->GetName());

            switch (inheritance) {
                case ClassInheritance::TH1: {
                    auto hist = HistPtr(root_ext::ReadObject<TH1>(*dir, key.name));
                    hist->SetDirectory(nullptr);
                    objects.hists[key].AddHistogram(std::move(hist));
                    break;
                }
                case ClassInheritance::TTree: {
                    objects.trees[key].AddFile(file_name);
                    break;
                } case ClassInheritance::TDirectory: {
                    auto subdir = root_ext::ReadObject<TDirectory>(*dir, key.name);
                    ProcessDirectory(file_name, key.full_name + "/", subdir, objects);
                    break;
                }
            }
        }
    }

private:
    Arguments args;
    std::vector<std::string> input_files;
    std::shared_ptr<TFile> output;
    ObjectCollection objects;
};

PROGRAM_MAIN(MergeRootFiles, Arguments)

