/*! Definition of SmartTree class.
This file is part of https://github.com/hh-italian-group/AnalysisTools. */

#pragma once

#include <stdexcept>
#include <sstream>
#include <memory>
#include <iostream>

#include <TFile.h>
#include <TTree.h>
#include <Rtypes.h>

#define DECLARE_BRANCH_VARIABLE(type, name) type name;
#define ADD_DATA_TREE_BRANCH(name) AddBranch(#name, data.name);

#define DECLARE_TREE(namespace_name, data_class_name, tree_class_name, data_macro, tree_name) \
    namespace namespace_name { \
    struct data_class_name : public root_ext::detail::BaseDataClass { data_macro() }; \
    using data_class_name##Vector = std::vector< data_class_name >; \
    class tree_class_name : public root_ext::detail::BaseSmartTree<data_class_name> { \
    public: \
        static const std::string& Name() { static const std::string name = tree_name; return name; } \
        tree_class_name(TDirectory* directory, bool readMode) \
            : BaseSmartTree(Name(), directory, readMode) { Initialize(); } \
        tree_class_name(const std::string& name, TDirectory* directory, bool readMode) \
            : BaseSmartTree(name, directory, readMode) { Initialize(); } \
    private: \
        inline void Initialize(); \
    }; \
    } \
    /**/

#define INITIALIZE_TREE(namespace_name, tree_class_name, data_macro) \
    namespace namespace_name { \
        inline void tree_class_name::Initialize() { \
            data_macro() \
            if (GetEntries() > 0) GetEntry(0); \
        } \
    } \
    /**/

namespace root_ext {
namespace detail {
    struct BaseSmartTreeEntry {
        virtual ~BaseSmartTreeEntry() {}
        virtual void clear() = 0;
    };

    template<typename DataType>
    struct SmartTreeVectorPtrEntry : public BaseSmartTreeEntry {
        std::vector<DataType>* value;
        SmartTreeVectorPtrEntry(std::vector<DataType>& origin)
            : value(&origin) {}
        virtual void clear() { value->clear(); }
    };

    struct BaseDataClass {
        virtual ~BaseDataClass() {}
    };

    using SmartTreeEntryMap = std::map<std::string, std::shared_ptr<detail::BaseSmartTreeEntry>>;

    inline void EnableBranch(TTree *tree, const std::string& branch_name)
    {
        UInt_t n_found = 0;
        tree->SetBranchStatus(branch_name.c_str(), 1, &n_found);
        if(n_found != 1) {
            std::ostringstream ss;
            ss << "Branch '" << branch_name << "' not found.";
            throw std::runtime_error(ss.str());
        }
    }

    template<typename DataType>
    struct BranchCreator {
        static void Create(TTree *tree, const std::string& branch_name, DataType& value, bool readMode,
                           SmartTreeEntryMap& entries)
        {
            if(readMode) {
                try {
                    EnableBranch(tree, branch_name);
                    tree->SetBranchAddress(branch_name.c_str(), &value);
                    if(tree->GetReadEntry() >= 0)
                        tree->GetBranch(branch_name.c_str())->GetEntry(tree->GetReadEntry());
                } catch(std::runtime_error& error) {
                    std::cerr << "ERROR: " << error.what() << std::endl;
                }
            } else {
                TBranch* branch = tree->Branch(branch_name.c_str(), &value);
                const Long64_t n_entries = tree->GetEntries();
                for(Long64_t n = 0; n < n_entries; ++n)
                    branch->Fill();
            }
        }
    };

    template<typename DataType>
    struct BranchCreator<std::vector<DataType>> {
        static void Create(TTree *tree, const std::string& branch_name, std::vector<DataType>& value, bool readMode,
                           SmartTreeEntryMap& entries)
        {
            using PtrEntry = detail::SmartTreeVectorPtrEntry<DataType>;
            std::shared_ptr<PtrEntry> entry(new PtrEntry(value));
            if(entries.count(branch_name))
                throw std::runtime_error("Entry is already defined.");
            entries[branch_name] = entry;
            if(readMode) {
                try {
                    EnableBranch(tree, branch_name);
                    tree->SetBranchAddress(branch_name.c_str(), &entry->value);
                    if(tree->GetReadEntry() >= 0)
                        tree->GetBranch(branch_name.c_str())->GetEntry(tree->GetReadEntry());
                } catch(std::runtime_error& error) {
                    std::cerr << "ERROR: " << error.what() << std::endl;
                }
            } else {
                TBranch* branch = tree->Branch(branch_name.c_str(), entry->value);
                const Long64_t n_entries = tree->GetEntries();
                for(Long64_t n = 0; n < n_entries; ++n)
                    branch->Fill();
            }
        }
    };

} // detail

class SmartTree {
public:
    SmartTree(const std::string& _name, TDirectory* _directory, bool _readMode)
        : name(_name), directory(_directory), readMode(_readMode)
    {
        static const Long64_t maxVirtualSize = 10000000;

        if(readMode) {
            if(!directory)
                throw std::runtime_error("Can't read tree from nonexistent directory.");
            tree = dynamic_cast<TTree*>(directory->Get(name.c_str()));
            if(!tree)
                throw std::runtime_error("Tree not found.");
            if(tree->GetNbranches())
                tree->SetBranchStatus("*", 0);
        } else {
            tree = new TTree(name.c_str(), name.c_str());
            tree->SetDirectory(directory);
            if(directory)
                tree->SetMaxVirtualSize(maxVirtualSize);
        }
    }

    SmartTree(const SmartTree&& other)
    {
        name = other.name;
        directory = other.directory;
        entries = other.entries;
        readMode = other.readMode;
        tree = other.tree;
    }

    virtual ~SmartTree()
    {
        if(directory) directory->Delete(name.c_str());
        else delete tree;
    }

    void Fill()
    {
        tree->Fill();
        for(auto& entry : entries)
            entry.second->clear();
    }

    Long64_t GetEntries() const { return tree->GetEntries(); }
    Long64_t GetReadEntry() const { return tree->GetReadEntry(); }
    Int_t GetEntry(Long64_t entry) { return tree->GetEntry(entry); }
    void Write()
    {
        if(directory)
            directory->WriteTObject(tree, tree->GetName(), "Overwrite");
    }

protected:
    template<typename DataType>
    void AddBranch(const std::string& branch_name, DataType& value)
    {
        detail::BranchCreator<DataType>::Create(tree, branch_name, value, readMode, entries);
    }

private:
    SmartTree(const SmartTree& other) { throw std::runtime_error("Can't copy a smart tree"); }

private:
    std::string name;
    TDirectory* directory;
    std::map< std::string, std::shared_ptr<detail::BaseSmartTreeEntry> > entries;
    bool readMode;
    TTree* tree;
};

namespace detail {
template<typename Data>
class BaseSmartTree : public SmartTree {
public:
    using SmartTree::SmartTree;
    Data& operator()() { return data; }
    const Data& operator()() const { return data; }
protected:
    Data data;
};
} // detail

} // root_ext
