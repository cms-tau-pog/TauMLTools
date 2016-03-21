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

#define SIMPLE_TREE_BRANCH(type, name) \
private: type _##name; \
public:  type& name() { return _##name; }

#define VECTOR_TREE_BRANCH(type, name) \
private: std::vector< type > _##name; \
public:  std::vector< type >& name() { return _##name; }

#define SIMPLE_DATA_TREE_BRANCH(type, name) \
    type& name() { return data.name; }

#define VECTOR_DATA_TREE_BRANCH(type, name) \
    std::vector< type >& name() { return data.name; }

#define DECLARE_SIMPLE_BRANCH_VARIABLE(type, name) type name;
#define DECLARE_VECTOR_BRANCH_VARIABLE(type, name) std::vector< type > name;

#define ADD_SIMPLE_TREE_BRANCH(name) AddSimpleBranch(#name, _##name);
#define ADD_SIMPLE_DATA_TREE_BRANCH(name) AddSimpleBranch(#name, data.name);
#define ADD_VECTOR_TREE_BRANCH(name) AddVectorBranch(#name, _##name);
#define ADD_VECTOR_DATA_TREE_BRANCH(name) AddVectorBranch(#name, data.name);

#define DATA_CLASS(namespace_name, class_name, data_macro) \
    namespace namespace_name { \
        struct class_name : public root_ext::detail::BaseDataClass { data_macro() }; \
        using class_name##Vector = std::vector< class_name >; \
    } \
    /**/

#define TREE_CLASS(namespace_name, tree_class_name, data_macro, data_class_name, tree_name, is_mc_truth) \
    namespace namespace_name { \
    class tree_class_name : public root_ext::SmartTree { \
    public: \
        static bool IsMCtruth() { return is_mc_truth; } \
        static const std::string& Name() { static const std::string name = tree_name; return name; } \
        tree_class_name(TDirectory* directory, bool readMode) \
            : SmartTree(Name(), directory, readMode) { Initialize(); } \
        tree_class_name(const std::string& name, TDirectory* directory, bool readMode) \
            : SmartTree(name, directory, readMode) { Initialize(); } \
        data_class_name data; \
        data_macro() \
    private: \
        inline void Initialize(); \
    }; \
    } \
    /**/

#define TREE_CLASS_WITH_EVENT_ID(namespace_name, tree_class_name, data_macro, data_class_name, tree_name, is_mc_truth) \
    namespace namespace_name { \
    class tree_class_name : public root_ext::SmartTree { \
    public: \
        static bool IsMCtruth() { return is_mc_truth; } \
        static const std::string& Name() { static const std::string name = tree_name; return name; } \
        tree_class_name(TDirectory* directory, bool readMode) \
            : SmartTree(Name(), directory, readMode) { Initialize(); } \
        tree_class_name(const std::string& name, TDirectory* directory, bool readMode) \
            : SmartTree(name, directory, readMode) { Initialize(); } \
        data_class_name data; \
        SIMPLE_TREE_BRANCH(UInt_t, RunId) \
        SIMPLE_TREE_BRANCH(UInt_t, LumiBlock) \
        SIMPLE_TREE_BRANCH(UInt_t, EventId) \
        data_macro() \
    private: \
        inline void Initialize(); \
    }; \
    } \
    /**/

#define TREE_CLASS_INITIALIZE(namespace_name, tree_class_name, data_macro) \
    namespace namespace_name { \
        inline void tree_class_name::Initialize() { \
            data_macro() \
            if (GetEntries() > 0) GetEntry(0); \
        } \
    } \
    /**/

#define TREE_CLASS_WITH_EVENT_ID_INITIALIZE(namespace_name, tree_class_name, data_macro) \
    namespace namespace_name { \
        inline void tree_class_name::Initialize() { \
            ADD_SIMPLE_TREE_BRANCH(RunId) \
            ADD_SIMPLE_TREE_BRANCH(LumiBlock) \
            ADD_SIMPLE_TREE_BRANCH(EventId) \
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
            directory->WriteTObject(tree, tree->GetName(), "WriteDelete");
    }

protected:
    template<typename DataType>
    void AddSimpleBranch(const std::string& branch_name, DataType& value)
    {
        if(readMode) {
            try {
                EnableBranch(branch_name);
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

    template<typename DataType>
    void AddVectorBranch(const std::string& branch_name, std::vector<DataType>& value)
    {
        using PtrEntry = detail::SmartTreeVectorPtrEntry<DataType>;
        auto entry = std::shared_ptr<PtrEntry>( new PtrEntry(value) );
        if(entries.count(branch_name))
            throw std::runtime_error("Entry is already defined.");
        entries[branch_name] = entry;
        if(readMode) {
            try {
                EnableBranch(branch_name);
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

    void EnableBranch(const std::string& branch_name)
    {
        UInt_t n_found = 0;
        tree->SetBranchStatus(branch_name.c_str(), 1, &n_found);
        if(n_found != 1) {
            std::ostringstream ss;
            ss << "Branch '" << branch_name << "' is not found.";
            throw std::runtime_error(ss.str());
        }
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

} // root_ext
