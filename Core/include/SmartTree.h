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
#define ADD_DATA_TREE_BRANCH(name) AddBranch(#name, _data->name);

#define DECLARE_TREE(namespace_name, data_class_name, tree_class_name, data_macro, tree_name) \
    namespace namespace_name { \
    struct data_class_name : public root_ext::detail::BaseDataClass { data_macro() }; \
    using data_class_name##Vector = std::vector< data_class_name >; \
    class tree_class_name : public root_ext::detail::BaseSmartTree<data_class_name> { \
    public: \
        static const std::string& Name() { static const std::string name = tree_name; return name; } \
        tree_class_name(TDirectory* directory, bool readMode, const std::set<std::string>& disabled_branches = {}) \
            : BaseSmartTree(Name(), directory, readMode, disabled_branches) { Initialize(); } \
        tree_class_name(const std::string& name, TDirectory* directory, bool readMode, \
                        const std::set<std::string>& disabled_branches = {}) \
            : BaseSmartTree(name, directory, readMode, disabled_branches) { Initialize(); } \
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
template<typename type>
using strmap = std::map<std::string, type>;

namespace detail {
    struct BaseSmartTreeEntry {
        virtual ~BaseSmartTreeEntry() {}
        virtual void clear() {}
    };

    template<typename DataType>
    struct SmartTreePtrEntry : BaseSmartTreeEntry {
        DataType* value;
        SmartTreePtrEntry(DataType& origin)
            : value(&origin) {}
    };

    template<typename DataType>
    struct SmartTreeCollectionEntry : SmartTreePtrEntry<DataType> {
        using SmartTreePtrEntry<DataType>::SmartTreePtrEntry;
        virtual void clear() override { this->value->clear(); }
    };

    template<typename DataType>
    struct EntryTypeSelector { using PtrEntry = SmartTreePtrEntry<DataType>; };
    template<typename DataType>
    struct EntryTypeSelector<std::vector<DataType>> {
        using PtrEntry = SmartTreeCollectionEntry<std::vector<DataType>>;
    };
    template<typename DataType>
    struct EntryTypeSelector<strmap<DataType>> {
        using PtrEntry = SmartTreeCollectionEntry<strmap<DataType>>;
    };

    struct BaseDataClass {
        virtual ~BaseDataClass() {}
    };

    using SmartTreeEntryMap = std::map<std::string, std::shared_ptr<BaseSmartTreeEntry>>;

    inline void EnableBranch(TBranch& branch)
    {
        branch.SetStatus(1);
        auto sub_branches = branch.GetListOfBranches();
        for(auto sub_branch : *sub_branches)
            EnableBranch(*dynamic_cast<TBranch*>(sub_branch));
    }

    inline void EnableBranch(TTree& tree, const std::string& branch_name)
    {
        TBranch* branch = tree.GetBranch(branch_name.c_str());
        if(!branch) {
            std::ostringstream ss;
            ss << "Branch '" << branch_name << "' not found.";
            throw std::runtime_error(ss.str());
        }
        EnableBranch(*branch);
    }

    template<typename DataType>
    struct BranchCreator {
        static void Create(TTree& tree, const std::string& branch_name, DataType& value, bool readMode,
                           SmartTreeEntryMap& entries)
        {
            static const std::map<std::string, std::string> class_fixes = {
                { "ROOT::Math::LorentzVector<ROOT::Math::PtEtaPhiM4D<double> >",
                  "ROOT::Math::LorentzVector<ROOT::Math::PtEtaPhiM4D<Double32_t> >" }
            };

            using PtrEntry = typename EntryTypeSelector<DataType>::PtrEntry;
            std::shared_ptr<PtrEntry> entry(new PtrEntry(value));
            if(entries.count(branch_name))
                throw std::runtime_error("Entry is already defined.");
            entries[branch_name] = entry;

            TClass *cl = TClass::GetClass(typeid(DataType));
            if(cl && class_fixes.count(cl->GetName()))
                cl = TClass::GetClass(class_fixes.at(cl->GetName()).c_str());

            if(readMode) {
                try {
                    EnableBranch(tree, branch_name);
                    if(cl)
                        tree.SetBranchAddress(branch_name.c_str(), &entry->value, nullptr, cl, kOther_t, true);
                    else
                        tree.SetBranchAddress(branch_name.c_str(), entry->value);
                    if(tree.GetReadEntry() >= 0)
                        tree.GetBranch(branch_name.c_str())->GetEntry(tree.GetReadEntry());
                } catch(std::runtime_error& error) {
                    std::cerr << "ERROR: " << error.what() << std::endl;
                }
            } else {
                TBranch* branch = tree.Branch(branch_name.c_str(), entry->value);
                const Long64_t n_entries = tree.GetEntries();
                for(Long64_t n = 0; n < n_entries; ++n)
                    branch->Fill();
            }
        }
    };

} // detail

class SmartTree {
public:
    SmartTree(const std::string& _name, TDirectory* _directory, bool _readMode,
              const std::set<std::string>& _disabled_branches = {})
        : name(_name), directory(_directory), readMode(_readMode), disabled_branches(_disabled_branches)
    {
        static constexpr Long64_t maxVirtualSize = 10000000;

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
        : name(other.name), directory(other.directory), entries(other.entries), tree(other.tree),
          disabled_branches(other.disabled_branches) {}

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
        if(!disabled_branches.count(branch_name))
            detail::BranchCreator<DataType>::Create(*tree, branch_name, value, readMode, entries);
    }

private:
    SmartTree(const SmartTree& other) { throw std::runtime_error("Can't copy a smart tree"); }

private:
    std::string name;
    TDirectory* directory;
    detail::SmartTreeEntryMap entries;
    bool readMode;
    TTree* tree;
    std::set<std::string> disabled_branches;
};

namespace detail {
template<typename Data>
class BaseSmartTree : public SmartTree {
public:
    using SmartTree::SmartTree;
    BaseSmartTree(const BaseSmartTree&& other)
        : SmartTree(other), _data(other._data) {}

    Data& operator()() { return *_data; }
    const Data& operator()() const { return *_data; }
    const Data& data() const { return *_data; }

protected:
    std::shared_ptr<Data> _data{new Data()};
};
} // detail
} // root_ext
