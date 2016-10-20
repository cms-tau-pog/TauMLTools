/*! Wrapper around ROOT TBranch.
This file is part of https://github.com/hh-italian-group/AnalysisTools. */

#pragma once

#include <TBranch.h>
#include <TTree.h>
#include <TClass.h>
#include "exception.h"
#include "AnalysisMath.h"

namespace root_ext {

namespace detail {

struct BaseSmartBranchEntry {
    virtual ~BaseSmartBranchEntry() {}
    virtual void SetBranchAddress(TBranch& branch) = 0;
    virtual bool IsCollection() const = 0;
    virtual std::string ToString() const = 0;
};

template<typename DataType, bool startNewLine = false>
struct SmartBranchEntry : BaseSmartBranchEntry {
    DataType* value;
    SmartBranchEntry() : value(new DataType()) {}
    virtual ~SmartBranchEntry() { delete value; }

    virtual void SetBranchAddress(TBranch& branch) override
    {
        TClass *branch_class;
        EDataType branch_type;
        branch.GetExpectedType(branch_class, branch_type);

        if(branch_class)
            branch.GetTree()->SetBranchAddress(branch.GetName(), &value, nullptr, branch_class, kOther_t, true);
        else
            branch.GetTree()->SetBranchAddress(branch.GetName(), value);
    }

    virtual bool IsCollection() const override { return false; }
    std::string ToString() const override
    {
        std::ostringstream ss;
        if(startNewLine)
            ss << std::endl;
        ss << *value;
        return ss.str();
    }

    static BaseSmartBranchEntry* Make() { return new SmartBranchEntry(); }
};

template<typename KeyType, typename ValueType>
struct SmartBranchEntry<std::map<KeyType, ValueType>, false> : BaseSmartBranchEntry {
    using DataType = std::map<KeyType, ValueType>;
    DataType* value;
    SmartBranchEntry() : value(new DataType()) {}
    virtual ~SmartBranchEntry() { delete value; }

    virtual void SetBranchAddress(TBranch& branch) override
    {
        branch.GetTree()->SetBranchAddress(branch.GetName(), &value);
    }

    virtual bool IsCollection() const override { return true; }
    std::string ToString() const override
    {
        std::ostringstream ss;
        ss << "size = " << value->size() << std::endl;
        for(const auto& entry : *value)
            ss << boost::format("\t%1%: %2%\n") % entry.first % entry.second;
        return ss.str();
    }

    static BaseSmartBranchEntry* Make() { return new SmartBranchEntry(); }
};

template<typename ValueType>
struct SmartBranchEntry<std::vector<ValueType>, false> : BaseSmartBranchEntry {
    using DataType = std::vector<ValueType>;
    DataType* value;
    SmartBranchEntry() : value(new DataType()) {}
    virtual ~SmartBranchEntry() { delete value; }

    virtual void SetBranchAddress(TBranch& branch) override
    {
        branch.GetTree()->SetBranchAddress(branch.GetName(), &value);
    }

    virtual bool IsCollection() const override { return true; }
    std::string ToString() const override
    {
        std::ostringstream ss;
        ss << "size = " << value->size() << std::endl;
        for(size_t n = 0; n < value->size(); ++n)
            ss << boost::format("\t%1%: %2%\n") % n % value->at(n);
        return ss.str();
    }

    static BaseSmartBranchEntry* Make() { return new SmartBranchEntry(); }
};

struct BranchEntryFactory {
    static BaseSmartBranchEntry* Make(TBranch& branch)
    {
        using MakeMethodPtr = BaseSmartBranchEntry* (*)();
        using SimpleTypeMakeMethodMap = std::map<EDataType, MakeMethodPtr>;
        using CompositeTypeMakeMethodMap = std::map<std::string, MakeMethodPtr>;

        static const SimpleTypeMakeMethodMap simpleTypeMakeMethods = {
            { kInt_t, &SmartBranchEntry<Int_t>::Make },
            { kUInt_t, &SmartBranchEntry<UInt_t>::Make },
            { kULong64_t, &SmartBranchEntry<ULong64_t>::Make },
            { kFloat_t, &SmartBranchEntry<Float_t>::Make },
            { kDouble_t, &SmartBranchEntry<Double_t>::Make },
            { kChar_t, &SmartBranchEntry<Char_t>::Make },
            { kBool_t, &SmartBranchEntry<Bool_t>::Make }
        };

        static const CompositeTypeMakeMethodMap compositeTypeMakeMethods = {
            { "ROOT::Math::LorentzVector<ROOT::Math::PtEtaPhiM4D<Double32_t> >",
              &SmartBranchEntry<ROOT::Math::LorentzVector<ROOT::Math::PtEtaPhiM4D<Double32_t> >>::Make },
            { "map<string,float>", &SmartBranchEntry<std::map<std::string,float>>::Make },
            { "ROOT::Math::SMatrix<double,2,2,ROOT::Math::MatRepSym<double,2> >",
              &SmartBranchEntry<ROOT::Math::SMatrix<double,2,2,ROOT::Math::MatRepSym<double,2> >, true>::Make },
            { "vector<ROOT::Math::LorentzVector<ROOT::Math::PtEtaPhiE4D<double> > >",
              &SmartBranchEntry<std::vector<ROOT::Math::LorentzVector<ROOT::Math::PtEtaPhiE4D<double> > >>::Make },
            { "vector<float>", &SmartBranchEntry<std::vector<float>>::Make },
            { "vector<double>", &SmartBranchEntry<std::vector<double>>::Make },
            { "vector<int>", &SmartBranchEntry<std::vector<int>>::Make },
            { "vector<unsigned long>", &SmartBranchEntry<std::vector<unsigned long>>::Make },
            { "vector<ROOT::Math::LorentzVector<ROOT::Math::PtEtaPhiM4D<double> > >",
              &SmartBranchEntry<std::vector<ROOT::Math::LorentzVector<ROOT::Math::PtEtaPhiM4D<double> > >>::Make }

        };

        TClass *branch_class;
        EDataType branch_type;
        branch.GetExpectedType(branch_class, branch_type);

        if(!branch_class) {
            if(!simpleTypeMakeMethods.count(branch_type))
                throw analysis::exception("%1%: simple type %2% not supported.") % branch.GetName() % branch_type;
            return (*simpleTypeMakeMethods.at(branch_type))();
        }

        if(compositeTypeMakeMethods.count(branch_class->GetName()))
            return (*compositeTypeMakeMethods.at(branch_class->GetName()))();

        throw analysis::exception("%1%: type '%2%' not supported.") % branch.GetName() % branch_class->GetName();
    }
};

template<typename OutputType>
struct BranchValueGetter;

} // namespace detail

class SmartBranch {
public:
    explicit SmartBranch(TBranch& _branch) :
        branch(&_branch), entry(detail::BranchEntryFactory::Make(_branch))
    {
    }

    SmartBranch(TTree& tree, const std::string& name)
    {
        branch = tree.GetBranch(name.c_str());
        if(!branch)
            throw analysis::exception("Branch '%1%' not found.") % name;
        entry = std::shared_ptr<detail::BaseSmartBranchEntry>(detail::BranchEntryFactory::Make(*branch));
    }

    void Enable()
    {
        EnableBranch(*branch);
        entry->SetBranchAddress(*branch);
    }

    TBranch* operator->() { return branch; }
    const TBranch* operator->() const { return branch; }

    template<typename OutputType>
    OutputType GetValue() const
    {
        return detail::BranchValueGetter<OutputType>::Get(*this);
    }

    template<typename OriginalType, typename OutputType>
    bool TryGetValue(OutputType& value) const
    {
        auto typed_entry = dynamic_cast<detail::SmartBranchEntry<OriginalType>*>(entry.get());
        if(!typed_entry) return false;
        value = *typed_entry->value;
        return true;
    }

    void PrintValue(std::ostream& s) const
    {
        s << branch->GetName() << ": " << entry->ToString();
        if(!entry->IsCollection())
            s << std::endl;
    }

private:
    static void EnableBranch(TBranch& branch)
    {
        branch.SetStatus(1);
        auto sub_branches = branch.GetListOfBranches();
        for(auto sub_branch : *sub_branches)
            EnableBranch(*dynamic_cast<TBranch*>(sub_branch));
    }

private:
    TBranch* branch;
    std::shared_ptr<detail::BaseSmartBranchEntry> entry;
};

namespace detail {
template<>
struct BranchValueGetter<unsigned long long> {
    using OutputType = unsigned long long;
    static unsigned long long Get(const SmartBranch& branch)
    {
        OutputType value;
        if(branch.TryGetValue<UInt_t, OutputType>(value)) return value;
        if(branch.TryGetValue<ULong64_t, OutputType>(value)) return value;
        throw analysis::exception("Can't read value of branch '%1%' as 'unsigned long long'.") % branch->GetName();
    }
};

} // namespace detail
} // namespace root_ext
