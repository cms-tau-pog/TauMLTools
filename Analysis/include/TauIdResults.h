/*! Definition of Tau ID discriminators.
This file is part of https://github.com/hh-italian-group/h-tautau. */

#pragma once

#include <bitset>
#include "AnalysisTypes.h"
#include "AnalysisTools/Core/include/TextIO.h"

namespace analysis {

#define TAU_IDS() \
    TAU_ID(againstElectronMVA6, "againstElectron{wp}MVA6{Raw}", "VLoose Loose Medium Tight VTight") \
    TAU_ID(againstMuon3, "againstMuon{wp}3", "Loose Tight") \
    TAU_ID(byCombinedIsolationDeltaBetaCorr3Hits, "by{wp}CombinedIsolationDeltaBetaCorr{Raw}3Hits", \
           "Loose Medium Tight") \
    TAU_ID(byPhotonPtSumOutsideSignalCone, "byPhotonPtSumOutsideSignalCone", "Medium") \
    TAU_ID(byIsolationMVArun2v1DBoldDMwLT, "by{wp}IsolationMVArun2v1DBoldDMwLT{raw}", \
           "VLoose Loose Medium Tight VTight VVTight") \
    TAU_ID(byIsolationMVArun2v1DBdR03oldDMwLT, "by{wp}IsolationMVArun2v1DBdR03oldDMwLT{raw}", \
           "VLoose Loose Medium Tight VTight VVTight") \
    TAU_ID(byIsolationMVArun2v1DBoldDMwLT2016, "by{wp}IsolationMVArun2v1DBoldDMwLT{raw}2016", \
           "VLoose Loose Medium Tight VTight VVTight") \
    TAU_ID(byIsolationMVArun2017v2DBoldDMwLT2017, "by{wp}IsolationMVArun2017v2DBoldDMwLT{raw}2017", \
           "VVLoose VLoose Loose Medium Tight VTight VVTight") \
    TAU_ID(byIsolationMVArun2017v2DBoldDMdR0p3wLT2017, "by{wp}IsolationMVArun2017v2DBoldDMdR0p3wLT{raw}2017", \
           "VVLoose VLoose Loose Medium Tight VTight VVTight") \
    /**/

#define TAU_ID(name, pattern, wp_list) name,
enum class TauIdDiscriminator { TAU_IDS() };
#undef TAU_ID

#define TAU_ID(name, pattern, wp_list) TauIdDiscriminator::name,
namespace tau_id {
inline const std::vector<TauIdDiscriminator>& GetOrderedTauIdDiscriminators()
{
    static const std::vector<TauIdDiscriminator> ordered_tau_ids = { TAU_IDS() };
    return ordered_tau_ids;
}
}
#undef TAU_ID

#define TAU_ID(name, pattern, wp_list) { TauIdDiscriminator::name, #name },
ENUM_NAMES(TauIdDiscriminator) = { TAU_IDS() };
#undef TAU_ID

namespace tau_id {
struct TauIdDescriptor {
    TauIdDiscriminator discriminator;
    std::string name_pattern;
    std::vector<DiscriminatorWP> working_points;

    TauIdDescriptor(TauIdDiscriminator _discriminator, const std::string& _name_pattern, const std::string& wp_list)
        : discriminator(_discriminator), name_pattern(_name_pattern)
    {
        auto wp_names = SplitValueList(wp_list, false, ", \t", true);
        for(const auto& wp_name : wp_names)
            working_points.push_back(analysis::Parse<DiscriminatorWP>(wp_name));
    }

    std::string ToString(DiscriminatorWP wp) const
    {
        std::string name = name_pattern;
        boost::algorithm::replace_all(name, "{wp}", analysis::ToString(wp));
        boost::algorithm::replace_all(name, "{raw}", "");
        boost::algorithm::replace_all(name, "{Raw}", "");
        return name;
    }

    std::string ToStringRaw() const
    {
        std::string name = name_pattern;
        boost::algorithm::replace_all(name, "{wp}", "");
        boost::algorithm::replace_all(name, "{raw}", "raw");
        boost::algorithm::replace_all(name, "{Raw}", "Raw");
        return name;
    }
};

using TauIdDescriptorCollection = std::map<TauIdDiscriminator, TauIdDescriptor>;

#define TAU_ID(name, pattern, wp_list) \
    { TauIdDiscriminator::name, TauIdDescriptor(TauIdDiscriminator::name, pattern, wp_list) },
inline const TauIdDescriptorCollection& GetTauIdDescriptors()
{
    static const TauIdDescriptorCollection descriptors = { TAU_IDS() };
    return descriptors;
}
#undef TAU_ID

}

#undef TAU_IDS

class TauIdResults {
public:
    using BitsContainer = unsigned long long;
    static constexpr size_t MaxNumberOfIds = std::numeric_limits<BitsContainer>::digits;
    using Bits = std::bitset<MaxNumberOfIds>;

    struct ResultDescriptor {
        TauIdDiscriminator discriminator;
        DiscriminatorWP wp;

        ResultDescriptor() {}
        ResultDescriptor(TauIdDiscriminator _discriminator, DiscriminatorWP _wp) :
            discriminator(_discriminator), wp(_wp) {}

        bool operator<(const ResultDescriptor& other) const
        {
            if(discriminator != other.discriminator) return discriminator < other.discriminator;
            return wp < other.wp;
        }

        std::string ToString() const
        {
            return tau_id::GetTauIdDescriptors().at(discriminator).ToString(wp);
        }
    };

    using ResultDescriptorCollection = std::vector<ResultDescriptor>;
    using BitRefByDescCollection = std::map<ResultDescriptor, size_t>;
    using BitRefByNameCollection = std::map<std::string, size_t>;

    static const ResultDescriptorCollection& GetResultDescriptors()
    {
        static const auto& discriminators = tau_id::GetOrderedTauIdDiscriminators();
        static const auto& id_descriptors = tau_id::GetTauIdDescriptors();

        auto createDescriptors = [&]() {
            ResultDescriptorCollection descs;
            for(const auto& discriminator : discriminators) {
                for(auto wp : id_descriptors.at(discriminator).working_points)
                    descs.emplace_back(discriminator, wp);
            }
            return descs;
        };

        static const ResultDescriptorCollection descriptors = createDescriptors();
        return descriptors;
    }

    static const BitRefByDescCollection& GetBitRefsByDesc()
    {
        auto createBitRefs = []() {
            BitRefByDescCollection bit_refs;
            const auto& descs = GetResultDescriptors();
            for(size_t n = 0; n < descs.size(); ++n) {
                const auto& desc = descs.at(n);
                if(bit_refs.count(desc))
                    throw exception("Duplicated descriptor of tau ID result = '%1%'") % desc.ToString();
                bit_refs[desc] = n;
            }
            return bit_refs;
        };
        static const BitRefByDescCollection bit_refs_by_desc = createBitRefs();
        return bit_refs_by_desc;
    }

    static const BitRefByNameCollection& GetBitRefsByName()
    {
        auto createBitRefs = []() {
            BitRefByNameCollection bit_refs;
            const auto& descs = GetBitRefsByDesc();
            for(const auto& item : descs)
                bit_refs[item.first.ToString()] = item.second;
            return bit_refs;
        };
        static const BitRefByNameCollection bit_refs_by_name = createBitRefs();
        return bit_refs_by_name;
    }

    TauIdResults() : result_bits(0) {}
    TauIdResults(BitsContainer _result_bits) : result_bits(_result_bits) {}

    BitsContainer GetResultBits() const { return result_bits.to_ullong(); }

    bool Result(size_t index) const { CheckIndex(index); return result_bits[index]; }
    void SetResult(size_t index, bool value) { CheckIndex(index); result_bits[index] = value; }

    bool Result(TauIdDiscriminator discriminator, DiscriminatorWP wp) const
    {
        const ResultDescriptor desc(discriminator, wp);
        const auto& bit_refs = GetBitRefsByDesc();
        auto iter = bit_refs.find(desc);
        if(iter == bit_refs.end())
            throw exception("Result bit not found for %1% working point of %2%.") % wp % discriminator;
        return Result(iter->second);
    }

    bool Result(const std::string& name) const
    {
        const auto& bit_refs = GetBitRefsByName();
        auto iter = bit_refs.find(name);
        if(iter == bit_refs.end())
            throw exception("Result bit not found for '%1%'.") % name;
        return Result(iter->second);
    }

private:
    void CheckIndex(size_t index) const
    {
        if(index >= MaxNumberOfIds)
            throw exception("Tau ID index is out of range.");
    }

private:
    Bits result_bits;
};

} // namespace analysis
