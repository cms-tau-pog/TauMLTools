/*! Definition of Tau ID discriminators.
This file is part of https://github.com/hh-italian-group/h-tautau. */

#pragma once

#include "AnalysisTypes.h"

namespace analysis {

#define TAU_IDS() \
    TAU_ID(againstElectronMVA6) \
    TAU_ID(againstMuon3) \
    TAU_ID(byCombinedIsolationDeltaBetaCorr3Hits) \
    TAU_ID(byPhotonPtSumOutsideSignalCone) \
    TAU_ID(byIsolationMVArun2v1DBoldDMwLT) \
    TAU_ID(byIsolationMVArun2v1DBdR03oldDMwLT) \
    TAU_ID(byIsolationMVArun2v1DBoldDMwLT2016) \
    TAU_ID(byIsolationMVArun2017v2DBoldDMwLT2017) \
    TAU_ID(byIsolationMVArun2017v2DBoldDMdR0p3wLT2017) \
    /**/

#define TAU_ID(name) name,
enum class TauIdDiscriminator { TAU_IDS() };
#undef TAU_ID

#define TAU_ID(name) TauIdDiscriminator::name,
namespace tau_id {
inline const std::vector<TauIdDiscriminator>& GetOrderedTauIdDiscriminators()
{
    static const std::vector<TauIdDiscriminator> ordered_tau_ids = { TAU_IDS() };
    return ordered_tau_ids;
}
}
#undef TAU_ID

#define TAU_ID(name) { TauIdDiscriminator::name, #name },
ENUM_NAMES(TauIdDiscriminator) = { TAU_IDS() };
#undef TAU_ID

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
            if(discriminator == TauIdDiscriminator::againstMuon3)
                return "againstMuon" + analysis::ToString(wp) + "3";
            if(discriminator == TauIdDiscriminator::againstElectronMVA6)
                return "againstElectron" + analysis::ToString(wp) + "MVA6";
            if(discriminator == TauIdDiscriminator::byPhotonPtSumOutsideSignalCone)
                return analysis::ToString(discriminator);
            return "by" + analysis::ToString(wp) + analysis::ToString(discriminator).substr(2);
        }
    };

    using ResultDescriptorCollection = std::vector<ResultDescriptor>;
    using BitRefByDescCollection = std::map<ResultDescriptor, size_t>;
    using BitRefByNameCollection = std::map<std::string, size_t>;

    static const ResultDescriptorCollection& GetResultDescriptors()
    {
        static const std::vector<TauIdDiscriminator>& discriminators = tau_id::GetOrderedTauIdDiscriminators();
        static const std::set<TauIdDiscriminator> discriminators_with_VVLoose_wp = {
            TauIdDiscriminator::byIsolationMVArun2017v2DBoldDMwLT2017,
            TauIdDiscriminator::byIsolationMVArun2017v2DBoldDMdR0p3wLT2017
        };

        auto createDescriptors = [&]() {
            ResultDescriptorCollection descs;
            for(const auto& discriminator : discriminators) {
                if(discriminator == TauIdDiscriminator::againstMuon3) {
                    descs.emplace_back(discriminator, DiscriminatorWP::Loose);
                    descs.emplace_back(discriminator, DiscriminatorWP::Tight);
                } else if(discriminator == TauIdDiscriminator::byCombinedIsolationDeltaBetaCorr3Hits) {
                    descs.emplace_back(discriminator, DiscriminatorWP::Loose);
                    descs.emplace_back(discriminator, DiscriminatorWP::Medium);
                    descs.emplace_back(discriminator, DiscriminatorWP::Tight);
                } else if(discriminator == TauIdDiscriminator::byPhotonPtSumOutsideSignalCone) {
                    descs.emplace_back(discriminator, DiscriminatorWP::Medium);
                } else {
                    if(discriminators_with_VVLoose_wp.count(discriminator))
                        descs.emplace_back(discriminator, DiscriminatorWP::VVLoose);
                    descs.emplace_back(discriminator, DiscriminatorWP::VLoose);
                    descs.emplace_back(discriminator, DiscriminatorWP::Loose);
                    descs.emplace_back(discriminator, DiscriminatorWP::Medium);
                    descs.emplace_back(discriminator, DiscriminatorWP::Tight);
                    descs.emplace_back(discriminator, DiscriminatorWP::VTight);
                    if(discriminator != TauIdDiscriminator::againstElectronMVA6)
                        descs.emplace_back(discriminator, DiscriminatorWP::VVTight);
                }
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
