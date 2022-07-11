/*! Definition of Tau ID discriminators.
This file is part of https://github.com/hh-italian-group/h-tautau. */

#pragma once

#include <bitset>
#include "AnalysisTypes.h"
#include "TauMLTools/Core/interface/TextIO.h"

namespace analysis {

struct TauIdResults {
    using BitsContainer = uint16_t;
    static constexpr size_t MaxNumberOfWorkingPoints = std::numeric_limits<BitsContainer>::digits;

    TauIdResults() : results(0) {}
    TauIdResults(BitsContainer _results) : results(_results) {}

    bool Passed(DiscriminatorWP wp) const
    {
        const unsigned bit_index = static_cast<unsigned>(wp);
        if(bit_index > MaxNumberOfWorkingPoints)
            throw exception("Discriminator WP = '{}' is not supported.") % wp;

        const BitsContainer mask = static_cast<BitsContainer>(BitsContainer(1) << bit_index);
        return (results & mask) != BitsContainer(0);
    }
    bool Failed(DiscriminatorWP wp) const { return !Passed(wp); }

    void SetResult(DiscriminatorWP wp, bool result)
    {
        const unsigned bit_index = static_cast<unsigned>(wp);
        if(bit_index > MaxNumberOfWorkingPoints)
            throw exception("Discriminator WP = '{}' is not supported.") % wp;
        const BitsContainer mask = static_cast<BitsContainer>(BitsContainer(1) << bit_index);
        results = (results & ~mask) | static_cast<BitsContainer>(BitsContainer(result) << bit_index);
    }

    BitsContainer GetResultBits() const { return results; }

private:
    BitsContainer results;
};

#define TAU_IDS() \
    TAU_ID(byCombinedIsolationDeltaBetaCorr3Hits, "by{wp}CombinedIsolationDeltaBetaCorr{Raw}3Hits", true, \
           "Loose Medium Tight") \
    TAU_ID(byDeepTau2017v2p1VSe, "by{wp}DeepTau2017v2p1VSe{raw}", true, \
       "VVVLoose VVLoose VLoose Loose Medium Tight VTight VVTight") \
    TAU_ID(byDeepTau2017v2p1VSmu, "by{wp}DeepTau2017v2p1VSmu{raw}", true, \
       "VLoose Loose Medium Tight") \
    TAU_ID(byDeepTau2017v2p1VSjet, "by{wp}DeepTau2017v2p1VSjet{raw}", true, \
       "VVVLoose VVLoose VLoose Loose Medium Tight VTight VVTight") \
    TAU_ID(byIsolationMVArun2017v2DBoldDMwLT2017, "by{wp}IsolationMVArun2017v2DBoldDMwLT{raw}2017", true, \
        "VVLoose VLoose Loose Medium Tight VTight VVTight") \
    TAU_ID(byIsolationMVArun2017v2DBnewDMwLT2017, "by{wp}IsolationMVArun2017v2DBnewDMwLT{raw}2017", true, \
        "VVLoose VLoose Loose Medium Tight VTight VVTight") \
    TAU_ID(byIsolationMVArun2017v2DBoldDMdR0p3wLT2017, "by{wp}IsolationMVArun2017v2DBoldDMdR0p3wLT{raw}2017", true, \
        "VVLoose VLoose Loose Medium Tight VTight VVTight") \
    TAU_ID(byIsolationMVADBnewDMwLTPhase2, "by{wp}IsolationMVADBnewDMwLTPhase2{raw}", true, \
        "VVLoose VLoose Loose Medium Tight VTight VVTight") \
    TAU_ID(byDeepTau2018v2p5VSe, "by{wp}DeepTau2018v2p5VSe{raw}", true, \
       "VVVLoose VVLoose VLoose Loose Medium Tight VTight VVTight") \
    TAU_ID(byDeepTau2018v2p5VSmu, "by{wp}DeepTau2018v2p5VSmu{raw}", true, \
       "VLoose Loose Medium Tight") \
    TAU_ID(byDeepTau2018v2p5VSjet, "by{wp}DeepTau2018v2p5VSjet{raw}", true, \
       "VVVLoose VVLoose VLoose Loose Medium Tight VTight VVTight") \
    /**/

#define TAU_ID(name, pattern, has_raw, wp_list) name,
enum class TauIdDiscriminator { TAU_IDS() };
#undef TAU_ID

#define TAU_ID(name, pattern, has_raw, wp_list) TauIdDiscriminator::name,
namespace tau_id {
inline const std::vector<TauIdDiscriminator>& GetOrderedTauIdDiscriminators()
{
    static const std::vector<TauIdDiscriminator> ordered_tau_ids = { TAU_IDS() };
    return ordered_tau_ids;
}
}
#undef TAU_ID

#define TAU_ID(name, pattern, has_raw, wp_list) { TauIdDiscriminator::name, #name },
ENUM_NAMES(TauIdDiscriminator) = { TAU_IDS() };
#undef TAU_ID

namespace tau_id {
struct TauIdDescriptor {
    TauIdDiscriminator discriminator;
    std::string name_pattern;
    bool has_raw;
    std::string raw_name;
    std::map<DiscriminatorWP, std::string> working_points;

    TauIdDescriptor(TauIdDiscriminator _discriminator, const std::string& _name_pattern, bool _has_raw,
                    const std::string& wp_list) :
        discriminator(_discriminator), name_pattern(_name_pattern), has_raw(_has_raw)
    {
        if(has_raw)
            raw_name = ToStringRaw();
        auto wp_names = SplitValueList(wp_list, false, ", \t", true);
        for(const auto& wp_name : wp_names) {
            const DiscriminatorWP wp = ::analysis::Parse<DiscriminatorWP>(wp_name);
            working_points[wp] = ToString(wp);
        }
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

    template<typename Tuple, typename Tau>
    void FillTuple(Tuple& tuple, const Tau* tau, float default_value, const std::string& prefix = "",
                   const std::string& raw_suffix = "raw") const
    {
        const std::string disc_name = ::analysis::ToString(discriminator);
        if(has_raw)
            tuple.template get<float>(prefix + disc_name + raw_suffix) = tau && tau->isTauIDAvailable(raw_name) ? tau->tauID(raw_name) : default_value;
        if(!working_points.empty()) {
            TauIdResults id_results;
            for(const auto& wp_entry : working_points) {
                const bool result = tau && tau->isTauIDAvailable(wp_entry.second) && tau->tauID(wp_entry.second) > 0.5;
                id_results.SetResult(wp_entry.first, result);
            }
            tuple.template get<TauIdResults::BitsContainer>(prefix + disc_name) = id_results.GetResultBits();
        }
    }
};

using TauIdDescriptorCollection = std::map<TauIdDiscriminator, TauIdDescriptor>;

#define TAU_ID(name, pattern, has_raw, wp_list) \
    { TauIdDiscriminator::name, TauIdDescriptor(TauIdDiscriminator::name, pattern, has_raw, wp_list) },
inline const TauIdDescriptorCollection& GetTauIdDescriptors()
{
    static const TauIdDescriptorCollection descriptors = { TAU_IDS() };
    return descriptors;
}
#undef TAU_ID

} // namespace tau_id

} // namespace analysis
