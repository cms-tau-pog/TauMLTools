/*! Common simple types for analysis purposes.
This file is part of https://github.com/hh-italian-group/h-tautau. */

#pragma once

#include "AnalysisTools/Core/include/PhysicalValue.h"
#include "AnalysisTools/Core/include/EnumNameMap.h"

namespace analysis {

enum class TauType { e = 0, mu = 1, tau = 2, jet = 3 };
ENUM_NAMES(TauType) = {
    { TauType::e, "e" }, { TauType::mu, "mu" }, { TauType::tau, "tau" }, { TauType::jet, "jet" }
};

enum class DiscriminatorWP { VVLoose, VLoose, Loose, Medium, Tight, VTight, VVTight };
ENUM_NAMES(DiscriminatorWP) = {
    { DiscriminatorWP::VVLoose, "VVLoose" }, { DiscriminatorWP::VLoose, "VLoose" }, { DiscriminatorWP::Loose, "Loose" },
    { DiscriminatorWP::Medium, "Medium" }, { DiscriminatorWP::Tight, "Tight" }, { DiscriminatorWP::VTight, "VTight" },
    { DiscriminatorWP::VVTight, "VVTight" }
};
const EnumNameMap<DiscriminatorWP> __DiscriminatorWP_short_names("ShortWPNames", {
    { DiscriminatorWP::VVLoose, "VVL" }, { DiscriminatorWP::VLoose, "VL" }, { DiscriminatorWP::Loose, "L" },
    { DiscriminatorWP::Medium, "M" }, { DiscriminatorWP::Tight, "T" }, { DiscriminatorWP::VTight, "VT" },
    { DiscriminatorWP::VVTight, "VVT" }
});

enum class Period { Run2015, Run2016, Run2017, Run2018 };
ENUM_NAMES(Period) = {
    { Period::Run2015, "Run2015" },
    { Period::Run2016, "Run2016" },
    { Period::Run2017, "Run2017" },
    { Period::Run2018, "Run2018" },
};

enum class GenMatch { Electron = 1, Muon = 2, TauElectron = 3,  TauMuon = 4, Tau = 5, NoMatch = 6 };
ENUM_NAMES(GenMatch) = {
    { GenMatch::Electron, "gen_electron" },
    { GenMatch::Muon, "gen_muon" },
    { GenMatch::TauElectron, "gen_electron_from_tau" },
    { GenMatch::TauMuon, "gen_muon_from_tau" },
    { GenMatch::Tau, "gen_tau" },
    { GenMatch::NoMatch, "no_gen_match" }
};

inline TauType GenMatchToTauType(GenMatch gen_match)
{
    if(gen_match == GenMatch::Electron || gen_match == GenMatch::TauElectron) return TauType::e;
    if(gen_match == GenMatch::Muon || gen_match == GenMatch::TauMuon) return TauType::mu;
    if(gen_match == GenMatch::Tau) return TauType::tau;
    return TauType::jet;
}

enum class GenEventType { Other = 0, TTbar_Hadronic = 1, TTbar_SemiLeptonic = 2, TTbar_Leptonic = 3 };
ENUM_NAMES(GenEventType) = {
    { GenEventType::Other, "other" },
    { GenEventType::TTbar_Hadronic, "TTbar_Hadronic" },
    { GenEventType::TTbar_SemiLeptonic, "TTbar_SemiLeptonic" },
    { GenEventType::TTbar_Leptonic, "TTbar_Leptonic" },
};

} // namespace analysis
