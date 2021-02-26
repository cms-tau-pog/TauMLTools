/*! Common simple types for analysis purposes.
This file is part of https://github.com/hh-italian-group/h-tautau. */

#pragma once

#include "TauMLTools/Core/interface/PhysicalValue.h"
#include "TauMLTools/Core/interface/EnumNameMap.h"

namespace analysis {

enum class DiscriminatorWP { VVVLoose = 0, VVLoose = 1, VLoose = 2, Loose = 3, Medium = 4, Tight = 5,
                             VTight = 6, VVTight = 7, VVVTight = 8 };
ENUM_NAMES(DiscriminatorWP) = {
    { DiscriminatorWP::VVVLoose, "VVVLoose" }, { DiscriminatorWP::VVLoose, "VVLoose" },
    { DiscriminatorWP::VLoose, "VLoose" }, { DiscriminatorWP::Loose, "Loose" }, { DiscriminatorWP::Medium, "Medium" },
    { DiscriminatorWP::Tight, "Tight" }, { DiscriminatorWP::VTight, "VTight" }, { DiscriminatorWP::VVTight, "VVTight" },
    { DiscriminatorWP::VVVTight, "VVVTight" }
};
const EnumNameMap<DiscriminatorWP> __DiscriminatorWP_short_names("ShortWPNames", {
    { DiscriminatorWP::VVVLoose, "VVVL" }, { DiscriminatorWP::VVLoose, "VVL" }, { DiscriminatorWP::VLoose, "VL" },
    { DiscriminatorWP::Loose, "L" }, { DiscriminatorWP::Medium, "M" }, { DiscriminatorWP::Tight, "T" },
    { DiscriminatorWP::VTight, "VT" }, { DiscriminatorWP::VVTight, "VVT" }, { DiscriminatorWP::VVVTight, "VVVT" }
});

enum class Period { Run2011, Run2012, Run2015, Run2016, Run2017, Run2018 };
ENUM_NAMES(Period) = {
    { Period::Run2011, "Run2011" }, { Period::Run2012, "Run2012" }, { Period::Run2015, "Run2015" },
    { Period::Run2016, "Run2016" }, { Period::Run2017, "Run2017" }, { Period::Run2018, "Run2018" },
};

enum class GenLeptonMatch { Electron = 1, Muon = 2, TauElectron = 3,  TauMuon = 4, Tau = 5, NoMatch = 6 };
ENUM_NAMES(GenLeptonMatch) = {
    { GenLeptonMatch::Electron, "gen_electron" },
    { GenLeptonMatch::Muon, "gen_muon" },
    { GenLeptonMatch::TauElectron, "gen_electron_from_tau" },
    { GenLeptonMatch::TauMuon, "gen_muon_from_tau" },
    { GenLeptonMatch::Tau, "gen_tau" },
    { GenLeptonMatch::NoMatch, "no_gen_match" }
};

enum class GenQcdMatch { NoMatch = 0, Down = 1, Up = 2, Strange = 3, Charm = 4, Bottom = 5, Top = 6, Gluon = 21 };
ENUM_NAMES(GenQcdMatch) = {
    { GenQcdMatch::NoMatch, "no_gen_match" },
    { GenQcdMatch::Down, "gen_down_quark" },
    { GenQcdMatch::Up, "gen_up_quark" },
    { GenQcdMatch::Strange, "gen_strange_quark" },
    { GenQcdMatch::Charm, "gen_charm_quark" },
    { GenQcdMatch::Bottom, "gen_bottom_quark" },
    { GenQcdMatch::Top, "gen_top_quark" },
    { GenQcdMatch::Gluon, "gen_gluon" }
};

enum class TauType { e = 0, mu = 1, tau = 2, jet = 3, emb_e = 4, emb_mu = 5, emb_tau = 6, emb_jet = 7, data = 8 };
ENUM_NAMES(TauType) = {
    { TauType::e, "e" }, { TauType::mu, "mu" }, { TauType::tau, "tau" }, { TauType::jet, "jet" },
    { TauType::emb_e, "emb_e" }, { TauType::emb_mu, "emb_mu" }, { TauType::emb_tau, "emb_tau" }, { TauType::emb_jet, "emb_jet" },
    { TauType::data, "data" }
};

enum class SampleType { Data = 0, MC = 1, Embedded = 2 };
ENUM_NAMES(SampleType) = {
    { SampleType::Data, "Data" }, { SampleType::MC, "MC" }, { SampleType::Embedded, "Embedded" }
};

inline TauType GenMatchToTauType(GenLeptonMatch gen_match, SampleType sample_type)
{
    if(sample_type == SampleType::MC && (gen_match == GenLeptonMatch::Electron
            || gen_match == GenLeptonMatch::TauElectron)) {
        return TauType::e;
    } else if(sample_type == SampleType::MC && (gen_match == GenLeptonMatch::Muon
            || gen_match == GenLeptonMatch::TauMuon)) {
        return TauType::mu;
    } else if(gen_match == GenLeptonMatch::Tau) {
        if(sample_type == SampleType::MC) return TauType::tau;
        if(sample_type == SampleType::Embedded) return TauType::emb_tau;
    } else if(gen_match == GenLeptonMatch::NoMatch) {
        if(sample_type == SampleType::MC) return TauType::jet;
        if(sample_type == SampleType::Data) return TauType::data;
        if(sample_type == SampleType::Embedded) return TauType::emb_jet;
    } else if(sample_type == SampleType::Embedded) {
        if(gen_match == GenLeptonMatch::TauMuon) return TauType::emb_mu;
        if(gen_match == GenLeptonMatch::TauElectron) return TauType::emb_e;
    }
    throw exception("Incompatible gen_lepton_match = %1% and sample_type = %2%.") % gen_match % sample_type;
}

// based on reco::PFCandidate::ParticleType
enum class PFParticleType {
    Undefined = 0,  // undefined
    h = 1,          // charged hadron
    e = 2,          // electron
    mu = 3,         // muon
    gamma = 4,      // photon
    h0 = 5,         // neutral hadron
    h_HF = 6,       // HF tower identified as a hadron
    egamma_HF = 7,  // HF tower identified as an EM particle
};
ENUM_NAMES(PFParticleType) = {
    { PFParticleType::Undefined, "Undefined" }, { PFParticleType::h, "h" }, { PFParticleType::e, "e" },
    { PFParticleType::mu, "mu" }, { PFParticleType::gamma, "gamma" }, { PFParticleType::h0, "h0" },
    { PFParticleType::h_HF, "h_HF" }, { PFParticleType::egamma_HF, "egamma_HF" },
};

inline PFParticleType TranslatePdgIdToPFParticleType(int pdgId)
{
    static const std::map<int, PFParticleType> type_map = {
        { 11, PFParticleType::e }, { 13, PFParticleType::mu }, { 22, PFParticleType::gamma },
        { 211, PFParticleType::h }, { 130, PFParticleType::h0 },
        { 1, PFParticleType::h_HF }, { 2, PFParticleType::egamma_HF },
    };
    auto iter = type_map.find(std::abs(pdgId));
    return iter == type_map.end() ? PFParticleType::Undefined : iter->second;
}

} // namespace analysis
