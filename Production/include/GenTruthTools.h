/*! Tools for working with MC generator truth.
This file is part of https://github.com/hh-italian-group/h-tautau. */

#pragma once

#include "DataFormats/HepMCCandidate/interface/GenParticle.h"
#include "SimDataFormats/PileupSummaryInfo/interface/PileupSummaryInfo.h"
#include "AnalysisTools/Core/include/AnalysisMath.h"
#include "AnalysisTools/Core/include/Tools.h"
#include "TauML/Analysis/include/AnalysisTypes.h"
#include "SimDataFormats/GeneratorProducts/interface/LHEEventProduct.h"


namespace analysis {

namespace gen_truth {

using MatchResult = std::pair<GenMatch, const reco::GenParticle*>;

inline void FindFinalStateDaughters(const reco::GenParticle& particle, std::vector<const reco::GenParticle*>& daughters,
                                    const std::set<int>& pdg_to_exclude = {})
{
    if(!particle.daughterRefVector().size()) {
        const int abs_pdg = std::abs(particle.pdgId());
        if(!pdg_to_exclude.count(abs_pdg))
            daughters.push_back(&particle);
    } else {
        for(const auto& daughter : particle.daughterRefVector())
            FindFinalStateDaughters(*daughter, daughters, pdg_to_exclude);
    }
}

inline LorentzVectorXYZ GetFinalStateMomentum(const reco::GenParticle& particle, bool excludeInvisible,
                                              bool excludeLightLeptons)
{
    using set = std::set<int>;
    using pair = std::pair<bool, bool>;
    static const set empty = {};
    static const set light_leptons = { 11, 13 };
    static const set invisible_particles = { 12, 14, 16 };
    static const set light_and_invisible = tools::union_sets({light_leptons, invisible_particles});

    static const std::map<pair, const set*> to_exclude {
        { pair(false, false), &empty }, { pair(true, false), &invisible_particles },
        { pair(false, true), &light_leptons }, { pair(true, true), &light_and_invisible },
    };

    std::vector<const reco::GenParticle*> daughters;
    FindFinalStateDaughters(particle, daughters, *to_exclude.at(pair(excludeInvisible, false)));

    LorentzVectorXYZ p4;
    for(auto daughter : daughters){
    if(excludeLightLeptons && light_leptons.count(std::abs(daughter->pdgId())) && daughter->statusFlags().isDirectTauDecayProduct()) continue;
    p4 += daughter->p4();
    }
    return p4;
}

template<typename LVector>
MatchResult LeptonGenMatch(const LVector& p4, const std::vector<reco::GenParticle>& genParticles)
{
    static constexpr int electronPdgId = 11, muonPdgId = 13, tauPdgId = 15;
    static constexpr double dR2_threshold = std::pow(0.2, 2);

    static const std::map<int, double> pt_thresholds = {
        { electronPdgId, 8 }, { muonPdgId, 8 }, { tauPdgId, 15 }
    };

    using pair = std::pair<int, bool>;
    static const std::map<pair, GenMatch> genMatches = {
        { { electronPdgId, false }, GenMatch::Electron }, { { electronPdgId, true }, GenMatch::TauElectron },
        { { muonPdgId, false }, GenMatch::Muon }, { { muonPdgId, true }, GenMatch::TauMuon },
        { { tauPdgId, false }, GenMatch::Tau }, { { tauPdgId, true }, GenMatch::Tau }
    };

    MatchResult result(GenMatch::NoMatch, nullptr);
    double match_dr2 = dR2_threshold;



    for(const reco::GenParticle& particle : genParticles) {
        const bool isTauProduct = particle.statusFlags().isDirectPromptTauDecayProduct();
        if((!particle.statusFlags().isPrompt() && !isTauProduct) || !particle.statusFlags().isLastCopy()) continue;

        const int abs_pdg = std::abs(particle.pdgId());
        if(!pt_thresholds.count(abs_pdg)) continue;

        const auto particle_p4 = abs_pdg == tauPdgId ? GetFinalStateMomentum(particle, true, true) : particle.p4();

        const double dr2 = ROOT::Math::VectorUtil::DeltaR2(p4, particle_p4);
        if(dr2 >= match_dr2) continue;
        if(particle_p4.pt() <= pt_thresholds.at(abs_pdg)) continue;

        match_dr2 = dr2;
        result.first = genMatches.at(pair(abs_pdg, isTauProduct));
        result.second = &particle;
    }
    return result;
}

inline float GetNumberOfPileUpInteractions(edm::Handle<std::vector<PileupSummaryInfo>>& pu_infos)
{
    if(pu_infos.isValid()) {
        for(const PileupSummaryInfo& pu : *pu_infos) {
            if(pu.getBunchCrossing() == 0)
                return pu.getTrueNumInteractions();
        }
    }
    return std::numeric_limits<float>::lowest();
}

struct LheSummary {
    size_t n_partons = 0, n_b_partons = 0, n_c_partons = 0;
    double HT = 0., m_H = 0., m_hh = 0., cosTheta_hh = 0.;
};

inline LheSummary ExtractLheSummary(const LHEEventProduct& lheEventProduct)
{
    static constexpr int c_quark = 4, b_quark = 5, H0 = 35, h0 = 25;
    static const std::set<int> quarks_and_gluons = { 1, 2, 3, 4, 5, 6, 21 };

    LheSummary summary;
    const lhef::HEPEUP& lheEvent = lheEventProduct.hepeup();
    const std::vector<lhef::HEPEUP::FiveVector>& lheParticles = lheEvent.PUP;
    std::vector<analysis::LorentzVectorXYZ> h0_p4;
    for(size_t n = 0; n < lheParticles.size(); ++n) {
        const int absPdgId = std::abs(lheEvent.IDUP[n]);
        const int status = lheEvent.ISTUP[n];
        if(absPdgId == H0) summary.m_H = lheParticles[n][4];
        if(absPdgId == h0)
            h0_p4.push_back(analysis::LorentzVectorXYZ(lheParticles[n][0], lheParticles[n][1],
                                                              lheParticles[n][2], lheParticles[n][3]));
        if(status != 1 || !quarks_and_gluons.count(absPdgId)) continue;
        ++summary.n_partons;
        if(absPdgId == c_quark) ++summary.n_c_partons;
        if(absPdgId == b_quark) ++summary.n_b_partons;
        summary.HT += std::sqrt(std::pow(lheParticles[n][0], 2) + std::pow(lheParticles[n][1], 2));
    }
    if(h0_p4.size() == 2) {
        const analysis::LorentzVectorXYZ H_p4 = h0_p4.at(0) + h0_p4.at(1);
        const auto boosted_h0 = ROOT::Math::VectorUtil::boost(h0_p4.at(0), H_p4.BoostToCM());
        summary.cosTheta_hh = ROOT::Math::VectorUtil::CosTheta(boosted_h0, ROOT::Math::Cartesian3D<>(0, 0, 1));
        summary.m_hh = H_p4.mass();
    }
    return summary;
}

} // namespace gen_truth
} // namespace analysis
