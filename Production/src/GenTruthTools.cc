/*! Tools for working with MC generator truth.
This file is part of https://github.com/hh-italian-group/h-tautau. */

#include "TauML/Production/include/GenTruthTools.h"

namespace tau_analysis {
namespace gen_truth {

void FindFinalStateDaughters(const GenParticle& particle, std::set<const GenParticle*>& daughters,
                             const std::set<int>& pdg_to_exclude)
{
    if(!particle.daughterRefVector().size()) {
        const int abs_pdg = std::abs(particle.pdgId());
        if(!pdg_to_exclude.count(abs_pdg))
            daughters.insert(&particle);
    } else {
        for(const auto& daughter : particle.daughterRefVector())
            FindFinalStateDaughters(*daughter, daughters, pdg_to_exclude);
    }
}

LorentzVectorXYZ GetFinalStateMomentum(const GenParticle& particle, std::vector<const GenParticle*>& visible_daughters,
                                       bool excludeInvisible, bool excludeLightLeptons)
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



    std::set<const GenParticle*> daughters_set;
    FindFinalStateDaughters(particle, daughters_set, *to_exclude.at(pair(excludeInvisible, false)));
    visible_daughters.clear();
    visible_daughters.insert(visible_daughters.begin(), daughters_set.begin(), daughters_set.end());

    LorentzVectorXYZ p4;
    for(auto daughter : visible_daughters) {
        if(excludeLightLeptons && light_leptons.count(std::abs(daughter->pdgId()))
                && daughter->statusFlags().isDirectTauDecayProduct()) continue;
            p4 += daughter->p4();
    }
    return p4;
}

LeptonMatchResult LeptonGenMatch(const LorentzVectorM& p4, const GenParticleCollection& genParticles)
{
    static constexpr int electronPdgId = 11, muonPdgId = 13, tauPdgId = 15;
    static constexpr double dR2_threshold = std::pow(0.2, 2);

    static const std::map<int, double> pt_thresholds = {
        { electronPdgId, 8 }, { muonPdgId, 8 }, { tauPdgId, 15 }
    };

    using pair = std::pair<int, bool>;
    static const std::map<pair, GenLeptonMatch> genMatches = {
        { { electronPdgId, false }, GenLeptonMatch::Electron }, { { electronPdgId, true }, GenLeptonMatch::TauElectron },
        { { muonPdgId, false }, GenLeptonMatch::Muon }, { { muonPdgId, true }, GenLeptonMatch::TauMuon },
        { { tauPdgId, false }, GenLeptonMatch::Tau }, { { tauPdgId, true }, GenLeptonMatch::Tau }
    };

    LeptonMatchResult result;
    double match_dr2 = dR2_threshold;

    for(const reco::GenParticle& particle : genParticles) {
        const bool isTauProduct = particle.statusFlags().isDirectPromptTauDecayProduct();
        if((!particle.statusFlags().isPrompt() && !isTauProduct) /*|| !particle.statusFlags().isLastCopy()*/) continue;

        const int abs_pdg = std::abs(particle.pdgId());
        if(!pt_thresholds.count(abs_pdg)) continue;

        std::vector<const GenParticle*> visible_daughters;
        const auto particle_p4 = abs_pdg == tauPdgId ? GetFinalStateMomentum(particle, visible_daughters, true, true)
                                                     : particle.p4();

        const double dr2 = ROOT::Math::VectorUtil::DeltaR2(p4, particle_p4);
        if(dr2 >= match_dr2) continue;
        if(particle_p4.pt() <= pt_thresholds.at(abs_pdg)) continue;

        match_dr2 = dr2;
        result.match = genMatches.at(pair(abs_pdg, isTauProduct));
        result.gen_particle = &particle;
        result.visible_daughters = visible_daughters;
    }
    return result;
}

QcdMatchResult QcdGenMatch(const LorentzVectorM& p4, const GenParticleCollection& genParticles)
{
    static const std::set<int> qcdPdg = { 1, 2, 3, 4, 5, 6, 21 };
    static constexpr double dR2_threshold = std::pow(0.5, 2);
    static constexpr double pt_threshold = 15;

    QcdMatchResult result;
    double match_dr2 = dR2_threshold;

    for(const reco::GenParticle& particle : genParticles) {
        if(!(particle.statusFlags().isPrompt() && particle.statusFlags().isHardProcess()
             && !particle.statusFlags().fromHardProcessBeforeFSR())) continue;

        const int abs_pdg = std::abs(particle.pdgId());
        const auto& particle_p4 = particle.p4();
        if(!(qcdPdg.count(abs_pdg) && particle_p4.pt() > pt_threshold)) continue;

        const double dr2 = ROOT::Math::VectorUtil::DeltaR2(p4, particle_p4);
        if(dr2 >= match_dr2) continue;

        match_dr2 = dr2;
        result.match = static_cast<GenQcdMatch>(abs_pdg);
        result.gen_particle = &particle;
    }
    return result;
}

float GetNumberOfPileUpInteractions(edm::Handle<std::vector<PileupSummaryInfo>>& pu_infos)
{
    if(pu_infos.isValid()) {
        for(const PileupSummaryInfo& pu : *pu_infos) {
            if(pu.getBunchCrossing() == 0)
                return pu.getTrueNumInteractions();
        }
    }
    return std::numeric_limits<float>::lowest();
}

} // namespace gen_truth
} // namespace analysis
