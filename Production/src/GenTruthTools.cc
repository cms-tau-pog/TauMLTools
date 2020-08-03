/*! Tools for working with MC generator truth.
This file is part of https://github.com/hh-italian-group/h-tautau. */

#include "TauMLTools/Production/interface/GenTruthTools.h"

namespace tau_analysis {

using namespace ::analysis;

namespace gen_truth {

FinalState::FinalState(const reco::GenParticle& particle, const std::set<int>& pdg_to_exclude,
                       const std::set<const reco::GenParticle*>& particles_to_exclude)
{
    findFinalStateParticles(particle, pdg_to_exclude, particles_to_exclude);
}

void FinalState::findFinalStateParticles(const reco::GenParticle& particle, const std::set<int>& pdg_to_exclude,
                                         const std::set<const reco::GenParticle*>& particles_to_exclude)
{
    if(!particles_to_exclude.count(&particle)) {
        if(particle.daughterRefVector().empty()) {
            const int abs_pdg = std::abs(particle.pdgId());
            if(!pdg_to_exclude.count(abs_pdg))
                addParticle(particle);
        } else {
            for(const auto& daughter : particle.daughterRefVector())
                findFinalStateParticles(*daughter, pdg_to_exclude, particles_to_exclude);
        }
    }
}

void FinalState::addParticle(const reco::GenParticle& particle)
{
    static constexpr int gamma = 22;
    static const std::set<int> light_leptons = { 11, 13 }, neutrinos = { 12, 14, 16 };
    const int abs_pdg = std::abs(particle.pdgId());
    std::set<ParticleType> types;
    if(neutrinos.count(abs_pdg)) {
        types.insert(ParticleType::neutrino);
    } else {
        types.insert(ParticleType::visible);
        if(light_leptons.count(abs_pdg)) {
            types.insert(ParticleType::light_lepton);
        } else if(abs_pdg == gamma) {
            types.insert(ParticleType::gamma);
        } else {
            if(particle.charge() == 0)
                types.insert(ParticleType::neutral_hadron);
            else
                types.insert(ParticleType::charged_hadron);
        }
    }
    for(ParticleType type : types) {
        if(!particles[type].count(&particle)) {
            particles[type].insert(&particle);
            momentum[type] += particle.p4();
        }
    }
}

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

const reco::GenParticle* FindTerminalCopy(const reco::GenParticle& genParticle, bool first)
{
    const reco::GenParticle* particle = &genParticle;
    while((first && !particle->statusFlags().isFirstCopy()) || (!first && !particle->statusFlags().isLastCopy())) {
        bool nextCopyFound = false;
        const auto& refVector = first ? particle->motherRefVector() : particle->daughterRefVector();
        for(const auto& p : refVector) {
            if(p->pdgId() == particle->pdgId()) {
                particle = &(*p);
                nextCopyFound = true;
                break;
            }
        }
        if(!nextCopyFound) {
            const std::string pos = first ? "first" : "last";
            throw analysis::exception("Unable to find the %1% copy") % pos;
        }
    }
    return particle;
}

bool FindLeptonGenMatch(const reco::GenParticle& particle, LeptonMatchResult& result,
                        const LorentzVectorM* ref_p4, double* best_match_dr2)
{
    static constexpr int electronPdgId = 11, muonPdgId = 13, tauPdgId = 15;

    static const std::map<int, double> pt_thresholds = {
        { electronPdgId, 8 }, { muonPdgId, 8 }, { tauPdgId, 15 }
    };

    using pair = std::pair<int, bool>;
    static const std::map<pair, GenLeptonMatch> genMatches = {
        { { electronPdgId, false }, GenLeptonMatch::Electron },
        { { electronPdgId, true }, GenLeptonMatch::TauElectron },
        { { muonPdgId, false }, GenLeptonMatch::Muon }, { { muonPdgId, true }, GenLeptonMatch::TauMuon },
        { { tauPdgId, false }, GenLeptonMatch::Tau }, { { tauPdgId, true }, GenLeptonMatch::Tau }
    };

    const bool isTauProduct = particle.statusFlags().isDirectPromptTauDecayProduct();
    if(!((particle.statusFlags().isPrompt() || isTauProduct) && particle.statusFlags().isFirstCopy())) return false;
    const int abs_pdg = std::abs(particle.pdgId());
    if(!pt_thresholds.count(abs_pdg)) return false;

    const reco::GenParticle* particle_lastCopy = FindTerminalCopy(particle, false);
    FinalState finalState(*particle_lastCopy), finalState_rad(particle, {}, {particle_lastCopy});
    const auto& vis_p4 = finalState.getMomentum(FinalState::ParticleType::visible);
    const auto& vis_rad_p4 = finalState_rad.getMomentum(FinalState::ParticleType::visible);
    const auto total_vis_p4 = vis_p4 + vis_rad_p4;

    GenLeptonMatch match;
    if(abs_pdg == tauPdgId && finalState.count(FinalState::ParticleType::light_lepton)) {
        auto light_lepton = *finalState.getParticles(FinalState::ParticleType::light_lepton).begin();
        const int abs_lep_pdg = std::abs(light_lepton->pdgId());
        const double pt_thr = pt_thresholds.at(abs_lep_pdg);
        if(light_lepton->pt() > pt_thr || total_vis_p4.pt() < pt_thr) return false;
        match = genMatches.at(pair(abs_lep_pdg, true));
    } else {
        if(total_vis_p4.pt() <= pt_thresholds.at(abs_pdg)) return false;
        match = genMatches.at(pair(abs_pdg, isTauProduct));
    }

    if(ref_p4 != nullptr && best_match_dr2 != nullptr) {
        const double dr2_vis = ROOT::Math::VectorUtil::DeltaR2(*ref_p4, vis_p4);
        const double dr2_tot_vis = ROOT::Math::VectorUtil::DeltaR2(*ref_p4, total_vis_p4);
        const double dr2 = std::min(dr2_vis, dr2_tot_vis);
        if(dr2 >= *best_match_dr2) return false;
        *best_match_dr2 = dr2;
    }

    result.match = match;
    result.gen_particle_firstCopy = &particle;
    result.gen_particle_lastCopy = particle_lastCopy;
    result.visible_daughters = finalState.getParticles(FinalState::ParticleType::visible);
    result.visible_rad = finalState_rad.getParticles(FinalState::ParticleType::visible);
    result.visible_p4 = finalState.getMomentum(FinalState::ParticleType::visible);
    result.visible_rad_p4 = finalState_rad.getMomentum(FinalState::ParticleType::visible);
    result.n_charged_hadrons = finalState.count(FinalState::ParticleType::charged_hadron);
    result.n_neutral_hadrons = finalState.count(FinalState::ParticleType::neutral_hadron);
    result.n_gammas = finalState.count(FinalState::ParticleType::gamma);
    result.n_gammas_rad = finalState_rad.count(FinalState::ParticleType::gamma);

    return true;
}

LeptonMatchResult LeptonGenMatch(const LorentzVectorM& p4, const reco::GenParticleCollection& genParticles)
{
    static const double dR2_threshold = std::pow(0.2, 2);
    LeptonMatchResult result;
    double best_match_dr2 = dR2_threshold;
    for(const reco::GenParticle& particle : genParticles)
        FindLeptonGenMatch(particle, result, &p4, &best_match_dr2);
    return result;
}

LeptonMatchResult LeptonGenMatch(const LorentzVectorM& p4, const std::vector<LeptonMatchResult>& genLeptons)
{
    static const double dR2_threshold = std::pow(0.2, 2);
    LeptonMatchResult result;
    double best_match_dr2 = dR2_threshold;
    for(const LeptonMatchResult& lepton : genLeptons) {
        const auto total_vis_p4 = lepton.visible_p4 + lepton.visible_rad_p4;
        const double dr2_vis = ROOT::Math::VectorUtil::DeltaR2(p4, lepton.visible_p4);
        const double dr2_tot_vis = ROOT::Math::VectorUtil::DeltaR2(p4, total_vis_p4);
        const double dr2 = std::min(dr2_vis, dr2_tot_vis);
        if(dr2 >= best_match_dr2) continue;
        best_match_dr2 = dr2;
        result = lepton;
    }
    return result;
}

QcdMatchResult QcdGenMatch(const LorentzVectorM& p4, const GenParticleCollection& genParticles)
{
    static const std::set<int> qcdPdg = { 1, 2, 3, 4, 5, 6, 21 };
    static constexpr double dR2_threshold = std::pow(0.5, 2);
    static constexpr double pt_threshold = 10;

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
