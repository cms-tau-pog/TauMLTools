/*! Tools for working with MC generator truth.
This file is part of https://github.com/hh-italian-group/h-tautau. */

#pragma once

#include "DataFormats/HepMCCandidate/interface/GenParticle.h"
#include "SimDataFormats/PileupSummaryInfo/interface/PileupSummaryInfo.h"
#include "TauMLTools/Core/interface/AnalysisMath.h"
#include "TauMLTools/Core/interface/Tools.h"
#include "TauMLTools/Analysis/interface/AnalysisTypes.h"
#include "SimDataFormats/GeneratorProducts/interface/LHEEventProduct.h"

namespace tau_analysis {

using namespace ::analysis;

namespace gen_truth {

using GenParticle = reco::GenParticle;
using GenParticleCollection = reco::GenParticleCollection;

struct FinalState {
public:
    enum class ParticleType { visible, light_lepton, neutrino, gamma, charged_hadron, neutral_hadron };

    explicit FinalState(const reco::GenParticle& particle, const std::set<int>& pdg_to_exclude = {},
                        const std::set<const reco::GenParticle*>& particles_to_exclude = {});

    const std::set<const reco::GenParticle*>& getParticles(ParticleType type) { return particles[type]; }
    const LorentzVectorXYZ& getMomentum(ParticleType type) { return momentum[type]; }
    size_t count(ParticleType type) { return getParticles(type).size(); }

private:
    void findFinalStateParticles(const reco::GenParticle& particle, const std::set<int>& pdg_to_exclude,
                                 const std::set<const reco::GenParticle*>& particles_to_exclude);
    void addParticle(const reco::GenParticle& particle);

private:
    std::map<ParticleType, std::set<const reco::GenParticle*>> particles;
    std::map<ParticleType, LorentzVectorXYZ> momentum;
};

struct LeptonMatchResult {
    GenLeptonMatch match{GenLeptonMatch::NoMatch};
    const reco::GenParticle *gen_particle_firstCopy{nullptr}, *gen_particle_lastCopy{nullptr};
    std::set<const reco::GenParticle*> visible_daughters, visible_rad;
    LorentzVectorXYZ visible_p4, visible_rad_p4;
    unsigned n_charged_hadrons{0}, n_neutral_hadrons{0}, n_gammas{0}, n_gammas_rad{0};
};

struct QcdMatchResult {
    GenQcdMatch match{GenQcdMatch::NoMatch};
    const GenParticle* gen_particle{nullptr};
};

void FindFinalStateDaughters(const GenParticle& particle, std::set<const GenParticle*>& daughters,
                             const std::set<int>& pdg_to_exclude = {});

LorentzVectorXYZ GetFinalStateMomentum(const reco::GenParticle& particle,
                                       std::vector<const GenParticle*>& visible_daughters,
                                       bool excludeInvisible, bool excludeLightLeptons);

LeptonMatchResult LeptonGenMatch(const LorentzVectorM& p4, const GenParticleCollection& genParticles);

QcdMatchResult QcdGenMatch(const LorentzVectorM& p4, const GenParticleCollection& genParticles);

float GetNumberOfPileUpInteractions(edm::Handle<std::vector<PileupSummaryInfo>>& pu_infos);

} // namespace gen_truth
} // namespace analysis
