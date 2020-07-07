/*! Tools for working with MC generator truth.
This file is part of https://github.com/hh-italian-group/h-tautau. */

#pragma once

#include "DataFormats/HepMCCandidate/interface/GenParticle.h"
#include "SimDataFormats/PileupSummaryInfo/interface/PileupSummaryInfo.h"
#include "AnalysisTools/Core/include/AnalysisMath.h"
#include "AnalysisTools/Core/include/Tools.h"
#include "TauML/Analysis/include/AnalysisTypes.h"
#include "SimDataFormats/GeneratorProducts/interface/LHEEventProduct.h"

namespace tau_analysis {

using namespace ::analysis;

namespace gen_truth {

using GenParticle = reco::GenParticle;
using GenParticleCollection = reco::GenParticleCollection;

struct LeptonMatchResult {
    GenLeptonMatch match{GenLeptonMatch::NoMatch};
    const GenParticle* gen_particle{nullptr};
    std::vector<const GenParticle*> visible_daughters;
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
