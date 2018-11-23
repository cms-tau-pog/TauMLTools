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

using MatchResult = std::pair<GenLeptonMatch, const GenParticle*>;

void FindFinalStateDaughters(const GenParticle& particle, std::vector<const GenParticle*>& daughters,
                             const std::set<int>& pdg_to_exclude = {});

LorentzVectorXYZ GetFinalStateMomentum(const reco::GenParticle& particle, bool excludeInvisible,
                                       bool excludeLightLeptons);

MatchResult LeptonGenMatchImpl(const LorentzVectorM& p4, const GenParticleCollection& genParticles);

template<typename LVector>
MatchResult LeptonGenMatch(const LVector& p4, const GenParticleCollection& genParticles)
{
    return LeptonGenMatchImpl(LorentzVectorM(p4), genParticles);
}

float GetNumberOfPileUpInteractions(edm::Handle<std::vector<PileupSummaryInfo>>& pu_infos);

// Copied from https://github.com/cms-tau-pog/TauAnalysisTools/blob/master/TauAnalysisTools/plugins/TauIdMVATrainingNtupleProducer.cc#L808-L833
const reco::GenParticle* FindMatchingGenParticle(const LorentzVectorXYZ& recTauP4,
                                                 const GenParticleCollection& genParticles,
                                                 double minGenVisPt, const std::vector<int>& pdgIds,
                                                 double dRmatch, double& dRmin);

} // namespace gen_truth
} // namespace analysis
