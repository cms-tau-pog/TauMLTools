/* Tau jet selectors.
*/

#pragma once

#include "DataFormats/PatCandidates/interface/MET.h"
#include "FWCore/Framework/interface/Event.h"
#include "TauMLTools/Production/interface/TauJet.h"

namespace tau_analysis {
namespace selectors {

struct TagObject {
    pat::PackedCandidate::PolarLorentzVector p4;
    int charge;
    unsigned id;
    float isolation;
    bool extramuon;
    bool extraelectron;
    bool extradimuon;
};

bool muonveto (const std::vector<pat::Muon>& muons);

bool electronveto (const std::vector<pat::Electron>& electrons);

bool dimuonveto (const std::vector<pat::Muon>& muons);

struct TauJetSelector {
    using Result = std::tuple<std::vector<const TauJet*>, std::shared_ptr<TagObject>>;

    virtual ~TauJetSelector() {}
    virtual Result Select(const edm::Event& event, const std::deque<TauJet>& tauJets,
                          const std::vector<pat::Electron>& electrons,
                          const std::vector<pat::Muon>& muons, const pat::MET& met,
                          const reco::Vertex& primaryVertex,
                          const pat::TriggerObjectStandAloneCollection& triggerObjects,
                          const edm::TriggerResults& triggerResults, const float rho);

    static std::shared_ptr<TauJetSelector> Make(const std::string& name);
};

struct MuTau : TauJetSelector {
    virtual Result Select(const edm::Event& event, const std::deque<TauJet>& tauJets,
                          const std::vector<pat::Electron>& electrons,
                          const std::vector<pat::Muon>& muons, const pat::MET& met,
                          const reco::Vertex& primaryVertex,
                          const pat::TriggerObjectStandAloneCollection& triggerObjects,
                          const edm::TriggerResults& triggerResults, const float rho) override;
};

} // namespace selectors
} // namespace tau_analysis
