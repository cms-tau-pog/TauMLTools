
#include "TauMLTools/Production/interface/Selectors.h"
#include "TauMLTools/Core/interface/AnalysisMath.h"
#include "TauMLTools/Production/interface/TauAnalysis.h"

namespace tau_analysis {
namespace selectors {

std::shared_ptr<TauJetSelector> TauJetSelector::Make(const std::string& name)
{
    if(name.empty() || name == "None")
        return std::make_shared<TauJetSelector>();
    if(name == "MuTau")
        return std::make_shared<MuTau>();
    throw analysis::exception("Unknown selector name = '%1%'") % name;
}

TauJetSelector::Result TauJetSelector::Select(const edm::Event& event, const std::deque<TauJet>& tauJets,
                                      const std::vector<pat::Electron>& electrons,
                                      const std::vector<pat::Muon>& muons, const pat::MET& met,
                                      const reco::Vertex& primaryVertex,
                                      const pat::TriggerObjectStandAloneCollection& triggerObjects,
                                      const edm::TriggerResults& triggerResults)
{
    std::vector<const TauJet*> selected;
    for(const TauJet& tauJet :tauJets)
        selected.push_back(&tauJet);
    return Result(selected, nullptr);
}

TauJetSelector::Result MuTau::Select(const edm::Event& event, const std::deque<TauJet>& tauJets,
                                     const std::vector<pat::Electron>& electrons,
                                     const std::vector<pat::Muon>& muons, const pat::MET& met,
                                     const reco::Vertex& primaryVertex,
                                     const pat::TriggerObjectStandAloneCollection& triggerObjects,
                                     const edm::TriggerResults& triggerResults)
{
    static const std::string filterName = "hltL3crIsoL1sMu22Or25L1f0L2f10QL3f27QL3trkIsoFiltered0p07";
    static const std::set<int> decayModes = { 0, 1, 10, 11 };

    const pat::Muon *ref_muon = nullptr;
    for(const pat::Muon& muon : muons) {
        if(!(muon.pt() > 28 && std::abs(muon.eta()) < 2.1 && muon.isMediumMuon() && PFIsolation(muon) < 0.15
                && std::abs(muon.muonBestTrack()->dxy(primaryVertex.position())) < 0.2
                && std::abs(muon.muonBestTrack()->dz(primaryVertex.position())) < 0.0045))
            continue;
        if(!ref_muon || ref_muon->pt() < muon.pt())
            ref_muon = &muon;

    }
    if(!(ref_muon && analysis::Calculate_MT(ref_muon->polarP4(), met.polarP4()) < 30)) return {};

    bool passTrigger = false;
    for (const pat::TriggerObjectStandAlone& triggerObject : triggerObjects) {
        if(!(reco::deltaR(ref_muon->polarP4(), triggerObject.polarP4()) < 0.3)) continue;
        pat::TriggerObjectStandAlone unpackedTriggerObject(triggerObject);
        unpackedTriggerObject.unpackFilterLabels(event, triggerResults);
        if(unpackedTriggerObject.hasFilterLabel(filterName)) {
            passTrigger = true;
            break;
        }
    }
    if(!passTrigger) return {};

    const TauJet* selectedTau = nullptr;
    for(const TauJet& tauJet : tauJets) {
        if(!tauJet.tau) continue;
        const pat::Tau& tau = *tauJet.tau;

        if(!(tau.pt() > 20 && std::abs(tau.eta()) < 2.3 && tau.tauID("decayModeFindingNewDMs")
                && decayModes.count(tau.decayMode()) && tau.tauID("byMediumDeepTau2017v2p1VSjet") > 0.5f
                && reco::deltaR(ref_muon->polarP4(), tau.polarP4()) > 0.5)) continue;
        if(!selectedTau || selectedTau->tau->pt() < tau.pt())
            selectedTau = &tauJet;
    }
    if(!(selectedTau && (selectedTau->tau->charge() + ref_muon->charge()) == 0)) return {};
    std::vector<const TauJet*> selectedTauJets = { selectedTau };
    auto tagObject = std::make_shared<TagObject>();
    tagObject->p4 = ref_muon->polarP4();
    tagObject->charge = ref_muon->charge();
    tagObject->id = unsigned(ref_muon->isLooseMuon()) * 1 + unsigned(ref_muon->isMediumMuon()) * 2
                    + unsigned(ref_muon->isTightMuon(primaryVertex)) * 4;
    tagObject->isolation = PFIsolation(*ref_muon);
    return Result(selectedTauJets, tagObject);
}

} // namespace selectors
} // namespace tau_analysis
