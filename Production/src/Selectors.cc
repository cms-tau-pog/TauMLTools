
#include "TauMLTools/Production/interface/Selectors.h"
#include "TauMLTools/Core/interface/AnalysisMath.h"
#include "TauMLTools/Production/interface/TauAnalysis.h"

namespace {

bool hasExtraMuon (const std::vector<pat::Muon>& muons, const pat::Muon *ref_muon, const reco::Vertex& primaryVertex){
    for(const pat::Muon& muon : muons) {
        if(muon.pt() > 10 && std::abs(muon.eta()) < 2.4 && muon.isMediumMuon() && tau_analysis::PFRelIsolation(muon) < 0.30 && std::abs(muon.muonBestTrack()->dz(primaryVertex.position())) < 0.2
                && std::abs(muon.muonBestTrack()->dxy(primaryVertex.position())) < 0.0045 && &muon != ref_muon){
            return true;
        }
    }
    return false;
}

bool hasExtraElectron (const std::vector<pat::Electron>& electrons, float rho){
    for(const pat::Electron& electron : electrons) {
        if(electron.pt() > 10 && std::abs(electron.eta()) < 2.5 && electron.electronID("mvaEleID-Fall17-noIso-V2-wp90") > 0.5f && tau_analysis::PFRelIsolation(electron, rho)<0.3){
            return true;
        }
    }
    return false;
}

bool hasExtraDimuon (const std::vector<pat::Muon>& muons, const reco::Vertex& primaryVertex){
    std::vector<const pat::Muon*> dimuon_cands; // vector of all muons that pass selection
    for(const pat::Muon& muon : muons) {
        if(muon.pt() > 15 && std::abs(muon.eta()) < 2.4 && muon.isLooseMuon() && tau_analysis::PFRelIsolation(muon) < 0.30 && std::abs(muon.muonBestTrack()->dz(primaryVertex.position())) < 0.2
                && std::abs(muon.muonBestTrack()->dxy(primaryVertex.position())) < 0.0045)
                    dimuon_cands.push_back(&muon);
    }
    for(size_t m1 = 0; m1 < dimuon_cands.size(); ++m1) {
        for(size_t m2 = m1 + 1; m2 < dimuon_cands.size(); ++m2){
            if (reco::deltaR(dimuon_cands.at(m1)->polarP4(), dimuon_cands.at(m2)->polarP4()) > 0.15 && (dimuon_cands.at(m1)->charge() + dimuon_cands.at(m2)->charge()) == 0 )
                return true;
        }   
    }
    return false;
}

}

namespace tau_analysis {
namespace selectors {

std::shared_ptr<TauJetSelector> TauJetSelector::Make(const std::string& name)
{
    if(name.empty() || name == "None")
        return std::make_shared<TauJetSelector>();
    if(name == "MuTau")
        return std::make_shared<MuTau>();
    if(name == "genTauTau")
        return std::make_shared<genTauTau>();
    throw analysis::exception("Unknown selector name = '%1%'") % name;
}



TauJetSelector::Result TauJetSelector::Select(const edm::Event& event, const std::deque<TauJet>& tauJets,
                                      const std::vector<pat::Electron>& electrons,
                                      const std::vector<pat::Muon>& muons, const pat::MET& met,
                                      const reco::Vertex& primaryVertex,
                                      const pat::TriggerObjectStandAloneCollection& triggerObjects,
                                      const edm::TriggerResults& triggerResults, float rho)
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
                                     const edm::TriggerResults& triggerResults, float rho)
{
    static const std::string filterName = "hltL3crIsoL1sSingleMu22L1f0L2f10QL3f24QL3trkIsoFiltered0p07";
    static const std::set<int> decayModes = { 0, 1, 10, 11 };


    const pat::Muon *ref_muon = nullptr;
    for(const pat::Muon& muon : muons) {
        if(!(muon.pt() > 25 && std::abs(muon.eta()) < 2.1 && muon.isMediumMuon() && PFRelIsolation(muon) < 0.15
                && std::abs(muon.muonBestTrack()->dz(primaryVertex.position())) < 0.2
                && std::abs(muon.muonBestTrack()->dxy(primaryVertex.position())) < 0.0045))
            continue;
        if(!ref_muon || PFRelIsolation(*ref_muon) > PFRelIsolation(muon) || (PFRelIsolation(*ref_muon) == PFRelIsolation(muon) && ref_muon->pt() < muon.pt())){
            ref_muon = &muon;
        }          
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
                && reco::deltaR(ref_muon->polarP4(), tau.polarP4()) > 0.5
                && std::abs(dynamic_cast<const pat::PackedCandidate*>(tau.leadChargedHadrCand().get())->dz()) < 0.2)) continue;
        if(!selectedTau || selectedTau->tau->tauID("byDeepTau2017v2p1VSjetraw")< tau.tauID("byDeepTau2017v2p1VSjetraw") 
            || (selectedTau->tau->tauID("byDeepTau2017v2p1VSjetraw")== tau.tauID("byDeepTau2017v2p1VSjetraw") && selectedTau->tau->pt() < tau.pt())){
                selectedTau = &tauJet;
             }
            
    }
    if(!(selectedTau && (selectedTau->tau->charge() + ref_muon->charge()) == 0)) return {};
    std::vector<const TauJet*> selectedTauJets = { selectedTau };

    auto tagObject = std::make_shared<TagObject>();
    tagObject->p4 = ref_muon->polarP4();
    tagObject->charge = ref_muon->charge();
    tagObject->id = unsigned(ref_muon->isLooseMuon()) * 1 + unsigned(ref_muon->isMediumMuon()) * 2
                    + unsigned(ref_muon->isTightMuon(primaryVertex)) * 4;
    tagObject->isolation = PFRelIsolation(*ref_muon);
    tagObject->has_extramuon = hasExtraMuon(muons, ref_muon, primaryVertex);
    tagObject->has_extraelectron = hasExtraElectron(electrons, rho);
    tagObject->has_dimuon = hasExtraDimuon(muons, primaryVertex);
    return Result(selectedTauJets, tagObject);
}

TauJetSelector::Result genTauTau::Select(const edm::Event& event, const std::deque<TauJet>& tauJets,
                                     const std::vector<pat::Electron>& electrons,
                                     const std::vector<pat::Muon>& muons, const pat::MET& met,
                                     const reco::Vertex& primaryVertex,
                                     const pat::TriggerObjectStandAloneCollection& triggerObjects,
                                     const edm::TriggerResults& triggerResults, float rho)
{
    std::vector<const TauJet*> selected;
    for(const TauJet& tauJet :tauJets) {
      const ObjPtr<reco_tau::gen_truth::GenLepton>& genLepton = tauJet.genLepton;
      if(!genLepton) continue;
      else {
        if(!(genLepton->kind() == reco_tau::gen_truth::GenLepton::Kind::TauDecayedToHadrons
             && genLepton->visibleP4().pt() > 10                           
             && std::abs(genLepton->visibleP4().eta()) < 2.5)
          ) continue;
        selected.push_back(&tauJet);
      }
    }
    if(!(selected.size()==2)) return {};
    return Result(selected, nullptr);
}


} // namespace selectors
} // namespace tau_analysis
