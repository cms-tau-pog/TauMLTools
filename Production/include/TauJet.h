/* Tau jet candidate.
*/

#pragma once

#include "DataFormats/PatCandidates/interface/Electron.h"
#include "DataFormats/PatCandidates/interface/Muon.h"
#include "DataFormats/PatCandidates/interface/Tau.h"
#include "DataFormats/PatCandidates/interface/Jet.h"
#include "DataFormats/PatCandidates/interface/PackedCandidate.h"

#include "GenTruthTools.h"

namespace tau_analysis {

enum class JetTauMatch { NoMatch = 0, PF = 1, dR = 2 };

struct PFCandDesc {
    const pat::PackedCandidate* candidate;
    bool jetDaughter{false}, tauSignal{false}, leadChargedHadrCand{false}, tauIso{false};
};

struct TauJet {
    const pat::Jet* jet{nullptr};
    const pat::Tau* tau{nullptr};
    JetTauMatch jetTauMatch{JetTauMatch::NoMatch};
    int jetIndex{-1}, tauIndex{-1};

    std::vector<PFCandDesc> cands;
    std::vector<const pat::Electron*> electrons;
    std::vector<const pat::Muon*> muons;
    gen_truth::LeptonMatchResult jetGenLeptonMatchResult, tauGenLeptonMatchResult;
    gen_truth::QcdMatchResult jetGenQcdMatchResult, tauGenQcdMatchResult;

    TauJet(const pat::Jet* _jet, size_t _jetIndex);
    TauJet(const pat::Tau* _tau, size_t _tauIndex);
    TauJet(const pat::Jet* _jet, const pat::Tau* _tau, JetTauMatch _jetTauMatch, size_t _jetIndex, size_t _tauIndex);
};

struct TauJetBuilderSetup {
    double minJetPt{10.};
    double maxJetEta{3.};
    bool forceTauJetMatch{false}, useOnlyTauObjectMatch{false};
    double tauJetMatchDeltaR2Threshold{std::pow(0.2, 2)};
    double objectMatchDeltaR2ThresholdJet{std::pow(0.7, 2)};
    double objectMatchDeltaR2ThresholdTau{std::pow(0.5, 2)};
};

class TauJetBuilder {
public:
    using IndexSet = std::set<size_t>;
    using PolarLorentzVector = reco::LeafCandidate::PolarLorentzVector;

    TauJetBuilder(const TauJetBuilderSetup& setup, const pat::JetCollection& jets, const pat::TauCollection& taus,
                  const pat::PackedCandidateCollection& cands, const pat::ElectronCollection& electrons,
                  const pat::MuonCollection& muons, const reco::GenParticleCollection* genParticles);

    TauJetBuilder(const TauJetBuilder&) = delete;
    TauJetBuilder& operator=(const TauJetBuilder&) = delete;

    std::vector<TauJet> Build();

private:
    static bool IsJetDaughter(const pat::Jet& jet, const pat::PackedCandidate& cand);
    static bool IsTauSignalCand(const pat::Tau& tau, const pat::PackedCandidate& cand);
    static bool IsTauIsoCand(const pat::Tau& tau, const pat::PackedCandidate& cand);
    static bool IsLeadChargedHadrCand(const pat::Tau& tau, const pat::PackedCandidate& cand);

    void MatchJetsAndTaus(JetTauMatch matchStrategy, std::vector<TauJet>& tauJets);
    bool FindJet(const pat::Tau& tau, JetTauMatch matchStrategy, size_t& matchedJetIndex) const;
    bool GetMatchReferences(const pat::Jet* jet, const pat::Tau* tau,
                            PolarLorentzVector& ref_p4, double& deltaR2) const;
    std::vector<PFCandDesc> FindMatchedPFCandidates(const pat::Jet* jet, const pat::Tau* tau) const;
    std::vector<const pat::Electron*> FindMatchedElectrons(const pat::Jet* jet, const pat::Tau* tau) const;
    std::vector<const pat::Muon*> FindMatchedMuons(const pat::Jet* jet, const pat::Tau* tau) const;

private:
    const TauJetBuilderSetup setup_;
    const pat::JetCollection& jets_;
    const pat::TauCollection& taus_;
    const pat::PackedCandidateCollection& cands_;
    const pat::ElectronCollection& electrons_;
    const pat::MuonCollection& muons_;
    const reco::GenParticleCollection* genParticles_;

    IndexSet availableJets_, processedJets_;
    IndexSet availableTaus_, processedTaus_;
};

} // namespace tau_analysis
