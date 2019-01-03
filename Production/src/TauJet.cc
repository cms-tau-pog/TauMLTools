/* Tau jet candidate.
*/

#include "TauML/Production/include/TauJet.h"

namespace {

std::set<size_t> CreateIndexSet(size_t collection_size)
{
    std::vector<size_t> indices_vec(collection_size);
    std::iota(indices_vec.begin(), indices_vec.end(), 0);
    return std::set<size_t>(indices_vec.begin(), indices_vec.end());
}

void UpdateAvailable(std::set<size_t>& available, const std::set<size_t>& processed)
{
    for(size_t index : processed)
        available.erase(index);
}

} // anonymous namespace

namespace tau_analysis {

TauJet::TauJet(const pat::Jet* _jet, size_t _jetIndex) :
    jet(_jet), tau(nullptr), jetTauMatch(JetTauMatch::NoMatch), jetIndex(static_cast<int>(_jetIndex)), tauIndex(-1)
{
}

TauJet::TauJet(const pat::Tau* _tau, size_t _tauIndex) :
    jet(nullptr), tau(_tau), jetTauMatch(JetTauMatch::NoMatch), jetIndex(-1), tauIndex(static_cast<int>(_tauIndex))
{
}

TauJet::TauJet(const pat::Jet* _jet, const pat::Tau* _tau, JetTauMatch _jetTauMatch, size_t _jetIndex,
               size_t _tauIndex) :
    jet(_jet), tau(_tau), jetTauMatch(_jetTauMatch), jetIndex(static_cast<int>(_jetIndex)),
    tauIndex(static_cast<int>(_tauIndex))
{
}

TauJetBuilder::TauJetBuilder(const TauJetBuilderSetup& setup, const pat::JetCollection& jets,
                             const pat::TauCollection& taus, const pat::PackedCandidateCollection& cands,
                             const pat::ElectronCollection& electrons, const pat::MuonCollection& muons,
                             const reco::GenParticleCollection* genParticles) :
    setup_(setup), jets_(jets), taus_(taus), cands_(cands), electrons_(electrons), muons_(muons),
    genParticles_(genParticles)
{
}

std::vector<TauJet> TauJetBuilder::Build()
{
    std::vector<TauJet> tauJets;

    availableTaus_ = CreateIndexSet(taus_.size());
    processedJets_.clear();
    processedTaus_.clear();

    for(size_t n = 0; n < jets_.size(); ++n) {
        const pat::Jet& jet = jets_.at(n);
        if(jet.pt() > setup_.minJetPt && std::abs(jet.eta()) < setup_.maxJetEta)
            availableJets_.insert(n);
    }

    MatchJetsAndTaus(JetTauMatch::PF, tauJets);
    MatchJetsAndTaus(JetTauMatch::dR, tauJets);

    if(setup_.forceTauJetMatch && !availableTaus_.empty()) {
        for(size_t tau_index : availableTaus_) {
            const pat::Tau& tau = taus_.at(tau_index);
            std::cout << "Missing tau: (" << tau.pt() << ", " << tau.eta() << ", " << tau.phi() << ")" << std::endl;
        }
        for(const pat::Jet& jet : jets_) {
            std::cout << "Jet: (" << jet.pt() << ", " << jet.eta() << ", " << jet.phi() << ")" << std::endl;
        }

        throw cms::Exception("TauJetBuilder") << "Some taus are not found.";
    }

    for(size_t tauIndex : availableTaus_) {
        const pat::Tau& tau = taus_.at(tauIndex);
        tauJets.emplace_back(&tau, tauIndex);
        processedTaus_.insert(tauIndex);
    }
    availableTaus_.clear();

    for(size_t jetIndex : availableJets_) {
        const pat::Jet& jet = jets_.at(jetIndex);
        tauJets.emplace_back(&jet, jetIndex);
        processedJets_.insert(jetIndex);
    }
    availableJets_.clear();

    for(TauJet& tauJet : tauJets) {
        tauJet.cands = FindMatchedPFCandidates(tauJet.jet, tauJet.tau);
        tauJet.electrons = FindMatchedElectrons(tauJet.jet, tauJet.tau);
        tauJet.muons = FindMatchedMuons(tauJet.jet, tauJet.tau);
        if(genParticles_) {
            if(tauJet.jet) {
                tauJet.jetGenLeptonMatchResult = gen_truth::LeptonGenMatch(tauJet.jet->polarP4(), *genParticles_);
                tauJet.jetGenQcdMatchResult = gen_truth::QcdGenMatch(tauJet.jet->polarP4(), *genParticles_);
            }
            if(tauJet.tau) {
                tauJet.tauGenLeptonMatchResult = gen_truth::LeptonGenMatch(tauJet.tau->polarP4(), *genParticles_);
                tauJet.tauGenQcdMatchResult = gen_truth::QcdGenMatch(tauJet.tau->polarP4(), *genParticles_);
            }
        }
    }

    return tauJets;
}

bool TauJetBuilder::IsJetDaughter(const pat::Jet& jet, const pat::PackedCandidate& cand)
{
    const size_t nDaughters = jet.numberOfDaughters();
    for(size_t n = 0; n < nDaughters; ++n) {
        const auto& daughter = jet.daughterPtr(n);
        auto jetCand = dynamic_cast<const pat::PackedCandidate*>(daughter.get());
        if(jetCand == &cand)
            return true;
    }
    return false;
}

bool TauJetBuilder::IsTauSignalCand(const pat::Tau& tau, const pat::PackedCandidate& cand)
{
    for(const auto& signalCandBase : tau.signalCands()) {
        auto signalCand = dynamic_cast<const pat::PackedCandidate*>(signalCandBase.get());
        if(signalCand == &cand)
            return true;
    }
    return false;
}

bool TauJetBuilder::IsTauIsoCand(const pat::Tau& tau, const pat::PackedCandidate& cand)
{
    for(const auto& isoCandBase : tau.isolationCands()) {
        auto isoCand = dynamic_cast<const pat::PackedCandidate*>(isoCandBase.get());
        if(isoCand == &cand)
            return true;
    }
    return false;
}

bool TauJetBuilder::IsLeadChargedHadrCand(const pat::Tau& tau, const pat::PackedCandidate& cand)
{
    auto leadChargedHadrCand = dynamic_cast<const pat::PackedCandidate*>(tau.leadChargedHadrCand().get());
    return leadChargedHadrCand == &cand;
}

void TauJetBuilder::MatchJetsAndTaus(JetTauMatch matchStrategy, std::vector<TauJet>& tauJets)
{
    for(size_t tauIndex : availableTaus_) {
        const pat::Tau& tau = taus_.at(tauIndex);
        size_t jetIndex;
        if(FindJet(tau, matchStrategy, jetIndex)) {
            const pat::Jet& jet = jets_.at(jetIndex);
            tauJets.emplace_back(&jet, &tau, matchStrategy, jetIndex, tauIndex);
            processedJets_.insert(jetIndex);
            processedTaus_.insert(tauIndex);
        }
    }

    UpdateAvailable(availableJets_, processedJets_);
    UpdateAvailable(availableTaus_, processedTaus_);
}

bool TauJetBuilder::FindJet(const pat::Tau& tau, JetTauMatch matchStrategy, size_t& matchedJetIndex) const
{
    auto leadChargedHadrCand = dynamic_cast<const pat::PackedCandidate*>(tau.leadChargedHadrCand().get());
    double matchDR2 = setup_.tauJetMatchDeltaR2Threshold;
    for(size_t jetIndex : availableJets_) {
        if(processedJets_.count(jetIndex)) continue;
        const pat::Jet& jet = jets_.at(jetIndex);
        if(matchStrategy == JetTauMatch::PF) {
            if(IsJetDaughter(jet, *leadChargedHadrCand)) {
                matchedJetIndex = jetIndex;
                return true;
            }
        } else if(matchStrategy == JetTauMatch::dR) {
            const double dr2 = ROOT::Math::VectorUtil::DeltaR2(tau.p4(), jet.p4());
            if(dr2 < matchDR2) {
                matchedJetIndex = jetIndex;
                matchDR2 = dr2;
            }
        } else {
            throw cms::Exception("TauJetBuilder") << "JetTauMatch strategy not supported.";
        }
    }
    return matchStrategy == JetTauMatch::dR && matchDR2 < setup_.tauJetMatchDeltaR2Threshold;
}

bool TauJetBuilder::GetMatchReferences(const pat::Jet* jet, const pat::Tau* tau,
                                       PolarLorentzVector& ref_p4, double& deltaR2) const
{
    if(jet != nullptr && !setup_.useOnlyTauObjectMatch) {
        ref_p4 = jet->polarP4();
        deltaR2 = setup_.objectMatchDeltaR2ThresholdJet;
        return true;
    }
    if(tau != nullptr) {
        ref_p4 = tau->polarP4();
        deltaR2 = setup_.objectMatchDeltaR2ThresholdTau;
        return true;
    }
    return false;
}

std::vector<PFCandDesc> TauJetBuilder::FindMatchedPFCandidates(const pat::Jet* jet, const pat::Tau* tau) const
{
    PolarLorentzVector ref_p4;
    double deltaR2;
    const bool has_ref = GetMatchReferences(jet, tau, ref_p4, deltaR2);
    std::vector<PFCandDesc> matched_cands;
    if(has_ref) {
        for(const auto& cand : cands_) {
            if(ROOT::Math::VectorUtil::DeltaR2(ref_p4, cand.polarP4()) >= deltaR2) continue;
            PFCandDesc desc;
            desc.candidate = &cand;
            desc.jetDaughter = jet != nullptr && IsJetDaughter(*jet, cand);
            desc.tauSignal = tau != nullptr && IsTauSignalCand(*tau, cand);
            desc.leadChargedHadrCand = tau != nullptr && IsLeadChargedHadrCand(*tau, cand);
            desc.tauIso = tau != nullptr && IsTauIsoCand(*tau, cand);
            matched_cands.push_back(desc);
        }
    }
    return matched_cands;
}

std::vector<const pat::Electron*> TauJetBuilder::FindMatchedElectrons(const pat::Jet* jet, const pat::Tau* tau) const
{
    PolarLorentzVector ref_p4;
    double deltaR2;
    const bool has_ref = GetMatchReferences(jet, tau, ref_p4, deltaR2);
    std::vector<const pat::Electron*> matched_electrons;
    if(has_ref) {
        for(const auto& ele : electrons_) {
            if(ROOT::Math::VectorUtil::DeltaR2(ref_p4, ele.polarP4()) >= deltaR2) continue;
            matched_electrons.push_back(&ele);
        }
    }
    return matched_electrons;
}

std::vector<const pat::Muon*> TauJetBuilder::FindMatchedMuons(const pat::Jet* jet, const pat::Tau* tau) const
{
    PolarLorentzVector ref_p4;
    double deltaR2;
    const bool has_ref = GetMatchReferences(jet, tau, ref_p4, deltaR2);
    std::vector<const pat::Muon*> matched_muons;
    if(has_ref) {
        for(const auto& muon : muons_) {
            if(ROOT::Math::VectorUtil::DeltaR2(ref_p4, muon.polarP4()) >= deltaR2) continue;
            matched_muons.push_back(&muon);
        }
    }
    return matched_muons;
}

} // namespace tau_analysis
