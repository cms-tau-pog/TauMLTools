/* Tau jet candidate.
*/

#include "TauMLTools/Production/interface/TauJet.h"
#include "TauMLTools/Core/interface/TextIO.h"

#include <random>
#include <Math/VectorUtil.h>

namespace {

std::set<size_t> CreateIndexSet(size_t collection_size)
{
    std::vector<size_t> indices_vec(collection_size);
    std::iota(indices_vec.begin(), indices_vec.end(), 0);
    return std::set<size_t>(indices_vec.begin(), indices_vec.end());
}

} // anonymous namespace

namespace tau_analysis {

bool TauJetBuilder::MatchResult::HasMatch() const
{
    return dR_genLepton < inf || dR_genJet < inf || dR_tau < inf || dR_boostedTau || dR_jet < inf;
}

void TauJetBuilder::MatchResult::SetDeltaR_genLepton(size_t index_in, double dR_in, double dR_thr)
{
    SetDeltaR(index_in, dR_in, dR_thr, dR_genLepton);
}

void TauJetBuilder::MatchResult::SetDeltaR_genJet(size_t index_in, double dR_in, double dR_thr)
{
    SetDeltaR(index_in, dR_in, dR_thr, dR_genJet);
}

void TauJetBuilder::MatchResult::SetDeltaR_tau(size_t index_in, double dR_in, double dR_thr)
{
    SetDeltaR(index_in, dR_in, dR_thr, dR_tau);
}

void TauJetBuilder::MatchResult::SetDeltaR_boostedTau(size_t index_in, double dR_in, double dR_thr)
{
    SetDeltaR(index_in, dR_in, dR_thr, dR_boostedTau);
}

void TauJetBuilder::MatchResult::SetDeltaR_jet(size_t index_in, double dR_in, double dR_thr)
{
    SetDeltaR(index_in, dR_in, dR_thr, dR_jet);
}

void TauJetBuilder::MatchResult::SetDeltaR(size_t index_in, double dR_in, double dR_thr, double& dR_out)
{
    if(dR_in < dR_thr && dR_in < dR_out) {
        dR_out = dR_in;
        index = static_cast<int>(index_in);
    }
}

bool TauJetBuilder::MatchResult::operator <(const TauJetBuilder::MatchResult& other) const
{
    if(dR_genLepton != other.dR_genLepton) return dR_genLepton < other.dR_genLepton;
    if(dR_genJet != other.dR_genJet) return dR_genJet < other.dR_genJet;
    if(dR_tau != other.dR_tau) return dR_tau < other.dR_tau;
    if(dR_boostedTau != other.dR_boostedTau) return dR_boostedTau < other.dR_boostedTau;
    if(dR_jet != other.dR_jet) return dR_jet < other.dR_jet;
    return index < other.index;
}

std::ostream& operator<<(std::ostream& os, const TauJetBuilder::MatchResult& match)
{
    const auto print_dR = [&](double dR, const std::string& name) {
        if(dR < TauJetBuilder::MatchResult::inf)
            os << ", " << name << " = " << dR;
    };
    if(match.index) {
        os << "index = " << *match.index;
        print_dR(match.dR_genLepton, "dR_genLepton");
        print_dR(match.dR_genJet, "dR_genJet");
        print_dR(match.dR_tau, "dR_tau");
        print_dR(match.dR_boostedTau, "dR_boostedTau");
        print_dR(match.dR_jet, "dR_jet");
    } else {
        os << "no_match";
    }
    return os;
}

TauJetBuilder::TauJetBuilder(const TauJetBuilderSetup& setup, const pat::TauCollection& taus,
              const pat::TauCollection& boostedTaus, const pat::JetCollection& jets,
              const pat::JetCollection& fatJets, const pat::PackedCandidateCollection& cands,
              const pat::ElectronCollection& electrons, const pat::PhotonCollection& photons, const pat::MuonCollection& muons,
              const pat::IsolatedTrackCollection& isoTracks, const pat::PackedCandidateCollection& lostTracks, 
	      const std::vector<reco::VertexCompositePtrCandidate>& secondVertices,
              const reco::GenParticleCollection* genParticles, const reco::GenJetCollection* genJets,
              bool requireGenMatch, bool requireGenORRecoTauMatch, bool applyRecoPtSieve) :
    setup_(setup), taus_(taus), boostedTaus_(boostedTaus), jets_(jets), fatJets_(fatJets), cands_(cands),
    electrons_(electrons), photons_(photons), muons_(muons), isoTracks_(isoTracks), lostTracks_(lostTracks), secondVertices_(secondVertices), 
    genParticles_(genParticles), genJets_(genJets), requireGenMatch_(requireGenMatch), 
    requireGenORRecoTauMatch_(requireGenORRecoTauMatch), applyRecoPtSieve_(applyRecoPtSieve)
{
    if(genParticles)
        genLeptons_ = reco_tau::gen_truth::GenLepton::fromGenParticleCollection(*genParticles);
    Build();
}

void TauJetBuilder::Build()
{
    for(size_t genLeptonIndex = 0; genLeptonIndex < genLeptons_.size(); ++genLeptonIndex) {
        TauJet tauJet;
        tauJet.genLepton.reset(genLeptons_.at(genLeptonIndex), genLeptonIndex);
        tauJets_.push_back(tauJet);
    }

    std::set<size_t> unmatched;
    if(genJets_) {
        unmatched = CreateIndexSet(genJets_->size());
        for(TauJet& tauJet : tauJets_) {
            MatchResult bestMatch;
            for(size_t genJetIndex = 0; genJetIndex < genJets_->size(); ++genJetIndex) {
                const auto& genJet = genJets_->at(genJetIndex);
                const auto& p4 = genJet.polarP4();
                MatchResult match;
                match.SetDeltaR_genLepton(genJetIndex, deltaR(p4, tauJet.genLepton->visibleP4()),
                                          setup_.genLepton_genJet_dR);
                if(match < bestMatch)
                    bestMatch = match;
            }
            if(bestMatch.index) {
                tauJet.genJet.reset(genJets_->at(*bestMatch.index), *bestMatch.index);
                unmatched.erase(*bestMatch.index);
            }
        }

        for(size_t idx : unmatched) {
            TauJet tauJet;
            tauJet.genJet.reset(genJets_->at(idx), idx);
            tauJets_.push_back(tauJet);
        }
    }

    unmatched = CreateIndexSet(taus_.size());
    for(TauJet& tauJet : tauJets_) {
        MatchResult bestMatch;
        for(size_t tauIndex = 0; tauIndex < taus_.size(); ++tauIndex) {
            const auto& tau = taus_.at(tauIndex);
            const auto& p4 = tau.polarP4();
            MatchResult match;
            if(tauJet.genLepton)
                match.SetDeltaR_genLepton(tauIndex, deltaR(p4, tauJet.genLepton->visibleP4()),
                                          setup_.genLepton_tau_dR);
            if(tauJet.genJet)
                match.SetDeltaR_genJet(tauIndex, deltaR(p4, tauJet.genJet->polarP4()), setup_.genJet_tau_dR);
            if(match < bestMatch)
                bestMatch = match;
        }
        if(bestMatch.index) {
            tauJet.tau.reset(taus_.at(*bestMatch.index), *bestMatch.index);
            unmatched.erase(*bestMatch.index);
        }
    }

    if(!requireGenMatch_) {
        for(size_t idx : unmatched) {
            TauJet tauJet;
            tauJet.tau.reset(taus_.at(idx), idx);
            tauJets_.push_back(tauJet);
        }
    }

    unmatched = CreateIndexSet(boostedTaus_.size());
    for(TauJet& tauJet : tauJets_) {
        MatchResult bestMatch;
        for(size_t boostedTauIndex = 0; boostedTauIndex < boostedTaus_.size(); ++boostedTauIndex) {
            const auto& boostedTau = boostedTaus_.at(boostedTauIndex);
            const auto& p4 = boostedTau.polarP4();

            MatchResult match;
            if(tauJet.genLepton)
                match.SetDeltaR_genLepton(boostedTauIndex, deltaR(p4, tauJet.genLepton->visibleP4()),
                                          setup_.genLepton_boostedTau_dR);
            if(tauJet.genJet)
                match.SetDeltaR_genJet(boostedTauIndex, deltaR(p4, tauJet.genJet->polarP4()),
                                       setup_.genJet_boostedTau_dR);
            if(tauJet.tau)
                match.SetDeltaR_tau(boostedTauIndex, deltaR(p4, tauJet.tau->polarP4()), setup_.tau_boostedTau_dR);
            if(match < bestMatch)
                bestMatch = match;
        }

        if(bestMatch.index) {
            tauJet.boostedTau.reset(boostedTaus_.at(*bestMatch.index), *bestMatch.index);
            unmatched.erase(*bestMatch.index);
        }
    }

    if(!requireGenMatch_) {
        for(size_t idx : unmatched) {
            TauJet tauJet;
            tauJet.boostedTau.reset(boostedTaus_.at(idx), idx);
            tauJets_.push_back(tauJet);
        }
    }

    unmatched = CreateIndexSet(jets_.size());
    for(TauJet& tauJet : tauJets_) {
        MatchResult bestMatch;
        for(size_t jetIndex = 0; jetIndex < jets_.size(); ++jetIndex) {
            const auto& jet = jets_.at(jetIndex);
            const auto& p4 = jet.polarP4();

            if(!(std::abs(p4.eta()) < setup_.jet_maxAbsEta)) continue;

            MatchResult match;
            if(tauJet.genLepton)
                match.SetDeltaR_genLepton(jetIndex, deltaR(p4, tauJet.genLepton->visibleP4()),
                                          setup_.genLepton_jet_dR);
            if(tauJet.genJet)
                match.SetDeltaR_genJet(jetIndex, deltaR(p4, tauJet.genJet->polarP4()), setup_.genJet_jet_dR);
            if(tauJet.tau) {
                match.SetDeltaR_tau(jetIndex, deltaR(p4, tauJet.tau->polarP4()), setup_.tau_jet_dR);
            }
            if(match < bestMatch)
                bestMatch = match;
        }

        if(bestMatch.index) {
            tauJet.jet.reset(jets_.at(*bestMatch.index), *bestMatch.index);
            unmatched.erase(*bestMatch.index);
        }
    }

    if(!requireGenMatch_) {
        for(size_t idx : unmatched) {
            TauJet tauJet;
            tauJet.jet.reset(jets_.at(idx), idx);
            tauJets_.push_back(tauJet);
        }
    }

    unmatched = CreateIndexSet(fatJets_.size());
    for(TauJet& tauJet : tauJets_) {
        MatchResult bestMatch;
        for(size_t fatJetIndex = 0; fatJetIndex < fatJets_.size(); ++fatJetIndex) {
            const auto& fatJet = fatJets_.at(fatJetIndex);
            const auto& p4 = fatJet.polarP4();

            if(!(std::abs(p4.eta()) < setup_.fatJet_maxAbsEta)) continue;

            MatchResult match;
            if(tauJet.genLepton)
                match.SetDeltaR_genLepton(fatJetIndex, deltaR(p4, tauJet.genLepton->visibleP4()),
                                          setup_.genLepton_fatJet_dR);
            if(tauJet.genJet)
                match.SetDeltaR_genJet(fatJetIndex, deltaR(p4, tauJet.genJet->polarP4()), setup_.genJet_fatJet_dR);
            if(tauJet.tau)
                match.SetDeltaR_tau(fatJetIndex, deltaR(p4, tauJet.tau->polarP4()), setup_.tau_fatJet_dR);
            if(tauJet.jet)
                match.SetDeltaR_jet(fatJetIndex, deltaR(p4, tauJet.jet->polarP4()), setup_.jet_fatJet_dR);
            if(match < bestMatch)
                bestMatch = match;
        }

        if(bestMatch.index) {
            tauJet.fatJet.reset(fatJets_.at(*bestMatch.index), *bestMatch.index);
            unmatched.erase(*bestMatch.index);
        }
    }

    if(!requireGenMatch_) {
        for(size_t idx : unmatched) {
            TauJet tauJet;
            tauJet.fatJet.reset(fatJets_.at(idx), idx);
            tauJets_.push_back(tauJet);
        }
    }

    if(requireGenORRecoTauMatch_ || applyRecoPtSieve_) {
        std::deque<TauJet> prunedTauJets;
        for(const TauJet& tauJet : tauJets_) {
            const bool is_true_tau = tauJet.genLepton
                    && tauJet.genLepton->kind() == reco_tau::gen_truth::GenLepton::Kind::TauDecayedToHadrons;
            if(requireGenORRecoTauMatch_ && !(tauJet.tau || tauJet.boostedTau || is_true_tau))
                continue;
            if(applyRecoPtSieve_ && !tauJet.genLepton) {
                static std::mt19937 gen(12345);
                static std::uniform_real_distribution<> dist(0., 1.);
                static constexpr double slop = -5e-2;
                static constexpr double ref_pt = 80; // GeV
                static const double exp_ref = std::exp(slop * ref_pt);

                double pt = -1;
                if(tauJet.tau)
                    pt = tauJet.tau->polarP4().pt();
                if(tauJet.boostedTau)
                    pt = std::max(pt, tauJet.boostedTau->polarP4().pt());
                if(pt < 0) continue;
                if(pt < ref_pt) {
                    double prob_thr = exp_ref / std::exp(slop * pt);
                    if(dist(gen) > prob_thr) continue;
                }
            }
            prunedTauJets.push_back(tauJet);
        }
        tauJets_ = prunedTauJets;
    }

   for(TauJet& tauJet : tauJets_) {

        auto hasSVPFMatch = [&](const reco::VertexCompositePtrCandidate& sv, const reco::Candidate* pfCandPtr) {
            for(size_t it = 0; it < sv.numberOfSourceCandidatePtrs(); ++it) {
                const edm::Ptr<reco::Candidate>& recoCandPtr = sv.sourceCandidatePtr(it);
                if(pfCandPtr == &(*recoCandPtr))
                    return true;
            }
            return false;
        };

        const auto hasMatch = [&](const PolarLorentzVector& p4) {
            if(tauJet.genLepton && deltaR(p4, tauJet.genLepton->visibleP4()) < setup_.genLepton_cone)
                return true;
            if(tauJet.genJet && deltaR(p4, tauJet.genJet->polarP4()) < setup_.genJet_cone)
                return true;
            if(tauJet.tau && deltaR(p4, tauJet.tau->polarP4()) < setup_.tau_cone)
                return true;
            if(tauJet.boostedTau && deltaR(p4, tauJet.boostedTau->polarP4()) < setup_.boostedTau_cone)
                return true;
            if(tauJet.jet && deltaR(p4, tauJet.jet->polarP4()) < setup_.jet_cone)
                return true;
            if(tauJet.fatJet && deltaR(p4, tauJet.fatJet->polarP4()) < setup_.fatJet_cone)
                return true;
            return false;
        };

        const auto fillMatched = [&](auto& out_col, const auto& in_col) {
            for(size_t index = 0; index < in_col.size(); ++index) {
                const auto& obj = in_col.at(index);
                if(hasMatch(obj.polarP4()))
                    out_col.emplace_back(obj, index);
            }
        };

        for(size_t pfCandIndex = 0; pfCandIndex < lostTracks_.size(); ++pfCandIndex) {
            const auto& pfCand = lostTracks_.at(pfCandIndex);
            if(!hasMatch(pfCand.polarP4())) continue;
            PFCandDesc pfCandDesc;
            pfCandDesc.candidate = &pfCand;
            pfCandDesc.index = static_cast<int>(pfCandIndex);
            tauJet.lostTracks.push_back(pfCandDesc);
        }

        std::map<int, PFCandDesc> selCands;
        std::map<int, const reco::VertexCompositePtrCandidate*> selSVs;
        size_t nSelCands = 0, nSelSVs = 0;
        while(true) {
            for(size_t pfCandIndex = 0; pfCandIndex < cands_.size(); ++pfCandIndex) {
                const auto& pfCand = cands_.at(pfCandIndex);
                if(selCands.count(pfCandIndex)) continue;
                bool belongsToSelectedSV = false;
                for(const auto& [svIndex, sv] : selSVs) {
                    if(hasSVPFMatch(*sv, &pfCand)) {
                        belongsToSelectedSV = true;
                        break;
                    }
                }
                if(!(belongsToSelectedSV || hasMatch(pfCand.polarP4()))) continue;
                PFCandDesc pfCandDesc;
                pfCandDesc.candidate = &pfCand;
                pfCandDesc.index = static_cast<int>(pfCandIndex);
                if(tauJet.tau) {
                    pfCandDesc.tauSignal = IsTauSignalCand(*tauJet.tau, pfCand);
                    pfCandDesc.tauLeadChargedHadrCand = IsLeadChargedHadrCand(*tauJet.tau, pfCand);
                    pfCandDesc.tauIso = IsTauIsoCand(*tauJet.tau, pfCand);
                }
                if(tauJet.boostedTau) {
                    pfCandDesc.boostedTauSignal = IsTauSignalCand(*tauJet.boostedTau, pfCand);
                    pfCandDesc.boostedTauLeadChargedHadrCand = IsLeadChargedHadrCand(*tauJet.boostedTau, pfCand);
                    pfCandDesc.boostedTauIso = IsTauIsoCand(*tauJet.boostedTau, pfCand);
                }
                if(tauJet.jet)
                    pfCandDesc.jetDaughter = IsJetDaughter(*tauJet.jet, pfCand);
                if(tauJet.fatJet)
                    pfCandDesc.subJetDaughter = GetMatchedSubJetIndex(*tauJet.fatJet, pfCand);
                selCands[pfCandDesc.index] = pfCandDesc;
            }

            for(size_t svIndex = 0; svIndex < secondVertices_.size(); ++svIndex) {
                const auto& sv = secondVertices_.at(svIndex);
                if(selSVs.count(svIndex)) continue;
                bool hasSelectedPF = false;
                for(const auto& [pfCandIndex, pfCandDesc] : selCands) {
                    if(hasSVPFMatch(sv, pfCandDesc.candidate)) {
                        hasSelectedPF = true;
                        break;
                    }
                }
                if(!(hasSelectedPF || hasMatch(sv.polarP4()))) continue;
                selSVs[svIndex] = &sv;
            }
            if(nSelCands == selCands.size() && nSelSVs == selSVs.size()) break;
            nSelCands = selCands.size();
            nSelSVs = selSVs.size();
        }

        for(const auto& [pfCandIndex, pfCandDesc] : selCands)
            tauJet.cands.push_back(pfCandDesc);
        for(const auto& [svIndex, sv] : selSVs)
            tauJet.secondVertices.emplace_back(*sv, svIndex);

        fillMatched(tauJet.electrons, electrons_);
        fillMatched(tauJet.photons, photons_);
        fillMatched(tauJet.muons, muons_);
        fillMatched(tauJet.isoTracks, isoTracks_);
    }

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

int TauJetBuilder::GetMatchedSubJetIndex(const pat::Jet& fatJet, const pat::PackedCandidate& cand)
{
    static const std::string subjetCollection = "SoftDropPuppi";
    if(fatJet.hasSubjets(subjetCollection)) {
        const auto& subJets = fatJet.subjets(subjetCollection);
        for(size_t n = 0; n < subJets.size(); ++n) {
            const auto& subJet = subJets.at(n);
            if(IsJetDaughter(subJet, cand))
                return static_cast<int>(n);
        }
    }
    return -1;
}

} // namespace tau_analysis
