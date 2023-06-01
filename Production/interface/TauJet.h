/* Tau jet candidate.
*/

#pragma once

#include <boost/optional.hpp>

#include "DataFormats/PatCandidates/interface/Electron.h"
#include "DataFormats/PatCandidates/interface/Photon.h"
#include "DataFormats/PatCandidates/interface/Muon.h"
#include "DataFormats/PatCandidates/interface/Tau.h"
#include "DataFormats/PatCandidates/interface/Jet.h"
#include "DataFormats/PatCandidates/interface/PackedCandidate.h"
#include "DataFormats/PatCandidates/interface/IsolatedTrack.h"

#include "TauMLTools/Analysis/interface/GenLepton.h"

namespace tau_analysis {

struct PFCandDesc {
    const pat::PackedCandidate* candidate;
    int index{-1}, subJetDaughter{-1};
    bool tauSignal{false}, tauLeadChargedHadrCand{false}, tauIso{false};
    bool boostedTauSignal{false}, boostedTauLeadChargedHadrCand{false}, boostedTauIso{false};
    bool jetDaughter{false}, fatJetDaughter{false};
};

template<typename T>
struct ObjPtr {
    T* obj{nullptr};
    int index{-1};

    ObjPtr() {}
    ObjPtr(T& _obj, size_t _index) : obj(&_obj), index(static_cast<int>(_index)) {}
    operator bool() const { return obj != nullptr; }
    T& operator*() const { return *obj; }
    T* operator->() const { return obj; }

    void reset(T& _obj, size_t _index)
    {
        obj = &_obj;
        index = static_cast<int>(_index);
    }
};

struct TauJet {
    using PFCandCollection = std::vector<PFCandDesc>;
    using ElectronCollection = std::vector<ObjPtr<const pat::Electron>>;
    using PhotonCollection = std::vector<ObjPtr<const pat::Photon>>;
    using MuonCollection = std::vector<ObjPtr<const pat::Muon>>;
    using IsoTrackCollection = std::vector<ObjPtr<const pat::IsolatedTrack>>;
    using LostTrackCollection = std::vector<ObjPtr<const pat::PackedCandidate>>;
    using SVCollection = std::vector<ObjPtr<const reco::VertexCompositePtrCandidate>>;

    ObjPtr<reco_tau::gen_truth::GenLepton> genLepton;
    ObjPtr<const reco::GenJet> genJet;
    ObjPtr<const pat::Tau> tau;
    ObjPtr<const pat::Tau> boostedTau;
    ObjPtr<const pat::Jet> jet;
    ObjPtr<const pat::Jet> fatJet;

    PFCandCollection cands;
    ElectronCollection electrons;
    PhotonCollection photons;
    MuonCollection muons;
    IsoTrackCollection isoTracks;
    PFCandCollection lostTracks;
    SVCollection secondVertices;
};

struct TauJetBuilderSetup {
    double genLepton_genJet_dR{0.4}, genLepton_tau_dR{0.2}, genLepton_boostedTau_dR{0.2}, genLepton_jet_dR{0.4},
           genLepton_fatJet_dR{0.8};
    double genJet_tau_dR{0.4}, genJet_boostedTau_dR{0.4}, genJet_jet_dR{0.4}, genJet_fatJet_dR{0.8};
    double tau_boostedTau_dR{0.2}, tau_jet_dR{0.4}, tau_fatJet_dR{0.8};
    double jet_fatJet_dR{0.8};

    double jet_maxAbsEta{3.4}, fatJet_maxAbsEta{3.8};

    double genLepton_cone{0.5}, genJet_cone{0.5}, tau_cone{0.5}, boostedTau_cone{0.5}, jet_cone{0.8}, fatJet_cone{0.8};
};

class TauJetBuilder {
public:
    using IndexSet = std::set<size_t>;

    struct MatchResult {
        static constexpr double inf = std::numeric_limits<double>::infinity();

        boost::optional<size_t> index;
        double dR_genLepton{inf};
        double dR_genJet{inf};
        double dR_tau{inf};
        double dR_boostedTau{inf};
        double dR_jet{inf};

        bool HasMatch() const;

        void SetDeltaR_genLepton(size_t index_in, double dR_in, double dR_thr);
        void SetDeltaR_genJet(size_t index_in, double dR_in, double dR_thr);
        void SetDeltaR_tau(size_t index_in, double dR_in, double dR_thr);
        void SetDeltaR_boostedTau(size_t index_in, double dR_in, double dR_thr);
        void SetDeltaR_jet(size_t index_in, double dR_in, double dR_thr);
        void SetDeltaR(size_t index_in, double dR_in, double dR_thr, double& dR_out);

        bool operator <(const MatchResult& other) const;
    };

    using PolarLorentzVector = reco::LeafCandidate::PolarLorentzVector;

    TauJetBuilder(const TauJetBuilderSetup& setup, const pat::TauCollection& taus,
                  const pat::TauCollection& boostedTaus, const pat::JetCollection& jets,
                  const pat::JetCollection& fatJets, const pat::PackedCandidateCollection& cands,
                  const pat::ElectronCollection& electrons, const pat::PhotonCollection& photons, const pat::MuonCollection& muons,
                  const pat::IsolatedTrackCollection& isoTracks, const pat::PackedCandidateCollection& lostTracks, 
	          const std::vector<reco::VertexCompositePtrCandidate>& secondVertices,
                  const reco::GenParticleCollection* genParticles, const reco::GenJetCollection* genJets,
                  bool requireGenMatch, bool requireGenORRecoTauMatch, bool applyRecoPtSieve);

    TauJetBuilder(const TauJetBuilder&) = delete;
    TauJetBuilder& operator=(const TauJetBuilder&) = delete;

    const std::deque<TauJet>& GetTauJets() const { return tauJets_; }
    const std::vector<reco_tau::gen_truth::GenLepton>& GetGenLeptons() const { return genLeptons_; }

private:
    static bool IsTauSignalCand(const pat::Tau& tau, const pat::PackedCandidate& cand);
    static bool IsTauIsoCand(const pat::Tau& tau, const pat::PackedCandidate& cand);
    static bool IsLeadChargedHadrCand(const pat::Tau& tau, const pat::PackedCandidate& cand);
    static bool IsJetDaughter(const pat::Jet& jet, const pat::PackedCandidate& cand);
    static int GetMatchedSubJetIndex(const pat::Jet& jet, const pat::PackedCandidate& cand);

    void Build();

private:
    const TauJetBuilderSetup& setup_;
    const pat::TauCollection& taus_;
    const pat::TauCollection& boostedTaus_;
    const pat::JetCollection& jets_;
    const pat::JetCollection& fatJets_;
    const pat::PackedCandidateCollection& cands_;
    const pat::ElectronCollection& electrons_;
    const pat::PhotonCollection& photons_;
    const pat::MuonCollection& muons_;
    const pat::IsolatedTrackCollection& isoTracks_;
    const pat::PackedCandidateCollection& lostTracks_;
    const std::vector<reco::VertexCompositePtrCandidate>& secondVertices_;
    const reco::GenParticleCollection* genParticles_;
    const reco::GenJetCollection* genJets_;
    const bool requireGenMatch_, requireGenORRecoTauMatch_, applyRecoPtSieve_;

    std::deque<TauJet> tauJets_;
    std::vector<reco_tau::gen_truth::GenLepton> genLeptons_;
};

std::ostream& operator<<(std::ostream& os, const TauJetBuilder::MatchResult& match);

} // namespace tau_analysis
