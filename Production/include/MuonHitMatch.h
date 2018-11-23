/*! Match hits in the muon system.
*/

#pragma once

#include "DataFormats/PatCandidates/interface/Muon.h"
#include "DataFormats/PatCandidates/interface/Tau.h"
#include "TauML/Analysis/include/TauTuple.h"

namespace tau_analysis {

namespace MuonSubdetId {
enum { DT = 1, CSC = 2, RPC = 3, GEM = 4, ME0 = 5 };
}

struct MuonHitMatch {
    static constexpr int n_muon_stations = 4;

    std::map<int, std::vector<UInt_t>> n_matches, n_hits;
    unsigned n_muons{0};
    const pat::Muon* best_matched_muon{nullptr};
    double deltaR2_best_match{-1};

    MuonHitMatch();
    void AddMatchedMuon(const pat::Muon& muon, const pat::Tau& tau);
    static std::vector<const pat::Muon*> FindMatchedMuons(const pat::Tau& tau, const pat::MuonCollection& muons,
                                                          double deltaR, double minPt);
    void FillTuple(tau_tuple::Tau& tau, const pat::Tau& reco_tau) const;

private:
    unsigned CountMuonStationsWithMatches(size_t first_station, size_t last_station) const;

    unsigned CountMuonStationsWithHits(size_t first_station, size_t last_station) const;
};

} // namespace tau_analysis
