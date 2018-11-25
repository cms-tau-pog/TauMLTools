/*! Match hits in the muon system.
*/

#include "TauML/Production/include/MuonHitMatch.h"
#include "TauML/Production/include/TauAnalysis.h"

namespace tau_analysis {

MuonHitMatch::MuonHitMatch()
{
    n_matches[MuonSubdetId::DT].assign(n_muon_stations, 0);
    n_matches[MuonSubdetId::CSC].assign(n_muon_stations, 0);
    n_matches[MuonSubdetId::RPC].assign(n_muon_stations, 0);
    n_hits[MuonSubdetId::DT].assign(n_muon_stations, 0);
    n_hits[MuonSubdetId::CSC].assign(n_muon_stations, 0);
    n_hits[MuonSubdetId::RPC].assign(n_muon_stations, 0);
}

void MuonHitMatch::AddMatchedMuon(const pat::Muon& muon, const pat::Tau& tau)
{
    static constexpr int n_stations = 4;

    ++n_muons;
    const double dR2 = reco::deltaR2(tau.p4(), muon.p4());
    if(!best_matched_muon || dR2 < deltaR2_best_match) {
        best_matched_muon = &muon;
        deltaR2_best_match = dR2;
    }

    for(const auto& segment : muon.matches()) {
        if(segment.segmentMatches.empty()) continue;
        if(n_matches.count(segment.detector()))
            ++n_matches.at(segment.detector()).at(segment.station() - 1);
    }

    if(muon.outerTrack().isNonnull()) {
        const auto& hit_pattern = muon.outerTrack()->hitPattern();
        for(int hit_index = 0; hit_index < hit_pattern.numberOfAllHits(reco::HitPattern::TRACK_HITS);
            ++hit_index) {
            auto hit_id = hit_pattern.getHitPattern(reco::HitPattern::TRACK_HITS, hit_index);
            if(hit_id == 0) break;
            if(hit_pattern.muonHitFilter(hit_id) && (hit_pattern.getHitType(hit_id) == TrackingRecHit::valid
                                                     || hit_pattern.getHitType(hit_id == TrackingRecHit::bad))) {
                const int station = hit_pattern.getMuonStation(hit_id) - 1;
                if(station >= 0 && station < n_stations) {
                    std::vector<UInt_t>* muon_n_hits = nullptr;
                    if(hit_pattern.muonDTHitFilter(hit_id))
                        muon_n_hits = &n_hits.at(MuonSubdetId::DT);
                    else if(hit_pattern.muonCSCHitFilter(hit_id))
                        muon_n_hits = &n_hits.at(MuonSubdetId::CSC);
                    else if(hit_pattern.muonRPCHitFilter(hit_id))
                        muon_n_hits = &n_hits.at(MuonSubdetId::RPC);

                    if(muon_n_hits)
                        ++muon_n_hits->at(station);
                }
            }
        }
    }
}

void MuonHitMatch::FillTuple(tau_tuple::Tau& tau, const pat::Tau& reco_tau) const
{
    static constexpr float default_value = tau_tuple::DefaultFillValue<float>();

    tau.n_matched_muons = n_muons;
    tau.muon_pt = best_matched_muon != nullptr ? best_matched_muon->p4().pt() : default_value;
    tau.muon_dEta = best_matched_muon != nullptr ? dEta(best_matched_muon->p4(), reco_tau.p4()) : default_value;
    tau.muon_dPhi = best_matched_muon != nullptr ? dPhi(best_matched_muon->p4(), reco_tau.p4()) : default_value;
    tau.muon_n_matches_DT_1 = n_matches.at(MuonSubdetId::DT).at(0);
    tau.muon_n_matches_DT_2 = n_matches.at(MuonSubdetId::DT).at(1);
    tau.muon_n_matches_DT_3 = n_matches.at(MuonSubdetId::DT).at(2);
    tau.muon_n_matches_DT_4 = n_matches.at(MuonSubdetId::DT).at(3);
    tau.muon_n_matches_CSC_1 = n_matches.at(MuonSubdetId::CSC).at(0);
    tau.muon_n_matches_CSC_2 = n_matches.at(MuonSubdetId::CSC).at(1);
    tau.muon_n_matches_CSC_3 = n_matches.at(MuonSubdetId::CSC).at(2);
    tau.muon_n_matches_CSC_4 = n_matches.at(MuonSubdetId::CSC).at(3);
    tau.muon_n_matches_RPC_1 = n_matches.at(MuonSubdetId::RPC).at(0);
    tau.muon_n_matches_RPC_2 = n_matches.at(MuonSubdetId::RPC).at(1);
    tau.muon_n_matches_RPC_3 = n_matches.at(MuonSubdetId::RPC).at(2);
    tau.muon_n_matches_RPC_4 = n_matches.at(MuonSubdetId::RPC).at(3);
    tau.muon_n_hits_DT_1 = n_hits.at(MuonSubdetId::DT).at(0);
    tau.muon_n_hits_DT_2 = n_hits.at(MuonSubdetId::DT).at(1);
    tau.muon_n_hits_DT_3 = n_hits.at(MuonSubdetId::DT).at(2);
    tau.muon_n_hits_DT_4 = n_hits.at(MuonSubdetId::DT).at(3);
    tau.muon_n_hits_CSC_1 = n_hits.at(MuonSubdetId::CSC).at(0);
    tau.muon_n_hits_CSC_2 = n_hits.at(MuonSubdetId::CSC).at(1);
    tau.muon_n_hits_CSC_3 = n_hits.at(MuonSubdetId::CSC).at(2);
    tau.muon_n_hits_CSC_4 = n_hits.at(MuonSubdetId::CSC).at(3);
    tau.muon_n_hits_RPC_1 = n_hits.at(MuonSubdetId::RPC).at(0);
    tau.muon_n_hits_RPC_2 = n_hits.at(MuonSubdetId::RPC).at(1);
    tau.muon_n_hits_RPC_3 = n_hits.at(MuonSubdetId::RPC).at(2);
    tau.muon_n_hits_RPC_4 = n_hits.at(MuonSubdetId::RPC).at(3);
}

unsigned MuonHitMatch::CountMuonStationsWithMatches(size_t first_station, size_t last_station) const
{
    static const std::map<int, std::vector<bool>> masks = {
        { MuonSubdetId::DT, { false, false, false, false } },
        { MuonSubdetId::CSC, { true, false, false, false } },
        { MuonSubdetId::RPC, { false, false, false, false } },
    };
    unsigned cnt = 0;
    for(size_t n = first_station; n <= last_station; ++n) {
        for(const auto& match : n_matches) {
            if(!masks.at(match.first).at(n) && match.second.at(n) > 0) ++cnt;
        }
    }
    return cnt;
}

unsigned MuonHitMatch::CountMuonStationsWithHits(size_t first_station, size_t last_station) const
{
    static const std::map<int, std::vector<bool>> masks = {
        { MuonSubdetId::DT, { false, false, false, false } },
        { MuonSubdetId::CSC, { false, false, false, false } },
        { MuonSubdetId::RPC, { false, false, false, false } },
    };

    unsigned cnt = 0;
    for(size_t n = first_station; n <= last_station; ++n) {
        for(const auto& hit : n_hits) {
            if(!masks.at(hit.first).at(n) && hit.second.at(n) > 0) ++cnt;
        }
    }
    return cnt;
}

} // namespace tau_analysis
