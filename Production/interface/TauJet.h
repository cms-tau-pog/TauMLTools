/* Tau jet candidate.
*/

#pragma once

#include <random>
#include <boost/optional.hpp>

#include "DataFormats/PatCandidates/interface/Electron.h"
#include "DataFormats/PatCandidates/interface/Muon.h"
#include "DataFormats/PatCandidates/interface/Tau.h"
#include "DataFormats/PatCandidates/interface/Jet.h"
#include "DataFormats/PatCandidates/interface/PackedCandidate.h"
#include "DataFormats/PatCandidates/interface/IsolatedTrack.h"

#include "TauMLTools/Analysis/interface/GenLepton.h"

#include "Geometry/CaloGeometry/interface/CaloCellGeometry.h"
#include "Geometry/CaloGeometry/interface/CaloGeometry.h"
#include "Geometry/CaloTopology/interface/HcalTopology.h"
#include "Geometry/Records/interface/CaloGeometryRecord.h"
#include "DataFormats/CaloRecHit/interface/CaloRecHit.h"
#include "DataFormats/EcalRecHit/interface/EcalRecHit.h"
#include "DataFormats/EcalRecHit/interface/EcalRecHitCollections.h"
#include "DataFormats/EcalDetId/interface/EcalDetIdCollections.h"
#include "DataFormats/HcalDetId/interface/HcalDetId.h"
#include "DataFormats/HcalRecHit/interface/HBHERecHit.h"
#include "DataFormats/HcalRecHit/interface/HcalRecHitDefs.h"
#include "DataFormats/HcalRecHit/interface/HFRecHit.h"
#include "DataFormats/HcalRecHit/interface/HORecHit.h"
#include "DataFormats/HLTReco/interface/TriggerTypeDefs.h"
#include "DataFormats/HLTReco/interface/TriggerFilterObjectWithRefs.h"
#include "TrackingTools/TrajectoryParametrization/interface/CurvilinearTrajectoryError.h"
#include "RecoPixelVertexing/PixelTrackFitting/interface/FitUtils.h"
#include "TrackingTools/TrajectoryParametrization/interface/GlobalTrajectoryParameters.h"
#include "DataFormats/TrackReco/interface/HitPattern.h"
#include "TrackingTools/AnalyticalJacobians/interface/JacobianLocalToCurvilinear.h"
#include "DataFormats/TrajectoryState/interface/LocalTrajectoryParameters.h"
#include "DataFormats/GeometrySurface/interface/Plane.h"
#include "DataFormats/BeamSpot/interface/BeamSpot.h"
#include "CUDADataFormats/Track/interface/PixelTrackHeterogeneous.h"
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"
#include "CUDADataFormats/SiPixelCluster/interface/gpuClusteringConstants.h"
#include "CUDADataFormats/Track/interface/TrackSoAHeterogeneousT.h"
#include "CUDADataFormats/Vertex/interface/ZVertexSoA.h"
#include "CUDADataFormats/Vertex/interface/ZVertexHeterogeneous.h"

namespace tau_analysis {

template<typename _CandType>
struct PFCandDesc {
    using CandType = _CandType;

    const CandType* candidate;
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

enum class CaloHitType {
    Undefined = -1, HBHE = 0, HO = 1, EcalBarrel = 2, EcalEndcap = 3
};

struct CaloHit {
    using PolarLorentzVector = reco::LeafCandidate::PolarLorentzVector;

    CaloHitType hitType{CaloHitType::Undefined};
    GlobalPoint position;
    double energy{-1}, chi2{-1};
    std::tuple<const HBHERecHit*, const HORecHit*, const EcalRecHit*> hitPtr;

    bool isValid() const { return hitType != CaloHitType::Undefined; }
    // for compatibility with matching function
    PolarLorentzVector polarP4() const
    {
        return PolarLorentzVector(position.perp(), position.eta(), position.phi(), 0);
    }

    static std::vector<CaloHit> MakeHitCollection(const CaloGeometry& geom, const HBHERecHitCollection* hbheRecHits,
                                                  const HORecHitCollection* hoRecHits,
                                                  const EcalRecHitCollection* ebRecHits,
                                                  const EcalRecHitCollection* eeRecHits);
};

struct PataTrack {
    using PolarLorentzVector = reco::LeafCandidate::PolarLorentzVector;
    int index{-1};
    PolarLorentzVector p4;
    const pixelTrack::TrackSoA* tsoa{nullptr};
    // for compatibility with matching function
    const PolarLorentzVector& polarP4() const { return p4; }

    static std::vector<PataTrack> MakeTrackCollection(const pixelTrack::TrackSoA& tracks);
};

template<typename _PFCand, typename _Tau, typename _BoostedTau, typename _Jet, typename _FatJet, typename _Electron,
         typename _Muon, typename _IsoTrack, typename _LostTrack, typename _L1Tau>
struct TauJetT {
    using PFCand = _PFCand;
    using Tau = _Tau;
    using BoostedTau = _BoostedTau;
    using Jet = _Jet;
    using FatJet = _FatJet;
    using Electron = _Electron;
    using Muon = _Muon;
    using IsoTrack = _IsoTrack;
    using LostTrack = _LostTrack;
    using L1Tau = _L1Tau;
    using PFCandCollection = std::vector<PFCandDesc<PFCand>>;
    using ElectronCollection = std::vector<ObjPtr<const Electron>>;
    using MuonCollection = std::vector<ObjPtr<const Muon>>;
    using IsoTrackCollection = std::vector<ObjPtr<const IsoTrack>>;
    using LostTrackCollection = std::vector<PFCandDesc<LostTrack>>;
    using CaloHitCollection = std::vector<ObjPtr<const CaloHit>>;
    using PataTrackCollection = std::vector<ObjPtr<const PataTrack>>;

    ObjPtr<reco_tau::gen_truth::GenLepton> genLepton;
    ObjPtr<const reco::GenJet> genJet;
    ObjPtr<const Tau> tau;
    ObjPtr<const BoostedTau> boostedTau;
    ObjPtr<const Jet> jet;
    ObjPtr<const FatJet> fatJet;
    ObjPtr<const L1Tau> l1Tau;

    PFCandCollection cands;
    ElectronCollection electrons;
    MuonCollection muons;
    IsoTrackCollection isoTracks;
    LostTrackCollection lostTracks;
    CaloHitCollection caloHits;
    PataTrackCollection pataTracks;
};

struct TauJetBuilderSetup {
    double genLepton_genJet_dR{0.4}, genLepton_tau_dR{0.2}, genLepton_boostedTau_dR{0.2}, genLepton_jet_dR{0.4},
           genLepton_fatJet_dR{0.8}, genLepton_l1Tau_dR{0.4};
    double genJet_tau_dR{0.4}, genJet_boostedTau_dR{0.4}, genJet_jet_dR{0.4}, genJet_fatJet_dR{0.8},
           genJet_l1Tau_dR{0.4};
    double tau_boostedTau_dR{0.2}, tau_jet_dR{0.4}, tau_fatJet_dR{0.8}, tau_l1Tau_dR{0.4};
    double boostedTau_jet_dR{0.4}, boostedTau_fatJet_dR{0.8}, boostedTau_l1Tau_dR{0.4};
    double jet_fatJet_dR{0.8}, jet_l1Tau_dR{0.4};
    double fatJet_l1Tau_dR{0.8};

    double jet_maxAbsEta{3.4}, fatJet_maxAbsEta{3.8};

    double genLepton_cone{0.5}, genJet_cone{0.5}, tau_cone{0.5}, boostedTau_cone{0.5}, jet_cone{0.8}, fatJet_cone{0.8},
           l1Tau_cone{0.5};


    static TauJetBuilderSetup fromPSet(const edm::ParameterSet& builderParams);
};

struct TauJetMatchResult {
    static constexpr double inf = std::numeric_limits<double>::infinity();

    boost::optional<size_t> index;
    double dR_genLepton{inf};
    double dR_genJet{inf};
    double dR_tau{inf};
    double dR_boostedTau{inf};
    double dR_jet{inf};
    double dR_fatJet{inf};
    double dR_l1Tau{inf};

    bool HasMatch() const;

    void SetDeltaR_genLepton(size_t index_in, double dR_in, double dR_thr);
    void SetDeltaR_genJet(size_t index_in, double dR_in, double dR_thr);
    void SetDeltaR_tau(size_t index_in, double dR_in, double dR_thr);
    void SetDeltaR_boostedTau(size_t index_in, double dR_in, double dR_thr);
    void SetDeltaR_jet(size_t index_in, double dR_in, double dR_thr);
    void SetDeltaR_fatJet(size_t index_in, double dR_in, double dR_thr);
    void SetDeltaR_l1Tau(size_t index_in, double dR_in, double dR_thr);
    void SetDeltaR(size_t index_in, double dR_in, double dR_thr, double& dR_out);

    bool operator <(const TauJetMatchResult& other) const;
};

template<typename _TauJet>
class TauJetBuilder {
public:
    using IndexSet = std::set<size_t>;
    using PolarLorentzVector = reco::LeafCandidate::PolarLorentzVector;
    using TauJet = _TauJet;
    using PFCand = typename TauJet::PFCand;
    using Tau = typename TauJet::Tau;
    using BoostedTau = typename TauJet::BoostedTau;
    using Jet = typename TauJet::Jet;
    using FatJet = typename TauJet::FatJet;
    using Electron = typename TauJet::Electron;
    using Muon = typename TauJet::Muon;
    using IsoTrack = typename TauJet::IsoTrack;
    using LostTrack = typename TauJet::LostTrack;
    using L1Tau = typename TauJet::L1Tau;
    using TauCollection = std::vector<Tau>;
    using BoostedTauCollection = std::vector<BoostedTau>;
    using JetCollection = std::vector<Jet>;
    using FatJetCollection = std::vector<FatJet>;
    using PFCandCollection = std::vector<PFCand>;
    using ElectronCollection = std::vector<Electron>;
    using MuonCollection = std::vector<Muon>;
    using IsoTrackCollection = std::vector<IsoTrack>;
    using LostTrackCollection = std::vector<LostTrack>;
    using L1TauCollection = std::vector<L1Tau>;
    using CaloHitCollection = std::vector<CaloHit>;
    using PataTrackCollection = std::vector<PataTrack>;

    TauJetBuilder(const TauJetBuilderSetup& setup, const TauCollection* taus, const BoostedTauCollection* boostedTaus,
                  const JetCollection* jets, const FatJetCollection* fatJets, const PFCandCollection* cands,
                  const ElectronCollection* electrons, const MuonCollection* muons,
                  const IsoTrackCollection* isoTracks, const LostTrackCollection* lostTracks,
                  const L1TauCollection* l1Taus, const CaloHitCollection* caloHits,
                  const PataTrackCollection* pataTracks,
                  const reco::GenParticleCollection* genParticles, const reco::GenJetCollection* genJets,
                  bool requireGenMatch, bool requireGenORRecoTauMatch, bool applyRecoPtSieve) :
        setup_(setup), taus_(taus), boostedTaus_(boostedTaus), jets_(jets), fatJets_(fatJets), cands_(cands),
        electrons_(electrons), muons_(muons), isoTracks_(isoTracks), lostTracks_(lostTracks), l1Taus_(l1Taus),
        caloHits_(caloHits), pataTracks_(pataTracks), genParticles_(genParticles), genJets_(genJets),
        requireGenMatch_(requireGenMatch), requireGenORRecoTauMatch_(requireGenORRecoTauMatch),
        applyRecoPtSieve_(applyRecoPtSieve)
    {
        if(genParticles)
            genLeptons_ = reco_tau::gen_truth::GenLepton::fromGenParticleCollection(*genParticles);
        Build();
    }

    TauJetBuilder(const TauJetBuilder&) = delete;
    TauJetBuilder& operator=(const TauJetBuilder&) = delete;

    const std::deque<TauJet>& GetTauJets() const { return tauJets_; }
    const std::vector<reco_tau::gen_truth::GenLepton>& GetGenLeptons() const { return genLeptons_; }

private:
    template<typename TauType>
    static bool IsTauSignalCand(const TauType& tau, const PFCand& cand)
    {
        for(const auto& signalCandBase : tau.signalCands()) {
            auto signalCand = dynamic_cast<const PFCand*>(signalCandBase.get());
            if(signalCand == &cand)
                return true;
        }
        return false;
    }

    template<typename TauType>
    static bool IsTauIsoCand(const TauType& tau, const PFCand& cand)
    {
        for(const auto& isoCandBase : tau.isolationCands()) {
            auto isoCand = dynamic_cast<const PFCand*>(isoCandBase.get());
            if(isoCand == &cand)
                return true;
        }
        return false;
    }

    template<typename TauType>
    static bool IsLeadChargedHadrCand(const TauType& tau, const PFCand& cand)
    {
        auto leadChargedHadrCand = dynamic_cast<const PFCand*>(tau.leadChargedHadrCand().get());
        return leadChargedHadrCand == &cand;
    }

    template<typename JetType>
    static bool IsJetDaughter(const JetType& jet, const PFCand& cand)
    {
        const size_t nDaughters = jet.numberOfDaughters();
        for(size_t n = 0; n < nDaughters; ++n) {
            const auto& daughter = jet.daughterPtr(n);
            auto jetCand = dynamic_cast<const PFCand*>(daughter.get());
            if(jetCand == &cand)
                return true;
        }
        return false;
    }

    static int GetMatchedSubJetIndex(const FatJet& fatJet, const PFCand& cand)
    {
        static const std::string subjetCollection = "SoftDropPuppi";
        if(fatJet.hasSubjets(subjetCollection)) {
            const auto& subJets = fatJet.subjets(subjetCollection);
            for(size_t n = 0; n < subJets.size(); ++n) {
                const auto& subJet = subJets.at(n);
                if(IsJetDaughter(*subJet, cand))
                    return static_cast<int>(n);
            }
        }
        return -1;
    }

    static std::set<size_t> CreateIndexSet(size_t collection_size)
    {
        std::vector<size_t> indices_vec(collection_size);
        std::iota(indices_vec.begin(), indices_vec.end(), 0);
        return std::set<size_t>(indices_vec.begin(), indices_vec.end());
    }

    template<typename Obj, typename Fn>
    void BuildStep(const std::vector<Obj>* objs, Fn getTauJetObj, bool isGenObj,
                   double genLepton_dR = -1, double genJet_dR = -1, double tau_dR = -1, double boostedTau_dR = -1,
                   double jet_dR = -1, double fatJet_dR = -1, double l1Tau_dR = -1)
    {
        if(!objs) return;
        std::set<size_t> unmatched = CreateIndexSet(objs->size());
        for(TauJet& tauJet : tauJets_) {
            TauJetMatchResult bestMatch;
            for(size_t objIndex = 0; objIndex < objs->size(); ++objIndex) {
                const auto& obj = objs->at(objIndex);
                const auto& p4 = obj.polarP4();
                TauJetMatchResult match;
                if(genLepton_dR >= 0 && tauJet.genLepton)
                    match.SetDeltaR_genLepton(objIndex, deltaR(p4, tauJet.genLepton->visibleP4()), genLepton_dR);
                if(genJet_dR >= 0 && tauJet.genJet)
                    match.SetDeltaR_genJet(objIndex, deltaR(p4, tauJet.genJet->polarP4()), genJet_dR);
                if(tau_dR >= 0 && tauJet.tau)
                    match.SetDeltaR_tau(objIndex, deltaR(p4, tauJet.tau->polarP4()), tau_dR);
                if(boostedTau_dR >= 0 && tauJet.boostedTau)
                    match.SetDeltaR_boostedTau(objIndex, deltaR(p4, tauJet.boostedTau->polarP4()), boostedTau_dR);
                if(jet_dR >= 0 && tauJet.jet)
                    match.SetDeltaR_jet(objIndex, deltaR(p4, tauJet.jet->polarP4()), jet_dR);
                if(fatJet_dR >= 0 && tauJet.fatJet)
                    match.SetDeltaR_fatJet(objIndex, deltaR(p4, tauJet.fatJet->polarP4()), fatJet_dR);
                if(l1Tau_dR >= 0 && tauJet.l1Tau)
                    match.SetDeltaR_l1Tau(objIndex, deltaR(p4, tauJet.l1Tau->polarP4()), l1Tau_dR);

                if(match < bestMatch)
                    bestMatch = match;
            }
            if(bestMatch.index) {
                getTauJetObj(tauJet).reset(objs->at(*bestMatch.index), *bestMatch.index);
                unmatched.erase(*bestMatch.index);
            }
        }

        if(isGenObj || !requireGenMatch_) {
            for(size_t idx : unmatched) {
                TauJet tauJet;
                getTauJetObj(tauJet).reset(objs->at(idx), idx);
                tauJets_.push_back(tauJet);
            }
        }
    }

    void Build()
    {
        for(size_t genLeptonIndex = 0; genLeptonIndex < genLeptons_.size(); ++genLeptonIndex) {
            TauJet tauJet;
            tauJet.genLepton.reset(genLeptons_.at(genLeptonIndex), genLeptonIndex);
            tauJets_.push_back(tauJet);
        }

        BuildStep(genJets_, [](TauJet& tauJet) -> ObjPtr<const reco::GenJet>& { return tauJet.genJet; }, true,
                  setup_.genLepton_genJet_dR);
        BuildStep(taus_, [](TauJet& tauJet) -> ObjPtr<const Tau>& { return tauJet.tau; }, false,
                  setup_.genLepton_tau_dR, setup_.genJet_tau_dR);
        BuildStep(boostedTaus_, [](TauJet& tauJet) -> ObjPtr<const BoostedTau>& { return tauJet.boostedTau; }, false,
                  setup_.genLepton_boostedTau_dR, setup_.genJet_boostedTau_dR, setup_.tau_boostedTau_dR);
        BuildStep(jets_, [](TauJet& tauJet) -> ObjPtr<const Jet>& { return tauJet.jet; }, false,
                  setup_.genLepton_jet_dR, setup_.genJet_jet_dR, setup_.tau_jet_dR, setup_.boostedTau_jet_dR);
        BuildStep(fatJets_, [](TauJet& tauJet) -> ObjPtr<const FatJet>& { return tauJet.fatJet; }, false,
                  setup_.genLepton_fatJet_dR, setup_.genJet_fatJet_dR, setup_.tau_fatJet_dR,
                  setup_.boostedTau_fatJet_dR, setup_.jet_fatJet_dR);
        BuildStep(l1Taus_, [](TauJet& tauJet) -> ObjPtr<const L1Tau>& { return tauJet.l1Tau; }, false,
                  setup_.genLepton_l1Tau_dR, setup_.genJet_l1Tau_dR, setup_.tau_l1Tau_dR, setup_.boostedTau_l1Tau_dR,
                  setup_.jet_l1Tau_dR, setup_.fatJet_l1Tau_dR);

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
                if(tauJet.l1Tau && deltaR(p4, tauJet.l1Tau->polarP4()) < setup_.l1Tau_cone)
                    return true;
                return false;
            };

            const auto fillMatched = [&](auto& out_col, auto in_col) {
                if(!in_col) return;
                for(size_t index = 0; index < in_col->size(); ++index) {
                    const auto& obj = in_col->at(index);
                    if(hasMatch(obj.polarP4()))
                        out_col.emplace_back(obj, index);
                }
            };
            if(cands_) {
                for(size_t pfCandIndex = 0; pfCandIndex < cands_->size(); ++pfCandIndex) {
                    const auto& pfCand = cands_->at(pfCandIndex);
                    if(!hasMatch(pfCand.polarP4())) continue;
                    PFCandDesc<PFCand> pfCandDesc;
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
                    tauJet.cands.push_back(pfCandDesc);
                }
            }

            if(lostTracks_) {
                for(size_t pfCandIndex = 0; pfCandIndex < lostTracks_->size(); ++pfCandIndex) {
                    const auto& pfCand = lostTracks_->at(pfCandIndex);
                    if(!hasMatch(pfCand.polarP4())) continue;
                    PFCandDesc<LostTrack> pfCandDesc;
                    pfCandDesc.candidate = &pfCand;
                    pfCandDesc.index = static_cast<int>(pfCandIndex);
                    tauJet.lostTracks.push_back(pfCandDesc);
                }
            }

            fillMatched(tauJet.electrons, electrons_);
            fillMatched(tauJet.muons, muons_);
            fillMatched(tauJet.isoTracks, isoTracks_);
            fillMatched(tauJet.caloHits, caloHits_);
            fillMatched(tauJet.pataTracks, pataTracks_);
        }
    }

private:
    const TauJetBuilderSetup setup_;
    const TauCollection* taus_;
    const BoostedTauCollection* boostedTaus_;
    const JetCollection* jets_;
    const FatJetCollection* fatJets_;
    const PFCandCollection* cands_;
    const ElectronCollection* electrons_;
    const MuonCollection* muons_;
    const IsoTrackCollection* isoTracks_;
    const LostTrackCollection* lostTracks_;
    const L1TauCollection* l1Taus_;
    const CaloHitCollection* caloHits_;
    const PataTrackCollection* pataTracks_;
    const reco::GenParticleCollection* genParticles_;
    const reco::GenJetCollection* genJets_;
    const bool requireGenMatch_, requireGenORRecoTauMatch_, applyRecoPtSieve_;

    std::deque<TauJet> tauJets_;
    std::vector<reco_tau::gen_truth::GenLepton> genLeptons_;
};

std::ostream& operator<<(std::ostream& os, const TauJetMatchResult& match);

} // namespace tau_analysis
