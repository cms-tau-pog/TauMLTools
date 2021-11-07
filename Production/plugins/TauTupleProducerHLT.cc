/*! Creates tuple for tau analysis.
*/

#include "Compression.h"

#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ServiceRegistry/interface/Service.h"

#include "DataFormats/JetMatching/interface/JetFlavourInfoMatching.h"
#include "DataFormats/JetReco/interface/Jet.h"
#include "DataFormats/L1Trigger/interface/Tau.h"
#include "DataFormats/ParticleFlowCandidate/interface/PFCandidate.h"
#include "DataFormats/TauReco/interface/PFTau.h"
#include "DataFormats/TauReco/interface/PFTauTransverseImpactParameterAssociation.h"
#include "DataFormats/TrackReco/interface/HitPattern.h"

#include "SimDataFormats/GeneratorProducts/interface/GenEventInfoProduct.h"

#include "AnalysisDataFormats/TopObjects/interface/TtGenEvent.h"
#include "CommonTools/UtilAlgos/interface/TFileService.h"
#include "RecoTauTag/RecoTau/interface/PFRecoTauClusterVariables.h"




#include "TauMLTools/Core/interface/Tools.h"
#include "TauMLTools/Core/interface/TextIO.h"
#include "TauMLTools/Analysis/interface/TauTupleHLT.h"
#include "TauMLTools/Analysis/interface/SummaryTuple.h"
#include "TauMLTools/Analysis/interface/TauIdResults.h"
#include "TauMLTools/Production/interface/GenTruthTools.h"
#include "TauMLTools/Production/interface/TauAnalysis.h"
#include "TauMLTools/Production/interface/MuonHitMatch.h"
#include "TauMLTools/Production/interface/TauJet.h"

namespace tau_hlt {

struct TauTupleProducerData {
    using clock = std::chrono::system_clock;

    const clock::time_point start;
    tau_hlt::TauTuple tauTuple;
    tau_tuple::SummaryTuple summaryTuple;
    std::mutex mutex;

private:
    size_t n_producers;

    TauTupleProducerData(TFile& file) :
        start(clock::now()),
        tauTuple("taus", &file, false),
        summaryTuple("summary", &file, false),
        n_producers(0)
    {
        summaryTuple().numberOfProcessedEvents = 0;
    }

    ~TauTupleProducerData() {}

public:

    static TauTupleProducerData* RequestGlobalData()
    {
        TauTupleProducerData* data = GetGlobalData();
        if(data == nullptr)
            throw cms::Exception("TauTupleProducerData") << "Request after all data copies were released.";
        {
            std::lock_guard<std::mutex> lock(data->mutex);
            ++data->n_producers;
            std::cout << "New request of TauTupleProducerData. Total number of producers = " << data->n_producers
                      << "." << std::endl;
        }
        return data;
    }

    static void ReleaseGlobalData()
    {
        TauTupleProducerData*& data = GetGlobalData();
        if(data == nullptr)
            throw cms::Exception("TauTupleProducerData") << "Another release after all data copies were released.";
        {
            std::lock_guard<std::mutex> lock(data->mutex);
            if(!data->n_producers)
                throw cms::Exception("TauTupleProducerData") << "Release before any request.";
            --data->n_producers;
            std::cout << "TauTupleProducerData has been released. Total number of producers = " << data->n_producers
                      << "." << std::endl;
            if(!data->n_producers) {
                data->tauTuple.Write();
                const auto stop = clock::now();
                data->summaryTuple().exeTime = static_cast<unsigned>(
                            std::chrono::duration_cast<std::chrono::seconds>(stop - data->start).count());
                data->summaryTuple.Fill();
                data->summaryTuple.Write();
                delete data;
                data = nullptr;
                std::cout << "TauTupleProducerData has been destroyed." << std::endl;
            }
        }

    }

private:
    static TauTupleProducerData*& GetGlobalData()
    {
        static TauTupleProducerData* data = InitializeGlobalData();
        return data;
    }

    static TauTupleProducerData* InitializeGlobalData()
    {
        TFile& file = edm::Service<TFileService>()->file();
        file.SetCompressionAlgorithm(ROOT::kLZMA);
        file.SetCompressionLevel(9);
        TauTupleProducerData* data = new TauTupleProducerData(file);
        std::cout << "TauTupleProducerData has been created." << std::endl;
        return data;
    }
};

class TauTupleProducer : public edm::EDAnalyzer {
public:
    // using pat::* as placeholders
    using TauJet = tau_analysis::TauJetT<reco::PFCandidate, reco::PFTau, pat::Tau, reco::PFJet, pat::Jet,
                                         pat::Electron, pat::Muon, pat::IsolatedTrack, pat::PackedCandidate, l1t::Tau>;
    using TauIPCollection = edm::AssociationVector<reco::PFTauRefProd,
                                                   std::vector<reco::PFTauTransverseImpactParameterRef>>;

    TauTupleProducer(const edm::ParameterSet& cfg) :
        isMC_(cfg.getParameter<bool>("isMC")),
        requireGenMatch_(cfg.getParameter<bool>("requireGenMatch")),
        requireGenORRecoTauMatch_(cfg.getParameter<bool>("requireGenORRecoTauMatch")),
        applyRecoPtSieve_(cfg.getParameter<bool>("applyRecoPtSieve")),
        builderSetup_(tau_analysis::TauJetBuilderSetup::fromPSet(cfg.getParameterSet("tauJetBuilderSetup"))),
        genEventToken_(mayConsume<GenEventInfoProduct>(cfg.getParameter<edm::InputTag>("genEvent"))),
        genParticlesToken_(mayConsume<reco::GenParticleCollection>(cfg.getParameter<edm::InputTag>("genParticles"))),
        genJetsToken_(mayConsume<reco::GenJetCollection>(cfg.getParameter<edm::InputTag>("genJets"))),
        genJetFlavourInfosToken_(consumes<reco::JetFlavourInfoMatchingCollection>(
                                 cfg.getParameter<edm::InputTag>("genJetFlavourInfos"))),
        puInfoToken_(mayConsume<std::vector<PileupSummaryInfo>>(cfg.getParameter<edm::InputTag>("puInfo"))),
        beamSpotToken_(consumes<reco::BeamSpot>(cfg.getParameter<edm::InputTag>("beamSpot"))),
        rhoToken_(consumes<double>(cfg.getParameter<edm::InputTag>("rho"))),
        hbheRecHitsToken_(consumes<HBHERecHitCollection>(cfg.getParameter<edm::InputTag>("hbheRecHits"))),
        hoRecHitsToken_(consumes<HORecHitCollection>(cfg.getParameter<edm::InputTag>("hoRecHits"))),
        ebRecHitsToken_(consumes<EcalRecHitCollection>(cfg.getParameter<edm::InputTag>("ebRecHits"))),
        eeRecHitsToken_(consumes<EcalRecHitCollection>(cfg.getParameter<edm::InputTag>("eeRecHits"))),
        pataTracksToken_(consumes<PixelTrackHeterogeneous>(cfg.getParameter<edm::InputTag>("pataTracks"))),
        pataVerticesToken_(consumes<ZVertexHeterogeneous>(cfg.getParameter<edm::InputTag>("pataVertices"))),
        candsToken_(consumes<reco::PFCandidateCollection>(cfg.getParameter<edm::InputTag>("pfCandidates"))),
        l1TausToken_(consumes<l1t::TauBxCollection>(cfg.getParameter<edm::InputTag>("l1Taus"))),
        jetsToken_(consumes<reco::PFJetCollection>(cfg.getParameter<edm::InputTag>("jets"))),
        tausToken_(consumes<reco::PFTauCollection>(cfg.getParameter<edm::InputTag>("taus"))),
        tauIPToken_(consumes<TauIPCollection>(cfg.getParameter<edm::InputTag>("tauIP"))),
        geometryToken_(esConsumes<CaloGeometry, CaloGeometryRecord>()),
        bFieldToken_(esConsumes<MagneticField, IdealMagneticFieldRecord>()),
        data(TauTupleProducerData::RequestGlobalData()),
        tauTuple(data->tauTuple),
        summaryTuple(data->summaryTuple)
    {
    }

private:
    static constexpr float default_value = tau_hlt::DefaultFillValue<float>();
    static constexpr int default_int_value = tau_hlt::DefaultFillValue<int>();

    virtual void analyze(const edm::Event& event, const edm::EventSetup& eventSetup) override
    {
        std::lock_guard<std::mutex> lock(data->mutex);
        summaryTuple().numberOfProcessedEvents++;

        tauTuple().run  = event.id().run();
        tauTuple().lumi = event.id().luminosityBlock();
        tauTuple().evt  = event.id().event();
        tauTuple().sampleType = isMC_ ? static_cast<int>(tau_analysis::SampleType::MC)
                                      : static_cast<int>(tau_analysis::SampleType::Data);
        tauTuple().tauType = -1;
        tauTuple().dataset_id = -1;
        tauTuple().dataset_group_id = -1;

        const auto& vertices_SoA = *event.get(pataVerticesToken_);
        tauTuple().npv = static_cast<int>(vertices_SoA.nvFinal);
        const double rho = event.get(rhoToken_);
        tauTuple().rho = static_cast<float>(rho);

        const auto& beamSpot = event.get(beamSpotToken_);
        tauTuple().beamSpot_x = beamSpot.x0();
        tauTuple().beamSpot_y = beamSpot.y0();
        tauTuple().beamSpot_z = beamSpot.z0();

        if(isMC_) {
            const auto& genEvent = event.get(genEventToken_);
            tauTuple().genEventWeight = static_cast<float>(genEvent.weight());

            const auto& puInfo = event.get(puInfoToken_);
            tauTuple().npu = tau_analysis::gen_truth::GetNumberOfPileUpInteractions(puInfo);
        }

        const auto& taus = event.get(tausToken_);
        const auto& pfTauTransverseImpactParameters = event.get(tauIPToken_);
        const auto& jets = event.get(jetsToken_);
        const auto& cands = event.get(candsToken_);
        const auto& hL1Taus = event.get(l1TausToken_);

        std::vector<l1t::Tau> l1Taus;
        for(auto iter = hL1Taus.begin(0); iter != hL1Taus.end(0); ++iter)
            l1Taus.push_back(*iter);

        const auto& hbheRecHits = event.get(hbheRecHitsToken_);
        const auto& hoRecHits = event.get(hoRecHitsToken_);
        const auto& ebRecHits = event.get(ebRecHitsToken_);
        const auto& eeRecHits = event.get(eeRecHitsToken_);
        const auto& patatracks_SoA = *event.get(pataTracksToken_);


        const auto& geometry = eventSetup.getHandle(geometryToken_);

        const auto caloHits = tau_analysis::CaloHit::MakeHitCollection(*geometry, &hbheRecHits, &hoRecHits, &ebRecHits,
                                                                       &eeRecHits);
        const auto pataTracks = tau_analysis::PataTrack::MakeTrackCollection(patatracks_SoA);

        edm::Handle<reco::GenParticleCollection> hGenParticles;
        edm::Handle<reco::GenJetCollection> hGenJets;
        edm::Handle<reco::JetFlavourInfoMatchingCollection> hGenJetFlavourInfos;
        if(isMC_) {
            event.getByToken(genParticlesToken_, hGenParticles);
            event.getByToken(genJetsToken_, hGenJets);
            event.getByToken(genJetFlavourInfosToken_, hGenJetFlavourInfos);
        }

        auto genParticles = hGenParticles.isValid() ? hGenParticles.product() : nullptr;
        auto genJets = hGenJets.isValid() ? hGenJets.product() : nullptr;
        auto genJetFlavourInfos = hGenJetFlavourInfos.isValid() ? hGenJetFlavourInfos.product() : nullptr;

        tau_analysis::TauJetBuilder<TauJet> builder(builderSetup_, &taus, nullptr, &jets, nullptr, &cands, nullptr,
                                                    nullptr, nullptr, nullptr, &l1Taus, &caloHits, &pataTracks,
                                                    genParticles, genJets,
                                                    requireGenMatch_, requireGenORRecoTauMatch_, applyRecoPtSieve_);
        const auto& tauJets = builder.GetTauJets();
        tauTuple().total_entries = static_cast<int>(tauJets.size());
        for(size_t tauJetIndex = 0; tauJetIndex < tauJets.size(); ++tauJetIndex) {
            const TauJet& tauJet = tauJets.at(tauJetIndex);
            tauTuple().entry_index = static_cast<int>(tauJetIndex);

            FillGenLepton(tauJet.genLepton);
            FillGenJet(tauJet.genJet, genJetFlavourInfos);
            FillTau(tauJet.tau, pfTauTransverseImpactParameters, "tau_");
            FillJet(tauJet.jet, "jet_");
            FillL1Tau(tauJet.l1Tau);
            FillPFCandidates(tauJet.cands, "pfCand_");
            FillCaloHits(tauJet.caloHits);
            FillPixelTracks(tauJet.pataTracks, patatracks_SoA, vertices_SoA);

            tauTuple.Fill();
        }
    }

    virtual void endJob() override
    {
        TauTupleProducerData::ReleaseGlobalData();
    }

private:
    void FillGenLepton(const tau_analysis::ObjPtr<reco_tau::gen_truth::GenLepton>& genLepton)
    {
        tauTuple().genLepton_index = genLepton.index;
        tauTuple().genLepton_kind = genLepton ? static_cast<int>(genLepton->kind()) : default_int_value;
        tauTuple().genLepton_charge = genLepton ? genLepton->charge() : default_int_value;
        tauTuple().genLepton_vis_pt = genLepton ? static_cast<float>(genLepton->visibleP4().pt()) : default_value;
        tauTuple().genLepton_vis_eta = genLepton ? static_cast<float>(genLepton->visibleP4().eta()) : default_value;
        tauTuple().genLepton_vis_phi = genLepton ? static_cast<float>(genLepton->visibleP4().phi()) : default_value;
        tauTuple().genLepton_vis_mass = genLepton ? static_cast<float>(genLepton->visibleP4().mass()) : default_value;
        tauTuple().genLepton_lastMotherIndex = default_int_value;

        if(genLepton) {
            const auto ref_ptr = genLepton->allParticles().data();

            auto getIndex = [&](const reco_tau::gen_truth::GenParticle* particle) {
                int pos = -1;
                if(particle) {
                    pos = static_cast<int>(particle - ref_ptr);
                    if(pos < 0 || pos >= static_cast<int>(genLepton->allParticles().size()))
                        throw cms::Exception("TauTupleProducer") << "Unable to determine a gen particle index.";
                }
                return pos;
            };

            auto encodeMotherIndex = [&](const std::set<const reco_tau::gen_truth::GenParticle*>& mothers) -> Long64_t {
                static constexpr Long64_t shift_scale =
                        static_cast<Long64_t>(reco_tau::gen_truth::GenLepton::MaxNumberOfParticles);

                if(mothers.empty()) return -1;
                if(mothers.size() > 6)
                    throw cms::Exception("TauTupleProducer") << "Gen particle with > 6 mothers.";
                if(mothers.size() > 1 && genLepton->allParticles().size() > static_cast<size_t>(shift_scale))
                    throw cms::Exception("TauTupleProducer") << "Too many gen particles per gen lepton.";
                Long64_t pos = 0;
                Long64_t shift = 1;
                std::set<int> mother_indices;
                for(auto mother : mothers)
                    mother_indices.insert(getIndex(mother));
                for(int mother_idx : mother_indices) {
                    pos = pos + shift * mother_idx;
                    shift *= shift_scale;
                }
                return pos;
            };

            tauTuple().genLepton_lastMotherIndex = static_cast<int>(genLepton->mothers().size()) - 1;
            for(const auto& p : genLepton->allParticles()) {
                tauTuple().genParticle_pdgId.push_back(p.pdgId);
                tauTuple().genParticle_mother.push_back(encodeMotherIndex(p.mothers));
                tauTuple().genParticle_charge.push_back(p.charge);
                tauTuple().genParticle_isFirstCopy.push_back(p.isFirstCopy);
                tauTuple().genParticle_isLastCopy.push_back(p.isLastCopy);
                tauTuple().genParticle_pt.push_back(p.p4.pt());
                tauTuple().genParticle_eta.push_back(p.p4.eta());
                tauTuple().genParticle_phi.push_back(p.p4.phi());
                tauTuple().genParticle_mass.push_back(p.p4.mass());
                tauTuple().genParticle_vtx_x.push_back(p.vertex.x());
                tauTuple().genParticle_vtx_y.push_back(p.vertex.y());
                tauTuple().genParticle_vtx_z.push_back(p.vertex.z());
            }
        }
    }

    void FillGenJet(const tau_analysis::ObjPtr<const reco::GenJet>& genJet,
                    const reco::JetFlavourInfoMatchingCollection* genJetFlavourInfos)
    {
        tauTuple().genJet_index = genJet.index;
        tauTuple().genJet_pt = genJet ? static_cast<float>(genJet->polarP4().pt()) : default_value;
        tauTuple().genJet_eta = genJet ? static_cast<float>(genJet->polarP4().eta()) : default_value;
        tauTuple().genJet_phi = genJet ? static_cast<float>(genJet->polarP4().phi()) : default_value;
        tauTuple().genJet_mass = genJet ? static_cast<float>(genJet->polarP4().mass()) : default_value;
        tauTuple().genJet_emEnergy = genJet ? genJet->emEnergy() : default_value;
        tauTuple().genJet_hadEnergy = genJet ? genJet->hadEnergy() : default_value;
        tauTuple().genJet_invisibleEnergy = genJet ? genJet->invisibleEnergy() : default_value;
        tauTuple().genJet_auxiliaryEnergy = genJet ? genJet->auxiliaryEnergy() : default_value;
        tauTuple().genJet_chargedHadronEnergy = genJet ? genJet->chargedHadronEnergy() : default_value;
        tauTuple().genJet_neutralHadronEnergy = genJet ? genJet->neutralHadronEnergy() : default_value;
        tauTuple().genJet_chargedEmEnergy = genJet ? genJet->chargedEmEnergy() : default_value;
        tauTuple().genJet_neutralEmEnergy = genJet ? genJet->neutralEmEnergy() : default_value;
        tauTuple().genJet_muonEnergy = genJet ? genJet->muonEnergy() : default_value;
        tauTuple().genJet_chargedHadronMultiplicity = genJet ? genJet->chargedHadronMultiplicity() : default_int_value;
        tauTuple().genJet_neutralHadronMultiplicity = genJet ? genJet->neutralHadronMultiplicity() : default_int_value;
        tauTuple().genJet_chargedEmMultiplicity = genJet ? genJet->chargedEmMultiplicity() : default_int_value;
        tauTuple().genJet_neutralEmMultiplicity = genJet ? genJet->neutralEmMultiplicity() : default_int_value;
        tauTuple().genJet_muonMultiplicity = genJet ? genJet->muonMultiplicity() : default_int_value;

        auto genJetFI = genJet && genJetFlavourInfos ? &(*genJetFlavourInfos)[genJet.index].second : nullptr;
        tauTuple().genJet_n_bHadrons = genJetFI ? static_cast<int>(genJetFI->getbHadrons().size()) : default_int_value;
        tauTuple().genJet_n_cHadrons = genJetFI ? static_cast<int>(genJetFI->getcHadrons().size()) : default_int_value;
        tauTuple().genJet_n_partons = genJetFI ? static_cast<int>(genJetFI->getPartons().size()) : default_int_value;
        tauTuple().genJet_n_leptons = genJetFI ? static_cast<int>(genJetFI->getLeptons().size()) : default_int_value;
        tauTuple().genJet_hadronFlavour = genJetFI ? genJetFI->getHadronFlavour() : default_int_value;
        tauTuple().genJet_partonFlavour = genJetFI ? genJetFI->getPartonFlavour() : default_int_value;
    }

    void FillTau(const tau_analysis::ObjPtr<const reco::PFTau>& tau, const TauIPCollection& tauIPs,
                 const std::string& prefix)
    {
        tauTuple.get<int>(prefix + "index") = tau.index;
        tauTuple.get<float>(prefix + "pt") = tau ? static_cast<float>(tau->polarP4().pt()) : default_value;
        tauTuple.get<float>(prefix + "eta") = tau ? static_cast<float>(tau->polarP4().eta()) : default_value;
        tauTuple.get<float>(prefix + "phi") = tau ? static_cast<float>(tau->polarP4().phi()) : default_value;
        tauTuple.get<float>(prefix + "mass") = tau ? static_cast<float>(tau->polarP4().mass()) : default_value;
        tauTuple.get<int>(prefix + "charge") = tau ? tau->charge() : default_int_value;

        tauTuple.get<int>(prefix + "decayMode") = tau ? tau->decayMode() : default_int_value;

        const reco::PFTauTransverseImpactParameter* ip = tau && tauIPs.value(tau.index).isNonnull()
                                                         ? &*tauIPs.value(tau.index) : nullptr;
        tauTuple.get<float>(prefix + "dxy") = ip ? ip->dxy() : default_value;
        tauTuple.get<float>(prefix + "dxy_error") = ip ? ip->dxy_error() : default_value;
        tauTuple.get<float>(prefix + "ip3d") = ip ? ip->ip3d() : default_value;
        tauTuple.get<float>(prefix + "ip3d_error") = ip ? ip->ip3d_error() : default_value;
        const bool has_sv = ip && ip->hasSecondaryVertex();
        tauTuple.get<int>(prefix + "hasSecondaryVertex") = ip ? ip->hasSecondaryVertex() : default_int_value;
        tauTuple.get<float>(prefix + "sv_x") = has_sv ? ip->secondaryVertexPos().x() : default_value;
        tauTuple.get<float>(prefix + "sv_y") = has_sv ? ip->secondaryVertexPos().y() : default_value;
        tauTuple.get<float>(prefix + "sv_z") = has_sv ? ip->secondaryVertexPos().z() : default_value;
        tauTuple.get<float>(prefix + "flightLength_x") = ip ? ip->flightLength().x() : default_value;
        tauTuple.get<float>(prefix + "flightLength_y") = ip ? ip->flightLength().y() : default_value;
        tauTuple.get<float>(prefix + "flightLength_z") = ip ? ip->flightLength().z() : default_value;
        tauTuple.get<float>(prefix + "flightLength_sig") = ip ? ip->flightLengthSig() : default_value;

        auto leadChargedHadrCand = tau && tau->leadChargedHadrCand().isNonnull()
                                   ? dynamic_cast<const reco::PFCandidate*>(tau->leadChargedHadrCand().get())
                                   : nullptr;

        auto bestTrack = leadChargedHadrCand ? leadChargedHadrCand->bestTrack() : nullptr;
        tauTuple.get<float>(prefix + "dz") = bestTrack ? bestTrack->dz() : default_value;
        tauTuple.get<float>(prefix + "dz_error") = bestTrack ? bestTrack->dzError() : default_value;
        tauTuple.get<float>(prefix + "pt_weighted_deta_strip") = tau
                ? reco::tau::pt_weighted_deta_strip(*tau, tau->decayMode()) : default_value;
        tauTuple.get<float>(prefix + "pt_weighted_dphi_strip") = tau
                ? reco::tau::pt_weighted_dphi_strip(*tau, tau->decayMode()) : default_value;
        tauTuple.get<float>(prefix + "pt_weighted_dr_signal") = tau
                ? reco::tau::pt_weighted_dr_signal(*tau, tau->decayMode()) : default_value;
        tauTuple.get<float>(prefix + "pt_weighted_dr_iso") = tau
                ? reco::tau::pt_weighted_dr_iso(*tau, tau->decayMode()) : default_value;
        tauTuple.get<float>(prefix + "leadingTrackNormChi2") = tau ? reco::tau::lead_track_chi2(*tau) : default_value;
        tauTuple.get<float>(prefix + "e_ratio") = tau ? reco::tau::eratio(*tau) : default_value;
        tauTuple.get<float>(prefix + "gj_angle_diff") = has_sv
                ? CalculateGottfriedJacksonAngleDifference(*tau, *ip) : default_value;
        tauTuple.get<int>(prefix + "n_photons") = tau
                ? static_cast<int>(reco::tau::n_photons_total(*tau)) : default_int_value;
        tauTuple.get<float>(prefix + "emFraction") = tau ? tau->emFraction() : default_value;
        tauTuple.get<float>(prefix + "leadChargedCand_etaAtEcalEntrance") = leadChargedHadrCand
                ? leadChargedHadrCand->positionAtECALEntrance().eta() : default_value;
    }

    void FillJet(const tau_analysis::ObjPtr<const reco::PFJet>& jet, const std::string& prefix)
    {
        tauTuple.get<int>(prefix + "index") = jet.index;
        tauTuple.get<float>(prefix + "pt") = jet ? static_cast<float>(jet->polarP4().pt()) : default_value;
        tauTuple.get<float>(prefix + "eta") = jet ? static_cast<float>(jet->polarP4().eta()) : default_value;
        tauTuple.get<float>(prefix + "phi") = jet ? static_cast<float>(jet->p4().phi()) : default_value;
        tauTuple.get<float>(prefix + "mass") = jet ? static_cast<float>(jet->p4().mass()) : default_value;
        tauTuple.get<int>(prefix + "nConstituents") = jet ? jet->nConstituents() : default_int_value;
        tauTuple.get<int>(prefix + "chargedMultiplicity") = jet
                ? jet->chargedMultiplicity() : default_int_value;
        tauTuple.get<int>(prefix + "neutralMultiplicity") = jet
                ? jet->neutralMultiplicity() : default_int_value;
    }

    void FillL1Tau(const tau_analysis::ObjPtr<const l1t::Tau>& l1Tau)
    {
        tauTuple().l1Tau_index = l1Tau.index;
        tauTuple().l1Tau_pt = l1Tau ? static_cast<float>(l1Tau->polarP4().pt()) : default_value;
        tauTuple().l1Tau_eta = l1Tau ? static_cast<float>(l1Tau->polarP4().eta()) : default_value;
        tauTuple().l1Tau_phi = l1Tau ? static_cast<float>(l1Tau->polarP4().phi()) : default_value;
        tauTuple().l1Tau_mass = l1Tau ? static_cast<float>(l1Tau->polarP4().mass()) : default_value;

        tauTuple().l1Tau_towerIEta = l1Tau ? l1Tau->towerIEta() : default_int_value;
        tauTuple().l1Tau_towerIPhi = l1Tau ? l1Tau->towerIPhi() : default_int_value;
        tauTuple().l1Tau_rawEt = l1Tau ? l1Tau->rawEt() : default_int_value;
        tauTuple().l1Tau_isoEt = l1Tau ? l1Tau->isoEt() : default_int_value;
        tauTuple().l1Tau_hasEM = l1Tau ? l1Tau->hasEM() : default_int_value;
        tauTuple().l1Tau_isMerged = l1Tau ? l1Tau->isMerged() : default_int_value;

        tauTuple().l1Tau_hwIso = l1Tau ? l1Tau->hwIso() : default_int_value;
        tauTuple().l1Tau_hwQual = l1Tau ? l1Tau->hwQual() : default_int_value;
    }

    void FillPFCandidates(const TauJet::PFCandCollection& cands, const std::string& prefix)
    {
        auto push_back = [&](const std::string& name, auto value) {
            tauTuple.get<std::vector<decltype(value)>>(prefix + name).push_back(value);
        };

        for(const auto& cand_desc : cands) {
            const reco::PFCandidate* cand = cand_desc.candidate;

            push_back("index", cand_desc.index);
            push_back("tauSignal", int(cand_desc.tauSignal));
            push_back("tauLeadChargedHadrCand", int(cand_desc.tauLeadChargedHadrCand));
            push_back("tauIso", int(cand_desc.tauIso));
            push_back("jetDaughter", int(cand_desc.jetDaughter));

            push_back("pt", static_cast<float>(cand->polarP4().pt()));
            push_back("eta", static_cast<float>(cand->polarP4().eta()));
            push_back("phi", static_cast<float>(cand->polarP4().phi()));
            push_back("mass", static_cast<float>(cand->polarP4().mass()));

            push_back("particleType", static_cast<int>(analysis::TranslatePdgIdToPFParticleType(cand->pdgId())));
            push_back("charge", cand->charge());

            push_back("vertex_x", static_cast<float>(cand->vertex().x()));
            push_back("vertex_y", static_cast<float>(cand->vertex().y()));
            push_back("vertex_z", static_cast<float>(cand->vertex().z()));

            const auto track = cand->bestTrack();
            const reco::HitPattern* hitPattern = track ? &track->hitPattern() : nullptr;

            using nHitFn = int (reco::HitPattern::*)() const;
            using nHitFnEx = int (reco::HitPattern::*)(reco::HitPattern::HitCategory) const;

            static const std::map<std::string, nHitFn> nHitBranches = {
                { "numberOfValidHits", &reco::HitPattern::numberOfValidHits },
                { "numberOfValidTrackerHits", &reco::HitPattern::numberOfValidTrackerHits },
                { "numberOfValidPixelHits", &reco::HitPattern::numberOfValidPixelHits },
                { "numberOfValidPixelBarrelHits", &reco::HitPattern::numberOfValidPixelBarrelHits },
                { "numberOfValidPixelEndcapHits", &reco::HitPattern::numberOfValidPixelEndcapHits },
                { "numberOfValidStripHits", &reco::HitPattern::numberOfValidStripHits },
                { "numberOfValidStripTIBHits", &reco::HitPattern::numberOfValidStripTIBHits },
                { "numberOfValidStripTIDHits", &reco::HitPattern::numberOfValidStripTIDHits },
                { "numberOfValidStripTOBHits", &reco::HitPattern::numberOfValidStripTOBHits },
                { "numberOfValidStripTECHits", &reco::HitPattern::numberOfValidStripTECHits },
                { "numberOfMuonHits", &reco::HitPattern::numberOfMuonHits },
                { "numberOfLostMuonHits", &reco::HitPattern::numberOfLostMuonHits },
                { "numberOfBadHits", &reco::HitPattern::numberOfBadHits },
                { "numberOfBadMuonHits", &reco::HitPattern::numberOfBadMuonHits },
                { "numberOfInactiveHits", &reco::HitPattern::numberOfInactiveHits },
                { "numberOfInactiveTrackerHits", &reco::HitPattern::numberOfInactiveTrackerHits },
                { "trackerLayersWithMeasurement", &reco::HitPattern::trackerLayersWithMeasurement },
                { "pixelLayersWithMeasurement", &reco::HitPattern::pixelLayersWithMeasurement },
                { "stripLayersWithMeasurement", &reco::HitPattern::stripLayersWithMeasurement },
                { "pixelBarrelLayersWithMeasurement", &reco::HitPattern::pixelBarrelLayersWithMeasurement },
                { "pixelEndcapLayersWithMeasurement", &reco::HitPattern::pixelEndcapLayersWithMeasurement },
                { "stripTIBLayersWithMeasurement", &reco::HitPattern::stripTIBLayersWithMeasurement },
                { "stripTIDLayersWithMeasurement", &reco::HitPattern::stripTIDLayersWithMeasurement },
                { "stripTOBLayersWithMeasurement", &reco::HitPattern::stripTOBLayersWithMeasurement },
                { "stripTECLayersWithMeasurement", &reco::HitPattern::stripTECLayersWithMeasurement },
            };

            static const std::map<reco::HitPattern::HitCategory, std::string> hitCategoryNames = {
                { reco::HitPattern::TRACK_HITS, "TRACK_HITS" },
                { reco::HitPattern::MISSING_INNER_HITS, "MISSING_INNER_HITS" },
                { reco::HitPattern::MISSING_OUTER_HITS, "MISSING_OUTER_HITS" },
            };

            static const std::map<std::string, nHitFnEx> nHitBranchesEx = {
                { "numberOfAllHits", &reco::HitPattern::numberOfAllHits },
                { "numberOfAllTrackerHits", &reco::HitPattern::numberOfAllTrackerHits },
                { "numberOfLostHits", &reco::HitPattern::numberOfLostHits },
                { "numberOfLostTrackerHits", &reco::HitPattern::numberOfLostTrackerHits },
                { "numberOfLostPixelHits", &reco::HitPattern::numberOfLostPixelHits },
                { "numberOfLostPixelBarrelHits", &reco::HitPattern::numberOfLostPixelBarrelHits },
                { "numberOfLostPixelEndcapHits", &reco::HitPattern::numberOfLostPixelEndcapHits },
                { "numberOfLostStripHits", &reco::HitPattern::numberOfLostStripHits },
                { "numberOfLostStripTIBHits", &reco::HitPattern::numberOfLostStripTIBHits },
                { "numberOfLostStripTIDHits", &reco::HitPattern::numberOfLostStripTIDHits },
                { "numberOfLostStripTOBHits", &reco::HitPattern::numberOfLostStripTOBHits },
                { "numberOfLostStripTECHits", &reco::HitPattern::numberOfLostStripTECHits },
                { "trackerLayersWithoutMeasurement", &reco::HitPattern::trackerLayersWithoutMeasurement },
                { "pixelLayersWithoutMeasurement", &reco::HitPattern::pixelLayersWithoutMeasurement },
                { "stripLayersWithoutMeasurement", &reco::HitPattern::stripLayersWithoutMeasurement },
                { "pixelBarrelLayersWithoutMeasurement", &reco::HitPattern::pixelBarrelLayersWithoutMeasurement },
                { "pixelEndcapLayersWithoutMeasurement", &reco::HitPattern::pixelEndcapLayersWithoutMeasurement },
                { "stripTIBLayersWithoutMeasurement", &reco::HitPattern::stripTIBLayersWithoutMeasurement },
                { "stripTIDLayersWithoutMeasurement", &reco::HitPattern::stripTIDLayersWithoutMeasurement },
                { "stripTOBLayersWithoutMeasurement", &reco::HitPattern::stripTOBLayersWithoutMeasurement },
                { "stripTECLayersWithoutMeasurement", &reco::HitPattern::stripTECLayersWithoutMeasurement },
            };

            for(const auto& [br_name, fn_ptr] : nHitBranches) {
                push_back(br_name, hitPattern ? (hitPattern->*fn_ptr)() : default_int_value);
            }

            for(const auto& [br_name, fn_ptr] : nHitBranchesEx) {
                for(const auto& [hit_category, hit_category_name] : hitCategoryNames) {
                    const std::string full_br_name = br_name + "_" + hit_category_name;
                    push_back(full_br_name, hitPattern ? (hitPattern->*fn_ptr)(hit_category) : default_int_value);
                }
            }

            push_back("hasTrackDetails", int(track != nullptr));
            push_back("dxy", track ? static_cast<float>(track->dxy()) : default_value);
            push_back("dxy_error", track ? static_cast<float>(track->dxyError()) : default_value);
            push_back("dz", track ? static_cast<float>(track->dz()) : default_value);
            push_back("dz_error", track ? static_cast<float>(cand->dzError()) : default_value);
            push_back("track_pt", track ? static_cast<float>(track->pt()) : default_value);
            push_back("track_etaAtVtx", track ? static_cast<float>(track->eta()) : default_value);
            push_back("track_phiAtVtx", track ? static_cast<float>(track->phi()) : default_value);
            push_back("track_chi2", track ? static_cast<float>(track->chi2()) : default_value);
            push_back("track_ndof", track ? static_cast<float>(track->ndof()) : default_value);

            push_back("etaAtECALEntrance", static_cast<float>(cand->positionAtECALEntrance().eta()));
            push_back("phiAtECALEntrance", static_cast<float>(cand->positionAtECALEntrance().phi()));

            push_back("ecalEnergy", static_cast<float>(cand->ecalEnergy()));
            push_back("hcalEnergy", static_cast<float>(cand->hcalEnergy()));

            push_back("rawEcalEnergy", static_cast<float>(cand->rawEcalEnergy()));
            push_back("rawHcalEnergy", static_cast<float>(cand->rawHcalEnergy()));
        }
    }

    void FillCaloHits(const TauJet::CaloHitCollection& caloHits)
    {
        for(const auto& caloHit : caloHits) {
            tauTuple().caloHit_type.push_back(static_cast<int>(caloHit->hitType));
            tauTuple().caloHit_r.push_back(static_cast<float>(caloHit->position.perp()));
            tauTuple().caloHit_eta.push_back(static_cast<float>(caloHit->position.eta()));
            tauTuple().caloHit_phi.push_back(static_cast<float>(caloHit->position.phi()));
            tauTuple().caloHit_energy.push_back(static_cast<float>(caloHit->energy));
            tauTuple().caloHit_chi2.push_back(static_cast<float>(caloHit->chi2));
        }
    }

    void FillPixelTracks(const TauJet::PataTrackCollection& selectedTracks, const pixelTrack::TrackSoA& tracks,
                         const ZVertexSoA& vertices)
    {
        for(const auto& track : selectedTracks) {
            const int idx = track->index;
            tauTuple().pixelTrack_pt.push_back(tracks.pt(idx));
            tauTuple().pixelTrack_eta.push_back(tracks.eta(idx));
            tauTuple().pixelTrack_phi.push_back(tracks.phi(idx));
            tauTuple().pixelTrack_charge.push_back(tracks.charge(idx));
            tauTuple().pixelTrack_quality.push_back(static_cast<int>(tracks.quality(idx)));
            tauTuple().pixelTrack_tip.push_back(tracks.tip(idx));
            tauTuple().pixelTrack_zip.push_back(tracks.zip(idx));
            const int idv = vertices.idv[idx];
            const bool has_vtx = idv >= 0;
            tauTuple().pixelTrack_vtx_index.push_back(idv);
            tauTuple().pixelTrack_vtx_z.push_back(has_vtx ? vertices.zv[idv] : default_value);
            tauTuple().pixelTrack_vtx_w.push_back(has_vtx ? vertices.wv[idv] : default_value);
            tauTuple().pixelTrack_vtx_chi2.push_back(has_vtx ? vertices.chi2[idv] : default_value);
            tauTuple().pixelTrack_vtx_pt2.push_back(has_vtx ? vertices.ptv2[idv] : default_value);
            tauTuple().pixelTrack_vtx_ndof.push_back(has_vtx ? vertices.ndof[idv] : default_int_value);
            tauTuple().pixelTrack_vtx_sortInd.push_back(has_vtx ? vertices.sortInd[idv] : default_int_value);
        }
    }

    static float CalculateGottfriedJacksonAngleDifference(const reco::PFTau& tau,
                                                          const reco::PFTauTransverseImpactParameter& tau_transverse_ip)
    {
        double gj_diff;
        if(::tau_analysis::CalculateGottfriedJacksonAngleDifference(tau, tau_transverse_ip, gj_diff))
            return static_cast<float>(gj_diff);
        return default_value;
    }

private:
    const bool isMC_, requireGenMatch_, requireGenORRecoTauMatch_, applyRecoPtSieve_;
    tau_analysis::TauJetBuilderSetup builderSetup_;

    const edm::EDGetTokenT<GenEventInfoProduct> genEventToken_;
    const edm::EDGetTokenT<reco::GenParticleCollection> genParticlesToken_;
    const edm::EDGetTokenT<reco::GenJetCollection> genJetsToken_;
    const edm::EDGetTokenT<reco::JetFlavourInfoMatchingCollection> genJetFlavourInfosToken_;
    const edm::EDGetTokenT<std::vector<PileupSummaryInfo>> puInfoToken_;
    const edm::EDGetTokenT<reco::BeamSpot> beamSpotToken_;
    const edm::EDGetTokenT<double> rhoToken_;

    const edm::EDGetTokenT<HBHERecHitCollection> hbheRecHitsToken_;
    const edm::EDGetTokenT<HORecHitCollection> hoRecHitsToken_;
    const edm::EDGetTokenT<EcalRecHitCollection> ebRecHitsToken_;
    const edm::EDGetTokenT<EcalRecHitCollection> eeRecHitsToken_;
    const edm::EDGetTokenT<PixelTrackHeterogeneous> pataTracksToken_;
    const edm::EDGetTokenT<ZVertexHeterogeneous> pataVerticesToken_;
    const edm::EDGetTokenT<reco::PFCandidateCollection> candsToken_;

    const edm::EDGetTokenT<l1t::TauBxCollection> l1TausToken_;
    const edm::EDGetTokenT<reco::PFJetCollection> jetsToken_;
    const edm::EDGetTokenT<reco::PFTauCollection> tausToken_;
    const edm::EDGetTokenT<TauIPCollection> tauIPToken_;

    const edm::ESGetToken<CaloGeometry, CaloGeometryRecord> geometryToken_;
    const edm::ESGetToken<MagneticField, IdealMagneticFieldRecord> bFieldToken_;

    TauTupleProducerData* data;
    tau_hlt::TauTuple& tauTuple;
    tau_tuple::SummaryTuple& summaryTuple;
};

} // namespace tau_hlt

#include "FWCore/Framework/interface/MakerMacros.h"
using TauTupleProducerHLT = tau_hlt::TauTupleProducer;
DEFINE_FWK_MODULE(TauTupleProducerHLT);
