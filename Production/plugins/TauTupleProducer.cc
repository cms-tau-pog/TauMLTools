/*! Creates tuple for tau analysis.
*/

#include "Compression.h"

#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ServiceRegistry/interface/Service.h"

#include "DataFormats/PatCandidates/interface/Electron.h"
#include "DataFormats/PatCandidates/interface/Photon.h"
#include "DataFormats/PatCandidates/interface/Muon.h"
#include "DataFormats/PatCandidates/interface/Tau.h"
#include "DataFormats/PatCandidates/interface/Jet.h"
#include "DataFormats/PatCandidates/interface/PackedCandidate.h"
#include "DataFormats/PatCandidates/interface/IsolatedTrack.h"
#include "DataFormats/TrackReco/interface/HitPattern.h"
#include "DataFormats/METReco/interface/PFMETCollection.h"

#include "SimDataFormats/GeneratorProducts/interface/GenEventInfoProduct.h"

#if __has_include ("DataFormats/JetMatching/interface/JetFlavourInfoMatching.h")
    #include "DataFormats/JetMatching/interface/JetFlavourInfoMatching.h"
#else
    #include "SimDataFormats/JetMatching/interface/JetFlavourInfoMatching.h"
#endif

#include "AnalysisDataFormats/TopObjects/interface/TtGenEvent.h"
#include "CommonTools/UtilAlgos/interface/TFileService.h"
#include "CommonTools/Utils/interface/StringCutObjectSelector.h"
#include "RecoTauTag/RecoTau/interface/PFRecoTauClusterVariables.h"

#include "TauMLTools/Core/interface/Tools.h"
#include "TauMLTools/Core/interface/TextIO.h"
#include "TauMLTools/Analysis/interface/TauTuple.h"
#include "TauMLTools/Analysis/interface/SummaryTuple.h"
#include "TauMLTools/Analysis/interface/TauIdResults.h"
#include "TauMLTools/Production/interface/GenTruthTools.h"
#include "TauMLTools/Production/interface/TauAnalysis.h"
#include "TauMLTools/Production/interface/MuonHitMatch.h"
#include "TauMLTools/Production/interface/TauJet.h"
#include "TauMLTools/Production/interface/Selectors.h"

namespace {
    template <typename T>
    edm::Handle<T> getHandle(const edm::Event& event, const edm::EDGetTokenT<T>& token, bool get = true) {
        edm::Handle<T> handle;
        if(get)
            event.getByToken(token, handle);
        return handle;
    }

    template <typename T>
    const T* getProduct(const edm::Event& event, const edm::EDGetTokenT<T>& token, bool get = true) {
        auto handle = getHandle(event, token, get);
        return handle.isValid() ? handle.product() : nullptr;
    }

} // anonymous namespace

namespace tau_analysis {

    inline constexpr int GetCMSSWVersion()
    {
        int d1 =  *(PROJECT_VERSION + 6) - '0';
        int d2 =  *(PROJECT_VERSION + 7) - '0';
        if(d2 >= 0 && d2 <= 9) return d1 * 10 + d2;
        return d1;
    }

    template<typename Elec, int cmssw_version>
    struct GetElecVer;

    template<typename Elec>
    struct GetElecVer<Elec, 12> {
        static float hcalDepth1OverEcal(const Elec* ele) { return ele->hcalOverEcal(1); }
        static float hcalDepth2OverEcal(const Elec* ele) { return ele->hcalOverEcal(2); }
        static float hcalDepth1OverEcalBc(const Elec* ele) { return ele->hcalOverEcalBc(1); }
        static float hcalDepth2OverEcalBc(const Elec* ele) { return ele->hcalOverEcalBc(2); }
        static float full5x5_hcalDepth1OverEcal(const Elec* ele) { return ele->full5x5_hcalOverEcal(1); }
        static float full5x5_hcalDepth2OverEcal(const Elec* ele) { return ele->full5x5_hcalOverEcal(2); }
        static float full5x5_hcalDepth1OverEcalBc(const Elec* ele) { return ele->full5x5_hcalOverEcalBc(1); }
        static float full5x5_hcalDepth2OverEcalBc(const Elec* ele) { return ele->full5x5_hcalOverEcalBc(2); }
    };

    template<typename Elec>
    struct GetElecVer<Elec, 10> {
        static float hcalDepth1OverEcal(const Elec* ele) { return ele->hcalDepth1OverEcal(); }
        static float hcalDepth2OverEcal(const Elec* ele) { return ele->hcalDepth2OverEcal(); }
        static float hcalDepth1OverEcalBc(const Elec* ele) { return ele->hcalDepth1OverEcalBc(); }
        static float hcalDepth2OverEcalBc(const Elec* ele) { return ele->hcalDepth2OverEcalBc(); }
        static float full5x5_hcalDepth1OverEcal(const Elec* ele) { return ele->full5x5_hcalDepth1OverEcal(); }
        static float full5x5_hcalDepth2OverEcal(const Elec* ele) { return ele->full5x5_hcalDepth2OverEcal(); }
        static float full5x5_hcalDepth1OverEcalBc(const Elec* ele) { return ele->full5x5_hcalDepth1OverEcalBc(); }
        static float full5x5_hcalDepth2OverEcalBc(const Elec* ele) { return ele->full5x5_hcalDepth2OverEcalBc(); }
    };

    template<typename Elec>
    struct GetElecVer<Elec, 11> : GetElecVer<Elec, 10> {};

struct TauTupleProducerData {
    using clock = std::chrono::system_clock;

    const clock::time_point start;
    tau_tuple::TauTuple tauTuple;
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
    TauTupleProducer(const edm::ParameterSet& cfg) :
        isMC(cfg.getParameter<bool>("isMC")),
        isEmbedded(cfg.getParameter<bool>("isEmbedded")),
        requireGenMatch(cfg.getParameter<bool>("requireGenMatch")),
        requireGenORRecoTauMatch(cfg.getParameter<bool>("requireGenORRecoTauMatch")),
        applyRecoPtSieve(cfg.getParameter<bool>("applyRecoPtSieve")),
        genEvent_token(mayConsume<GenEventInfoProduct>(cfg.getParameter<edm::InputTag>("genEvent"))),
        genParticles_token(mayConsume<reco::GenParticleCollection>(cfg.getParameter<edm::InputTag>("genParticles"))),
        genJets_token(mayConsume<reco::GenJetCollection>(cfg.getParameter<edm::InputTag>("genJets"))),
        genJetFlavourInfos_token(consumes<reco::JetFlavourInfoMatchingCollection>(
                                 cfg.getParameter<edm::InputTag>("genJetFlavourInfos"))),
        puInfo_token(mayConsume<std::vector<PileupSummaryInfo>>(cfg.getParameter<edm::InputTag>("puInfo"))),
        vertices_token(consumes<std::vector<reco::Vertex>>(cfg.getParameter<edm::InputTag>("vertices"))),
        secondVertices_token(consumes<std::vector<reco::VertexCompositePtrCandidate>>(cfg.getParameter<edm::InputTag>("secondVertices"))),
        rho_token(consumes<double>(cfg.getParameter<edm::InputTag>("rho"))),
        electrons_token(consumes<pat::ElectronCollection>(cfg.getParameter<edm::InputTag>("electrons"))),
        photons_token(consumes<pat::PhotonCollection>(cfg.getParameter<edm::InputTag>("photons"))),
        muons_token(consumes<pat::MuonCollection>(cfg.getParameter<edm::InputTag>("muons"))),
        taus_token(consumes<pat::TauCollection>(cfg.getParameter<edm::InputTag>("taus"))),
        boostedTaus_token(consumes<pat::TauCollection>(cfg.getParameter<edm::InputTag>("boostedTaus"))),
        jets_token(consumes<pat::JetCollection>(cfg.getParameter<edm::InputTag>("jets"))),
        fatJets_token(consumes<pat::JetCollection>(cfg.getParameter<edm::InputTag>("fatJets"))),
        cands_token(consumes<pat::PackedCandidateCollection>(cfg.getParameter<edm::InputTag>("pfCandidates"))),
        isoTracks_token(consumes<pat::IsolatedTrackCollection>(cfg.getParameter<edm::InputTag>("isoTracks"))),
        lostTracks_token(consumes<pat::PackedCandidateCollection>(cfg.getParameter<edm::InputTag>("lostTracks"))),
        METs_token(consumes<edm::View<pat::MET>>(cfg.getParameter<edm::InputTag>("METs"))),
        puppiMETs_token(consumes<edm::View<pat::MET>>(cfg.getParameter<edm::InputTag>("puppiMETs"))),
        deepMETs_token(consumes<edm::View<pat::MET>>(cfg.getParameter<edm::InputTag>("deepMETs"))),
        genMETs_token(consumes<edm::View<reco::GenMET>>(cfg.getParameter<edm::InputTag>("genMETs"))),
        triggerResults_token(consumes<edm::TriggerResults>(cfg.getParameter<edm::InputTag>("triggerResults"))),
        triggerObjects_token(consumes<pat::TriggerObjectStandAloneCollection>(cfg.getParameter<edm::InputTag>("triggerObjects"))),
        tauSpinnerWTEven_token(consumes<double>(cfg.getParameter<edm::InputTag>("tauSpinnerWTEven"))),
        tauSpinnerWTOdd_token(consumes<double>(cfg.getParameter<edm::InputTag>("tauSpinnerWTOdd"))),
        tauSpinnerWTMM_token(consumes<double>(cfg.getParameter<edm::InputTag>("tauSpinnerWTMM"))),

        data(TauTupleProducerData::RequestGlobalData()),
        tauTuple(data->tauTuple),
        summaryTuple(data->summaryTuple),
        selector(selectors::TauJetSelector::Make(cfg.getParameter<std::string>("selector")))
    {
        const std::map<std::string, double*> builderParamNames = {
            { "genLepton_genJet_dR", &builderSetup.genLepton_genJet_dR },
            { "genLepton_tau_dR", &builderSetup.genLepton_tau_dR },
            { "genLepton_boostedTau_dR", &builderSetup.genLepton_boostedTau_dR },
            { "genLepton_jet_dR", &builderSetup.genLepton_jet_dR },
            { "genLepton_fatJet_dR", &builderSetup.genLepton_fatJet_dR },
            { "genJet_tau_dR", &builderSetup.genJet_tau_dR },
            { "genJet_boostedTau_dR", &builderSetup.genJet_boostedTau_dR },
            { "genJet_jet_dR", &builderSetup.genJet_jet_dR },
            { "genJet_fatJet_dR", &builderSetup.genJet_fatJet_dR },
            { "tau_boostedTau_dR", &builderSetup.tau_boostedTau_dR },
            { "tau_jet_dR", &builderSetup.tau_jet_dR },
            { "tau_fatJet_dR", &builderSetup.tau_fatJet_dR },
            { "jet_fatJet_dR", &builderSetup.jet_fatJet_dR },
            { "jet_maxAbsEta", &builderSetup.jet_maxAbsEta },
            { "fatJet_maxAbsEta", &builderSetup.fatJet_maxAbsEta },
            { "genLepton_cone", &builderSetup.genLepton_cone },
            { "genJet_cone", &builderSetup.genJet_cone },
            { "tau_cone", &builderSetup.tau_cone },
            { "boostedTau_cone", &builderSetup.boostedTau_cone },
            { "jet_cone", &builderSetup.jet_cone },
            { "fatJet_cone", &builderSetup.fatJet_cone },
        };
        const auto& builderParams = cfg.getParameterSet("tauJetBuilderSetup");
        for(const auto& paramName : builderParams.getParameterNames()) {
            auto iter = builderParamNames.find(paramName);
            if(iter == builderParamNames.end())
                throw cms::Exception("TauTupleProducer") << "Unknown parameter '" << paramName <<
                                                            "' in tauJetBuilderSetup.";
            *iter->second = builderParams.getParameter<double>(paramName);
        }
    }

private:
    static constexpr float default_value = tau_tuple::DefaultFillValue<float>();
    static constexpr int default_int_value = tau_tuple::DefaultFillValue<int>();

    virtual void analyze(const edm::Event& event, const edm::EventSetup& eventSetup) override
    {
        std::lock_guard<std::mutex> lock(data->mutex);
        summaryTuple().numberOfProcessedEvents++;

        tauTuple().run  = event.id().run();
        tauTuple().lumi = event.id().luminosityBlock();
        tauTuple().evt  = event.id().event();
        tauTuple().sampleType = isEmbedded ? static_cast<int>(SampleType::Embedded) :
                                isMC ? static_cast<int>(SampleType::MC) : static_cast<int>(SampleType::Data);
        tauTuple().dataset_id = -1;
        tauTuple().dataset_group_id = -1;

        auto vertices = getHandle(event, vertices_token);
        tauTuple().npv = static_cast<int>(vertices->size());
	auto secondVertices = getHandle(event, secondVertices_token);
        auto rho = getHandle(event, rho_token);
        tauTuple().rho = static_cast<float>(*rho);

        if(isMC) {
            auto genEvent = getHandle(event, genEvent_token);
            tauTuple().genEventWeight = static_cast<float>(genEvent->weight());

            auto puInfo = getHandle(event, puInfo_token);
            tauTuple().npu = gen_truth::GetNumberOfPileUpInteractions(puInfo);
        }

        const auto& PV = vertices->at(0);
        tauTuple().pv_x = static_cast<float>(PV.position().x());
        tauTuple().pv_y = static_cast<float>(PV.position().y());
        tauTuple().pv_z = static_cast<float>(PV.position().z());
        tauTuple().pv_t = static_cast<float>(PV.t());
        tauTuple().pv_xE = static_cast<float>(PV.xError());
        tauTuple().pv_yE = static_cast<float>(PV.yError());
        tauTuple().pv_zE = static_cast<float>(PV.zError());
        tauTuple().pv_tE = static_cast<float>(PV.tError());
        tauTuple().pv_chi2 = static_cast<float>(PV.chi2());
        tauTuple().pv_ndof = static_cast<float>(PV.ndof());

        auto electrons = getHandle(event, electrons_token);
        auto photons = getHandle(event, photons_token);
        auto muons = getHandle(event, muons_token);
        auto taus = getHandle(event, taus_token);
        auto boostedTaus = getHandle(event, boostedTaus_token);
        auto jets = getHandle(event, jets_token);
        auto fatJets = getHandle(event, fatJets_token);
        auto cands = getHandle(event, cands_token);
        auto isoTracks = getHandle(event, isoTracks_token);
        auto lostTracks = getHandle(event, lostTracks_token);
        auto METs = getHandle(event, METs_token);
	auto puppiMETs = getHandle(event, puppiMETs_token);
	auto deepMETs = getHandle(event, deepMETs_token);
	auto genMETs = getHandle(event, genMETs_token);
        auto triggerResults = getHandle(event, triggerResults_token);
        auto triggerObjects = getHandle(event, triggerObjects_token);
        auto tauSpinnerWTEven = getHandle(event, tauSpinnerWTEven_token);
        auto tauSpinnerWTOdd = getHandle(event, tauSpinnerWTOdd_token);
        auto tauSpinnerWTMM = getHandle(event, tauSpinnerWTMM_token);

        auto genParticles = getProduct(event, genParticles_token, isMC);
        auto genJets = getProduct(event, genJets_token, isMC);
        auto genJetFlavourInfos = getProduct(event, genJetFlavourInfos_token, isMC);

        tauTuple().met_pt = METs->at(0).pt();
        tauTuple().met_phi = METs->at(0).phi();
	tauTuple().metcov_00 = METs->at(0).getSignificanceMatrix()[0][0];
        tauTuple().metcov_01 = METs->at(0).getSignificanceMatrix()[0][1];
        tauTuple().metcov_11 = METs->at(0).getSignificanceMatrix()[1][1];
	tauTuple().puppimet_pt = puppiMETs->at(0).pt();
        tauTuple().puppimet_phi = puppiMETs->at(0).phi();
        tauTuple().puppimetcov_00 = puppiMETs->at(0).getSignificanceMatrix()[0][0];
        tauTuple().puppimetcov_01 = puppiMETs->at(0).getSignificanceMatrix()[0][1];
        tauTuple().puppimetcov_11 = puppiMETs->at(0).getSignificanceMatrix()[1][1];
	tauTuple().deepmet_pt = deepMETs->at(0).pt();
	tauTuple().deepmet_phi = deepMETs->at(0).phi();
        tauTuple().genmet_pt = genMETs->at(0).pt();
        tauTuple().genmet_phi = genMETs->at(0).phi();
	tauTuple().tauSpinnerWTEven = (*tauSpinnerWTEven);
        tauTuple().tauSpinnerWTOdd = (*tauSpinnerWTOdd);
        tauTuple().tauSpinnerWTMM = (*tauSpinnerWTMM);

        TauJetBuilder builder(builderSetup, *taus, *boostedTaus, *jets, *fatJets, *cands, *electrons, *photons, *muons,
                              *isoTracks, *lostTracks, *secondVertices, genParticles, genJets, requireGenMatch,
                              requireGenORRecoTauMatch, applyRecoPtSieve);
        const auto [tauJets, tagObj] = selector->Select(event, builder.GetTauJets(), *electrons, *muons,
                                                           METs->at(0), PV, *triggerObjects, *triggerResults, *rho);
        tauTuple().tagObj_valid = tagObj != nullptr;                                                    
        tauTuple().tagObj_pt = tagObj ? tagObj->p4.pt() : default_value;
        tauTuple().tagObj_eta = tagObj ? tagObj->p4.eta() : default_value;
        tauTuple().tagObj_phi = tagObj ? tagObj->p4.phi() : default_value;
        tauTuple().tagObj_mass = tagObj ? tagObj->p4.mass() : default_value;
        tauTuple().tagObj_charge = tagObj ? tagObj->charge : default_int_value;
        tauTuple().tagObj_id = tagObj ? tagObj->id : 0;
        tauTuple().tagObj_iso = tagObj ? tagObj->isolation : default_value;
        tauTuple().has_extramuon = tagObj ? tagObj->has_extramuon : default_value;
        tauTuple().has_extraelectron = tagObj ? tagObj->has_extraelectron : default_value;
        tauTuple().has_dimuon = tagObj ? tagObj->has_dimuon : default_value;

        tauTuple().total_entries = static_cast<int>(tauJets.size());
        for(size_t tauJetIndex = 0; tauJetIndex < tauJets.size(); ++tauJetIndex) {
            const TauJet& tauJet = *tauJets.at(tauJetIndex);
            tauTuple().entry_index = static_cast<int>(tauJetIndex);
	    
            FillGenLepton(tauJet.genLepton);
            FillGenJet(tauJet.genJet, genJetFlavourInfos);

            FillTau(tauJet.tau, "tau_");
            FillTau(tauJet.boostedTau, "boostedTau_");
            FillJet(tauJet.jet, "jet_");
            FillJet(tauJet.fatJet, "fatJet_");

            FillPFCandidates(tauJet.cands, "pfCand_");
            FillPFCandidates(tauJet.lostTracks, "lostTrack_");
            FillElectrons(tauJet.electrons);
            FillPhotons(tauJet.photons);
            FillMuons(tauJet.muons, PV);
            FillIsoTracks(tauJet.isoTracks);
	    FillSV(tauJet.secondVertices);

            tauTuple.Fill();
        }
    }

    virtual void endJob() override
    {
        TauTupleProducerData::ReleaseGlobalData();
    }

private:
    static bool PrintTauIdNames(const pat::Tau& tau)
    {
        static const std::string header(40, '-');

        std::set<std::string> tauId_names;
        for(const auto& id : tau.tauIDs())
            tauId_names.insert(id.first);
        std::cout << "Tau IDs:\n" << header << "\n";
        for(const std::string& name : tauId_names)
            std::cout << name << "\n";
        std::cout << header << std::endl;

        return true;
    }

    template<typename Candidate>
    static float GetUserFloat(const Candidate* cand, const std::string& key)
    {
        return cand && cand->hasUserFloat(key) ? cand->userFloat(key) : default_value;
    }

    void FillGenLepton(const ObjPtr<reco_tau::gen_truth::GenLepton>& genLepton)
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

    void FillGenJet(const ObjPtr<const reco::GenJet>& genJet,
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

    void FillTau(const ObjPtr<const pat::Tau>& tau, const std::string& prefix)
    {
        if(tau) {
            static const bool id_names_printed = PrintTauIdNames(*tau);
            (void)id_names_printed;
        }

        tauTuple.get<int>(prefix + "index") = tau.index;
        tauTuple.get<float>(prefix + "pt") = tau ? static_cast<float>(tau->polarP4().pt()) : default_value;
        tauTuple.get<float>(prefix + "eta") = tau ? static_cast<float>(tau->polarP4().eta()) : default_value;
        tauTuple.get<float>(prefix + "phi") = tau ? static_cast<float>(tau->polarP4().phi()) : default_value;
        tauTuple.get<float>(prefix + "mass") = tau ? static_cast<float>(tau->polarP4().mass()) : default_value;
        tauTuple.get<int>(prefix + "charge") = tau ? tau->charge() : default_int_value;

        tauTuple.get<int>(prefix + "decayMode") = tau ? tau->decayMode() : default_int_value;
        tauTuple.get<int>(prefix + "decayModeFinding") = tau
                ? tau->tauID("decayModeFinding") > 0.5f : default_int_value;
        tauTuple.get<int>(prefix + "decayModeFindingNewDMs") = tau
                ? tau->tauID("decayModeFindingNewDMs") > 0.5f : default_int_value;
        tauTuple.get<float>(prefix + "chargedIsoPtSum") = tau ? tau->tauID("chargedIsoPtSum") : default_value;
        tauTuple.get<float>(prefix + "chargedIsoPtSumdR03") = tau ? tau->tauID("chargedIsoPtSumdR03") : default_value;
        tauTuple.get<float>(prefix + "footprintCorrection") = tau ? tau->tauID("footprintCorrection") : default_value;
        tauTuple.get<float>(prefix + "footprintCorrectiondR03") = tau
                ? tau->tauID("footprintCorrectiondR03") : default_value;
        tauTuple.get<float>(prefix + "neutralIsoPtSum") = tau ? tau->tauID("neutralIsoPtSum") : default_value;
        tauTuple.get<float>(prefix + "neutralIsoPtSumWeight") = tau
                ? tau->tauID("neutralIsoPtSumWeight") : default_value;
        tauTuple.get<float>(prefix + "neutralIsoPtSumWeightdR03") = tau
                ? tau->tauID("neutralIsoPtSumWeightdR03") : default_value;
        tauTuple.get<float>(prefix + "neutralIsoPtSumdR03") = tau ? tau->tauID("neutralIsoPtSumdR03") : default_value;
        tauTuple.get<float>(prefix + "photonPtSumOutsideSignalCone") = tau
                ? tau->tauID("photonPtSumOutsideSignalCone") : default_value;
        tauTuple.get<float>(prefix + "photonPtSumOutsideSignalConedR03") = tau
                ? tau->tauID("photonPtSumOutsideSignalConedR03") : default_value;
        tauTuple.get<float>(prefix + "puCorrPtSum") = tau ? tau->tauID("puCorrPtSum") : default_value;

        for(const auto& tau_id_entry : analysis::tau_id::GetTauIdDescriptors()) {
            const auto& desc = tau_id_entry.second;
            desc.FillTuple(tauTuple, tau.obj, default_value, prefix);
        }

        tauTuple.get<float>(prefix + "dxy_pca_x") = tau ? tau->dxy_PCA().x() : default_value;
        tauTuple.get<float>(prefix + "dxy_pca_y") = tau ? tau->dxy_PCA().y() : default_value;
        tauTuple.get<float>(prefix + "dxy_pca_z") = tau ? tau->dxy_PCA().z() : default_value;
        tauTuple.get<float>(prefix + "dxy") = tau ? tau->dxy() : default_value;
        tauTuple.get<float>(prefix + "dxy_error") = tau ? tau->dxy_error() : default_value;
        tauTuple.get<float>(prefix + "ip3d") = tau ? tau->ip3d() : default_value;
        tauTuple.get<float>(prefix + "ip3d_error") = tau ? tau->ip3d_error() : default_value;
        const bool has_sv = tau && tau->hasSecondaryVertex();
        tauTuple.get<int>(prefix + "hasSecondaryVertex") = tau ? tau->hasSecondaryVertex() : default_int_value;
        tauTuple.get<float>(prefix + "sv_x") = has_sv ? tau->secondaryVertexPos().x() : default_value;
        tauTuple.get<float>(prefix + "sv_y") = has_sv ? tau->secondaryVertexPos().y() : default_value;
        tauTuple.get<float>(prefix + "sv_z") = has_sv ? tau->secondaryVertexPos().z() : default_value;
        tauTuple.get<float>(prefix + "flightLength_x") = tau ? tau->flightLength().x() : default_value;
        tauTuple.get<float>(prefix + "flightLength_y") = tau ? tau->flightLength().y() : default_value;
        tauTuple.get<float>(prefix + "flightLength_z") = tau ? tau->flightLength().z() : default_value;
        tauTuple.get<float>(prefix + "flightLength_sig") = tau ? tau->flightLengthSig() : default_value;

        const pat::PackedCandidate* leadChargedHadrCand =
                tau ? dynamic_cast<const pat::PackedCandidate*>(tau->leadChargedHadrCand().get()) : nullptr;
        tauTuple.get<float>(prefix + "dz") = leadChargedHadrCand ? leadChargedHadrCand->dz() : default_value;
        tauTuple.get<float>(prefix + "dz_error") = leadChargedHadrCand && leadChargedHadrCand->hasTrackDetails()
                ? leadChargedHadrCand->dzError() : default_value;

        tauTuple.get<float>(prefix + "pt_weighted_deta_strip") = tau
                ? reco::tau::pt_weighted_deta_strip(*tau, tau->decayMode()) : default_value;
        tauTuple.get<float>(prefix + "pt_weighted_dphi_strip") = tau
                ? reco::tau::pt_weighted_dphi_strip(*tau, tau->decayMode()) : default_value;
        tauTuple.get<float>(prefix + "pt_weighted_dr_signal") = tau
                ? reco::tau::pt_weighted_dr_signal(*tau, tau->decayMode()) : default_value;
        tauTuple.get<float>(prefix + "pt_weighted_dr_iso") = tau
                ? reco::tau::pt_weighted_dr_iso(*tau, tau->decayMode()) : default_value;
        tauTuple.get<float>(prefix + "leadingTrackNormChi2") = tau ? tau->leadingTrackNormChi2() : default_value;
        tauTuple.get<float>(prefix + "e_ratio") = tau ? reco::tau::eratio(*tau) : default_value;
        tauTuple.get<float>(prefix + "gj_angle_diff") = tau
                ? CalculateGottfriedJacksonAngleDifference(*tau) : default_value;
        tauTuple.get<int>(prefix + "n_photons") = tau
                ? static_cast<int>(reco::tau::n_photons_total(*tau)) : default_int_value;

        tauTuple.get<float>(prefix + "emFraction") = tau ? tau->emFraction_MVA() : default_value;
        tauTuple.get<int>(prefix + "inside_ecal_crack") = tau ? IsInEcalCrack(tau->polarP4().eta()) : default_int_value;
        tauTuple.get<float>(prefix + "leadChargedCand_etaAtEcalEntrance") = tau
                ? tau->etaAtEcalEntranceLeadChargedCand() : default_value;
    }

    void FillJet(const ObjPtr<const pat::Jet>& jet, const std::string& prefix)
    {
        tauTuple.get<int>(prefix + "index") = jet.index;
        tauTuple.get<float>(prefix + "pt") = jet ? static_cast<float>(jet->polarP4().pt()) : default_value;
        tauTuple.get<float>(prefix + "eta") = jet ? static_cast<float>(jet->polarP4().eta()) : default_value;
        tauTuple.get<float>(prefix + "phi") = jet ? static_cast<float>(jet->p4().phi()) : default_value;
        tauTuple.get<float>(prefix + "mass") = jet ? static_cast<float>(jet->p4().mass()) : default_value;
        boost::optional<pat::Jet> uncorrected_jet;
        if(jet)
            uncorrected_jet = jet->correctedJet("Uncorrected");
        const bool has_details = jet && (jet->isPFJet() || jet->isJPTJet());
        tauTuple.get<float>(prefix + "neutralHadronEnergyFraction") = has_details
                ? uncorrected_jet->neutralHadronEnergyFraction() : default_value;
        tauTuple.get<float>(prefix + "neutralEmEnergyFraction") = has_details
                ? uncorrected_jet->neutralEmEnergyFraction() : default_value;
        tauTuple.get<int>(prefix + "nConstituents") = jet ? uncorrected_jet->nConstituents() : default_int_value;
        tauTuple.get<int>(prefix + "chargedMultiplicity") = has_details
                ? uncorrected_jet->chargedMultiplicity() : default_int_value;
        tauTuple.get<int>(prefix + "neutralMultiplicity") = has_details
                ? uncorrected_jet->neutralMultiplicity() : default_int_value;
        tauTuple.get<int>(prefix + "partonFlavour") = jet ? jet->partonFlavour() : default_int_value;
        tauTuple.get<int>(prefix + "hadronFlavour") = jet ? jet->hadronFlavour() : default_int_value;
        tauTuple.get<float>(prefix + "m_softDrop") = GetUserFloat(jet.obj, "ak8PFJetsPuppiSoftDropMass");
        tauTuple.get<float>(prefix + "nJettiness_tau1") = GetUserFloat(jet.obj, "NjettinessAK8Puppi:tau1");
        tauTuple.get<float>(prefix + "nJettiness_tau2") = GetUserFloat(jet.obj, "NjettinessAK8Puppi:tau2");
        tauTuple.get<float>(prefix + "nJettiness_tau3") = GetUserFloat(jet.obj, "NjettinessAK8Puppi:tau3");
        tauTuple.get<float>(prefix + "nJettiness_tau4") = GetUserFloat(jet.obj, "NjettinessAK8Puppi:tau4");

        if(jet && jet->hasSubjets("SoftDropPuppi")) {
            const auto& sub_jets = jet->subjets("SoftDropPuppi");
            for(const auto& sub_jet : sub_jets) {
                tauTuple.get<std::vector<float>>(prefix + "subJet_pt").push_back(
                        static_cast<float>(sub_jet->polarP4().pt()));
                tauTuple.get<std::vector<float>>(prefix + "subJet_eta").push_back(
                        static_cast<float>(sub_jet->polarP4().eta()));
                tauTuple.get<std::vector<float>>(prefix + "subJet_phi").push_back(
                        static_cast<float>(sub_jet->polarP4().phi()));
                tauTuple.get<std::vector<float>>(prefix + "subJet_mass").push_back(
                        static_cast<float>(sub_jet->polarP4().mass()));
            }
        }
    }

    void FillPFCandidates(const TauJet::PFCandCollection& cands, const std::string& prefix)
    {
        auto push_back = [&](const std::string& name, auto value) {
            tauTuple.get<std::vector<decltype(value)>>(prefix + name).push_back(value);
        };

        for(const PFCandDesc& cand_desc : cands) {
            const pat::PackedCandidate* cand = cand_desc.candidate;

            push_back("index", cand_desc.index);
            push_back("tauSignal", int(cand_desc.tauSignal));
            push_back("tauLeadChargedHadrCand", int(cand_desc.tauLeadChargedHadrCand));
            push_back("tauIso", int(cand_desc.tauIso));
            push_back("boostedTauSignal", int(cand_desc.boostedTauSignal));
            push_back("boostedTauLeadChargedHadrCand", int(cand_desc.boostedTauLeadChargedHadrCand));
            push_back("boostedTauIso", int(cand_desc.boostedTauIso));
            push_back("jetDaughter", int(cand_desc.jetDaughter));
            push_back("fatJetDaughter", int(cand_desc.fatJetDaughter));
            push_back("subJetDaughter", cand_desc.subJetDaughter);

            push_back("pt", static_cast<float>(cand->polarP4().pt()));
            push_back("eta", static_cast<float>(cand->polarP4().eta()));
            push_back("phi", static_cast<float>(cand->polarP4().phi()));
            push_back("mass", static_cast<float>(cand->polarP4().mass()));

            push_back("pvAssociationQuality", static_cast<int>(cand->pvAssociationQuality()));
            push_back("fromPV", static_cast<int>(cand->fromPV()));
            push_back("puppiWeight", cand->puppiWeight());
            push_back("puppiWeightNoLep", cand->puppiWeightNoLep());
            push_back("particleType", static_cast<int>(TranslatePdgIdToPFParticleType(cand->pdgId())));
            push_back("charge", cand->charge());
            push_back("lostInnerHits", static_cast<int>(cand->lostInnerHits()));
            push_back("nPixelHits", cand->numberOfPixelHits());
            push_back("nHits", cand->numberOfHits());
            push_back("nPixelLayers", cand->pixelLayersWithMeasurement());
            push_back("nStripLayers", cand->stripLayersWithMeasurement());

            push_back("vertex_x", static_cast<float>(cand->vertex().x()));
            push_back("vertex_y", static_cast<float>(cand->vertex().y()));
            push_back("vertex_z", static_cast<float>(cand->vertex().z()));
            push_back("vertex_t", static_cast<float>(cand->vertexRef()->t()));

            push_back("time", cand->time());
            push_back("timeError", cand->timeError());

            const bool hasTrackDetails = cand->hasTrackDetails();
            push_back("hasTrackDetails", int(hasTrackDetails));
            push_back("dxy", cand->dxy());
            push_back("dxy_error", hasTrackDetails ? cand->dxyError() : default_value);
            push_back("dz", cand->dz());
            push_back("dz_error", hasTrackDetails ? cand->dzError() : default_value);
            push_back("track_pt", static_cast<float>(cand->ptTrk()));
            push_back("track_eta", cand->etaAtVtx());
            push_back("track_phi", cand->phiAtVtx());
            push_back("track_chi2", hasTrackDetails ? static_cast<float>(cand->bestTrack()->chi2()) : default_value);
            push_back("track_ndof", hasTrackDetails ? static_cast<float>(cand->bestTrack()->ndof()) : default_value);

            push_back("caloFraction", cand->caloFraction());
            push_back("hcalFraction", cand->hcalFraction());

            push_back("rawCaloFraction", cand->rawCaloFraction());
            push_back("rawHcalFraction", cand->rawHcalFraction());
        }
    }

    void FillElectrons(const TauJet::ElectronCollection& electrons)
    {
        for(const auto& ele_ptr : electrons) {
            const pat::Electron* ele = ele_ptr.obj;
            tauTuple().ele_index.push_back(ele_ptr.index);
            tauTuple().ele_pt.push_back(static_cast<float>(ele->polarP4().pt()));
            tauTuple().ele_eta.push_back(static_cast<float>(ele->polarP4().eta()));
            tauTuple().ele_phi.push_back(static_cast<float>(ele->polarP4().phi()));
            tauTuple().ele_mass.push_back(static_cast<float>(ele->polarP4().mass()));
            float cc_ele_energy, cc_gamma_energy;
            int cc_n_gamma;
            CalculateElectronClusterVars(*ele, cc_ele_energy, cc_gamma_energy, cc_n_gamma);
            tauTuple().ele_cc_ele_energy.push_back(cc_ele_energy);
            tauTuple().ele_cc_gamma_energy.push_back(cc_gamma_energy);
            tauTuple().ele_cc_n_gamma.push_back(cc_n_gamma);
            tauTuple().ele_trackMomentumAtVtx.push_back(ele->trackMomentumAtVtx().R());
            tauTuple().ele_trackMomentumAtCalo.push_back(ele->trackMomentumAtCalo().R());
            tauTuple().ele_trackMomentumOut.push_back(ele->trackMomentumOut().R());
            tauTuple().ele_trackMomentumAtEleClus.push_back(ele->trackMomentumAtEleClus().R());
            tauTuple().ele_trackMomentumAtVtxWithConstraint.push_back(ele->trackMomentumAtVtxWithConstraint().R());
            tauTuple().ele_dxy.push_back(ele->dB(pat::Electron::PV2D));
            tauTuple().ele_dxy_error.push_back(ele->edB(pat::Electron::PV2D));
            tauTuple().ele_ip3d.push_back(ele->ip3d());
            tauTuple().ele_ecalEnergy.push_back(ele->ecalEnergy());
            tauTuple().ele_ecalEnergy_error.push_back(ele->ecalEnergyError());
            tauTuple().ele_eSuperClusterOverP.push_back(ele->eSuperClusterOverP());
            tauTuple().ele_eSeedClusterOverP.push_back(ele->eSeedClusterOverP());
            tauTuple().ele_eSeedClusterOverPout.push_back(ele->eSeedClusterOverPout());
            tauTuple().ele_eEleClusterOverPout.push_back(ele->eEleClusterOverPout());
            tauTuple().ele_deltaEtaSuperClusterTrackAtVtx.push_back(ele->deltaEtaSuperClusterTrackAtVtx());
            tauTuple().ele_deltaEtaSeedClusterTrackAtCalo.push_back(ele->deltaEtaSeedClusterTrackAtCalo());
            tauTuple().ele_deltaEtaEleClusterTrackAtCalo.push_back(ele->deltaEtaEleClusterTrackAtCalo());
            tauTuple().ele_deltaEtaSeedClusterTrackAtVtx.push_back(ele->deltaEtaSeedClusterTrackAtVtx());
            tauTuple().ele_deltaPhiEleClusterTrackAtCalo.push_back(ele->deltaPhiEleClusterTrackAtCalo());
            tauTuple().ele_deltaPhiSuperClusterTrackAtVtx.push_back(ele->deltaPhiSuperClusterTrackAtVtx());
            tauTuple().ele_deltaPhiSeedClusterTrackAtCalo.push_back(ele->deltaPhiSeedClusterTrackAtCalo());

            const bool isHGCAL = ele->hasUserFloat("hgcElectronID:sigmaUU");
            tauTuple().ele_mvaInput_earlyBrem.push_back(!isHGCAL ? ele->mvaInput().earlyBrem : default_int_value);
            tauTuple().ele_mvaInput_lateBrem.push_back(!isHGCAL ? ele->mvaInput().lateBrem : default_int_value);
            tauTuple().ele_mvaInput_sigmaEtaEta.push_back(!isHGCAL ? ele->mvaInput().sigmaEtaEta : default_value);
            tauTuple().ele_mvaInput_hadEnergy.push_back(!isHGCAL ? ele->mvaInput().hadEnergy : default_value);
            tauTuple().ele_mvaInput_deltaEta.push_back(!isHGCAL ? ele->mvaInput().deltaEta : default_value);

            // shower shape variables are available for non-HGCal electrons with pt > 5
            const bool hasShapeVars = !isHGCAL && ele->polarP4().pt() > 5;
            tauTuple().ele_sigmaIetaIphi.push_back(hasShapeVars ? ele->sigmaIetaIphi() : default_value);
            tauTuple().ele_sigmaEtaEta.push_back(hasShapeVars ? ele->sigmaEtaEta() : default_value);
            tauTuple().ele_sigmaIetaIeta.push_back(hasShapeVars ? ele->sigmaIetaIeta() : default_value);
            tauTuple().ele_sigmaIphiIphi.push_back(hasShapeVars ? ele->sigmaIphiIphi() : default_value);
            tauTuple().ele_e1x5.push_back(hasShapeVars ? ele->e1x5() : default_value);
            tauTuple().ele_e2x5Max.push_back(hasShapeVars ? ele->e2x5Max() : default_value);
            tauTuple().ele_e5x5.push_back(hasShapeVars ? ele->e5x5() : default_value);
            tauTuple().ele_r9.push_back(hasShapeVars ? ele->r9() : default_value);
            tauTuple().ele_hcalDepth1OverEcal.push_back(hasShapeVars ? GetElecVer<pat::Electron, GetCMSSWVersion()>::hcalDepth1OverEcal(ele) : default_value);
            tauTuple().ele_hcalDepth2OverEcal.push_back(hasShapeVars ? GetElecVer<pat::Electron, GetCMSSWVersion()>::hcalDepth2OverEcal(ele) : default_value);
            tauTuple().ele_hcalDepth1OverEcalBc.push_back(hasShapeVars ? GetElecVer<pat::Electron, GetCMSSWVersion()>::hcalDepth1OverEcalBc(ele) : default_value);
            tauTuple().ele_hcalDepth2OverEcalBc.push_back(hasShapeVars ? GetElecVer<pat::Electron, GetCMSSWVersion()>::hcalDepth2OverEcalBc(ele) : default_value);
            tauTuple().ele_eLeft.push_back(hasShapeVars ? ele->eLeft() : default_value);
            tauTuple().ele_eRight.push_back(hasShapeVars ? ele->eRight() : default_value);
            tauTuple().ele_eTop.push_back(hasShapeVars ? ele->eTop() : default_value);
            tauTuple().ele_eBottom.push_back(hasShapeVars ? ele->eBottom() : default_value);

            tauTuple().ele_full5x5_sigmaEtaEta.push_back(hasShapeVars ? ele->full5x5_sigmaEtaEta() : default_value);
            tauTuple().ele_full5x5_sigmaIetaIeta.push_back(hasShapeVars ? ele->full5x5_sigmaIetaIeta() : default_value);
            tauTuple().ele_full5x5_sigmaIphiIphi.push_back(hasShapeVars ? ele->full5x5_sigmaIphiIphi() : default_value);
            tauTuple().ele_full5x5_sigmaIetaIphi.push_back(hasShapeVars ? ele->full5x5_sigmaIetaIphi() : default_value);
            tauTuple().ele_full5x5_e1x5.push_back(hasShapeVars ? ele->full5x5_e1x5() : default_value);
            tauTuple().ele_full5x5_e2x5Max.push_back(hasShapeVars ? ele->full5x5_e2x5Max() : default_value);
            tauTuple().ele_full5x5_e5x5.push_back(hasShapeVars ? ele->full5x5_e5x5() : default_value);
            tauTuple().ele_full5x5_r9.push_back(hasShapeVars ? ele->full5x5_r9() : default_value);
            tauTuple().ele_full5x5_hcalDepth1OverEcal.push_back(hasShapeVars ? GetElecVer<pat::Electron, GetCMSSWVersion()>::full5x5_hcalDepth1OverEcal(ele) : default_value);
            tauTuple().ele_full5x5_hcalDepth2OverEcal.push_back(hasShapeVars ? GetElecVer<pat::Electron, GetCMSSWVersion()>::full5x5_hcalDepth2OverEcal(ele) : default_value);
            tauTuple().ele_full5x5_hcalDepth1OverEcalBc.push_back(hasShapeVars ? GetElecVer<pat::Electron, GetCMSSWVersion()>::full5x5_hcalDepth1OverEcalBc(ele) : default_value);
            tauTuple().ele_full5x5_hcalDepth2OverEcalBc.push_back(hasShapeVars ? GetElecVer<pat::Electron, GetCMSSWVersion()>::full5x5_hcalDepth2OverEcalBc(ele) : default_value);
            tauTuple().ele_full5x5_eLeft.push_back(hasShapeVars ? ele->full5x5_eLeft() : default_value);
            tauTuple().ele_full5x5_eRight.push_back(hasShapeVars ? ele->full5x5_eRight() : default_value);
            tauTuple().ele_full5x5_eTop.push_back(hasShapeVars ? ele->full5x5_eTop() : default_value);
            tauTuple().ele_full5x5_eBottom.push_back(hasShapeVars ? ele->full5x5_eBottom() : default_value);
            tauTuple().ele_full5x5_e2x5Left.push_back(hasShapeVars ? ele->full5x5_e2x5Left() : default_value);
            tauTuple().ele_full5x5_e2x5Right.push_back(hasShapeVars ? ele->full5x5_e2x5Right() : default_value);
            tauTuple().ele_full5x5_e2x5Top.push_back(hasShapeVars ? ele->full5x5_e2x5Top() : default_value);
            tauTuple().ele_full5x5_e2x5Bottom.push_back(hasShapeVars ? ele->full5x5_e2x5Bottom() : default_value);

            // Only phase2 electrons with !ele->isEB() obtain a value != default_{int_,}value via hasUserFloat() decision
            const auto fillUserFloat = [&ele](std::vector<float>& output, const std::string& name) {
                output.push_back(ele->hasUserFloat(name) ? ele->userFloat(name) : default_value);
            };

            const auto fillUserInt = [&ele](std::vector<int>& output, const std::string& name) {
                output.push_back(ele->hasUserFloat(name) ? static_cast<int>(ele->userFloat(name)) : default_int_value);
            };

            fillUserFloat(tauTuple().ele_hgcal_sigmaUU, "hgcElectronID:sigmaUU");
            fillUserFloat(tauTuple().ele_hgcal_sigmaVV, "hgcElectronID:sigmaVV");
            fillUserFloat(tauTuple().ele_hgcal_sigmaEE, "hgcElectronID:sigmaEE");
            fillUserFloat(tauTuple().ele_hgcal_sigmaPP, "hgcElectronID:sigmaPP");
            fillUserInt(tauTuple().ele_hgcal_nLayers, "hgcElectronID:nLayers");
            fillUserInt(tauTuple().ele_hgcal_firstLayer, "hgcElectronID:firstLayer");
            fillUserInt(tauTuple().ele_hgcal_lastLayer, "hgcElectronID:lastLayer");
            fillUserInt(tauTuple().ele_hgcal_layerEfrac10, "hgcElectronID:layerEfrac10");
            fillUserInt(tauTuple().ele_hgcal_layerEfrac90, "hgcElectronID:layerEfrac90");
            fillUserFloat(tauTuple().ele_hgcal_e4oEtot, "hgcElectronID:e4oEtot");
            fillUserFloat(tauTuple().ele_hgcal_ecEnergy, "hgcElectronID:ecEnergy");
            fillUserFloat(tauTuple().ele_hgcal_ecEnergyEE, "hgcElectronID:ecEnergyEE");
            fillUserFloat(tauTuple().ele_hgcal_ecEnergyBH, "hgcElectronID:ecEnergyBH");
            fillUserFloat(tauTuple().ele_hgcal_ecEnergyFH, "hgcElectronID:ecEnergyFH");
            fillUserFloat(tauTuple().ele_hgcal_ecEt, "hgcElectronID:ecEt");
            fillUserFloat(tauTuple().ele_hgcal_ecOrigEnergy, "hgcElectronID:ecOrigEnergy");
            fillUserFloat(tauTuple().ele_hgcal_ecOrigEt, "hgcElectronID:ecOrigEt");
            fillUserFloat(tauTuple().ele_hgcal_caloIsoRing0, "hgcElectronID:caloIsoRing0");
            fillUserFloat(tauTuple().ele_hgcal_caloIsoRing1, "hgcElectronID:caloIsoRing1");
            fillUserFloat(tauTuple().ele_hgcal_caloIsoRing2, "hgcElectronID:caloIsoRing2");
            fillUserFloat(tauTuple().ele_hgcal_caloIsoRing3, "hgcElectronID:caloIsoRing3");
            fillUserFloat(tauTuple().ele_hgcal_caloIsoRing4, "hgcElectronID:caloIsoRing4");
            fillUserFloat(tauTuple().ele_hgcal_depthCompatibility, "hgcElectronID:depthCompatibility");
            fillUserFloat(tauTuple().ele_hgcal_expectedDepth, "hgcElectronID:expectedDepth");
            fillUserFloat(tauTuple().ele_hgcal_expectedSigma, "hgcElectronID:expectedSigma");
            fillUserFloat(tauTuple().ele_hgcal_measuredDepth, "hgcElectronID:measuredDepth");
            fillUserFloat(tauTuple().ele_hgcal_pcaAxisX, "hgcElectronID:pcaAxisX");
            fillUserFloat(tauTuple().ele_hgcal_pcaAxisY, "hgcElectronID:pcaAxisY");
            fillUserFloat(tauTuple().ele_hgcal_pcaAxisZ, "hgcElectronID:pcaAxisZ");
            fillUserFloat(tauTuple().ele_hgcal_pcaPositionX, "hgcElectronID:pcaPositionX");
            fillUserFloat(tauTuple().ele_hgcal_pcaPositionY, "hgcElectronID:pcaPositionY");
            fillUserFloat(tauTuple().ele_hgcal_pcaPositionZ, "hgcElectronID:pcaPositionZ");
            fillUserFloat(tauTuple().ele_hgcal_pcaEig1, "hgcElectronID:pcaEig1");
            fillUserFloat(tauTuple().ele_hgcal_pcaEig2, "hgcElectronID:pcaEig2");
            fillUserFloat(tauTuple().ele_hgcal_pcaEig3, "hgcElectronID:pcaEig3");
            fillUserFloat(tauTuple().ele_hgcal_pcaSig1, "hgcElectronID:pcaSig1");
            fillUserFloat(tauTuple().ele_hgcal_pcaSig2, "hgcElectronID:pcaSig2");
            fillUserFloat(tauTuple().ele_hgcal_pcaSig3, "hgcElectronID:pcaSig3");

            const auto& gsfTrack = ele->gsfTrack();
            tauTuple().ele_gsfTrack_normalizedChi2.push_back(
                        gsfTrack.isNonnull() ? static_cast<float>(gsfTrack->normalizedChi2()) : default_value);
            tauTuple().ele_gsfTrack_numberOfValidHits.push_back(
                        gsfTrack.isNonnull() ? gsfTrack->numberOfValidHits() : default_int_value);
            tauTuple().ele_gsfTrack_pt.push_back(
                        gsfTrack.isNonnull() ? static_cast<float>(gsfTrack->pt()) : default_value);
            tauTuple().ele_gsfTrack_pt_error.push_back(
                        gsfTrack.isNonnull() ? static_cast<float>(gsfTrack->ptError()) : default_value);

            const auto& closestCtfTrack = ele->closestCtfTrackRef();
            tauTuple().ele_closestCtfTrack_normalizedChi2.push_back(closestCtfTrack.isNonnull()
                        ? static_cast<float>(closestCtfTrack->normalizedChi2()) : default_value);
            tauTuple().ele_closestCtfTrack_numberOfValidHits.push_back(
                        closestCtfTrack.isNonnull() ? closestCtfTrack->numberOfValidHits() : default_int_value);
        }
    }

    void FillPhotons(const TauJet::PhotonCollection& photons)
    {
        auto fillShape = [&](const pat::Photon*& gamma, const std::string& prefix) {
	    if(prefix ==  "photon_shape_") {
            	tauTuple.get<std::vector<float>>(prefix + "sigmaEtaEta").push_back(gamma->showerShapeVariables().sigmaEtaEta);
	    	tauTuple.get<std::vector<float>>(prefix + "sigmaIetaIeta").push_back(gamma->showerShapeVariables().sigmaIetaIeta);
            	tauTuple.get<std::vector<float>>(prefix + "e1x5").push_back(gamma->showerShapeVariables().e1x5);
            	tauTuple.get<std::vector<float>>(prefix + "e2x5").push_back(gamma->showerShapeVariables().e2x5);
            	tauTuple.get<std::vector<float>>(prefix + "e3x3").push_back(gamma->showerShapeVariables().e3x3);
            	tauTuple.get<std::vector<float>>(prefix + "e5x5").push_back(gamma->showerShapeVariables().e5x5);
            	tauTuple.get<std::vector<float>>(prefix + "maxEnergyXtal").push_back(gamma->showerShapeVariables().maxEnergyXtal);
            	tauTuple.get<std::vector<float>>(prefix + "hcalDepth1OverEcal").push_back(gamma->showerShapeVariables().hcalDepth1OverEcal);
            	tauTuple.get<std::vector<float>>(prefix + "hcalDepth2OverEcal").push_back(gamma->showerShapeVariables().hcalDepth2OverEcal);
            	tauTuple.get<std::vector<float>>(prefix + "hcalDepth1OverEcalBc").push_back(gamma->showerShapeVariables().hcalDepth1OverEcalBc);
            	tauTuple.get<std::vector<float>>(prefix + "hcalDepth2OverEcalBc").push_back(gamma->showerShapeVariables().hcalDepth2OverEcalBc);
            	tauTuple.get<std::vector<float>>(prefix + "effSigmaRR").push_back(gamma->showerShapeVariables().effSigmaRR);
            	tauTuple.get<std::vector<float>>(prefix + "sigmaIetaIphi").push_back(gamma->showerShapeVariables().sigmaIetaIphi);
            	tauTuple.get<std::vector<float>>(prefix + "sigmaIphiIphi").push_back(gamma->showerShapeVariables().sigmaIphiIphi);
            	tauTuple.get<std::vector<float>>(prefix + "e2nd").push_back(gamma->showerShapeVariables().e2nd);
           	tauTuple.get<std::vector<float>>(prefix + "eTop").push_back(gamma->showerShapeVariables().eTop);
            	tauTuple.get<std::vector<float>>(prefix + "eLeft").push_back(gamma->showerShapeVariables().eLeft);
            	tauTuple.get<std::vector<float>>(prefix + "eRight").push_back(gamma->showerShapeVariables().eRight);
            	tauTuple.get<std::vector<float>>(prefix + "eBottom").push_back(gamma->showerShapeVariables().eBottom);
            	tauTuple.get<std::vector<float>>(prefix + "e1x3").push_back(gamma->showerShapeVariables().e1x3);
            	tauTuple.get<std::vector<float>>(prefix + "e2x2").push_back(gamma->showerShapeVariables().e2x2);
            	tauTuple.get<std::vector<float>>(prefix + "e2x5Max").push_back(gamma->showerShapeVariables().e2x5Max);
            	tauTuple.get<std::vector<float>>(prefix + "e2x5Left").push_back(gamma->showerShapeVariables().e2x5Left);
            	tauTuple.get<std::vector<float>>(prefix + "e2x5Right").push_back(gamma->showerShapeVariables().e2x5Right);
            	tauTuple.get<std::vector<float>>(prefix + "e2x5Top").push_back(gamma->showerShapeVariables().e2x5Top);
            	tauTuple.get<std::vector<float>>(prefix + "e2x5Bottom").push_back(gamma->showerShapeVariables().e2x5Bottom);
            	tauTuple.get<std::vector<float>>(prefix + "smMajor").push_back(gamma->showerShapeVariables().smMajor);
            	tauTuple.get<std::vector<float>>(prefix + "smMinor").push_back(gamma->showerShapeVariables().smMinor);
            	tauTuple.get<std::vector<float>>(prefix + "smAlpha").push_back(gamma->showerShapeVariables().smAlpha);	
	    }
            if(prefix ==  "photon_full5x5shape_") {
                tauTuple.get<std::vector<float>>(prefix + "sigmaEtaEta").push_back(gamma->full5x5_showerShapeVariables().sigmaEtaEta);
                tauTuple.get<std::vector<float>>(prefix + "sigmaIetaIeta").push_back(gamma->full5x5_showerShapeVariables().sigmaIetaIeta);
                tauTuple.get<std::vector<float>>(prefix + "e1x5").push_back(gamma->full5x5_showerShapeVariables().e1x5);
                tauTuple.get<std::vector<float>>(prefix + "e2x5").push_back(gamma->full5x5_showerShapeVariables().e2x5);
                tauTuple.get<std::vector<float>>(prefix + "e3x3").push_back(gamma->full5x5_showerShapeVariables().e3x3);
                tauTuple.get<std::vector<float>>(prefix + "e5x5").push_back(gamma->full5x5_showerShapeVariables().e5x5);
                tauTuple.get<std::vector<float>>(prefix + "maxEnergyXtal").push_back(gamma->full5x5_showerShapeVariables().maxEnergyXtal);
                tauTuple.get<std::vector<float>>(prefix + "hcalDepth1OverEcal").push_back(gamma->full5x5_showerShapeVariables().hcalDepth1OverEcal);
                tauTuple.get<std::vector<float>>(prefix + "hcalDepth2OverEcal").push_back(gamma->full5x5_showerShapeVariables().hcalDepth2OverEcal);
                tauTuple.get<std::vector<float>>(prefix + "hcalDepth1OverEcalBc").push_back(gamma->full5x5_showerShapeVariables().hcalDepth1OverEcalBc);
                tauTuple.get<std::vector<float>>(prefix + "hcalDepth2OverEcalBc").push_back(gamma->full5x5_showerShapeVariables().hcalDepth2OverEcalBc);
                tauTuple.get<std::vector<float>>(prefix + "effSigmaRR").push_back(gamma->full5x5_showerShapeVariables().effSigmaRR);
                tauTuple.get<std::vector<float>>(prefix + "sigmaIetaIphi").push_back(gamma->full5x5_showerShapeVariables().sigmaIetaIphi);
                tauTuple.get<std::vector<float>>(prefix + "sigmaIphiIphi").push_back(gamma->full5x5_showerShapeVariables().sigmaIphiIphi);
                tauTuple.get<std::vector<float>>(prefix + "e2nd").push_back(gamma->full5x5_showerShapeVariables().e2nd);
                tauTuple.get<std::vector<float>>(prefix + "eTop").push_back(gamma->full5x5_showerShapeVariables().eTop);
                tauTuple.get<std::vector<float>>(prefix + "eLeft").push_back(gamma->full5x5_showerShapeVariables().eLeft);
                tauTuple.get<std::vector<float>>(prefix + "eRight").push_back(gamma->full5x5_showerShapeVariables().eRight);
                tauTuple.get<std::vector<float>>(prefix + "eBottom").push_back(gamma->full5x5_showerShapeVariables().eBottom);
                tauTuple.get<std::vector<float>>(prefix + "e1x3").push_back(gamma->full5x5_showerShapeVariables().e1x3);
                tauTuple.get<std::vector<float>>(prefix + "e2x2").push_back(gamma->full5x5_showerShapeVariables().e2x2);
                tauTuple.get<std::vector<float>>(prefix + "e2x5Max").push_back(gamma->full5x5_showerShapeVariables().e2x5Max);
                tauTuple.get<std::vector<float>>(prefix + "e2x5Left").push_back(gamma->full5x5_showerShapeVariables().e2x5Left);
                tauTuple.get<std::vector<float>>(prefix + "e2x5Right").push_back(gamma->full5x5_showerShapeVariables().e2x5Right);
                tauTuple.get<std::vector<float>>(prefix + "e2x5Top").push_back(gamma->full5x5_showerShapeVariables().e2x5Top);
                tauTuple.get<std::vector<float>>(prefix + "e2x5Bottom").push_back(gamma->full5x5_showerShapeVariables().e2x5Bottom);
                tauTuple.get<std::vector<float>>(prefix + "smMajor").push_back(gamma->full5x5_showerShapeVariables().smMajor);
                tauTuple.get<std::vector<float>>(prefix + "smMinor").push_back(gamma->full5x5_showerShapeVariables().smMinor);
                tauTuple.get<std::vector<float>>(prefix + "smAlpha").push_back(gamma->full5x5_showerShapeVariables().smAlpha);
            }
	};

        for(const auto& photon_ptr : photons) {
            const pat::Photon* photon = photon_ptr.obj;	    
            tauTuple().photon_index.push_back(photon_ptr.index);
            tauTuple().photon_pt.push_back(static_cast<float>(photon->polarP4().pt()));
            tauTuple().photon_eta.push_back(static_cast<float>(photon->polarP4().eta()));
            tauTuple().photon_phi.push_back(static_cast<float>(photon->polarP4().phi()));
            tauTuple().photon_energy.push_back(static_cast<float>(photon->polarP4().energy()));
	    tauTuple().photon_passElectronVeto.push_back(photon->passElectronVeto());
            tauTuple().photon_hasPixelSeed.push_back(photon->hasPixelSeed());
	    tauTuple().photon_eMax.push_back(photon->eMax());
            tauTuple().photon_e2nd.push_back(photon->e2nd());
            tauTuple().photon_e3x3.push_back(photon->e3x3());
            tauTuple().photon_eTop.push_back(photon->eTop());
            tauTuple().photon_eBottom.push_back(photon->eBottom());
            tauTuple().photon_eLeft.push_back(photon->eLeft());
            tauTuple().photon_eRight.push_back(photon->eRight());
            tauTuple().photon_see.push_back(photon->see());
            tauTuple().photon_spp.push_back(photon->spp());
            tauTuple().photon_sep.push_back(photon->sep());
            tauTuple().photon_maxDR.push_back(photon->maxDR());
            tauTuple().photon_maxDRDPhi.push_back(photon->maxDRDPhi());
            tauTuple().photon_maxDRDEta.push_back(photon->maxDRDEta());
            tauTuple().photon_maxDRRawEnergy.push_back(photon->maxDRRawEnergy());
            tauTuple().photon_subClusRawE1.push_back(photon->subClusRawE1());
            tauTuple().photon_subClusRawE2.push_back(photon->subClusRawE2());
            tauTuple().photon_subClusRawE3.push_back(photon->subClusRawE3());
            tauTuple().photon_subClusDPhi1.push_back(photon->subClusDPhi1());
            tauTuple().photon_subClusDPhi2.push_back(photon->subClusDPhi2());
            tauTuple().photon_subClusDPhi3.push_back(photon->subClusDPhi3());
            tauTuple().photon_subClusDEta1.push_back(photon->subClusDEta1());
            tauTuple().photon_subClusDEta2.push_back(photon->subClusDEta2());
            tauTuple().photon_subClusDEta3.push_back(photon->subClusDEta3());
            tauTuple().photon_cryPhi.push_back(photon->cryPhi());
            tauTuple().photon_cryEta.push_back(photon->cryEta());
            tauTuple().photon_iPhi.push_back(photon->iPhi());
            tauTuple().photon_iEta.push_back(photon->iEta());
	    //
	    fillShape(photon, "photon_shape_");
            fillShape(photon, "photon_full5x5shape_");
        }
    }

    void FillMuons(const TauJet::MuonCollection& muons, const reco::Vertex& PV)
    {
        for(const auto& muon_ptr : muons) {
            const pat::Muon* muon = muon_ptr.obj;
            tauTuple().muon_index.push_back(muon_ptr.index);
            tauTuple().muon_pt.push_back(static_cast<float>(muon->polarP4().pt()));
            tauTuple().muon_eta.push_back(static_cast<float>(muon->polarP4().eta()));
            tauTuple().muon_phi.push_back(static_cast<float>(muon->polarP4().phi()));
            tauTuple().muon_mass.push_back(static_cast<float>(muon->polarP4().mass()));
            tauTuple().muon_dxy.push_back(static_cast<float>(muon->dB(pat::Muon::PV2D)));
            tauTuple().muon_dxy_error.push_back(static_cast<float>(muon->edB(pat::Muon::PV2D)));
            tauTuple().muon_normalizedChi2.push_back(
                muon->globalTrack().isNonnull() ? static_cast<float>(muon->normChi2()) : default_value);
            tauTuple().muon_numberOfValidHits.push_back(
                muon->innerTrack().isNonnull() ? static_cast<int>(muon->numberOfValidHits()) : default_value);
            tauTuple().muon_segmentCompatibility.push_back(static_cast<float>(muon->segmentCompatibility()));
            tauTuple().muon_caloCompatibility.push_back(muon->caloCompatibility());
            tauTuple().muon_pfEcalEnergy.push_back(muon->pfEcalEnergy());
            tauTuple().muon_type.push_back(muon->type());
            tauTuple().muon_id.push_back(unsigned(muon->isLooseMuon()) * 1 + unsigned(muon->isMediumMuon()) * 2
                                         + unsigned(muon->isTightMuon(PV)) * 4);
            tauTuple().muon_pfRelIso04.push_back(static_cast<float>(PFRelIsolation(*muon)));

            const MuonHitMatch hit_match(*muon);
            for(int subdet : MuonHitMatch::ConsideredSubdets()) {
                const std::string& subdetName = MuonHitMatch::SubdetName(subdet);
                for(int station = MuonHitMatch::first_station_id; station <= MuonHitMatch::last_station_id; ++station) {
                    const std::string matches_branch_name = "muon_n_matches_" + subdetName + "_"
                            + std::to_string(station);
                    const std::string hits_branch_name = "muon_n_hits_" + subdetName + "_" + std::to_string(station);

                    const unsigned n_matches = hit_match.NMatches(subdet, station);
                    const unsigned n_hits = hit_match.NHits(subdet, station);
                    tauTuple.get<std::vector<int>>(matches_branch_name).push_back(static_cast<int>(n_matches));
                    tauTuple.get<std::vector<int>>(hits_branch_name).push_back(static_cast<int>(n_hits));
                }
            }
        }
    }

    void FillIsoTracks(const TauJet::IsoTrackCollection& tracks)
    {
        for(const auto& track_ptr : tracks) {
          const pat::IsolatedTrack* track = track_ptr.obj;
          tauTuple().isoTrack_index.push_back(track_ptr.index);
          tauTuple().isoTrack_pt.push_back(static_cast<float>(track->polarP4().pt()));
          tauTuple().isoTrack_eta.push_back(static_cast<float>(track->polarP4().eta()));
          tauTuple().isoTrack_phi.push_back(static_cast<float>(track->polarP4().phi()));
          tauTuple().isoTrack_fromPV.push_back(track->fromPV());
          tauTuple().isoTrack_charge.push_back(track->charge());
          tauTuple().isoTrack_dxy.push_back(track->dxy());
          tauTuple().isoTrack_dxy_error.push_back(track->dxyError());
          tauTuple().isoTrack_dz.push_back(track->dz());
          tauTuple().isoTrack_dz_error.push_back(track->dzError());
          tauTuple().isoTrack_isHighPurityTrack.push_back(track->isHighPurityTrack());
          tauTuple().isoTrack_isTightTrack.push_back(track->isTightTrack());
          tauTuple().isoTrack_isLooseTrack.push_back(track->isLooseTrack());
          tauTuple().isoTrack_dEdxStrip.push_back(track->dEdxStrip());
          tauTuple().isoTrack_dEdxPixel.push_back(track->dEdxPixel());
          tauTuple().isoTrack_deltaEta.push_back(track->deltaEta());
          tauTuple().isoTrack_deltaPhi.push_back(track->deltaPhi());

          const auto& hitPattern = track->hitPattern();
          tauTuple().isoTrack_n_ValidHits.push_back(hitPattern.numberOfValidHits());
          tauTuple().isoTrack_n_InactiveHits.push_back(hitPattern.numberOfInactiveHits());
          tauTuple().isoTrack_n_ValidPixelHits.push_back(hitPattern.numberOfValidPixelHits());
          tauTuple().isoTrack_n_ValidStripHits.push_back(hitPattern.numberOfValidStripHits());

          tauTuple().isoTrack_n_MuonHits.push_back(hitPattern.numberOfMuonHits());
          tauTuple().isoTrack_n_BadHits.push_back(hitPattern.numberOfBadHits());
          tauTuple().isoTrack_n_BadMuonHits.push_back(hitPattern.numberOfBadMuonHits());
          tauTuple().isoTrack_n_BadMuonDTHits.push_back(hitPattern.numberOfBadMuonDTHits());
          tauTuple().isoTrack_n_BadMuonCSCHits.push_back(hitPattern.numberOfBadMuonCSCHits());
          tauTuple().isoTrack_n_BadMuonRPCHits.push_back(hitPattern.numberOfBadMuonRPCHits());
          tauTuple().isoTrack_n_BadMuonGEMHits.push_back(hitPattern.numberOfBadMuonGEMHits());
          tauTuple().isoTrack_n_BadMuonME0Hits.push_back(hitPattern.numberOfBadMuonME0Hits());
          tauTuple().isoTrack_n_ValidMuonHits.push_back(hitPattern.numberOfValidMuonHits());
          tauTuple().isoTrack_n_ValidMuonDTHits.push_back(hitPattern.numberOfValidMuonDTHits());
          tauTuple().isoTrack_n_ValidMuonCSCHits.push_back(hitPattern.numberOfValidMuonCSCHits());
          tauTuple().isoTrack_n_ValidMuonRPCHits.push_back(hitPattern.numberOfValidMuonRPCHits());
          tauTuple().isoTrack_n_ValidMuonGEMHits.push_back(hitPattern.numberOfValidMuonGEMHits());
          tauTuple().isoTrack_n_ValidMuonME0Hits.push_back(hitPattern.numberOfValidMuonME0Hits());
          tauTuple().isoTrack_n_LostMuonHits.push_back(hitPattern.numberOfLostMuonHits());
          tauTuple().isoTrack_n_LostMuonDTHits.push_back(hitPattern.numberOfLostMuonDTHits());
          tauTuple().isoTrack_n_LostMuonCSCHits.push_back(hitPattern.numberOfLostMuonCSCHits());
          tauTuple().isoTrack_n_LostMuonRPCHits.push_back(hitPattern.numberOfLostMuonRPCHits());
          tauTuple().isoTrack_n_LostMuonGEMHits.push_back(hitPattern.numberOfLostMuonGEMHits());
          tauTuple().isoTrack_n_LostMuonME0Hits.push_back(hitPattern.numberOfLostMuonME0Hits());

          tauTuple().isoTrack_n_TimingHits.push_back(hitPattern.numberOfTimingHits());
          tauTuple().isoTrack_n_ValidTimingHits.push_back(hitPattern.numberOfValidTimingHits());
          tauTuple().isoTrack_n_LostTimingHits.push_back(hitPattern.numberOfLostTimingHits());

          using ctgr = reco::HitPattern;
          tauTuple().isoTrack_n_AllHits_TRACK.push_back(hitPattern.numberOfAllHits(ctgr::TRACK_HITS));
          tauTuple().isoTrack_n_AllHits_MISSING_INNER.push_back(hitPattern.numberOfAllHits(ctgr::MISSING_INNER_HITS));
          tauTuple().isoTrack_n_AllHits_MISSING_OUTER.push_back(hitPattern.numberOfAllHits(ctgr::MISSING_OUTER_HITS));
          tauTuple().isoTrack_n_LostHits_TRACK.push_back(hitPattern.numberOfLostHits(ctgr::TRACK_HITS));
          tauTuple().isoTrack_n_LostHits_MISSING_INNER.push_back(hitPattern.numberOfLostHits(ctgr::MISSING_INNER_HITS));
          tauTuple().isoTrack_n_LostHits_MISSING_OUTER.push_back(hitPattern.numberOfLostHits(ctgr::MISSING_OUTER_HITS));
          tauTuple().isoTrack_n_LostPixelHits_TRACK.push_back(hitPattern.numberOfLostPixelHits(ctgr::TRACK_HITS));
          tauTuple().isoTrack_n_LostPixelHits_MISSING_INNER.push_back(hitPattern.numberOfLostPixelHits(ctgr::MISSING_INNER_HITS));
          tauTuple().isoTrack_n_LostPixelHits_MISSING_OUTER.push_back(hitPattern.numberOfLostPixelHits(ctgr::MISSING_OUTER_HITS));
          tauTuple().isoTrack_n_LostStripHits_TRACK.push_back(hitPattern.numberOfLostStripHits(ctgr::TRACK_HITS));
          tauTuple().isoTrack_n_LostStripHits_MISSING_INNER.push_back(hitPattern.numberOfLostStripHits(ctgr::MISSING_INNER_HITS));
          tauTuple().isoTrack_n_LostStripHits_MISSING_OUTER.push_back(hitPattern.numberOfLostStripHits(ctgr::MISSING_OUTER_HITS));
        }
    }

    void FillSV(const TauJet::SVCollection sv)
    {
        for(const auto& sv_ptr : sv) {
	   const reco::VertexCompositePtrCandidate* SV = sv_ptr.obj;
           tauTuple().sv_cands_svIdx.push_back(sv_ptr.index);
	   tauTuple().sv_x.push_back(static_cast<float>(SV->position().x()));
           tauTuple().sv_y.push_back(static_cast<float>(SV->position().y()));
           tauTuple().sv_z.push_back(static_cast<float>(SV->position().z()));
           tauTuple().sv_t.push_back(static_cast<float>(SV->t()));
	   tauTuple().sv_xE.push_back(static_cast<float>(std::sqrt(SV->vertexCovariance(0,0))));
           tauTuple().sv_yE.push_back(static_cast<float>(std::sqrt(SV->vertexCovariance(1,1))));
           tauTuple().sv_zE.push_back(static_cast<float>(std::sqrt(SV->vertexCovariance(2,2))));
           tauTuple().sv_tE.push_back(static_cast<float>(SV->tError()));
	   tauTuple().sv_chi2.push_back(static_cast<float>(SV->vertexChi2()));
           tauTuple().sv_ndof.push_back(static_cast<float>(SV->vertexNdof()));
        }
        tauTuple().nsv = sv.size();
    }

    static float CalculateGottfriedJacksonAngleDifference(const pat::Tau& tau)
    {
        double gj_diff;
        if(::tau_analysis::CalculateGottfriedJacksonAngleDifference(tau, gj_diff))
            return static_cast<float>(gj_diff);
        return default_value;
    }

    static void CalculateElectronClusterVars(const pat::Electron& ele, float& cc_ele_energy, float& cc_gamma_energy,
                                             int& cc_n_gamma)
    {
        cc_ele_energy = cc_gamma_energy = 0;
        cc_n_gamma = 0;
        const auto& superCluster = ele.superCluster();
        if(superCluster.isNonnull() && superCluster.isAvailable() && superCluster->clusters().isNonnull()
                && superCluster->clusters().isAvailable()) {
            for(auto iter = superCluster->clustersBegin(); iter != superCluster->clustersEnd(); ++iter) {
                const float energy = static_cast<float>((*iter)->energy());
                if(iter == superCluster->clustersBegin())
                    cc_ele_energy += energy;
                else {
                    cc_gamma_energy += energy;
                    ++cc_n_gamma;
                }
            }
        } else {
            cc_ele_energy = cc_gamma_energy = default_value;
            cc_n_gamma = default_int_value;
        }
    }

private:
    const bool isMC, isEmbedded, requireGenMatch, requireGenORRecoTauMatch, applyRecoPtSieve;
    TauJetBuilderSetup builderSetup;

    edm::EDGetTokenT<GenEventInfoProduct> genEvent_token;
    edm::EDGetTokenT<reco::GenParticleCollection> genParticles_token;
    edm::EDGetTokenT<reco::GenJetCollection> genJets_token;
    edm::EDGetTokenT<reco::JetFlavourInfoMatchingCollection> genJetFlavourInfos_token;
    edm::EDGetTokenT<std::vector<PileupSummaryInfo>> puInfo_token;
    edm::EDGetTokenT<std::vector<reco::Vertex>> vertices_token;
    edm::EDGetTokenT<std::vector<reco::VertexCompositePtrCandidate>> secondVertices_token;
    edm::EDGetTokenT<double> rho_token;
    edm::EDGetTokenT<pat::ElectronCollection> electrons_token;
    edm::EDGetTokenT<pat::PhotonCollection> photons_token;
    edm::EDGetTokenT<pat::MuonCollection> muons_token;
    edm::EDGetTokenT<pat::TauCollection> taus_token, boostedTaus_token;
    edm::EDGetTokenT<pat::JetCollection> jets_token, fatJets_token;
    edm::EDGetTokenT<pat::PackedCandidateCollection> cands_token;
    edm::EDGetTokenT<pat::IsolatedTrackCollection> isoTracks_token;
    edm::EDGetTokenT<pat::PackedCandidateCollection> lostTracks_token;
    edm::EDGetTokenT<edm::View<pat::MET>> METs_token;
    edm::EDGetTokenT<edm::View<pat::MET>> puppiMETs_token;
    edm::EDGetTokenT<edm::View<pat::MET>> deepMETs_token;
    edm::EDGetTokenT<edm::View<reco::GenMET>> genMETs_token;
    edm::EDGetTokenT<edm::TriggerResults> triggerResults_token;
    edm::EDGetTokenT<pat::TriggerObjectStandAloneCollection> triggerObjects_token;
    edm::EDGetTokenT<double> tauSpinnerWTEven_token;
    edm::EDGetTokenT<double> tauSpinnerWTOdd_token;
    edm::EDGetTokenT<double> tauSpinnerWTMM_token;

    TauTupleProducerData* data;
    tau_tuple::TauTuple& tauTuple;
    tau_tuple::SummaryTuple& summaryTuple;
    std::shared_ptr<selectors::TauJetSelector> selector;
};

} // namespace tau_analysis

#include "FWCore/Framework/interface/MakerMacros.h"
using TauTupleProducer = tau_analysis::TauTupleProducer;
DEFINE_FWK_MODULE(TauTupleProducer);
