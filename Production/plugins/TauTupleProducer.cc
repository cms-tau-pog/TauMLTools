/*! Creates tuple for tau analysis.
*/

#include "Compression.h"

#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ServiceRegistry/interface/Service.h"

#include "DataFormats/PatCandidates/interface/Electron.h"
#include "DataFormats/PatCandidates/interface/Muon.h"
#include "DataFormats/PatCandidates/interface/Tau.h"
#include "DataFormats/PatCandidates/interface/Jet.h"
#include "DataFormats/PatCandidates/interface/PackedCandidate.h"
#include "DataFormats/PatCandidates/interface/IsolatedTrack.h"
#include "DataFormats/TrackReco/interface/HitPattern.h"

#include "SimDataFormats/GeneratorProducts/interface/GenEventInfoProduct.h"
#include "AnalysisDataFormats/TopObjects/interface/TtGenEvent.h"
#include "CommonTools/UtilAlgos/interface/TFileService.h"
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

namespace tau_analysis {

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
        file.SetCompressionAlgorithm(ROOT::kZLIB);
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
        storeJetsWithoutTau(cfg.getParameter<bool>("storeJetsWithoutTau")),
        requireGenMatch(cfg.getParameter<bool>("requireGenMatch")),
        genEvent_token(mayConsume<GenEventInfoProduct>(cfg.getParameter<edm::InputTag>("genEvent"))),
        genParticles_token(mayConsume<reco::GenParticleCollection>(cfg.getParameter<edm::InputTag>("genParticles"))),
        genJets_token(mayConsume<reco::GenJetCollection>(cfg.getParameter<edm::InputTag>("genJets"))),
        puInfo_token(mayConsume<std::vector<PileupSummaryInfo>>(cfg.getParameter<edm::InputTag>("puInfo"))),
        vertices_token(consumes<std::vector<reco::Vertex> >(cfg.getParameter<edm::InputTag>("vertices"))),
        rho_token(consumes<double>(cfg.getParameter<edm::InputTag>("rho"))),
        electrons_token(consumes<pat::ElectronCollection>(cfg.getParameter<edm::InputTag>("electrons"))),
        muons_token(consumes<pat::MuonCollection>(cfg.getParameter<edm::InputTag>("muons"))),
        taus_token(consumes<pat::TauCollection>(cfg.getParameter<edm::InputTag>("taus"))),
        boostedTaus_token(consumes<pat::TauCollection>(cfg.getParameter<edm::InputTag>("boostedTaus"))),
        jets_token(consumes<pat::JetCollection>(cfg.getParameter<edm::InputTag>("jets"))),
        fatJets_token(consumes<pat::JetCollection>(cfg.getParameter<edm::InputTag>("fatJets"))),
        cands_token(consumes<pat::PackedCandidateCollection>(cfg.getParameter<edm::InputTag>("pfCandidates"))),
        isoTracks_token(consumes<pat::IsolatedTrackCollection>(cfg.getParameter<edm::InputTag>("isoTracks"))),
        lostTracks_token(consumes<pat::PackedCandidateCollection>(cfg.getParameter<edm::InputTag>("lostTracks"))),
        data(TauTupleProducerData::RequestGlobalData()),
        tauTuple(data->tauTuple),
        summaryTuple(data->summaryTuple)
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

    virtual void analyze(const edm::Event& event, const edm::EventSetup&) override
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

        edm::Handle<std::vector<reco::Vertex>> vertices;
        event.getByToken(vertices_token, vertices);
        tauTuple().npv = static_cast<int>(vertices->size());
        edm::Handle<double> rho;
        event.getByToken(rho_token, rho);
        tauTuple().rho = static_cast<float>(*rho);

        if(isMC) {
            edm::Handle<GenEventInfoProduct> genEvent;
            event.getByToken(genEvent_token, genEvent);
            tauTuple().genEventWeight = static_cast<float>(genEvent->weight());

            edm::Handle<std::vector<PileupSummaryInfo>> puInfo;
            event.getByToken(puInfo_token, puInfo);
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

        edm::Handle<pat::ElectronCollection> electrons;
        event.getByToken(electrons_token, electrons);

        edm::Handle<pat::MuonCollection> muons;
        event.getByToken(muons_token, muons);

        edm::Handle<pat::TauCollection> taus;
        event.getByToken(taus_token, taus);

        edm::Handle<pat::TauCollection> boostedTaus;
        event.getByToken(boostedTaus_token, boostedTaus);

        edm::Handle<pat::JetCollection> jets;
        event.getByToken(jets_token, jets);

        edm::Handle<pat::JetCollection> fatJets;
        event.getByToken(fatJets_token, fatJets);

        edm::Handle<pat::PackedCandidateCollection> cands;
        event.getByToken(cands_token, cands);

        edm::Handle<pat::IsolatedTrackCollection> isoTracks;
        event.getByToken(isoTracks_token, isoTracks);

        edm::Handle<pat::PackedCandidateCollection> lostTracks;
        event.getByToken(lostTracks_token, lostTracks);

        edm::Handle<reco::GenParticleCollection> hGenParticles;
        edm::Handle<reco::GenJetCollection> hGenJets;
        if(isMC) {
            event.getByToken(genParticles_token, hGenParticles);
            event.getByToken(genJets_token, hGenJets);
        }

        auto genParticles = hGenParticles.isValid() ? hGenParticles.product() : nullptr;
        auto genJets = hGenJets.isValid() ? hGenJets.product() : nullptr;

        TauJetBuilder builder(builderSetup, *taus, *boostedTaus, *jets, *fatJets, *cands, *electrons, *muons,
                              *isoTracks, *lostTracks, genParticles, genJets, requireGenMatch);
        const auto& tauJets = builder.GetTauJets();

        for(const TauJet& tauJet : tauJets) {
            const bool has_jet = tauJet.jet;
            const bool has_tau = tauJet.tau;
            //if(!has_tau && !storeJetsWithoutTau) continue;

            // const auto& leptonGenMatch = has_tau ? tauJet.tauGenLeptonMatchResult : tauJet.jetGenLeptonMatchResult;
            // const auto& qcdGenMatch = has_tau ? tauJet.tauGenQcdMatchResult : tauJet.jetGenQcdMatchResult;

            tauTuple().jet_index = tauJet.jet.index;
            tauTuple().jet_pt = has_jet ? static_cast<float>(tauJet.jet->p4().pt()) : default_value;
            tauTuple().jet_eta = has_jet ? static_cast<float>(tauJet.jet->p4().eta()) : default_value;
            tauTuple().jet_phi = has_jet ? static_cast<float>(tauJet.jet->p4().phi()) : default_value;
            tauTuple().jet_mass = has_jet ? static_cast<float>(tauJet.jet->p4().mass()) : default_value;
            boost::optional<pat::Jet> uncorrected_jet;
            if(has_jet)
                uncorrected_jet = tauJet.jet->correctedJet("Uncorrected");
            tauTuple().jet_neutralHadronEnergyFraction = has_jet
                    ? uncorrected_jet->neutralHadronEnergyFraction() : default_value;
            tauTuple().jet_neutralEmEnergyFraction = has_jet
                    ? uncorrected_jet->neutralEmEnergyFraction() : default_value;
            tauTuple().jet_nConstituents = has_jet ? uncorrected_jet->nConstituents() : default_int_value;
            tauTuple().jet_chargedMultiplicity = has_jet ? uncorrected_jet->chargedMultiplicity() : default_int_value;
            tauTuple().jet_neutralMultiplicity = has_jet ? uncorrected_jet->neutralMultiplicity() : default_int_value;
            tauTuple().jet_partonFlavour = has_jet ? tauJet.jet->partonFlavour() : default_int_value;
            tauTuple().jet_hadronFlavour = has_jet ? tauJet.jet->hadronFlavour() : default_int_value;
            // const reco::GenJet* genJet = has_jet ? tauJet.jet->genJet() : nullptr;
            // tauTuple().jet_has_gen_match = genJet != nullptr;
            // tauTuple().jet_gen_pt = genJet != nullptr ? static_cast<float>(genJet->polarP4().pt()) : default_value;
            // tauTuple().jet_gen_eta = genJet != nullptr ? static_cast<float>(genJet->polarP4().eta()) : default_value;
            // tauTuple().jet_gen_phi = genJet != nullptr ? static_cast<float>(genJet->polarP4().phi()) : default_value;
            // tauTuple().jet_gen_mass = genJet != nullptr ? static_cast<float>(genJet->polarP4().mass()) : default_value;
            // tauTuple().jet_gen_n_b = genJet != nullptr
            //         ? static_cast<int>(tauJet.jet->jetFlavourInfo().getbHadrons().size()) : default_int_value;
            // tauTuple().jet_gen_n_c = genJet != nullptr
            //         ? static_cast<int>(tauJet.jet->jetFlavourInfo().getcHadrons().size()) : default_int_value;


            const pat::Tau* tau = tauJet.tau.obj;
            if(has_tau) {
                static const bool id_names_printed = PrintTauIdNames(*tau);
                (void)id_names_printed;
            }

            //tauTuple().jetTauMatch = static_cast<int>(tauJet.jetTauMatch);
            tauTuple().tau_index = tauJet.tau.index;
            tauTuple().tau_pt = has_tau ? static_cast<float>(tau->polarP4().pt()) : default_value;
            tauTuple().tau_eta = has_tau ? static_cast<float>(tau->polarP4().eta()) : default_value;
            tauTuple().tau_phi = has_tau ? static_cast<float>(tau->polarP4().phi()) : default_value;
            tauTuple().tau_mass = has_tau ? static_cast<float>(tau->polarP4().mass()) : default_value;
            tauTuple().tau_charge = has_tau ? tau->charge() : default_int_value;

            //FillGenMatchResult(leptonGenMatch, qcdGenMatch);

            tauTuple().tau_decayMode = has_tau ? tau->decayMode() : default_int_value;
            tauTuple().tau_decayModeFinding = has_tau ? tau->tauID("decayModeFinding") > 0.5f : default_int_value;
            tauTuple().tau_decayModeFindingNewDMs = has_tau ? tau->tauID("decayModeFindingNewDMs") > 0.5f
                                                            : default_int_value;
            tauTuple().tau_chargedIsoPtSum = has_tau ? tau->tauID("chargedIsoPtSum") : default_value;
            tauTuple().tau_chargedIsoPtSumdR03 = has_tau ? tau->tauID("chargedIsoPtSumdR03") : default_value;
            tauTuple().tau_footprintCorrection = has_tau ? tau->tauID("footprintCorrection") : default_value;
            tauTuple().tau_footprintCorrectiondR03 = has_tau ? tau->tauID("footprintCorrectiondR03") : default_value;
            tauTuple().tau_neutralIsoPtSum = has_tau ? tau->tauID("neutralIsoPtSum") : default_value;
            tauTuple().tau_neutralIsoPtSumWeight = has_tau ? tau->tauID("neutralIsoPtSumWeight") : default_value;
            tauTuple().tau_neutralIsoPtSumWeightdR03 = has_tau ? tau->tauID("neutralIsoPtSumWeightdR03") : default_value;
            tauTuple().tau_neutralIsoPtSumdR03 = has_tau ? tau->tauID("neutralIsoPtSumdR03") : default_value;
            tauTuple().tau_photonPtSumOutsideSignalCone = has_tau ? tau->tauID("photonPtSumOutsideSignalCone")
                                                              : default_value;
            tauTuple().tau_photonPtSumOutsideSignalConedR03 = has_tau ? tau->tauID("photonPtSumOutsideSignalConedR03")
                                                                  : default_value;
            tauTuple().tau_puCorrPtSum = has_tau ? tau->tauID("puCorrPtSum") : default_value;

            for(const auto& tau_id_entry : analysis::tau_id::GetTauIdDescriptors()) {
                const auto& desc = tau_id_entry.second;
                desc.FillTuple(tauTuple, tau, default_value);
            }

            tauTuple().tau_dxy_pca_x = has_tau ? tau->dxy_PCA().x() : default_value;
            tauTuple().tau_dxy_pca_y = has_tau ? tau->dxy_PCA().y() : default_value;
            tauTuple().tau_dxy_pca_z = has_tau ? tau->dxy_PCA().z() : default_value;
            tauTuple().tau_dxy = has_tau ? tau->dxy() : default_value;
            tauTuple().tau_dxy_error = has_tau ? tau->dxy_error() : default_value;
            tauTuple().tau_ip3d = has_tau ? tau->ip3d() : default_value;
            tauTuple().tau_ip3d_error = has_tau ? tau->ip3d_error() : default_value;
            const bool has_sv = has_tau && tau->hasSecondaryVertex();
            tauTuple().tau_hasSecondaryVertex = has_tau ? tau->hasSecondaryVertex() : default_int_value;
            tauTuple().tau_sv_x = has_sv ? tau->secondaryVertexPos().x() : default_value;
            tauTuple().tau_sv_y = has_sv ? tau->secondaryVertexPos().y() : default_value;
            tauTuple().tau_sv_z = has_sv ? tau->secondaryVertexPos().z() : default_value;
            tauTuple().tau_flightLength_x = has_tau ? tau->flightLength().x() : default_value;
            tauTuple().tau_flightLength_y = has_tau ? tau->flightLength().y() : default_value;
            tauTuple().tau_flightLength_z = has_tau ? tau->flightLength().z() : default_value;
            tauTuple().tau_flightLength_sig = has_tau ? tau->flightLengthSig() : default_value;

            const pat::PackedCandidate* leadChargedHadrCand =
                    has_tau ? dynamic_cast<const pat::PackedCandidate*>(tau->leadChargedHadrCand().get()) : nullptr;
            tauTuple().tau_dz = leadChargedHadrCand ? leadChargedHadrCand->dz() : default_value;
            tauTuple().tau_dz_error = leadChargedHadrCand && leadChargedHadrCand->hasTrackDetails()
                    ? leadChargedHadrCand->dzError() : default_value;

            tauTuple().tau_pt_weighted_deta_strip =
                    has_tau ? reco::tau::pt_weighted_deta_strip(*tau, tau->decayMode()) : default_value;
            tauTuple().tau_pt_weighted_dphi_strip =
                    has_tau ? reco::tau::pt_weighted_dphi_strip(*tau, tau->decayMode()) : default_value;
            tauTuple().tau_pt_weighted_dr_signal =
                    has_tau ? reco::tau::pt_weighted_dr_signal(*tau, tau->decayMode()) : default_value;
            tauTuple().tau_pt_weighted_dr_iso =
                    has_tau ? reco::tau::pt_weighted_dr_iso(*tau, tau->decayMode()) : default_value;
            tauTuple().tau_leadingTrackNormChi2 = has_tau ? tau->leadingTrackNormChi2() : default_value;
            tauTuple().tau_e_ratio = has_tau ? reco::tau::eratio(*tau) : default_value;
            tauTuple().tau_gj_angle_diff = has_tau ? CalculateGottfriedJacksonAngleDifference(*tau) : default_value;
            tauTuple().tau_n_photons =
                    has_tau ? static_cast<int>(reco::tau::n_photons_total(*tau)) : default_int_value;

            tauTuple().tau_emFraction = has_tau ? tau->emFraction_MVA() : default_value;
            tauTuple().tau_inside_ecal_crack = has_tau ? IsInEcalCrack(tau->p4().Eta()) : default_value;
            tauTuple().tau_leadChargedCand_etaAtEcalEntrance =
                    has_tau ? tau->etaAtEcalEntranceLeadChargedCand() : default_value;

            FillPFCandidates(tauJet.cands);
            FillElectrons(tauJet.electrons);
            FillMuons(tauJet.muons);
            FillIsoTracks(tauJet.isoTracks);

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

    // void FillGenMatchResult(const gen_truth::LeptonMatchResult& leptonMatch, const gen_truth::QcdMatchResult& qcdMatch)
    // {
    //     const bool has_lepton = leptonMatch.match != GenLeptonMatch::NoMatch;
    //     tauTuple().lepton_gen_match = static_cast<int>(leptonMatch.match);
    //     tauTuple().lepton_gen_charge = has_lepton ? leptonMatch.gen_particle_lastCopy->charge() : default_int_value;
    //     tauTuple().lepton_gen_pt = has_lepton ? static_cast<float>(leptonMatch.gen_particle_lastCopy->polarP4().pt())
    //                                           : default_value;
    //     tauTuple().lepton_gen_eta = has_lepton ? static_cast<float>(leptonMatch.gen_particle_lastCopy->polarP4().eta())
    //                                            : default_value;
    //     tauTuple().lepton_gen_phi = has_lepton ? static_cast<float>(leptonMatch.gen_particle_lastCopy->polarP4().phi())
    //                                            : default_value;
    //     tauTuple().lepton_gen_mass = has_lepton ? static_cast<float>(leptonMatch.gen_particle_lastCopy->polarP4().mass())
    //                                             : default_value;
    //
    //     tauTuple().lepton_gen_firstCopy_pt = has_lepton ? static_cast<float>(leptonMatch.gen_particle_firstCopy->polarP4().pt())
    //                                           : default_value;
    //     tauTuple().lepton_gen_firstCopy_eta = has_lepton ? static_cast<float>(leptonMatch.gen_particle_firstCopy->polarP4().eta())
    //                                            : default_value;
    //     tauTuple().lepton_gen_firstCopy_phi = has_lepton ? static_cast<float>(leptonMatch.gen_particle_firstCopy->polarP4().phi())
    //                                            : default_value;
    //     tauTuple().lepton_gen_firstCopy_mass = has_lepton ? static_cast<float>(leptonMatch.gen_particle_firstCopy->polarP4().mass())
    //                                             : default_value;
    //
    //     for(auto daughter : leptonMatch.visible_daughters) {
    //         tauTuple().lepton_gen_vis_pdg.push_back(daughter->pdgId());
    //         tauTuple().lepton_gen_vis_pt.push_back(static_cast<float>(daughter->polarP4().pt()));
    //         tauTuple().lepton_gen_vis_eta.push_back(static_cast<float>(daughter->polarP4().eta()));
    //         tauTuple().lepton_gen_vis_phi.push_back(static_cast<float>(daughter->polarP4().phi()));
    //         tauTuple().lepton_gen_vis_mass.push_back(static_cast<float>(daughter->polarP4().mass()));
    //     }
    //
    //     for(auto rad : leptonMatch.visible_rad) {
    //         tauTuple().lepton_gen_vis_rad_pdg.push_back(rad->pdgId());
    //         tauTuple().lepton_gen_vis_rad_pt.push_back(static_cast<float>(rad->polarP4().pt()));
    //         tauTuple().lepton_gen_vis_rad_eta.push_back(static_cast<float>(rad->polarP4().eta()));
    //         tauTuple().lepton_gen_vis_rad_phi.push_back(static_cast<float>(rad->polarP4().phi()));
    //         tauTuple().lepton_gen_vis_rad_mass.push_back(static_cast<float>(rad->polarP4().mass()));
    //     }
    //
    //     const bool has_qcd = qcdMatch.match != GenQcdMatch::NoMatch;
    //     tauTuple().qcd_gen_match = static_cast<int>(qcdMatch.match);
    //     tauTuple().qcd_gen_charge = has_qcd ? qcdMatch.gen_particle->charge() : default_int_value;
    //     tauTuple().qcd_gen_pt = has_qcd ? static_cast<float>(qcdMatch.gen_particle->polarP4().pt()) : default_value;
    //     tauTuple().qcd_gen_eta = has_qcd ? static_cast<float>(qcdMatch.gen_particle->polarP4().eta()) : default_value;
    //     tauTuple().qcd_gen_phi = has_qcd ? static_cast<float>(qcdMatch.gen_particle->polarP4().phi()) : default_value;
    //     tauTuple().qcd_gen_mass = has_qcd ? static_cast<float>(qcdMatch.gen_particle->polarP4().mass()) : default_value;
    // }

    void FillPFCandidates(const TauJet::PFCandCollection& cands)
    {
        for(const PFCandDesc& cand_desc : cands) {
            const pat::PackedCandidate* cand = cand_desc.candidate;

            tauTuple().pfCand_jetDaughter.push_back(cand_desc.jetDaughter);
            tauTuple().pfCand_tauSignal.push_back(cand_desc.tauSignal);
            tauTuple().pfCand_tauLeadChargedHadrCand.push_back(cand_desc.tauLeadChargedHadrCand);
            tauTuple().pfCand_tauIso.push_back(cand_desc.tauIso);

            tauTuple().pfCand_pt.push_back(static_cast<float>(cand->polarP4().pt()));
            tauTuple().pfCand_eta.push_back(static_cast<float>(cand->polarP4().eta()));
            tauTuple().pfCand_phi.push_back(static_cast<float>(cand->polarP4().phi()));
            tauTuple().pfCand_mass.push_back(static_cast<float>(cand->polarP4().mass()));

            tauTuple().pfCand_pvAssociationQuality.push_back(cand->pvAssociationQuality());
            tauTuple().pfCand_fromPV.push_back(cand->fromPV());
            tauTuple().pfCand_puppiWeight.push_back(cand->puppiWeight());
            tauTuple().pfCand_puppiWeightNoLep.push_back(cand->puppiWeightNoLep());
            tauTuple().pfCand_pdgId.push_back(cand->pdgId());
            tauTuple().pfCand_charge.push_back(cand->charge());
            tauTuple().pfCand_lostInnerHits.push_back(cand->lostInnerHits());
            tauTuple().pfCand_numberOfPixelHits.push_back(cand->numberOfPixelHits());
            tauTuple().pfCand_numberOfHits.push_back(cand->numberOfHits());

            tauTuple().pfCand_vertex_x.push_back(static_cast<float>(cand->vertex().x()));
            tauTuple().pfCand_vertex_y.push_back(static_cast<float>(cand->vertex().y()));
            tauTuple().pfCand_vertex_z.push_back(static_cast<float>(cand->vertex().z()));
            tauTuple().pfCand_vertex_t.push_back(static_cast<float>(cand->vertexRef()->t()));

            tauTuple().pfCand_time.push_back(cand->time());
            tauTuple().pfCand_timeError.push_back(cand->timeError());

            const bool hasTrackDetails = cand->hasTrackDetails();
            tauTuple().pfCand_hasTrackDetails.push_back(hasTrackDetails);
            tauTuple().pfCand_dxy.push_back(cand->dxy());
            tauTuple().pfCand_dxy_error.push_back(hasTrackDetails ? cand->dxyError() : default_value);
            tauTuple().pfCand_dz.push_back(cand->dz());
            tauTuple().pfCand_dz_error.push_back(hasTrackDetails ? cand->dzError() : default_value);
            tauTuple().pfCand_track_chi2.push_back(
                        hasTrackDetails ? static_cast<float>(cand->pseudoTrack().chi2()) : default_value);
            tauTuple().pfCand_track_ndof.push_back(
                        hasTrackDetails ? static_cast<float>(cand->pseudoTrack().ndof()) : default_value);

            tauTuple().pfCand_hcalFraction.push_back(cand->hcalFraction());
            tauTuple().pfCand_caloFraction.push_back(cand->caloFraction());
            tauTuple().pfCand_rawCaloFraction.push_back(cand->rawCaloFraction());
            tauTuple().pfCand_rawHcalFraction.push_back(cand->rawHcalFraction());
        }
    }

    void FillElectrons(const TauJet::ElectronCollection& electrons)
    {
        for(const auto& ele_ptr : electrons) {
            const pat::Electron* ele = ele_ptr.obj;
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
            tauTuple().ele_hcalDepth1OverEcal.push_back(hasShapeVars ? ele->hcalDepth1OverEcal() : default_value);
            tauTuple().ele_hcalDepth2OverEcal.push_back(hasShapeVars ? ele->hcalDepth2OverEcal() : default_value);
            tauTuple().ele_hcalDepth1OverEcalBc.push_back(hasShapeVars ? ele->hcalDepth1OverEcalBc() : default_value);
            tauTuple().ele_hcalDepth2OverEcalBc.push_back(hasShapeVars ? ele->hcalDepth2OverEcalBc() : default_value);
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
            tauTuple().ele_full5x5_hcalDepth1OverEcal.push_back(hasShapeVars ? ele->full5x5_hcalDepth1OverEcal() : default_value);
            tauTuple().ele_full5x5_hcalDepth2OverEcal.push_back(hasShapeVars ? ele->full5x5_hcalDepth2OverEcal() : default_value);
            tauTuple().ele_full5x5_hcalDepth1OverEcalBc.push_back(hasShapeVars ? ele->full5x5_hcalDepth1OverEcalBc() : default_value);
            tauTuple().ele_full5x5_hcalDepth2OverEcalBc.push_back(hasShapeVars ? ele->full5x5_hcalDepth2OverEcalBc() : default_value);
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

    void FillMuons(const TauJet::MuonCollection& muons)
    {
        for(const auto& muon_ptr : muons) {
            const pat::Muon* muon = muon_ptr.obj;
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
    const bool isMC, isEmbedded, storeJetsWithoutTau, requireGenMatch;
    TauJetBuilderSetup builderSetup;

    edm::EDGetTokenT<GenEventInfoProduct> genEvent_token;
    edm::EDGetTokenT<reco::GenParticleCollection> genParticles_token;
    edm::EDGetTokenT<reco::GenJetCollection> genJets_token;
    edm::EDGetTokenT<std::vector<PileupSummaryInfo>> puInfo_token;
    edm::EDGetTokenT<std::vector<reco::Vertex>> vertices_token;
    edm::EDGetTokenT<double> rho_token;
    edm::EDGetTokenT<pat::ElectronCollection> electrons_token;
    edm::EDGetTokenT<pat::MuonCollection> muons_token;
    edm::EDGetTokenT<pat::TauCollection> taus_token, boostedTaus_token;
    edm::EDGetTokenT<pat::JetCollection> jets_token, fatJets_token;
    edm::EDGetTokenT<pat::PackedCandidateCollection> cands_token;
    edm::EDGetTokenT<pat::IsolatedTrackCollection> isoTracks_token;
    edm::EDGetTokenT<pat::PackedCandidateCollection> lostTracks_token;


    TauTupleProducerData* data;
    tau_tuple::TauTuple& tauTuple;
    tau_tuple::SummaryTuple& summaryTuple;
};

} // namespace tau_analysis

#include "FWCore/Framework/interface/MakerMacros.h"
using TauTupleProducer = tau_analysis::TauTupleProducer;
DEFINE_FWK_MODULE(TauTupleProducer);
