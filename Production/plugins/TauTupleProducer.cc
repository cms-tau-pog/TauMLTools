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

#include "SimDataFormats/GeneratorProducts/interface/GenEventInfoProduct.h"
#include "AnalysisDataFormats/TopObjects/interface/TtGenEvent.h"
#include "CommonTools/UtilAlgos/interface/TFileService.h"
#include "RecoTauTag/RecoTau/interface/PFRecoTauClusterVariables.h"

#include "AnalysisTools/Core/include/Tools.h"
#include "AnalysisTools/Core/include/TextIO.h"
#include "TauML/Analysis/include/TauTuple.h"
#include "TauML/Analysis/include/SummaryTuple.h"
#include "TauML/Analysis/include/TauIdResults.h"
#include "TauML/Production/include/GenTruthTools.h"
#include "TauML/Production/include/TauAnalysis.h"
#include "TauML/Production/include/MuonHitMatch.h"
#include "TauML/Production/include/TauJet.h"

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
        storeJetsWithoutTau(cfg.getParameter<bool>("storeJetsWithoutTau")),
        requireGenMatch(cfg.getParameter<bool>("requireGenMatch")),
        genEvent_token(mayConsume<GenEventInfoProduct>(cfg.getParameter<edm::InputTag>("genEvent"))),
        genParticles_token(mayConsume<std::vector<reco::GenParticle>>(cfg.getParameter<edm::InputTag>("genParticles"))),
        puInfo_token(mayConsume<std::vector<PileupSummaryInfo>>(cfg.getParameter<edm::InputTag>("puInfo"))),
        vertices_token(consumes<std::vector<reco::Vertex> >(cfg.getParameter<edm::InputTag>("vertices"))),
        rho_token(consumes<double>(cfg.getParameter<edm::InputTag>("rho"))),
        electrons_token(consumes<pat::ElectronCollection>(cfg.getParameter<edm::InputTag>("electrons"))),
        muons_token(consumes<pat::MuonCollection>(cfg.getParameter<edm::InputTag>("muons"))),
        taus_token(consumes<pat::TauCollection>(cfg.getParameter<edm::InputTag>("taus"))),
        jets_token(consumes<pat::JetCollection>(cfg.getParameter<edm::InputTag>("jets"))),
        cands_token(consumes<pat::PackedCandidateCollection>(cfg.getParameter<edm::InputTag>("pfCandidates"))),
        data(TauTupleProducerData::RequestGlobalData()),
        tauTuple(data->tauTuple),
        summaryTuple(data->summaryTuple)
    {
        builderSetup.minJetPt = cfg.getParameter<double>("minJetPt");
        builderSetup.maxJetEta = cfg.getParameter<double>("maxJetEta");
        builderSetup.forceTauJetMatch = cfg.getParameter<bool>("forceTauJetMatch");
        builderSetup.useOnlyTauObjectMatch = !storeJetsWithoutTau;
        builderSetup.tauJetMatchDeltaR2Threshold = std::pow(cfg.getParameter<double>("tauJetMatchDeltaRThreshold"), 2);
        builderSetup.objectMatchDeltaR2ThresholdJet =
                std::pow(cfg.getParameter<double>("objectMatchDeltaRThresholdJet"), 2);
        builderSetup.objectMatchDeltaR2ThresholdTau =
                std::pow(cfg.getParameter<double>("objectMatchDeltaRThresholdTau"), 2);
    }

private:
    static constexpr float default_value = tau_tuple::DefaultFillValue<float>();
    static constexpr int default_int_value = tau_tuple::DefaultFillValue<int>();

    virtual void analyze(const edm::Event& event, const edm::EventSetup&) override
    {
        static const TauIdMVAAuxiliaries clusterVariables;

        std::lock_guard<std::mutex> lock(data->mutex);
        summaryTuple().numberOfProcessedEvents++;

        tauTuple().run  = event.id().run();
        tauTuple().lumi = event.id().luminosityBlock();
        tauTuple().evt  = event.id().event();

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
        tauTuple().pv_chi2 = static_cast<float>(PV.chi2());
        tauTuple().pv_ndof = static_cast<float>(PV.ndof());

        edm::Handle<pat::ElectronCollection> electrons;
        event.getByToken(electrons_token, electrons);

        edm::Handle<pat::MuonCollection> muons;
        event.getByToken(muons_token, muons);

        edm::Handle<pat::TauCollection> taus;
        event.getByToken(taus_token, taus);

        edm::Handle<pat::JetCollection> jets;
        event.getByToken(jets_token, jets);

        edm::Handle<pat::PackedCandidateCollection> cands;
        event.getByToken(cands_token, cands);

        edm::Handle<std::vector<reco::GenParticle>> hGenParticles;
        if(isMC)
            event.getByToken(genParticles_token, hGenParticles);

        auto genParticles = hGenParticles.isValid() ? hGenParticles.product() : nullptr;

        TauJetBuilder builder(builderSetup, *jets, *taus, *cands, *electrons, *muons, genParticles);
        const auto tauJets = builder.Build();

        for(const TauJet& tauJet : tauJets) {
            const bool has_jet = tauJet.jetIndex >= 0;
            const bool has_tau = tauJet.tauIndex >= 0;
            if(!has_tau && !storeJetsWithoutTau) continue;

            const auto& leptonGenMatch = has_tau ? tauJet.tauGenLeptonMatchResult : tauJet.jetGenLeptonMatchResult;
            const auto& qcdGenMatch = has_tau ? tauJet.tauGenQcdMatchResult : tauJet.jetGenQcdMatchResult;

            if(requireGenMatch && leptonGenMatch.match == GenLeptonMatch::NoMatch
                    && qcdGenMatch.match == GenQcdMatch::NoMatch) continue;

            tauTuple().jet_index = tauJet.jetIndex;
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
            const reco::GenJet* genJet = has_jet ? tauJet.jet->genJet() : nullptr;
            tauTuple().jet_has_gen_match = genJet != nullptr;
            tauTuple().jet_gen_pt = genJet != nullptr ? static_cast<float>(genJet->polarP4().pt()) : default_value;
            tauTuple().jet_gen_eta = genJet != nullptr ? static_cast<float>(genJet->polarP4().eta()) : default_value;
            tauTuple().jet_gen_phi = genJet != nullptr ? static_cast<float>(genJet->polarP4().phi()) : default_value;
            tauTuple().jet_gen_mass = genJet != nullptr ? static_cast<float>(genJet->polarP4().mass()) : default_value;
            tauTuple().jet_gen_n_b = genJet != nullptr
                    ? static_cast<int>(tauJet.jet->jetFlavourInfo().getbHadrons().size()) : default_int_value;
            tauTuple().jet_gen_n_c = genJet != nullptr
                    ? static_cast<int>(tauJet.jet->jetFlavourInfo().getcHadrons().size()) : default_int_value;


            const pat::Tau* tau = tauJet.tau;
            if(has_tau) {
                static const bool id_names_printed = PrintTauIdNames(*tau);
                (void)id_names_printed;
            }

            tauTuple().jetTauMatch = static_cast<int>(tauJet.jetTauMatch);
            tauTuple().tau_index = tauJet.tauIndex;
            tauTuple().tau_pt = has_tau ? static_cast<float>(tau->polarP4().pt()) : default_value;
            tauTuple().tau_eta = has_tau ? static_cast<float>(tau->polarP4().eta()) : default_value;
            tauTuple().tau_phi = has_tau ? static_cast<float>(tau->polarP4().phi()) : default_value;
            tauTuple().tau_mass = has_tau ? static_cast<float>(tau->polarP4().mass()) : default_value;
            tauTuple().tau_charge = has_tau ? tau->charge() : default_int_value;

            FillGenMatchResult(leptonGenMatch, qcdGenMatch);

            tauTuple().tau_decayMode = has_tau ? tau->decayMode() : default_int_value;
            tauTuple().tau_decayModeFinding = has_tau ? tau->tauID("decayModeFinding") > 0.5f : default_int_value;
            tauTuple().tau_decayModeFindingNewDMs = has_tau ? tau->tauID("decayModeFindingNewDMs") > 0.5f
                                                            : default_int_value;
            tauTuple().chargedIsoPtSum = has_tau ? tau->tauID("chargedIsoPtSum") : default_value;
            tauTuple().chargedIsoPtSumdR03 = has_tau ? tau->tauID("chargedIsoPtSumdR03") : default_value;
            tauTuple().footprintCorrection = has_tau ? tau->tauID("footprintCorrection") : default_value;
            tauTuple().footprintCorrectiondR03 = has_tau ? tau->tauID("footprintCorrectiondR03") : default_value;
            tauTuple().neutralIsoPtSum = has_tau ? tau->tauID("neutralIsoPtSum") : default_value;
            tauTuple().neutralIsoPtSumWeight = has_tau ? tau->tauID("neutralIsoPtSumWeight") : default_value;
            tauTuple().neutralIsoPtSumWeightdR03 = has_tau ? tau->tauID("neutralIsoPtSumWeightdR03") : default_value;
            tauTuple().neutralIsoPtSumdR03 = has_tau ? tau->tauID("neutralIsoPtSumdR03") : default_value;
            tauTuple().photonPtSumOutsideSignalCone = has_tau ? tau->tauID("photonPtSumOutsideSignalCone")
                                                              : default_value;
            tauTuple().photonPtSumOutsideSignalConedR03 = has_tau ? tau->tauID("photonPtSumOutsideSignalConedR03")
                                                                  : default_value;
            tauTuple().puCorrPtSum = has_tau ? tau->tauID("puCorrPtSum") : default_value;

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
                    has_tau ? clusterVariables.tau_pt_weighted_deta_strip(*tau, tau->decayMode()) : default_value;
            tauTuple().tau_pt_weighted_dphi_strip =
                    has_tau ? clusterVariables.tau_pt_weighted_dphi_strip(*tau, tau->decayMode()) : default_value;
            tauTuple().tau_pt_weighted_dr_signal =
                    has_tau ? clusterVariables.tau_pt_weighted_dr_signal(*tau, tau->decayMode()) : default_value;
            tauTuple().tau_pt_weighted_dr_iso =
                    has_tau ? clusterVariables.tau_pt_weighted_dr_iso(*tau, tau->decayMode()) : default_value;
            tauTuple().tau_leadingTrackNormChi2 = has_tau ? tau->leadingTrackNormChi2() : default_value;
            tauTuple().tau_e_ratio = has_tau ? clusterVariables.tau_Eratio(*tau) : default_value;
            tauTuple().tau_gj_angle_diff = has_tau ? CalculateGottfriedJacksonAngleDifference(*tau) : default_value;
            tauTuple().tau_n_photons =
                    has_tau ? static_cast<int>(clusterVariables.tau_n_photons_total(*tau)) : default_int_value;

            tauTuple().tau_emFraction = has_tau ? tau->emFraction_MVA() : default_value;
            tauTuple().tau_inside_ecal_crack = has_tau ? IsInEcalCrack(tau->p4().Eta()) : default_value;
            tauTuple().leadChargedCand_etaAtEcalEntrance =
                    has_tau ? tau->etaAtEcalEntranceLeadChargedCand() : default_value;

            FillPFCandidates(tauJet.cands);
            FillElectrons(tauJet.electrons);
            FillMuons(tauJet.muons);

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

    void FillGenMatchResult(const gen_truth::LeptonMatchResult& leptonMatch, const gen_truth::QcdMatchResult& qcdMatch)
    {
        const bool has_lepton = leptonMatch.match != GenLeptonMatch::NoMatch;
        tauTuple().lepton_gen_match = static_cast<int>(leptonMatch.match);
        tauTuple().lepton_gen_charge = has_lepton ? leptonMatch.gen_particle->charge() : default_int_value;
        tauTuple().lepton_gen_pt = has_lepton ? static_cast<float>(leptonMatch.gen_particle->polarP4().pt())
                                              : default_value;
        tauTuple().lepton_gen_eta = has_lepton ? static_cast<float>(leptonMatch.gen_particle->polarP4().eta())
                                               : default_value;
        tauTuple().lepton_gen_phi = has_lepton ? static_cast<float>(leptonMatch.gen_particle->polarP4().phi())
                                               : default_value;
        tauTuple().lepton_gen_mass = has_lepton ? static_cast<float>(leptonMatch.gen_particle->polarP4().mass())
                                                : default_value;
        for(auto daughter : leptonMatch.visible_daughters) {
            tauTuple().lepton_gen_vis_pdg.push_back(daughter->pdgId());
            tauTuple().lepton_gen_vis_pt.push_back(static_cast<float>(daughter->polarP4().pt()));
            tauTuple().lepton_gen_vis_eta.push_back(static_cast<float>(daughter->polarP4().eta()));
            tauTuple().lepton_gen_vis_phi.push_back(static_cast<float>(daughter->polarP4().phi()));
            tauTuple().lepton_gen_vis_mass.push_back(static_cast<float>(daughter->polarP4().mass()));
        }

        const bool has_qcd = qcdMatch.match != GenQcdMatch::NoMatch;
        tauTuple().qcd_gen_match = static_cast<int>(qcdMatch.match);
        tauTuple().qcd_gen_charge = has_qcd ? qcdMatch.gen_particle->charge() : default_int_value;
        tauTuple().qcd_gen_pt = has_qcd ? static_cast<float>(qcdMatch.gen_particle->polarP4().pt()) : default_value;
        tauTuple().qcd_gen_eta = has_qcd ? static_cast<float>(qcdMatch.gen_particle->polarP4().eta()) : default_value;
        tauTuple().qcd_gen_phi = has_qcd ? static_cast<float>(qcdMatch.gen_particle->polarP4().phi()) : default_value;
        tauTuple().qcd_gen_mass = has_qcd ? static_cast<float>(qcdMatch.gen_particle->polarP4().mass()) : default_value;
    }

    void FillPFCandidates(const std::vector<PFCandDesc>& cands)
    {
        for(const PFCandDesc& cand_desc : cands) {
            const pat::PackedCandidate* cand = cand_desc.candidate;

            tauTuple().pfCand_jetDaughter.push_back(cand_desc.jetDaughter);
            tauTuple().pfCand_tauSignal.push_back(cand_desc.tauSignal);
            tauTuple().pfCand_leadChargedHadrCand.push_back(cand_desc.leadChargedHadrCand);
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

            tauTuple().pfCand_vertex_x.push_back(static_cast<float>(cand->vertex().x()));
            tauTuple().pfCand_vertex_y.push_back(static_cast<float>(cand->vertex().y()));
            tauTuple().pfCand_vertex_z.push_back(static_cast<float>(cand->vertex().z()));

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
            tauTuple().pfCand_rawCaloFraction.push_back(cand->rawCaloFraction());
        }
    }

    void FillElectrons(const std::vector<const pat::Electron*>& electrons)
    {
        for(const pat::Electron* ele : electrons) {
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
            tauTuple().ele_ecalEnergy.push_back(ele->ecalEnergy());
            tauTuple().ele_ecalEnergy_error.push_back(ele->ecalEnergyError());
            tauTuple().ele_eSuperClusterOverP.push_back(ele->eSuperClusterOverP());
            tauTuple().ele_eSeedClusterOverP.push_back(ele->eSeedClusterOverP());
            tauTuple().ele_eSeedClusterOverPout.push_back(ele->eSeedClusterOverPout());
            tauTuple().ele_eEleClusterOverPout.push_back(ele->eEleClusterOverPout());
            tauTuple().ele_deltaEtaSuperClusterTrackAtVtx.push_back(ele->deltaEtaSuperClusterTrackAtVtx());
            tauTuple().ele_deltaEtaSeedClusterTrackAtCalo.push_back(ele->deltaEtaSeedClusterTrackAtCalo());
            tauTuple().ele_deltaEtaEleClusterTrackAtCalo.push_back(ele->deltaEtaEleClusterTrackAtCalo());
            tauTuple().ele_deltaPhiEleClusterTrackAtCalo.push_back(ele->deltaPhiEleClusterTrackAtCalo());
            tauTuple().ele_deltaPhiSuperClusterTrackAtVtx.push_back(ele->deltaPhiSuperClusterTrackAtVtx());
            tauTuple().ele_deltaPhiSeedClusterTrackAtCalo.push_back(ele->deltaPhiSeedClusterTrackAtCalo());

            tauTuple().ele_mvaInput_earlyBrem.push_back(ele->mvaInput().earlyBrem);
            tauTuple().ele_mvaInput_lateBrem.push_back(ele->mvaInput().lateBrem);
            tauTuple().ele_mvaInput_sigmaEtaEta.push_back(ele->mvaInput().sigmaEtaEta);
            tauTuple().ele_mvaInput_hadEnergy.push_back(ele->mvaInput().hadEnergy);
            tauTuple().ele_mvaInput_deltaEta.push_back(ele->mvaInput().deltaEta);

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

    void FillMuons(const std::vector<const pat::Muon*>& muons)
    {
        for(const pat::Muon* muon : muons) {
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
    const bool isMC, storeJetsWithoutTau, requireGenMatch;
    TauJetBuilderSetup builderSetup;

    edm::EDGetTokenT<GenEventInfoProduct> genEvent_token;
    edm::EDGetTokenT<std::vector<reco::GenParticle>> genParticles_token;
    edm::EDGetTokenT<std::vector<PileupSummaryInfo>> puInfo_token;
    edm::EDGetTokenT<std::vector<reco::Vertex>> vertices_token;
    edm::EDGetTokenT<double> rho_token;
    edm::EDGetTokenT<pat::ElectronCollection> electrons_token;
    edm::EDGetTokenT<pat::MuonCollection> muons_token;
    edm::EDGetTokenT<pat::TauCollection> taus_token;
    edm::EDGetTokenT<pat::JetCollection> jets_token;
    edm::EDGetTokenT<pat::PackedCandidateCollection> cands_token;

    TauTupleProducerData* data;
    tau_tuple::TauTuple& tauTuple;
    tau_tuple::SummaryTuple& summaryTuple;
};

} // namespace tau_analysis

#include "FWCore/Framework/interface/MakerMacros.h"
using TauTupleProducer = tau_analysis::TauTupleProducer;
DEFINE_FWK_MODULE(TauTupleProducer);
