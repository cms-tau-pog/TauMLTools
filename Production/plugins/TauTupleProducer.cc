/*! Creates tuple for tau analysis.
*/

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

class TauTupleProducer : public edm::EDAnalyzer {
public:
    using clock = std::chrono::system_clock;

    TauTupleProducer(const edm::ParameterSet& cfg) :
        start(clock::now()),
        isMC(cfg.getParameter<bool>("isMC")),
        genEvent_token(mayConsume<GenEventInfoProduct>(cfg.getParameter<edm::InputTag>("genEvent"))),
        genParticles_token(consumes<std::vector<reco::GenParticle>>(cfg.getParameter<edm::InputTag>("genParticles"))),
        puInfo_token(mayConsume<std::vector<PileupSummaryInfo>>(cfg.getParameter<edm::InputTag>("puInfo"))),
        vertices_token(mayConsume<std::vector<reco::Vertex> >(cfg.getParameter<edm::InputTag>("vertices"))),
        rho_token(consumes<double>(cfg.getParameter<edm::InputTag>("rho"))),
        electrons_token(mayConsume<pat::ElectronCollection>(cfg.getParameter<edm::InputTag>("electrons"))),
        muons_token(mayConsume<pat::MuonCollection>(cfg.getParameter<edm::InputTag>("muons"))),
        taus_token(mayConsume<pat::TauCollection>(cfg.getParameter<edm::InputTag>("taus"))),
        jets_token(mayConsume<pat::JetCollection>(cfg.getParameter<edm::InputTag>("jets"))),
        cands_token(mayConsume<pat::PackedCandidateCollection>(cfg.getParameter<edm::InputTag>("pfCandidates"))),
        tauTuple("taus", &edm::Service<TFileService>()->file(), false),
        summaryTuple("summary", &edm::Service<TFileService>()->file(), false)
    {
        summaryTuple().numberOfProcessedEvents = 0;
    }

private:
    static constexpr float default_value = tau_tuple::DefaultFillValue<float>();
    static constexpr int default_int_value = tau_tuple::DefaultFillValue<int>();

    virtual void analyze(const edm::Event& event, const edm::EventSetup&) override
    {
        static const TauIdMVAAuxiliaries clusterVariables;

        summaryTuple().numberOfProcessedEvents++;

        tauTuple().run  = event.id().run();
        tauTuple().lumi = event.id().luminosityBlock();
        tauTuple().evt  = event.id().event();

        edm::Handle<std::vector<reco::Vertex>> vertices;
        event.getByToken(vertices_token, vertices);
        tauTuple().npv = static_cast<unsigned>(vertices->size());
        edm::Handle<double> rho;
        event.getByToken(rho_token, rho);
        tauTuple().rho = static_cast<float>(*rho);

        if(isMC) {
            edm::Handle<GenEventInfoProduct> genEvent;
            event.getByToken(genEvent_token, genEvent);

            edm::Handle<std::vector<PileupSummaryInfo>> puInfo;
            event.getByToken(puInfo_token, puInfo);
            tauTuple().npu = gen_truth::GetNumberOfPileUpInteractions(puInfo);
            tauTuple().genEventWeight = static_cast<float>(genEvent->weight());
        }

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

        TauJetBuilder builder(*jets, *taus, *cands, genParticles);

        const auto tauJets = builder.Build();

        for(const TauJet& tauJet : tauJets) {
            tauTuple().jet_index = tauJet.jetIndex;
            tauTuple().jet_pt = static_cast<float>(tauJet.jet->p4().pt());
            tauTuple().jet_eta = static_cast<float>(tauJet.jet->p4().eta());
            tauTuple().jet_phi = static_cast<float>(tauJet.jet->p4().phi());
            tauTuple().jet_mass = static_cast<float>(tauJet.jet->p4().mass());

            const bool has_tau = tauJet.tauIndex >= 0;
            auto tau = tauJet.tau;
            if(has_tau) {
                static const bool id_names_printed = PrintTauIdNames(*tau);
                (void)id_names_printed;
            }


            tauTuple().tau_index = tauJet.tauIndex;
            tauTuple().tau_pt = has_tau ? static_cast<float>(tau->polarP4().pt()) : default_value;
            tauTuple().tau_eta = has_tau ? static_cast<float>(tau->polarP4().eta()) : default_value;
            tauTuple().tau_phi = has_tau ? static_cast<float>(tau->polarP4().phi()) : default_value;
            tauTuple().tau_mass = has_tau ? static_cast<float>(tau->polarP4().mass()) : default_value;
            tauTuple().tau_charge = has_tau ? tau->charge() : default_int_value;

            if(has_tau)
                FillGenMatchResult(tauJet.tauGenLeptonMatchResult, tauJet.tauGenQcdMatchResult);
            else
                FillGenMatchResult(tauJet.jetGenLeptonMatchResult, tauJet.jetGenQcdMatchResult);

            tauTuple().tau_decayMode = has_tau ? tau->decayMode() : default_int_value;
            tauTuple().tau_id_flags = has_tau ? CreateTauIdResults(*tau).GetResultBits() : 0;
            FillRawTauIds(tau);

            const pat::PackedCandidate* leadChargedHadrCand =
                    has_tau ? dynamic_cast<const pat::PackedCandidate*>(tau->leadChargedHadrCand().get()) : nullptr;
            tauTuple().tau_dxy = has_tau ? tau->dxy() : default_value;
            tauTuple().tau_dxy_sig = has_tau ? tau->dxy_Sig() : default_value;
            tauTuple().tau_dz = leadChargedHadrCand ? leadChargedHadrCand->dz() : default_value;
            tauTuple().tau_ip3d = has_tau ? tau->ip3d() : default_value;
            tauTuple().tau_ip3d_sig = has_tau ? tau->ip3d_Sig() : default_value;
            tauTuple().tau_hasSecondaryVertex = has_tau ? tau->hasSecondaryVertex() : default_int_value;
            tauTuple().tau_flightLength_r = has_tau ? tau->flightLength().R() : default_value;
            tauTuple().tau_flightLength_dEta = has_tau ? dEta(tau->flightLength(), tau->polarP4()) : default_value;
            tauTuple().tau_flightLength_dPhi = has_tau ? dPhi(tau->flightLength(), tau->polarP4()) : default_value;
            tauTuple().tau_flightLength_sig = has_tau ? tau->flightLengthSig() : default_value;

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
            tauTuple().tau_n_photons = has_tau ? clusterVariables.tau_n_photons_total(*tau) : default_value;

            tauTuple().tau_emFraction = has_tau ? tau->emFraction_MVA() : default_value;
            tauTuple().tau_inside_ecal_crack = has_tau ? IsInEcalCrack(tau->p4().Eta()) : default_value;

            tauTuple.Fill();
        }
    }

    virtual void endJob() override
    {
        tauTuple.Write();
        const auto stop = clock::now();
        summaryTuple().exeTime = static_cast<unsigned>(
                    std::chrono::duration_cast<std::chrono::seconds>(stop - start).count());
        summaryTuple.Fill();
        summaryTuple.Write();
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

    static analysis::TauIdResults CreateTauIdResults(const pat::Tau& tau)
    {
        analysis::TauIdResults results;
        const auto& descs = analysis::TauIdResults::GetResultDescriptors();
        for(size_t n = 0; n < descs.size(); ++n)
            results.SetResult(n, tau.tauID(descs.at(n).ToString()) > .5f);
        return results;
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

    void FillRawTauIds(const pat::Tau* tau)
    {
#define VAR(type, name) tauTuple().name = tau ? tau->tauID(#name) : default_value;
        RAW_TAU_IDS()
#undef VAR
    }
/*
    void FillExtendedVariables(const pat::Tau& tau, const pat::ElectronCollection& electrons,
                               const pat::MuonCollection& muons)
    {
        tauTuple().has_gsf_track = leadChargedHadrCand && std::abs(leadChargedHadrCand->pdgId()) == 11;
        auto gsf_ele = FindMatchedElectron(tau, electrons, 0.3);
        tauTuple().gsf_ele_matched = gsf_ele != nullptr;
        tauTuple().gsf_ele_pt = gsf_ele != nullptr ? gsf_ele->p4().Pt() : default_value;
        tauTuple().gsf_ele_dEta = gsf_ele != nullptr ? dEta(gsf_ele->p4(), tau.p4()) : default_value;
        tauTuple().gsf_ele_dPhi = gsf_ele != nullptr ? dPhi(gsf_ele->p4(), tau.p4()) : default_value;
        tauTuple().gsf_ele_energy = gsf_ele != nullptr ? gsf_ele->p4().E() : default_value;
        CalculateElectronClusterVars(gsf_ele, tauTuple().gsf_ele_Ee, tauTuple().gsf_ele_Egamma);
        tauTuple().gsf_ele_Pin = gsf_ele != nullptr ? gsf_ele->trackMomentumAtVtx().R() : default_value;
        tauTuple().gsf_ele_Pout = gsf_ele != nullptr ? gsf_ele->trackMomentumOut().R() : default_value;
        tauTuple().gsf_ele_Eecal = gsf_ele != nullptr ? gsf_ele->ecalEnergy() : default_value;
        tauTuple().gsf_ele_dEta_SeedClusterTrackAtCalo = gsf_ele != nullptr
                ? gsf_ele->deltaEtaSeedClusterTrackAtCalo() : default_value;
        tauTuple().gsf_ele_dPhi_SeedClusterTrackAtCalo = gsf_ele != nullptr
                ? gsf_ele->deltaPhiSeedClusterTrackAtCalo() : default_value;
        tauTuple().gsf_ele_mvaIn_sigmaEtaEta = gsf_ele != nullptr ? gsf_ele->mvaInput().sigmaEtaEta : default_value;
        tauTuple().gsf_ele_mvaIn_hadEnergy = gsf_ele != nullptr ? gsf_ele->mvaInput().hadEnergy : default_value;
        tauTuple().gsf_ele_mvaIn_deltaEta = gsf_ele != nullptr ? gsf_ele->mvaInput().deltaEta : default_value;
        tauTuple().gsf_ele_Chi2NormGSF = default_value;
        tauTuple().gsf_ele_GSFNumHits = default_value;
        tauTuple().gsf_ele_GSFTrackResol = default_value;
        tauTuple().gsf_ele_GSFTracklnPt = default_value;
        if(gsf_ele != nullptr && gsf_ele->gsfTrack().isNonnull()) {
            tauTuple().gsf_ele_Chi2NormGSF = gsf_ele->gsfTrack()->normalizedChi2();
            tauTuple().gsf_ele_GSFNumHits = gsf_ele->gsfTrack()->numberOfValidHits();
            if(gsf_ele->gsfTrack()->pt() > 0) {
                tauTuple().gsf_ele_GSFTrackResol = gsf_ele->gsfTrack()->ptError() / gsf_ele->gsfTrack()->pt();
                tauTuple().gsf_ele_GSFTracklnPt = std::log10(gsf_ele->gsfTrack()->pt());
            }
        }

        tauTuple().gsf_ele_Chi2NormKF = default_value;
        tauTuple().gsf_ele_KFNumHits = default_value;
        if(gsf_ele != nullptr && gsf_ele->closestCtfTrackRef().isNonnull()) {
            tauTuple().gsf_ele_Chi2NormKF = gsf_ele->closestCtfTrackRef()->normalizedChi2();
            tauTuple().gsf_ele_KFNumHits = gsf_ele->closestCtfTrackRef()->numberOfValidHits();
        }


        MuonHitMatch muon_hit_match;
        if(tau.leadPFChargedHadrCand().isNonnull() && tau.leadPFChargedHadrCand()->muonRef().isNonnull())
            muon_hit_match.AddMatchedMuon(*tau.leadPFChargedHadrCand()->muonRef(), tau);

        auto matched_muons = muon_hit_match.FindMatchedMuons(tau, muons, 0.3, 5);
        for(auto muon : matched_muons)
            muon_hit_match.AddMatchedMuon(*muon, tau);
        muon_hit_match.FillTuple(tauTuple(), tau);
    }
*/
    static float CalculateGottfriedJacksonAngleDifference(const pat::Tau& tau)
    {
        double gj_diff;
        if(::tau_analysis::CalculateGottfriedJacksonAngleDifference(tau, gj_diff))
            return static_cast<float>(gj_diff);
        return default_value;
    }

    static void CalculateEtaPhiAtEcalEntrance(const pat::Tau& tau, float& eta, float& phi)
    {
        float sumEtaTimesEnergy = 0., sumPhiTimesEnergy = 0., sumEnergy = 0.;
        for(const auto& pfCandidate : tau.signalPFCands()) {
            sumEtaTimesEnergy += pfCandidate->positionAtECALEntrance().eta() * pfCandidate->energy();
            sumPhiTimesEnergy += pfCandidate->positionAtECALEntrance().phi() * pfCandidate->energy();
            sumEnergy += pfCandidate->energy();
        }
        if(sumEnergy > 0) {
            eta = sumEtaTimesEnergy / sumEnergy;
            phi = sumPhiTimesEnergy / sumEnergy;
        } else {
            eta = tau_tuple::DefaultFillValue<float>();
            phi = tau_tuple::DefaultFillValue<float>();
        }
    }

    static void CalculateElectronClusterVars(const pat::Electron* ele, float& elecEe, float& elecEgamma)
    {
        if(ele) {
            elecEe = elecEgamma = 0;
            auto superCluster = ele->superCluster();
            if(superCluster.isNonnull() && superCluster.isAvailable() && superCluster->clusters().isNonnull()
                    && superCluster->clusters().isAvailable()) {
                for(auto iter = superCluster->clustersBegin(); iter != superCluster->clustersEnd(); ++iter) {
                    const double energy = (*iter)->energy();
                    if(iter == superCluster->clustersBegin()) elecEe += energy;
                    else elecEgamma += energy;
                }
            }
        } else {
            elecEe = elecEgamma = tau_tuple::DefaultFillValue<float>();
        }
    }

private:
    const clock::time_point start;
    const bool isMC;

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

    tau_tuple::TauTuple tauTuple;
    tau_tuple::SummaryTuple summaryTuple;
};

} // namespace tau_analysis

#include "FWCore/Framework/interface/MakerMacros.h"
using TauTupleProducer = tau_analysis::TauTupleProducer;
DEFINE_FWK_MODULE(TauTupleProducer);
