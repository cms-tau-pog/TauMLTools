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
    virtual void analyze(const edm::Event& event, const edm::EventSetup&) override
    {
        static constexpr float default_value = tau_tuple::DefaultFillValue<float>();

        summaryTuple().numberOfProcessedEvents++;

        tauTuple().run  = event.id().run();
        tauTuple().lumi = event.id().luminosityBlock();
        tauTuple().evt  = event.id().event();

        edm::Handle<std::vector<reco::Vertex>> vertices;
        event.getByToken(vertices_token, vertices);
        tauTuple().npv = vertices->size();
        edm::Handle<double> rho;
        event.getByToken(rho_token, rho);
        tauTuple().rho = *rho;

        if(isMC) {
            edm::Handle<GenEventInfoProduct> genEvent;
            event.getByToken(genEvent_token, genEvent);

            edm::Handle<std::vector<PileupSummaryInfo>> puInfo;
            event.getByToken(puInfo_token, puInfo);
            tauTuple().npu = gen_truth::GetNumberOfPileUpInteractions(puInfo);
            tauTuple().genEventWeight = genEvent->weight();
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

        TauJetBuilder builder(jets, taus, cands, genParticles);

        const auto tauJets = builder.Build();

        for(const TauJet& tauJet : tauJets) {

            const pat::Tau& tau = taus->at(tau_index);

            static const bool id_names_printed = PrintTauIdNames(tau);
            (void)id_names_printed;

            tauTuple().tau_index = static_cast<int>(tau_index);
            tauTuple().tau_pt = tau.p4().pt();
            tauTuple().tau_eta = tau.p4().eta();
            tauTuple().tau_phi = tau.p4().phi();
            tauTuple().tau_mass = tau.p4().mass();
            tauTuple().charge = tau.charge();

            if(isMC) {
                edm::Handle<std::vector<reco::GenParticle>> genParticles;

                const auto match = gen_truth::LeptonGenMatch(tau.p4(), *genParticles);
                tauTuple().gen_match = static_cast<int>(match.first);
                tauTuple().gen_pt = match.second ? match.second->p4().pt() : default_value;
                tauTuple().gen_eta = match.second ? match.second->p4().eta() : default_value;
                tauTuple().gen_phi = match.second ? match.second->p4().phi() : default_value;
                tauTuple().gen_mass = match.second ? match.second->p4().mass() : default_value;

                static constexpr double minGenVisPt = 10;
                static const double dRmatch = 0.3;
                static const std::vector<int> pdgIdsGenElectron = { -11, 11 };
                static const std::vector<int> pdgIdsGenMuon = { -13, 13 };
                static const std::vector<int> pdgIdsGenQuarkOrGluon = { -6, -5, -4, -3, -2, -1, 1, 2, 3, 4, 5, 6, 21 };

                double dRmin;
                auto gen_ele = gen_truth::FindMatchingGenParticle(tau.p4(), *genParticles, minGenVisPt,
                                                                            pdgIdsGenElectron, dRmatch, dRmin);
                tauTuple().has_gen_ele_match = gen_ele != nullptr;
                tauTuple().gen_ele_match_dR = gen_ele != nullptr ? dRmin : default_value;
                tauTuple().gen_ele_pdg = gen_ele != nullptr ? gen_ele->pdgId() : 0;
                tauTuple().gen_ele_pt = gen_ele != nullptr ? gen_ele->p4().pt() : default_value;
                tauTuple().gen_ele_eta = gen_ele != nullptr ? gen_ele->p4().eta() : default_value;
                tauTuple().gen_ele_phi = gen_ele != nullptr ? gen_ele->p4().phi() : default_value;
                tauTuple().gen_ele_mass = gen_ele != nullptr ? gen_ele->p4().mass() : default_value;

                auto gen_muon = gen_truth::FindMatchingGenParticle(tau.p4(), *genParticles, minGenVisPt, pdgIdsGenMuon,
                                                       dRmatch, dRmin);
                tauTuple().has_gen_muon_match = gen_muon != nullptr;
                tauTuple().gen_muon_match_dR = gen_muon != nullptr ? dRmin : default_value;
                tauTuple().gen_muon_pdg = gen_muon != nullptr ? gen_muon->pdgId() : 0;
                tauTuple().gen_muon_pt = gen_muon != nullptr ? gen_muon->p4().pt() : default_value;
                tauTuple().gen_muon_eta = gen_muon != nullptr ? gen_muon->p4().eta() : default_value;
                tauTuple().gen_muon_phi = gen_muon != nullptr ? gen_muon->p4().phi() : default_value;
                tauTuple().gen_muon_mass = gen_muon != nullptr ? gen_muon->p4().mass() : default_value;

                auto gen_qg = gen_truth::FindMatchingGenParticle(tau.p4(), *genParticles, minGenVisPt, pdgIdsGenMuon,
                                                       dRmatch, dRmin);
                tauTuple().has_gen_qg_match = gen_qg != nullptr;
                tauTuple().gen_qg_match_dR = gen_qg != nullptr ? dRmin : default_value;
                tauTuple().gen_qg_pdg = gen_qg != nullptr ? gen_qg->pdgId() : 0;
                tauTuple().gen_qg_pt = gen_qg != nullptr ? gen_qg->p4().pt() : default_value;
                tauTuple().gen_qg_eta = gen_qg != nullptr ? gen_qg->p4().eta() : default_value;
                tauTuple().gen_qg_phi = gen_qg != nullptr ? gen_qg->p4().phi() : default_value;
                tauTuple().gen_qg_mass = gen_qg != nullptr ? gen_qg->p4().mass() : default_value;
            }

            tauTuple().decayMode = tau.decayMode();
            tauTuple().id_flags = CreateTauIdResults(tau).GetResultBits();
            FillRawTauIds(tau);
            FillExtendedVariables(tau, *electrons, *muons);
            FillComponents(tau);

            tauTuple.Fill();
        }
    }

    virtual void endJob() override
    {
        tauTuple.Write();
        const auto stop = clock::now();
        summaryTuple().exeTime = std::chrono::duration_cast<std::chrono::seconds>(stop - start).count();
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

    void FillRawTauIds(const pat::Tau& tau)
    {
#define VAR(type, name) tauTuple().name = tau.tauID(#name);
        RAW_TAU_IDS()
#undef VAR
    }

    void FillExtendedVariables(const pat::Tau& tau, const pat::ElectronCollection& electrons,
                               const pat::MuonCollection& muons)
    {
        static constexpr float default_value = tau_tuple::DefaultFillValue<float>();
        auto leadChargedHadrCand = dynamic_cast<const pat::PackedCandidate*>(tau.leadChargedHadrCand().get());

        tauTuple().dxy = tau.dxy();
        tauTuple().dxy_sig = tau.dxy_Sig();
        tauTuple().dz = leadChargedHadrCand ? leadChargedHadrCand->dz() : default_value;
        tauTuple().ip3d = tau.ip3d();
        tauTuple().ip3d_sig = tau.ip3d_Sig();
        tauTuple().hasSecondaryVertex = tau.hasSecondaryVertex();
        tauTuple().flightLength_r = tau.flightLength().R();
        tauTuple().flightLength_dEta = dEta(tau.flightLength(), tau.p4());
        tauTuple().flightLength_dPhi = dPhi(tau.flightLength(), tau.p4());
        tauTuple().flightLength_sig = tau.flightLengthSig();

        tauTuple().leadChargedHadrCand_pt = leadChargedHadrCand ? leadChargedHadrCand->p4().Pt() : default_value;
        tauTuple().leadChargedHadrCand_dEta = leadChargedHadrCand
                ? dEta(leadChargedHadrCand->p4(), tau.p4()) : default_value;
        tauTuple().leadChargedHadrCand_dPhi = leadChargedHadrCand
                ? dPhi(leadChargedHadrCand->p4(), tau.p4()) : default_value;
        tauTuple().leadChargedHadrCand_mass = leadChargedHadrCand ? leadChargedHadrCand->p4().mass() : default_value;
        tauTuple().leadChargedCand_etaAtEcalEntrance = tau.etaAtEcalEntranceLeadChargedCand();


        tauTuple().pt_weighted_deta_strip = clusterVariables.tau_pt_weighted_deta_strip(tau, tau.decayMode());
        tauTuple().pt_weighted_dphi_strip = clusterVariables.tau_pt_weighted_dphi_strip(tau, tau.decayMode());
        tauTuple().pt_weighted_dr_signal = clusterVariables.tau_pt_weighted_dr_signal(tau, tau.decayMode());
        tauTuple().pt_weighted_dr_iso = clusterVariables.tau_pt_weighted_dr_iso(tau, tau.decayMode());
        tauTuple().leadingTrackNormChi2 = tau.leadingTrackNormChi2();
        tauTuple().e_ratio = clusterVariables.tau_Eratio(tau);
        tauTuple().gj_angle_diff = CalculateGottfriedJacksonAngleDifference(tau);
        tauTuple().n_photons = clusterVariables.tau_n_photons_total(tau);

        tauTuple().emFraction = tau.emFraction_MVA();
        tauTuple().inside_ecal_crack = IsInEcalCrack(tau.p4().Eta());

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

#define PROCESS_COMP(prefix, flav, col, dR2_min, dR2_max) \
    ProcessComponentCollection(tau, tau.prefix##flav(), dR2_min, dR2_max, \
        tauTuple().col##_##flav##_sum_pt, tauTuple().col##_##flav##_sum_ht, tauTuple().col##_##flav##_sum_dPhi, \
        tauTuple().col##_##flav##_sum_dEta, tauTuple().col##_##flav##_sum_mass, tauTuple().col##_##flav##_sum_energy, \
        tauTuple().col##_##flav##_nTotal)
    /**/

#define PROCESS_COMP_SET(prefix, col, dR2_min, dR2_max) \
    PROCESS_COMP(prefix, ChargedHadrCands, col, dR2_min, dR2_max); \
    PROCESS_COMP(prefix, NeutrHadrCands, col, dR2_min, dR2_max); \
    PROCESS_COMP(prefix, GammaCands, col, dR2_min, dR2_max) \
    /**/

    void FillComponents(const pat::Tau& tau)
    {
        const double innerSigCone_radius = GetInnerSignalConeRadius(tau.pt());

        PROCESS_COMP_SET(signal, innerSigCone, 0, innerSigCone_radius);
        PROCESS_COMP_SET(signal, outerSigCone, innerSigCone_radius, 0.5);
        PROCESS_COMP_SET(isolation, isoRing02, 0, 0.2);
        PROCESS_COMP_SET(isolation, isoRing03, 0.2, 0.3);
        PROCESS_COMP_SET(isolation, isoRing04, 0.3, 0.4);
        PROCESS_COMP_SET(isolation, isoRing05, 0.4, 0.5);
    }

#undef PROCESS_COMP_SET
#undef PROCESS_COMP

    template<typename CandCollection>
    static void ProcessComponentCollection(const pat::Tau& tau, const CandCollection& cands,
                                           double dR2_min, double dR2_max,
                                           float& sum_pt, float& sum_ht, float& sum_dPhi, float& sum_dEta,
                                           float& sum_mass, float& sum_energy, unsigned& nTotal)
    {
        static constexpr float default_value = tau_tuple::DefaultFillValue<float>();

        analysis::LorentzVectorXYZ sum_p4(0, 0, 0, 0);
        nTotal = 0;
        sum_ht = 0;
        for(const auto& cand : cands) {
            if(!IsInsideRing(tau.p4(), cand->p4(), dR2_min, dR2_max)) continue;
            sum_p4 += cand->p4();
            sum_ht += cand->p4().pt();
            ++nTotal;
        }

        sum_pt = sum_p4.pt();
        sum_dPhi = nTotal > 0 ? dPhi(sum_p4, tau.p4()) : default_value;
        sum_dEta = nTotal > 0 ? dEta(sum_p4, tau.p4()) : default_value;
        sum_mass = sum_p4.mass();
        sum_energy = sum_p4.E();
    }

    template<typename LorentzVector1, typename LorentzVector2>
    static double IsInsideRing(const LorentzVector1& p4_tau, const LorentzVector2& p4_cand,
                               double dR2_min, double dR2_max)
    {
        const double dR2 = ROOT::Math::VectorUtil::DeltaR2(p4_tau, p4_cand);
        return dR2 >= dR2_min && dR2 < dR2_max;
    }

    static float CalculateGottfriedJacksonAngleDifference(const pat::Tau& tau)
    {
        double gj_diff;
        if(::tau_analysis::CalculateGottfriedJacksonAngleDifference(tau, gj_diff))
            return static_cast<float>(gj_diff);
        return tau_tuple::DefaultFillValue<float>();
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

    static const pat::Electron* FindMatchedElectron(const pat::Tau& tau, const pat::ElectronCollection& electrons,
                                                    double deltaR)
    {
        const double deltaR2 = std::pow(deltaR, 2);
        const pat::Electron* matched_ele = nullptr;
        for(const auto& ele : electrons) {
            if(ROOT::Math::VectorUtil::DeltaR2(tau.p4(), ele.p4()) < deltaR2 &&
                    (!matched_ele || matched_ele->pt() < ele.pt())) {
                matched_ele = &ele;
            }
        }
        return matched_ele;
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
    TauIdMVAAuxiliaries clusterVariables;
};

} // namespace tau_analysis

#include "FWCore/Framework/interface/MakerMacros.h"
using TauTupleProducer = tau_analysis::TauTupleProducer;
DEFINE_FWK_MODULE(TauTupleProducer);
