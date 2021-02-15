/*! Creates tuple for tau analysis.
*/

#include "Compression.h"

#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ServiceRegistry/interface/Service.h"

#include "SimDataFormats/GeneratorProducts/interface/GenEventInfoProduct.h"
#include "AnalysisDataFormats/TopObjects/interface/TtGenEvent.h"
#include "CommonTools/UtilAlgos/interface/TFileService.h"
#include "RecoTauTag/RecoTau/interface/PFRecoTauClusterVariables.h"

#include "Core/interface/Tools.h"
#include "Core/interface/TextIO.h"
#include "Analysis/interface/TrainTuple.h"
#include "Analysis/interface/SummaryTuple.h"
#include "Analysis/interface/TauIdResults.h"
#include "Production/interface/GenTruthTools.h"
#include "Production/interface/TauAnalysis.h"
#include "Production/interface/MuonHitMatch.h"
#include "Production/interface/TauJet.h"

#include "DataFormats/L1Trigger/interface/Tau.h"
#include "DataFormats/TauReco/interface/PFTau.h"
#include "DataFormats/TauReco/interface/PFTauDiscriminator.h"
#include "DataFormats/Candidate/interface/LeafCandidate.h"
#include "DataFormats/RecoCandidate/interface/RecoEcalCandidate.h"
#include "DataFormats/TrackReco/interface/HitPattern.h"
#include "DataFormats/MuonReco/interface/MuonSelectors.h"
#include "DataFormats/TauReco/interface/PFTauTransverseImpactParameterAssociation.h"
#include "DataFormats/TauReco/interface/PFTauTransverseImpactParameterFwd.h"
#include "DataFormats/TauReco/interface/PFTauTransverseImpactParameter.h"
#include "DataFormats/Common/interface/AssociationVector.h"
#include "DataFormats/Common/interface/Association.h"
#include "DataFormats/CaloTowers/interface/CaloTowerCollection.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/TrackReco/interface/Track.h"
// #include "TauTriggerTools/Common/interface/GenTruthTools.h"
#include "DataFormats/JetReco/interface/CaloJetCollection.h"

namespace tau_analysis {

struct TrainTupleProducerData {
    using clock = std::chrono::system_clock;

    const clock::time_point start;
    train_tuple::TrainTuple trainTuple;
    tau_tuple::SummaryTuple summaryTuple;
    std::mutex mutex;

private:
    size_t n_producers;

    TrainTupleProducerData(TFile& file) :
        start(clock::now()),
        trainTuple("taus", &file, false),
        summaryTuple("summary", &file, false),
        n_producers(0)
    {
        summaryTuple().numberOfProcessedEvents = 0;
    }

    ~TrainTupleProducerData() {}

public:

    static TrainTupleProducerData* RequestGlobalData()
    {
        TrainTupleProducerData* data = GetGlobalData();
        if(data == nullptr)
            throw cms::Exception("TrainTupleProducerData") << "Request after all data copies were released.";
        {
            std::lock_guard<std::mutex> lock(data->mutex);
            ++data->n_producers;
            std::cout << "New request of TrainTupleProducerData. Total number of producers = " << data->n_producers
                      << "." << std::endl;
        }
        return data;
    }

    static void ReleaseGlobalData()
    {
        TrainTupleProducerData*& data = GetGlobalData();
        if(data == nullptr)
            throw cms::Exception("TrainTupleProducerData") << "Another release after all data copies were released.";
        {
            std::lock_guard<std::mutex> lock(data->mutex);
            if(!data->n_producers)
                throw cms::Exception("TrainTupleProducerData") << "Release before any request.";
            --data->n_producers;
            std::cout << "TrainTupleProducerData has been released. Total number of producers = " << data->n_producers
                      << "." << std::endl;
            if(!data->n_producers) {
                data->trainTuple.Write();
                const auto stop = clock::now();
                data->summaryTuple().exeTime = static_cast<unsigned>(
                            std::chrono::duration_cast<std::chrono::seconds>(stop - data->start).count());
                data->summaryTuple.Fill();
                data->summaryTuple.Write();
                delete data;
                data = nullptr;
                std::cout << "TrainTupleProducerData has been destroyed." << std::endl;
            }
        }

    }

private:
    static TrainTupleProducerData*& GetGlobalData()
    {
        static TrainTupleProducerData* data = InitializeGlobalData();
        return data;
    }

    static TrainTupleProducerData* InitializeGlobalData()
    {
        TFile& file = edm::Service<TFileService>()->file();
        file.SetCompressionAlgorithm(ROOT::kZLIB);
        file.SetCompressionLevel(9);
        TrainTupleProducerData* data = new TrainTupleProducerData(file);
        std::cout << "TrainTupleProducerData has been created." << std::endl;
        return data;
    }
};

class TrainTupleProducer : public edm::EDAnalyzer {
public:
    using TauDiscriminator = reco::PFTauDiscriminator;

    TrainTupleProducer(const edm::ParameterSet& cfg) :
        isMC(cfg.getParameter<bool>("isMC")),
        genEvent_token(mayConsume<GenEventInfoProduct>(cfg.getParameter<edm::InputTag>("genEvent"))),
        puInfo_token(mayConsume<std::vector<PileupSummaryInfo>>(cfg.getParameter<edm::InputTag>("puInfo"))),
        l1Taus_token(consumes<l1t::TauBxCollection>(cfg.getParameter<edm::InputTag>("l1taus"))), // --> VBor
        caloTowers_token(consumes<CaloTowerCollection>(cfg.getParameter<edm::InputTag>("caloTowers"))),
        caloTaus_token(consumes<reco::CaloJetCollection>(cfg.getParameter<edm::InputTag>("caloTaus"))),
        vertices_token(consumes<std::vector<reco::Vertex> >(cfg.getParameter<edm::InputTag>("vertices"))),
        pataVertices_token(consumes<std::vector<reco::Vertex> >(cfg.getParameter<edm::InputTag>("pataVertices"))),
        pixelTracks_token(consumes<reco::TrackCollection>(cfg.getParameter<edm::InputTag>("pixelTracks"))),
        pataTracks_token(consumes<reco::TrackCollection>(cfg.getParameter<edm::InputTag>("pataTracks"))),
        VeryBigOR_result_token(consumes<bool>(cfg.getParameter<edm::InputTag>("VeryBigOR"))),
        hltDoubleL2Tau26eta2p2_result_token(consumes<bool>(cfg.getParameter<edm::InputTag>("hltDoubleL2Tau26eta2p2"))),
        hltDoubleL2IsoTau26eta2p2_result_token(consumes<bool>(cfg.getParameter<edm::InputTag>("hltDoubleL2IsoTau26eta2p2"))),
        data(TrainTupleProducerData::RequestGlobalData()),
        trainTuple(data->trainTuple),
        summaryTuple(data->summaryTuple)
    {

    }

private:
    static constexpr float default_value = train_tuple::DefaultFillValue<float>();
    static constexpr int default_int_value = train_tuple::DefaultFillValue<int>();
    static constexpr int default_unsigned_value = train_tuple::DefaultFillValue<unsigned>();

    virtual void analyze(const edm::Event& event, const edm::EventSetup&) override
    {
        std::lock_guard<std::mutex> lock(data->mutex);
        summaryTuple().numberOfProcessedEvents++;

        trainTuple().run  = event.id().run();
        trainTuple().lumi = event.id().luminosityBlock();
        trainTuple().evt  = event.id().event();

        edm::Handle<bool> VBOR_result;
        event.getByToken(VeryBigOR_result_token, VBOR_result);
        trainTuple().VeryBigOR_result=*VBOR_result;
        edm::Handle<bool> hltDoubleL2Tau26eta2p2_result;
        event.getByToken(hltDoubleL2Tau26eta2p2_result_token, hltDoubleL2Tau26eta2p2_result);
        trainTuple().hltDoubleL2Tau26eta2p2_result=*hltDoubleL2Tau26eta2p2_result;
        edm::Handle<bool> hltDoubleL2IsoTau26eta2p2_result;
        event.getByToken(hltDoubleL2IsoTau26eta2p2_result_token, hltDoubleL2IsoTau26eta2p2_result);
        trainTuple().hltDoubleL2IsoTau26eta2p2_result=*hltDoubleL2IsoTau26eta2p2_result;

        edm::Handle<std::vector<reco::Vertex>> vertices;
        event.getByToken(vertices_token, vertices);
        trainTuple().npv = static_cast<int>(vertices->size());

        edm::Handle<std::vector<reco::Vertex>> pataVertices;
        event.getByToken(pataVertices_token, pataVertices);
        trainTuple().nppv = static_cast<int>(pataVertices->size());

        if(isMC) {
            edm::Handle<GenEventInfoProduct> genEvent;
            event.getByToken(genEvent_token, genEvent);
            trainTuple().genEventWeight = static_cast<float>(genEvent->weight());

            edm::Handle<std::vector<PileupSummaryInfo>> puInfo;
            event.getByToken(puInfo_token, puInfo);
            trainTuple().npu = analysis::gen_truth::GetNumberOfPileUpInteractions(puInfo);
        }

        const auto& PV = vertices->at(0);
        trainTuple().pv_x = static_cast<float>(PV.position().x());
        trainTuple().pv_y = static_cast<float>(PV.position().y());
        trainTuple().pv_z = static_cast<float>(PV.position().z());
        trainTuple().pv_chi2 = static_cast<float>(PV.chi2());
        trainTuple().pv_ndof = static_cast<float>(PV.ndof());

        const auto& PPV = pataVertices->at(0);
        trainTuple().ppv_x = static_cast<float>(PPV.position().x());
        trainTuple().ppv_y = static_cast<float>(PPV.position().y());
        trainTuple().ppv_z = static_cast<float>(PPV.position().z());
        trainTuple().ppv_chi2 = static_cast<float>(PPV.chi2());
        trainTuple().ppv_ndof = static_cast<float>(PPV.ndof());


        edm::Handle<l1t::TauBxCollection> l1Taus;
        event.getByToken(l1Taus_token, l1Taus);

        edm::Handle<CaloTowerCollection> caloTowers;
        event.getByToken(caloTowers_token, caloTowers);

        edm::Handle<reco::TrackCollection> pixelTracks;
        event.getByToken(pixelTracks_token, pixelTracks);

        edm::Handle<reco::TrackCollection> pataTracks;
        event.getByToken(pataTracks_token, pataTracks);

        edm::Handle<reco::CaloJetCollection> caloTaus;
        event.getByToken(caloTaus_token, caloTaus);
        pat::JetCollection jets;

        //check inputs
        FillL1Objects(*l1Taus);
        FillPataTracks(*pataTracks);
        FillPixelTracks(*pixelTracks);
        FillCaloTowers(*caloTowers);
        FillCaloTaus(*caloTaus);

        trainTuple.Fill();
    }

    virtual void endJob() override
    {
        TrainTupleProducerData::ReleaseGlobalData();
    }

private:


    void FillL1Objects(const l1t::TauBxCollection& l1Taus)
    {
        for(auto iter = l1Taus.begin(0); iter != l1Taus.end(0); ++iter) {
            trainTuple().l1Tau_pt.push_back(static_cast<float>(iter->polarP4().pt()));
            trainTuple().l1Tau_eta.push_back(static_cast<float>(iter->polarP4().eta()));
            trainTuple().l1Tau_phi.push_back(static_cast<float>(iter->polarP4().phi()));
            trainTuple().l1Tau_mass.push_back(static_cast<float>(iter->polarP4().mass()));

            trainTuple().l1Tau_towerIEta.push_back(static_cast<short int>(iter->towerIEta()));
            trainTuple().l1Tau_towerIPhi.push_back(static_cast<short int>(iter->towerIPhi()));
            trainTuple().l1Tau_rawEt.push_back(static_cast<short int>(iter->rawEt()));
            trainTuple().l1Tau_isoEt.push_back(static_cast<short int>(iter->isoEt()));
            trainTuple().l1Tau_hasEM.push_back(static_cast<bool>(iter->hasEM()));
            trainTuple().l1Tau_isMerged.push_back(static_cast<bool>(iter->isMerged()));

            trainTuple().l1Tau_hwIso.push_back(iter->hwIso());
            trainTuple().l1Tau_hwQual.push_back(iter->hwQual());
        }
    }

    template<typename T>
    void FillTrackVar(const std::string& var_name, T var, const std::string& prefix){
        trainTuple.get<std::vector<T>>(prefix+"_"+var_name).push_back(var);
    }

    void FillTracks(const reco::TrackCollection& tracks, const std::string prefix)
    {
        const auto push_back= [&](const std::string& var_name, auto var){
          FillTrackVar(var_name, var, prefix);
        }

        for(unsigned n = 0; n < tracks.size(); ++n){
            push_back( "pt", pixelTracks.at(n).pt());

            trainTuple().track_pt.push_back(pixelTracks.at(n).pt());
            trainTuple().track_eta.push_back(pixelTracks.at(n).eta());
            trainTuple().track_phi.push_back(pixelTracks.at(n).phi());
            trainTuple().track_outerOk.push_back(pixelTracks.at(n).outerOk());
            trainTuple().track_innerOk.push_back(pixelTracks.at(n).innerOk());
            trainTuple().track_found.push_back(pixelTracks.at(n).found());
            trainTuple().track_lost.push_back(pixelTracks.at(n).lost());
            trainTuple().track_chi2.push_back(pixelTracks.at(n).chi2());
            trainTuple().track_ndof.push_back(pixelTracks.at(n).ndof());
            trainTuple().track_charge.push_back(pixelTracks.at(n).charge());
            trainTuple().track_algo.push_back(static_cast<int>(pixelTracks.at(n).algo()));
            trainTuple().track_qualityMask.push_back(static_cast<unsigned>(pixelTracks.at(n).qualityMask()));
            trainTuple().track_dxy.push_back(pixelTracks.at(n).dxy());
            trainTuple().track_dz.push_back(pixelTracks.at(n).dz());
            trainTuple().track_vx.push_back(pixelTracks.at(n).vx());
            trainTuple().track_vy.push_back(pixelTracks.at(n).vy());
            trainTuple().track_vz.push_back(pixelTracks.at(n).vz());
            trainTuple().track_ptError.push_back(pixelTracks.at(n).ptError());
            trainTuple().track_etaError.push_back(pixelTracks.at(n).etaError());
            trainTuple().track_phiError.push_back(pixelTracks.at(n).phiError());
            trainTuple().track_dxyError.push_back(pixelTracks.at(n).dxyError());
            trainTuple().track_dzError.push_back(pixelTracks.at(n).dzError());
        }
    }

    void FillCaloTowers(const CaloTowerCollection& caloTowers)
    {
        for(auto iter = caloTowers.begin(); iter != caloTowers.end(); ++iter){
            trainTuple().caloTower_pt.push_back(iter->polarP4().pt());
            trainTuple().caloTower_eta.push_back(iter->polarP4().eta());
            trainTuple().caloTower_phi.push_back(iter->polarP4().phi());
            trainTuple().caloTower_energy.push_back(iter->polarP4().energy());
            trainTuple().caloTower_emEnergy.push_back(iter->emEnergy());
            trainTuple().caloTower_hadEnergy.push_back(iter->hadEnergy());
            trainTuple().caloTower_outerEnergy.push_back(iter->outerEnergy());
            trainTuple().caloTower_emPosition_x.push_back(iter->emPosition().x());
            trainTuple().caloTower_emPosition_y.push_back(iter->emPosition().y());
            trainTuple().caloTower_emPosition_z.push_back(iter->emPosition().z());
            trainTuple().caloTower_hadPosition_x.push_back(iter->hadPosition().x());
            trainTuple().caloTower_hadPosition_y.push_back(iter->hadPosition().y());
            trainTuple().caloTower_hadPosition_z.push_back(iter->hadPosition().z());
            trainTuple().caloTower_hadEnergyHeOuterLayer.push_back(iter->hadEnergyHeOuterLayer());
            trainTuple().caloTower_hadEnergyHeInnerLayer.push_back(iter->hadEnergyHeInnerLayer());
            trainTuple().caloTower_energyInHB.push_back(iter->energyInHB());
            trainTuple().caloTower_energyInHE.push_back(iter->energyInHE());
            trainTuple().caloTower_energyInHF.push_back(iter->energyInHF());
            trainTuple().caloTower_energyInHO.push_back(iter->energyInHO());
            trainTuple().caloTower_numBadEcalCells.push_back(iter->numBadEcalCells());
            trainTuple().caloTower_numRecoveredEcalCells.push_back(iter->numRecoveredEcalCells());
            trainTuple().caloTower_numProblematicEcalCells.push_back(iter->numProblematicEcalCells());
            trainTuple().caloTower_numBadHcalCells.push_back(iter->numBadHcalCells());
            trainTuple().caloTower_numRecoveredHcalCells.push_back(iter->numRecoveredHcalCells());
            trainTuple().caloTower_numProblematicHcalCells.push_back(iter->numProblematicHcalCells());
            trainTuple().caloTower_ecalTime.push_back(iter->ecalTime());
            trainTuple().caloTower_hcalTime.push_back(iter->hcalTime());
        }
    }

    void FillCaloTaus(const reco::CaloJetCollection& caloTaus)
    {
        for(const auto& caloTau : caloTaus){
            trainTuple().caloTau_pt.push_back(caloTau.p4().pt());
            trainTuple().caloTau_eta.push_back(caloTau.p4().eta());
            trainTuple().caloTau_phi.push_back(caloTau.p4().phi());
            trainTuple().caloTau_energy.push_back(caloTau.p4().energy());
            trainTuple().caloTau_maxEInEmTowers.push_back(caloTau.maxEInEmTowers());
            trainTuple().caloTau_maxEInHadTowers.push_back(caloTau.maxEInHadTowers());
            trainTuple().caloTau_energyFractionHadronic.push_back(caloTau.energyFractionHadronic());
            trainTuple().caloTau_emEnergyFraction.push_back(caloTau.emEnergyFraction());
            trainTuple().caloTau_hadEnergyInHB.push_back(caloTau.hadEnergyInHB());
            trainTuple().caloTau_hadEnergyInHO.push_back(caloTau.hadEnergyInHO());
            trainTuple().caloTau_hadEnergyInHE.push_back(caloTau.hadEnergyInHE());
            trainTuple().caloTau_hadEnergyInHF.push_back(caloTau.hadEnergyInHF());
            trainTuple().caloTau_emEnergyInEB.push_back(caloTau.emEnergyInEB());
            trainTuple().caloTau_emEnergyInEE.push_back(caloTau.emEnergyInEE());
            trainTuple().caloTau_emEnergyInHF.push_back(caloTau.emEnergyInHF());
            trainTuple().caloTau_towersArea.push_back(caloTau.towersArea());
            trainTuple().caloTau_n90.push_back(caloTau.n90());
            trainTuple().caloTau_n60.push_back(caloTau.n60());
        }
    }

private:
    const bool isMC;
    TauJetBuilderSetup builderSetup;
    edm::EDGetTokenT<GenEventInfoProduct> genEvent_token;
    edm::EDGetTokenT<std::vector<PileupSummaryInfo>> puInfo_token;
    edm::EDGetTokenT<l1t::TauBxCollection> l1Taus_token;
    edm::EDGetTokenT<CaloTowerCollection> caloTowers_token;
    edm::EDGetTokenT<reco::CaloJetCollection> caloTaus_token;
    edm::EDGetTokenT<std::vector<reco::Vertex>> vertices_token;
    edm::EDGetTokenT<std::vector<reco::Vertex>> pataVertices_token;
    edm::EDGetTokenT<reco::TrackCollection> pixelTracks_token;
    edm::EDGetTokenT<reco::TrackCollection> pataTracks_token;
    edm::EDGetTokenT<bool> VeryBigOR_result_token;
    edm::EDGetTokenT<bool> hltDoubleL2Tau26eta2p2_result_token;
    edm::EDGetTokenT<bool> hltDoubleL2IsoTau26eta2p2_result_token;
    TrainTupleProducerData* data;
    train_tuple::TrainTuple& trainTuple;
    tau_tuple::SummaryTuple& summaryTuple;

};

} // namespace tau_analysis

#include "FWCore/Framework/interface/MakerMacros.h"
using TrainTupleProducer = tau_analysis::TrainTupleProducer;
DEFINE_FWK_MODULE(TrainTupleProducer);
