#include "CondFormats/DataRecord/interface/L1TCaloParamsRcd.h"
#include "CondFormats/DataRecord/interface/L1TCaloParamsO2ORcd.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/NanoAOD/interface/FlatTable.h"
#include "DataFormats/L1Trigger/interface/EGamma.h"
#include "DataFormats/L1Trigger/interface/Tau.h"
#include "DataFormats/L1Trigger/interface/Jet.h"
#include "DataFormats/L1Trigger/interface/Muon.h"
#include "DataFormats/L1TCalorimeter/interface/CaloTower.h"
#include "DataFormats/Math/interface/deltaR.h"
#include "L1Trigger/L1TCalorimeter/interface/CaloParamsHelper.h"
#include "L1Trigger/L1TCalorimeter/interface/CaloStage2Nav.h"
#include "L1Trigger/L1TCalorimeter/interface/CaloTools.h"
#include "L1Trigger/L1TCalorimeter/interface/Stage2Layer2ClusterAlgorithmFirmware.h"
#include "HLTrigger/HLTcore/interface/HLTFilter.h"

class L1TableProducer : public edm::stream::EDProducer<> {
public:
  using FCol = trigger::TriggerFilterObjectWithRefs;
  using FColToken = edm::EDGetTokenT<FCol>;
  using L2Tags = std::vector<float>;
  using L2TagsToken = edm::EDGetTokenT<L2Tags>;
  using L2OutToken = std::pair<FColToken, L2TagsToken>;
  using L2OutTokenCol = std::vector<L2OutToken>;

  L1TableProducer(const edm::ParameterSet& cfg) :
      egammasToken_(consumes<l1t::EGammaBxCollection>(cfg.getParameter<edm::InputTag>("egammas"))),
      muonsToken_(consumes<l1t::MuonBxCollection>(cfg.getParameter<edm::InputTag>("muons"))),
      jetsToken_(consumes<l1t::JetBxCollection>(cfg.getParameter<edm::InputTag>("jets"))),
      tausToken_(consumes<l1t::TauBxCollection>(cfg.getParameter<edm::InputTag>("taus"))),
      caloTowersToken_(consumes<l1t::CaloTowerBxCollection>(cfg.getParameter<edm::InputTag>("caloTowers"))),
      l2TauToken_(loadL2Tokens(cfg.getParameterSetVector("l2Taus"),
                               cfg.getParameter<std::string>("l2TauTagNNProducer"))),
      candidateToken_(esConsumes<l1t::CaloParams, L1TCaloParamsRcd, edm::Transition::BeginRun>()),
      o2oProtoToken_(esConsumes<l1t::CaloParams, L1TCaloParamsO2ORcd, edm::Transition::BeginRun>()),
      precision_(cfg.getParameter<int>("precision"))
  {
    produces<nanoaod::FlatTable>("L1Egamma");
    produces<nanoaod::FlatTable>("L1Muon");
    produces<nanoaod::FlatTable>("L1Jet");
    produces<nanoaod::FlatTable>("L1Tau");
    produces<nanoaod::FlatTable>("L1TauTowers");
  }

private:
  L2OutTokenCol loadL2Tokens(const edm::VParameterSet& cfg, const std::string& l2TauTagNNProducer)
  {
    L2OutTokenCol tokens;
    for(const edm::ParameterSet& pset : cfg) {
      tokens.emplace_back(
        consumes<FCol>(pset.getParameter<edm::InputTag>("L1TauTrigger")),
        consumes<L2Tags>(edm::InputTag(l2TauTagNNProducer, pset.getParameter<std::string>("L1CollectionName")))
      );
    }
    return tokens;
  }

  // adapted from https://github.com/cms-sw/cmssw/blob/166eab1458287d877b7b06a931153f21b74e2093/L1Trigger/L1TCalorimeter/plugins/L1TStage2Layer2Producer.cc
  void beginRun(const edm::Run& run, const edm::EventSetup& setup) override
  {
    const unsigned long long id = setup.get<L1TCaloParamsRcd>().cacheIdentifier();
    if(!caloParamsCacheId_ || id != *caloParamsCacheId_) {
      caloParamsCacheId_ = id;

      // fetch payload corresponding to the current run from the CondDB
      const auto candidateHandle = setup.getHandle(candidateToken_);
      const auto candidate = std::make_unique<l1t::CaloParams>(*candidateHandle.product());

      // fetch the latest greatest prototype (equivalent of static payload)
      const auto o2oProtoHandle = setup.getHandle(o2oProtoToken_);
      const auto prototype = std::make_unique<l1t::CaloParams>(*o2oProtoHandle.product());

      // compare the candidate payload misses some of the pnodes compared to the prototype,
      // if this is the case - the candidate is an old payload that'll crash the Stage2 emulator
      // and we better use the prototype for the emulator's configuration
      const size_t nNodes_cand = static_cast<l1t::CaloParamsHelper*>(candidate.get())->getNodes().size();
      const size_t nNodes_proto = static_cast<l1t::CaloParamsHelper*>(prototype.get())->getNodes().size();
      const auto& product = nNodes_cand < nNodes_proto ? o2oProtoHandle.product() : candidateHandle.product();
      caloParams_ = std::make_unique<l1t::CaloParamsHelper>(*product);
      //towerAlgo_ = std::make_unique<Stage2TowerDecompressAlgorithmFirmwareImp1>(caloParams_);
      tauClusterAlgo_ = std::make_unique<l1t::Stage2Layer2ClusterAlgorithmFirmwareImp1>(
        caloParams_.get(), l1t::Stage2Layer2ClusterAlgorithmFirmwareImp1::ClusterInput::EH);
    }
  }

  void produce(edm::Event& event, const edm::EventSetup& setup) override
  {
    FillEgammas(event);
    FillMuons(event);
    FillJets(event);
    FillTaus(event);
  }

  template<typename T, typename F>
  std::unique_ptr<nanoaod::FlatTable> Fill(const edm::Event& event, const edm::EDGetTokenT<T>& token,
                                           const std::string& name, F fn) const
  {
    const auto& collection = event.get(token);
    std::vector<float> pt, eta, phi;

    for(auto it = collection.begin(0); it != collection.end(0); ++it) {
      if(it->pt() <= 0) continue;
      pt.push_back(it->pt());
      eta.push_back(it->eta());
      phi.push_back(it->phi());
      fn(it);
    }

    auto table = std::make_unique<nanoaod::FlatTable>(pt.size(), name, false, false);
    table->addColumn<float>("pt", pt, "transverse momentum", precision_);
    table->addColumn<float>("eta", eta, "pseudorapidity", precision_);
    table->addColumn<float>("phi", phi, "azimuthal angle", precision_);
    return table;
  }

  void FillEgammas(edm::Event& event) const
  {
    static const std::string name = "L1Egamma";
    std::vector<int> hwIso;
    auto table = Fill(event, egammasToken_, name, [&](const auto& it) {
      hwIso.push_back(it->hwIso());
    });
    table->addColumn<int>("hwIso", hwIso, "hardware isolation");
    event.put(std::move(table), name);
  }

  void FillMuons(edm::Event& event) const
  {
    static const std::string name = "L1Muon";
    std::vector<float> ptUnconstrained;
    std::vector<int> charge, hwIso, hwQual, hwDXY;
    auto table = Fill(event, muonsToken_, name, [&](const auto& it) {
      ptUnconstrained.push_back(it->ptUnconstrained());
      charge.push_back(it->charge());
      hwIso.push_back(it->hwIso());
      hwQual.push_back(it->hwQual());
      hwDXY.push_back(it->hwDXY());
    });
    table->addColumn<float>("ptUnconstrained", ptUnconstrained, "unconstrained transverse momentum", precision_);
    table->addColumn<int>("charge", charge, "charge");
    table->addColumn<int>("hwIso", hwIso, "hardware isolation");
    table->addColumn<int>("hwQual", hwQual, "hardware quality");
    table->addColumn<int>("hwDXY", hwDXY, "hardware transverse impact parameter");
    event.put(std::move(table), name);
  }

  void FillJets(edm::Event& event) const
  {
    static const std::string name = "L1Jet";
    std::vector<int> hwQual;
    auto table = Fill(event, jetsToken_, name, [&](const auto& it) {
      hwQual.push_back(it->hwQual());
    });
    table->addColumn<int>("hwQual", hwQual, "hardware quality");
    event.put(std::move(table), name);
  }

  void FillTaus(edm::Event& event) const
  {
    static const std::string tauTableName = "L1Tau";
    static const std::string tauTowersTableName = "L1TauTowers";
    std::vector<int> hwPt, hwEta, hwPhi, hwIso, towerIEta, towerIPhi, rawEt, isoEt, nTT, hwEtSum;
    std::vector<bool> hasEM, isMerged;

    std::vector<int> tower_tauIdx, tower_relEta, tower_relPhi, tower_hwEtEm, tower_hwEtHad, tower_hwPt;
    std::vector<float> l2_scores;

    if(!caloParams_ || !tauClusterAlgo_)
      throw cms::Exception("L1NtupleProducer") << "CaloParams or TauClusterAlgo not initialized";

    const auto& caloTowers = event.get(caloTowersToken_);
    std::vector<l1t::CaloTower> towers;
    for(auto tower_iter = caloTowers.begin(0); tower_iter != caloTowers.end(0); ++tower_iter) {
      towers.push_back(*tower_iter);
    }

    std::vector<l1t::CaloCluster> tauClusters;
    tauClusterAlgo_->processEvent(towers, tauClusters);

    std::vector<std::pair<l1t::TauVectorRef, std::vector<float>>> l2TauOutcomes;
    for(const auto& [token_l1, token_l2] : l2TauToken_) {
      const auto& l1Objects = event.get(token_l1);
      l1t::TauVectorRef taus;
      l1Objects.getObjects(trigger::TriggerL1Tau, taus);
      const auto& l2Objects = event.get(token_l2);
      l2TauOutcomes.emplace_back(taus, l2Objects);
    }

    const auto getL2Score = [&](double tau_eta, double tau_phi) {
      float best_score = -1.f;
      double best_dr2 = 0.01;
      for(const auto& [taus, scores] : l2TauOutcomes) {
        for(size_t i = 0; i < taus.size(); ++i) {
          const double dr2 = reco::deltaR2(tau_eta, tau_phi, taus[i]->eta(), taus[i]->phi());
          if(dr2 < best_dr2) {
            best_dr2 = dr2;
            best_score = scores.at(i);
          }
        }
      }
      return best_score;
    };

    // adapted from https://github.com/cms-sw/cmssw/blob/17b3781658ba5a652d1d3e0cc74c31915b708235/L1Trigger/L1TCalorimeter/src/CaloTools.cc#LL125-L151C2
    const auto fillTowers = [&](int tau_idx, int iEta, int iPhi, int localEtaMin, int localEtaMax, int localPhiMin,
                                int localPhiMax, int iEtaAbsMax) {
      for (int etaNr = localEtaMin; etaNr <= localEtaMax; etaNr++) {
        for (int phiNr = localPhiMin; phiNr <= localPhiMax; phiNr++) {
          const int towerIEta = l1t::CaloStage2Nav::offsetIEta(iEta, etaNr);
          const int towerIPhi = l1t::CaloStage2Nav::offsetIPhi(iPhi, phiNr);
          if (abs(towerIEta) <= iEtaAbsMax) {
            const l1t::CaloTower& tower = l1t::CaloTools::getTower(towers, towerIEta, towerIPhi);
            tower_tauIdx.push_back(tau_idx);
            tower_relEta.push_back(etaNr);
            tower_relPhi.push_back(phiNr);
            tower_hwEtEm.push_back(tower.hwEtEm());
            tower_hwEtHad.push_back(tower.hwEtHad());
            tower_hwPt.push_back(tower.hwPt());
          }
        }
      }
    };

    const auto fillTauEx = [&](const l1t::Tau& tau, size_t tau_idx) {
      const int iEta = tau.towerIEta();
      const int iPhi = tau.towerIPhi();

      const l1t::CaloCluster* main_cluster = nullptr;
      for(const auto& cluster : tauClusters) {
          if(cluster.isValid() && cluster.hwEta() == iEta && cluster.hwPhi() == iPhi) {
            main_cluster = &cluster;
            break;
          }
      }
      if(!main_cluster)
        throw cms::Exception("L1TNanoProducer") << "Tau cluster not found for tau at iEta = "
                                                << iEta << ", iPhi = " << iPhi;
      int isoLeftExtension = caloParams_->tauIsoAreaNrTowersEta();
      int isoRightExtension = caloParams_->tauIsoAreaNrTowersEta();
      if (main_cluster->checkClusterFlag(l1t::CaloCluster::TRIM_LEFT))
        ++isoRightExtension;
      else
        ++isoLeftExtension;


      // adapted from https://github.com/cms-sw/cmssw/blob/166eab1458287d877b7b06a931153f21b74e2093/L1Trigger/L1TCalorimeter/src/firmware/Stage2Layer2TauAlgorithmFirmwareImp1.cc#L332
      const int tau_hwEtSum = l1t::CaloTools::calHwEtSum(iEta, iPhi, towers,
                                                         -isoLeftExtension, isoRightExtension,
                                                         -1 * caloParams_->tauIsoAreaNrTowersPhi(),
                                                         caloParams_->tauIsoAreaNrTowersPhi(),
                                                         caloParams_->tauPUSParam(2),
                                                         l1t::CaloTools::CALO);
      hwEtSum.push_back(tau_hwEtSum);
      fillTowers(tau_idx, iEta, iPhi, -isoLeftExtension, isoRightExtension,
                 -1 * caloParams_->tauIsoAreaNrTowersPhi(), caloParams_->tauIsoAreaNrTowersPhi(),
                 caloParams_->tauPUSParam(2));
    };

    auto table = Fill(event, tausToken_, tauTableName, [&](const auto& it) {
      hwPt.push_back(it->hwPt());
      hwEta.push_back(it->hwEta());
      hwPhi.push_back(it->hwPhi());
      towerIEta.push_back(it->towerIEta());
      towerIPhi.push_back(it->towerIPhi());
      hwIso.push_back(it->hwIso());
      rawEt.push_back(it->rawEt());
      isoEt.push_back(it->isoEt());
      nTT.push_back(it->nTT());
      fillTauEx(*it, hwPt.size() - 1);
      l2_scores.push_back(getL2Score(it->eta(), it->phi()));
    });

    table->addColumn<int>("hwPt", hwPt, "hardware pt");
    table->addColumn<int>("hwEta", hwEta, "hardware eta");
    table->addColumn<int>("hwPhi", hwPhi, "hardware phi");
    table->addColumn<int>("towerIEta", towerIEta, "ieta of the seeding tower");
    table->addColumn<int>("towerIPhi", towerIPhi, "iphi of the seeding tower");
    table->addColumn<int>("hwIso", hwIso, "hardware isolation");
    table->addColumn<int>("rawEt", rawEt, "raw Et");
    table->addColumn<int>("isoEt", isoEt, "iso Et");
    table->addColumn<int>("nTT", nTT, "number of TTs");
    table->addColumn<int>("hwEtSum", hwEtSum, "hardware Et sum towers around tau (including signal and isolation)");
    table->addColumn<float>("l2Tag", l2_scores, "L2 NN tau tag score");
    event.put(std::move(table), tauTableName);

    auto towerTable = std::make_unique<nanoaod::FlatTable>(tower_tauIdx.size(), tauTowersTableName, false, false);
    towerTable->addColumn<int>("tauIdx", tower_tauIdx, "index of the tau in the collection");
    towerTable->addColumn<int>("relEta", tower_relEta, "relative eta of the tower");
    towerTable->addColumn<int>("relPhi", tower_relPhi, "relative phi of the tower");
    towerTable->addColumn<int>("hwEtEm", tower_hwEtEm, "hardware Et of the EM component of the tower");
    towerTable->addColumn<int>("hwEtHad", tower_hwEtHad, "hardware Et of the hadronic component of the tower");
    towerTable->addColumn<int>("hwPt", tower_hwPt, "hardware Et of the tower");
    event.put(std::move(towerTable), tauTowersTableName);
  }

private:
  const edm::EDGetTokenT<l1t::EGammaBxCollection> egammasToken_;
  const edm::EDGetTokenT<l1t::MuonBxCollection> muonsToken_;
  const edm::EDGetTokenT<l1t::JetBxCollection> jetsToken_;
  const edm::EDGetTokenT<l1t::TauBxCollection> tausToken_;
  const edm::EDGetTokenT<l1t::CaloTowerBxCollection> caloTowersToken_;
  const L2OutTokenCol l2TauToken_;
  const edm::ESGetToken<l1t::CaloParams, L1TCaloParamsRcd> candidateToken_;
  const edm::ESGetToken<l1t::CaloParams, L1TCaloParamsO2ORcd> o2oProtoToken_;
  const unsigned int precision_;

  std::optional<unsigned long long> caloParamsCacheId_;
  std::unique_ptr<l1t::CaloParamsHelper> caloParams_;
  std::unique_ptr<l1t::Stage2Layer2ClusterAlgorithmFirmwareImp1> tauClusterAlgo_;
  //std::unique_ptr<Stage2TowerDecompressAlgorithmFirmwareImp1> towerAlgo_;
};

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(L1TableProducer);