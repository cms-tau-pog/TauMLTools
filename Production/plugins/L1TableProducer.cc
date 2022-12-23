#include "FWCore/Framework/interface/global/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/NanoAOD/interface/FlatTable.h"
#include "DataFormats/L1Trigger/interface/EGamma.h"
#include "DataFormats/L1Trigger/interface/Tau.h"
#include "DataFormats/L1Trigger/interface/Jet.h"
#include "DataFormats/L1Trigger/interface/Muon.h"

class L1TableProducer : public edm::global::EDProducer<> {
public:
  L1TableProducer(const edm::ParameterSet& cfg) :
      egammasToken_(consumes<l1t::EGammaBxCollection>(cfg.getParameter<edm::InputTag>("egammas"))),
      muonsToken_(consumes<l1t::MuonBxCollection>(cfg.getParameter<edm::InputTag>("muons"))),
      jetsToken_(consumes<l1t::JetBxCollection>(cfg.getParameter<edm::InputTag>("jets"))),
      tausToken_(consumes<l1t::TauBxCollection>(cfg.getParameter<edm::InputTag>("taus"))),
      precision_(cfg.getParameter<int>("precision"))
  {
    produces<nanoaod::FlatTable>("L1Egamma");
    produces<nanoaod::FlatTable>("L1Muon");
    produces<nanoaod::FlatTable>("L1Jet");
    produces<nanoaod::FlatTable>("L1Tau");
  }

  void produce(edm::StreamID id, edm::Event& event, const edm::EventSetup& setup) const override
  {
    FillEgammas(event);
    FillMuons(event);
    FillJets(event);
    FillTaus(event);
  }

  template<typename T, typename F>
  std::unique_ptr<nanoaod::FlatTable> Fill(edm::Event& event, const edm::EDGetTokenT<T>& token,
                                           const std::string& name, F fn) const
  {
    const auto& collection = event.get(token);
    std::vector<float> pt, eta, phi;
    std::vector<int> hwQual;

    for(int bx = collection.getFirstBX(); bx <= collection.getLastBX(); ++bx) {
      if(bx != 0) continue;
      for(auto it = collection.begin(bx); it != collection.end(bx); ++it) {
        if(it->pt() <= 0) continue;
        pt.push_back(it->pt());
        eta.push_back(it->eta());
        phi.push_back(it->phi());
        hwQual.push_back(it->hwQual());
        fn(it);
      }
    }

    auto table = std::make_unique<nanoaod::FlatTable>(pt.size(), name, false, false);
    table->addColumn<float>("pt", pt, "transverse momentum", precision_);
    table->addColumn<float>("eta", eta, "pseudorapidity", precision_);
    table->addColumn<float>("phi", phi, "azimuthal angle", precision_);
    table->addColumn<int>("hwQual", hwQual, "hardware quality");
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
    std::vector<int> charge, hwIso, hwDXY;
    auto table = Fill(event, muonsToken_, name, [&](const auto& it) {
      ptUnconstrained.push_back(it->ptUnconstrained());
      charge.push_back(it->charge());
      hwIso.push_back(it->hwIso());
      hwDXY.push_back(it->hwDXY());
    });
    table->addColumn<float>("ptUnconstrained", ptUnconstrained, "unconstrained transverse momentum", precision_);
    table->addColumn<int>("charge", charge, "charge");
    table->addColumn<int>("hwIso", hwIso, "hardware isolation");
    table->addColumn<int>("hwDXY", hwDXY, "hardware transverse impact parameter");
    event.put(std::move(table), name);
  }

  void FillJets(edm::Event& event) const
  {
    static const std::string name = "L1Jet";
    std::vector<int> rawEt, seedEt, puEt;
    auto table = Fill(event, jetsToken_, name, [&](const auto& it) {
      rawEt.push_back(it->rawEt());
      seedEt.push_back(it->seedEt());
      puEt.push_back(it->puEt());
    });
    table->addColumn<int>("rawEt", rawEt, "raw (uncalibrated) cluster sum");
    table->addColumn<int>("seedEt", seedEt, "transverse energy of the seed");
    table->addColumn<int>("puEt", puEt, "transverse energy of the pile-up");
    event.put(std::move(table), name);
  }

  void FillTaus(edm::Event& event) const
  {
    static const std::string name = "L1Tau";
    std::vector<int> hwIso;
    auto table = Fill(event, tausToken_, name, [&](const auto& it) {
      hwIso.push_back(it->hwIso());
    });
    table->addColumn<int>("hwIso", hwIso, "hardware isolation");
    event.put(std::move(table), name);
  }

private:
  const edm::EDGetTokenT<l1t::EGammaBxCollection> egammasToken_;
  const edm::EDGetTokenT<l1t::MuonBxCollection> muonsToken_;
  const edm::EDGetTokenT<l1t::JetBxCollection> jetsToken_;
  const edm::EDGetTokenT<l1t::TauBxCollection> tausToken_;
  const unsigned int precision_;
};

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(L1TableProducer);