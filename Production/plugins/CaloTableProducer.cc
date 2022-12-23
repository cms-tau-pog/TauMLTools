#include "FWCore/Framework/interface/global/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/NanoAOD/interface/FlatTable.h"

#include "DataFormats/EcalRecHit/interface/EcalRecHit.h"
#include "DataFormats/EcalRecHit/interface/EcalRecHitCollections.h"
#include "DataFormats/HcalRecHit/interface/HBHERecHit.h"
#include "DataFormats/HcalRecHit/interface/HORecHit.h"
#include "DataFormats/HcalRecHit/interface/HcalRecHitCollections.h"
#include "Geometry/CaloGeometry/interface/CaloCellGeometry.h"
#include "Geometry/CaloGeometry/interface/CaloGeometry.h"
#include "Geometry/Records/interface/CaloGeometryRecord.h"


class CaloTableProducer : public edm::global::EDProducer<> {
public:
  CaloTableProducer(const edm::ParameterSet& cfg) :
      hbheToken_(consumes<HBHERecHitCollection>(cfg.getParameter<edm::InputTag>("hbhe"))),
      hoToken_(consumes<HORecHitCollection>(cfg.getParameter<edm::InputTag>("ho"))),
      ebToken_(consumes<EcalRecHitCollection>(cfg.getParameter<edm::InputTag>("eb"))),
      eeToken_(consumes<EcalRecHitCollection>(cfg.getParameter<edm::InputTag>("ee"))),
      geometryToken_(esConsumes<CaloGeometry, CaloGeometryRecord>()),
      precision_(cfg.getParameter<int>("precision"))
  {
    produces<nanoaod::FlatTable>("RecHitHBHE");
    produces<nanoaod::FlatTable>("RecHitHO");
    produces<nanoaod::FlatTable>("RecHitEB");
    produces<nanoaod::FlatTable>("RecHitEE");
  }

  void produce(edm::StreamID id, edm::Event& event, const edm::EventSetup& setup) const override
  {
    const auto& geometry = setup.getData(geometryToken_);

    FillHBHE(event, geometry);
    FillHO(event, geometry);
    FillEcal(event, ebToken_, geometry, "RecHitEB");
    FillEcal(event, eeToken_, geometry, "RecHitEE");
  }

  template<typename T, typename F>
  std::unique_ptr<nanoaod::FlatTable> Fill(const edm::Event& event, const edm::EDGetTokenT<T>& token,
                                           const CaloGeometry& geometry, const std::string& name, F fn) const
  {
    const auto& recHits = event.get(token);
    std::vector<float> rho, eta, phi, energy, time;

    for(const auto& hit : recHits) {
      const auto& position = geometry.getGeometry(hit.id())->getPosition();
      rho.push_back(position.perp());
      eta.push_back(position.eta());
      phi.push_back(position.phi());
      energy.push_back(hit.energy());
      time.push_back(hit.time());
      fn(hit);
    }

    auto table = std::make_unique<nanoaod::FlatTable>(rho.size(), name, false, false);
    table->addColumn<float>("rho", rho, "rho", precision_);
    table->addColumn<float>("eta", eta, "eta", precision_);
    table->addColumn<float>("phi", phi, "phi", precision_);
    table->addColumn<float>("energy", energy, "energy", precision_);
    table->addColumn<float>("time", time, "time", precision_);

    return table;
  }

  void FillHBHE(edm::Event& event, const CaloGeometry& geometry) const
  {
    static const std::string name = "RecHitHBHE";
    std::vector<float> eraw, eaux, rho_front, eta_front, phi_front, chi2, timeFalling;
    std::vector<bool> isMerged;
    auto table = Fill(event, hbheToken_, geometry, name, [&](const HBHERecHit& hit) {
      const auto& positionFront = geometry.getGeometry(hit.idFront())->getPosition();
      eraw.push_back(hit.eraw());
      eaux.push_back(hit.eaux());
      rho_front.push_back(positionFront.perp());
      eta_front.push_back(positionFront.eta());
      phi_front.push_back(positionFront.phi());
      chi2.push_back(hit.chi2());
      timeFalling.push_back(hit.timeFalling());
      isMerged.push_back(hit.isMerged());
    });

    table->addColumn<float>("eraw", eraw, "eraw", precision_);
    table->addColumn<float>("eaux", eaux, "eaux", precision_);
    table->addColumn<float>("rho_front", rho_front, "rho_front", precision_);
    table->addColumn<float>("eta_front", eta_front, "eta_front", precision_);
    table->addColumn<float>("phi_front", phi_front, "phi_front", precision_);
    table->addColumn<float>("chi2", chi2, "chi2", precision_);
    table->addColumn<float>("timeFalling", timeFalling, "timeFalling", precision_);
    table->addColumn<bool>("isMerged", isMerged, "isMerged", precision_);

    event.put(std::move(table), name);
  }

  void FillHO(edm::Event& event, const CaloGeometry& geometry) const
  {
    static const std::string name = "RecHitHO";
    auto table = Fill(event, hoToken_, geometry, name, [](const HORecHit&){});
    event.put(std::move(table), name);
  }

  void FillEcal(edm::Event& event, const edm::EDGetTokenT<EcalRecHitCollection>& token,
                const CaloGeometry& geometry, const std::string& name) const
  {
    std::vector<float> energyError, timeError, chi2;
    std::vector<bool> isRecovered, isTimeValid, isTimeErrorValid;
    auto table = Fill(event, token, geometry, name, [&](const EcalRecHit& hit) {
      energyError.push_back(hit.energyError());
      timeError.push_back(hit.timeError());
      chi2.push_back(hit.chi2());
      isRecovered.push_back(hit.isRecovered());
      isTimeValid.push_back(hit.isTimeValid());
      isTimeErrorValid.push_back(hit.isTimeErrorValid());
    });

    table->addColumn<float>("energyError", energyError, "energyError", precision_);
    table->addColumn<float>("timeError", timeError, "timeError", precision_);
    table->addColumn<float>("chi2", chi2, "chi2", precision_);
    table->addColumn<bool>("isRecovered", isRecovered, "isRecovered", precision_);
    table->addColumn<bool>("isTimeValid", isTimeValid, "isTimeValid", precision_);
    table->addColumn<bool>("isTimeErrorValid", isTimeErrorValid, "isTimeErrorValid", precision_);

    event.put(std::move(table), name);
  }

private:
  const edm::EDGetTokenT<HBHERecHitCollection> hbheToken_;
  const edm::EDGetTokenT<HORecHitCollection> hoToken_;
  const edm::EDGetTokenT<EcalRecHitCollection> ebToken_;
  const edm::EDGetTokenT<EcalRecHitCollection> eeToken_;
  const edm::ESGetToken<CaloGeometry, CaloGeometryRecord> geometryToken_;
  const unsigned int precision_;
};

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(CaloTableProducer);