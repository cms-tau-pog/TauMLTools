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

class L1TauProducer : public HLTFilter {
public:
  explicit L1TauProducer(const edm::ParameterSet& cfg) :
      HLTFilter(cfg),
      tausToken_(consumes<l1t::TauBxCollection>(cfg.getParameter<edm::InputTag>("taus")))
  {
  }

  bool hltFilter(edm::Event& event,
                 const edm::EventSetup& eventsetup,
                 trigger::TriggerFilterObjectWithRefs& filterproduct) const override
  {
    const auto& taus = event.getHandle(tausToken_);
    for(int bx_index = taus->getFirstBX(); bx_index <= taus->getLastBX(); ++bx_index) {
      const unsigned bx_index_shift = taus->begin(bx_index) - taus->begin();
      unsigned index_in_bx = 0;
      for(auto it = taus->begin(bx_index); it != taus->end(bx_index); ++it, ++index_in_bx) {
        if(it->pt() <= 0) continue;
        const l1t::TauRef tauRef(taus, bx_index_shift + index_in_bx);
        filterproduct.addObject(trigger::TriggerL1Tau, tauRef);
      }
    }
    return true;
  }

private:
  const edm::EDGetTokenT<l1t::TauBxCollection> tausToken_;
};

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(L1TauProducer);