#include "FWCore/Framework/interface/global/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/NanoAOD/interface/FlatTable.h"
#include "DataFormats/HLTReco/interface/TriggerFilterObjectWithRefs.h"
#include "DataFormats/Math/interface/deltaR.h"
#include "DataFormats/EgammaCandidates/interface/Electron.h"
#include "DataFormats/RecoCandidate/interface/RecoChargedCandidate.h"
#include "DataFormats/TauReco/interface/PFTau.h"
#include "DataFormats/RecoCandidate/interface/RecoEcalCandidate.h"

class FilterObjectTableProducer : public edm::global::EDProducer<> {
public:
  using ObjectCollection = trigger::TriggerFilterObjectWithRefs;
  using LorentzVectorM = math::PtEtaPhiMLorentzVector;

  struct InputDesc {
    edm::EDGetTokenT<ObjectCollection> token;
    std::string branchName;
    std::string description;
  };

  struct ObjectDesc {
    LorentzVectorM p4;
    std::vector<bool> matches;
  };

  static constexpr double deltaR2MatchingThreshold = std::pow(0.01, 2);
  static constexpr double ptMatchingThreshold = 0.01;
  static constexpr int precision = 10;

  FilterObjectTableProducer(const edm::ParameterSet& cfg) :
      inputs_(CreateInputDescs(cfg)),
      objectTypes_(cfg.getParameter<std::vector<int>>("types")),
      tableName_(cfg.getParameter<std::string>("tableName"))
  {
    extractMomenta();
    produces<nanoaod::FlatTable>();
  }

private:
  std::vector<InputDesc> CreateInputDescs(const edm::ParameterSet& cfg)
  {
    std::vector<InputDesc> inputs;
    for(const edm::ParameterSet& input_pset : cfg.getParameterSetVector("inputs")) {
      inputs.emplace_back(InputDesc{
        mayConsume<ObjectCollection>(input_pset.getParameter<edm::InputTag>("inputTag")),
        input_pset.getParameter<std::string>("branchName"),
        input_pset.getParameter<std::string>("description")
      });
    }
    return inputs;
  }

  void produce(edm::StreamID id, edm::Event& event, const edm::EventSetup& setup) const override
  {
    const auto& all_objects = extractMomenta(&event);

    const auto& merged_objects = mergeObjects(all_objects);
    const size_t n_objects = merged_objects.size();

    std::vector<float> pt(n_objects), eta(n_objects), phi(n_objects), mass(n_objects);
    std::vector<std::vector<bool>> filterBits(inputs_.size(), std::vector<bool>(n_objects, false));

    for(size_t obj_idx = 0; obj_idx < merged_objects.size(); ++obj_idx) {
      const auto& obj = merged_objects[obj_idx];
      pt[obj_idx] = obj.p4.pt();
      eta[obj_idx] = obj.p4.eta();
      phi[obj_idx] = obj.p4.phi();
      mass[obj_idx] = obj.p4.mass();
      for(size_t input_idx = 0; input_idx < inputs_.size(); ++input_idx) {
        filterBits[input_idx][obj_idx] = obj.matches[input_idx];
      }
    }

    auto table = std::make_unique<nanoaod::FlatTable>(n_objects, tableName_, false, false);
    table->addColumn<float>("pt", pt, "pt", precision);
    table->addColumn<float>("eta", eta, "eta", precision);
    table->addColumn<float>("phi", phi, "phi", precision);
    table->addColumn<float>("mass", mass, "mass", precision);
    for(size_t input_idx = 0; input_idx < inputs_.size(); ++input_idx) {
      table->addColumn<bool>(inputs_[input_idx].branchName, filterBits[input_idx], inputs_[input_idx].description);
    }
    event.put(std::move(table));
  }

std::vector<std::vector<LorentzVectorM>> extractMomenta(const edm::Event* event = nullptr) const {
    std::vector<std::vector<LorentzVectorM>> object_p4s(inputs_.size());
    for(size_t input_idx = 0; input_idx < inputs_.size(); ++input_idx) {
      const auto& input = inputs_[input_idx];
      const trigger::TriggerFilterObjectWithRefs* trigger_objects = nullptr;
      if (event) {
        const auto handle = event->getHandle(input.token);
        if (handle.isValid())
          trigger_objects = handle.product();
      }
      for (const int objectType : objectTypes_) {
        if (objectType == trigger::TriggerElectron) {
          extractMomenta<trigger::VRelectron>(trigger_objects, objectType, object_p4s[input_idx]);
        } else if (objectType == trigger::TriggerMuon) {
          extractMomenta<trigger::VRmuon>(trigger_objects, objectType, object_p4s[input_idx]);
        } else if (objectType == trigger::TriggerPhoton) {
          extractMomenta<trigger::VRphoton>(trigger_objects, objectType, object_p4s[input_idx]);
        } else if (objectType == trigger::TriggerL1Tau) {
          extractMomenta<l1t::TauVectorRef>(trigger_objects, objectType, object_p4s[input_idx]);
        } else if (objectType == trigger::TriggerL1Jet) {
          extractMomenta<l1t::JetVectorRef>(trigger_objects, objectType, object_p4s[input_idx]);
        } else if (objectType == trigger::TriggerTau) {
          extractMomenta<std::vector<reco::PFTauRef>>(trigger_objects, objectType, object_p4s[input_idx]);
        } else if (objectType == trigger::TriggerJet) {
          extractMomenta<std::vector<reco::PFJetRef>>(trigger_objects, objectType, object_p4s[input_idx]);
        } else
          throw cms::Exception("Invalid object type", "FilterObjectTableProducer::extractMomenta")
              << "Unsupported object type: " << objectType;
      }
    }
    return object_p4s;
  }

  template <typename Collection>
  static void extractMomenta(const trigger::TriggerRefsCollections* triggerObjects,
                             int objType,
                             std::vector<LorentzVectorM>& p4s) {
    if (triggerObjects) {
      Collection objects;
      triggerObjects->getObjects(objType, objects);
      for (const auto& obj : objects)
        p4s.push_back(obj->polarP4());
    }
  }

  std::vector<ObjectDesc> mergeObjects(const std::vector<std::vector<LorentzVectorM>>& object_p4s) const {
    std::vector<ObjectDesc> merged_objects;
    for(size_t input_idx = 0; input_idx < inputs_.size(); ++input_idx) {
      const auto& p4s = object_p4s[input_idx];
      for(const auto& p4 : p4s) {
        bool matched = false;
        for(auto& obj : merged_objects) {
          const double deltaR2 = reco::deltaR2(p4, obj.p4);
          const double deltaPt = std::abs(p4.pt() - obj.p4.pt());
          if(deltaR2 < deltaR2MatchingThreshold && deltaPt < ptMatchingThreshold) {
            matched = true;
            obj.matches[input_idx] = true;
            break;
          }
        }
        if (!matched) {
          merged_objects.push_back({p4, std::vector<bool>(inputs_.size(), false)});
          merged_objects.back().matches[input_idx] = true;
        }
      }
    }
    return merged_objects;
  }

private:
  const std::vector<InputDesc> inputs_;
  const std::vector<int> objectTypes_;
  const std::string tableName_;
};

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(FilterObjectTableProducer);