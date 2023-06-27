#include "FWCore/Framework/interface/global/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/BTauReco/interface/JetTag.h"
#include "DataFormats/Math/interface/deltaR.h"
#include "DataFormats/NanoAOD/interface/FlatTable.h"


class JetTableProducerHLT : public edm::global::EDProducer<> {
public:
  using JetCollection = edm::View<reco::Jet>;
  using JetTagCollection = reco::JetTagCollection;
  using JetTagTokenCollection = std::map<std::string, edm::EDGetTokenT<JetTagCollection>>;

  static constexpr float default_tag_value = -1.f;

  JetTableProducerHLT(const edm::ParameterSet& cfg) :
      jetsToken_(consumes<JetCollection>(cfg.getParameter<edm::InputTag>("jets"))),
      looseJetsToken_(consumes<JetCollection>(cfg.getParameter<edm::InputTag>("looseJets"))),
      tightJetsToken_(consumes<JetCollection>(cfg.getParameter<edm::InputTag>("tightJets"))),
      jetTagTokens_(loadTagTokens(cfg.getParameterSet("jetTags"))),
      maxDeltaR2_(std::pow(cfg.getParameter<double>("maxDeltaR"), 2)),
      precision_(cfg.getParameter<int>("precision"))
  {
    produces<nanoaod::FlatTable>("Jet");
  }

private:
  JetTagTokenCollection loadTagTokens(const edm::ParameterSet& cfg)
  {
    JetTagTokenCollection tokens;
    for(const auto& tag_name : cfg.getParameterNames())
      tokens[tag_name] = consumes<JetTagCollection>(cfg.getParameter<edm::InputTag>(tag_name));
    return tokens;
  }

  bool hasMatch(const reco::Jet& jet, const JetCollection& jets) const
  {
    for(const auto& other_jet : jets) {
      if(reco::deltaR2(jet, other_jet) < maxDeltaR2_)
        return true;
    }
    return false;
  }

  float getTag(const reco::Jet& jet, const JetTagCollection& jetTags) const
  {
    float best_value = default_tag_value;
    double best_dr2 = maxDeltaR2_;
    for(const auto& [other_jet, tag_value] : jetTags) {
      auto const dr2 = reco::deltaR2(jet, *other_jet);
      if (dr2 < best_dr2) {
        best_dr2 = dr2;
        best_value = tag_value;
      }
    }
    return best_value;
  }

  void produce(edm::StreamID id, edm::Event& event, const edm::EventSetup& setup) const override
  {
    const auto& jets = event.get(jetsToken_);
    const auto& looseJets = event.get(looseJetsToken_);
    const auto& tightJets = event.get(tightJetsToken_);

    std::vector<bool> isLooseJet(jets.size(), false);
    std::vector<bool> isTightJet(jets.size(), false);
    std::map<std::string, std::vector<float>> jetTags;

    for(size_t jet_idx = 0; jet_idx < jets.size(); ++jet_idx) {
      isLooseJet[jet_idx] = hasMatch(jets[jet_idx], looseJets);
      isTightJet[jet_idx] = hasMatch(jets[jet_idx], tightJets);
    }

    for(const auto& [tag_name, tag_token] : jetTagTokens_) {
      const auto& jetTagCollection = event.get(tag_token);
      auto& tags = jetTags[tag_name];
      tags.resize(jets.size(), default_tag_value);
      for(size_t jet_idx = 0; jet_idx < jets.size(); ++jet_idx)
        tags[jet_idx] = getTag(jets[jet_idx], jetTagCollection);
    }

    auto jetTable = std::make_unique<nanoaod::FlatTable>(jets.size(), "Jet", false, true);
    jetTable->addColumn<bool>("looseId", isLooseJet, "passes loose jet ID");
    jetTable->addColumn<bool>("tightId", isTightJet, "passes tight jet ID");
    for(const auto& [tag_name, tag_values] : jetTags)
      jetTable->addColumn<float>(tag_name, tag_values, tag_name + " score");

    event.put(std::move(jetTable), "Jet");
  }

private:
  const edm::EDGetTokenT<JetCollection> jetsToken_, looseJetsToken_, tightJetsToken_;
  const JetTagTokenCollection jetTagTokens_;
  const double maxDeltaR2_;
  const unsigned int precision_;
};

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(JetTableProducerHLT);