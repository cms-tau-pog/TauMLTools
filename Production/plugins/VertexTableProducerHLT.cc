// based on https://github.com/cms-sw/cmssw/blob/master/PhysicsTools/NanoAOD/plugins/VertexTableProducer.cc

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/StreamID.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/NanoAOD/interface/FlatTable.h"

class VertexTableProducerHLT : public edm::stream::EDProducer<> {
public:
  explicit VertexTableProducerHLT(const edm::ParameterSet& cfg)
    : verticesToken_(consumes(cfg.getParameter<edm::InputTag>("src"))),
      vertexName_(cfg.getParameter<std::string>("name"))
  {
    produces<nanoaod::FlatTable>(vertexName_);
  }

private:
  void produce(edm::Event& event, const edm::EventSetup&) override
  {
    const auto& vertices = event.get(verticesToken_);
    std::vector<float> x, y, z, ndof, normalizedChi2;

    for(const auto& pv : vertices) {
      x.push_back(pv.position().x());
      y.push_back(pv.position().y());
      z.push_back(pv.position().z());
      ndof.push_back(pv.ndof());
      normalizedChi2.push_back(pv.normalizedChi2());
    }

    auto pvTable = std::make_unique<nanoaod::FlatTable>(x.size(), vertexName_, false, false);
    pvTable->addColumn<float>("x", x, "position x coordinate", 10);
    pvTable->addColumn<float>("y", y, "position y coordinate", 10);
    pvTable->addColumn<float>("z", z, "position z coordinate", 16);
    pvTable->addColumn<float>("ndof", ndof, "number of degree of freedom", 8);
    pvTable->addColumn<float>("chi2", normalizedChi2, "reduced chi2, i.e. chi2/ndof", 8);

    event.put(std::move(pvTable), vertexName_);
  }

private:
  const edm::EDGetTokenT<std::vector<reco::Vertex>> verticesToken_;
  const std::string vertexName_;
};

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(VertexTableProducerHLT);
