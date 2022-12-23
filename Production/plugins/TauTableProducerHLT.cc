#include "FWCore/Framework/interface/global/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/NanoAOD/interface/FlatTable.h"
#include "DataFormats/TauReco/interface/PFTauTransverseImpactParameterAssociation.h"
#include "DataFormats/TauReco/interface/TauDiscriminatorContainer.h"

class TauTableProducerHLT : public edm::global::EDProducer<> {
public:
  using TauCollection = edm::View<reco::BaseTau>;
  using TauIPVector = edm::AssociationVector<reco::PFTauRefProd, std::vector<reco::PFTauTransverseImpactParameterRef>>;
  using TauDiscrMap = reco::TauDiscriminatorContainer;
  // TauCollection = deeptau.TauCollection;
  // using TauDeepTauVector = edm::AssociationVector<reco::PFTauRefProd, std::vector<reco::TauDiscriminatorContainer>>;
  TauTableProducerHLT(const edm::ParameterSet& cfg) :
      tauToken_(consumes<TauCollection>(cfg.getParameter<edm::InputTag>("taus"))),
      tauIPToken_(consumes<TauIPVector>(cfg.getParameter<edm::InputTag>("tauTransverseImpactParameters"))),
      deepTauVSeToken_(consumes<TauDiscrMap>(cfg.getParameter<edm::InputTag>("deepTauVSe"))),
      deepTauVSmuToken_(consumes<TauDiscrMap>(cfg.getParameter<edm::InputTag>("deepTauVSmu"))),
      deepTauVSjetToken_(consumes<TauDiscrMap>(cfg.getParameter<edm::InputTag>("deepTauVSjet"))),
      precision_(cfg.getParameter<int>("precision"))
  {
    produces<nanoaod::FlatTable>("Tau");
  }

  void produce(edm::StreamID id, edm::Event& event, const edm::EventSetup& setup) const override
  {
    const auto tausHandle = event.getHandle(tauToken_);
    const auto& tausProductId = tausHandle.id();
    const auto& taus = *tausHandle;
    const auto& tausIP = event.get(tauIPToken_);
    const auto& deepTauVSeMap = event.get(deepTauVSeToken_);
    const auto& deepTauVSmuMap = event.get(deepTauVSmuToken_);
    const auto& deepTauVSjetMap = event.get(deepTauVSjetToken_);
    std::vector<float> deepTauVSe(taus.size());
    std::vector<float> deepTauVSmu(taus.size());
    std::vector<float> deepTauVSjet(taus.size());
    // source: RecoTauTag/RecoTau/plugins/PFTauTransverseImpactParameters.cc
    std::vector<float> dxy(tausIP.size());
    std::vector<float> dxy_error(tausIP.size());
    std::vector<float> dxy_Sig(tausIP.size());
    std::vector<float> ip3d(tausIP.size());
    std::vector<float> ip3d_error(tausIP.size());
    std::vector<float> ip3d_Sig(tausIP.size());
    std::vector<float> hasSecondaryVertex(tausIP.size());
    std::vector<float> flightLength_x(tausIP.size());
    std::vector<float> flightLength_y(tausIP.size());
    std::vector<float> flightLength_z(tausIP.size());
    std::vector<float> flightLengthSig(tausIP.size());
    std::vector<float> secondaryVertex_x(tausIP.size());
    std::vector<float> secondaryVertex_y(tausIP.size());
    std::vector<float> secondaryVertex_z(tausIP.size());
    std::vector<float> secondaryVertex_t(tausIP.size());

    // for (size_t tau_index = 0; tau_index < tauScore.size(); tau_index++) {
    //   scores[tau_index] = tauScore.ValueMap(TauCollection.at(tau_index));.output_desc;
    // }

    for(size_t tau_index = 0; tau_index < taus.size(); ++tau_index) {
      deepTauVSe[tau_index] = deepTauVSeMap.get(tausProductId, tau_index).rawValues.at(0);
      deepTauVSmu[tau_index] = deepTauVSmuMap.get(tausProductId, tau_index).rawValues.at(0);
      deepTauVSjet[tau_index] = deepTauVSjetMap.get(tausProductId, tau_index).rawValues.at(0);
      dxy[tau_index] = tausIP.value(tau_index)->dxy();
      dxy_error[tau_index] =tausIP.value(tau_index)->dxy_error();
      dxy_Sig[tau_index] =tausIP.value(tau_index)->dxy_Sig();
      ip3d[tau_index] =tausIP.value(tau_index)->ip3d();
      ip3d_error[tau_index] =tausIP.value(tau_index)->ip3d_error();
      ip3d_Sig[tau_index] =tausIP.value(tau_index)->ip3d_Sig();
      hasSecondaryVertex[tau_index] = tausIP.value(tau_index)->hasSecondaryVertex();
      flightLength_x[tau_index] = tausIP.value(tau_index)->flightLength().x();
      flightLength_y[tau_index] = tausIP.value(tau_index)->flightLength().y();
      flightLength_z[tau_index] = tausIP.value(tau_index)->flightLength().z();
      flightLengthSig[tau_index] = tausIP.value(tau_index)->flightLengthSig();
      if (hasSecondaryVertex[tau_index] > 0) {
        secondaryVertex_x[tau_index] = tausIP.value(tau_index)->secondaryVertex()->x();
        secondaryVertex_y[tau_index] = tausIP.value(tau_index)->secondaryVertex()->y();
        secondaryVertex_z[tau_index] = tausIP.value(tau_index)->secondaryVertex()->z();
      }
      else {
        secondaryVertex_x[tau_index] = -999.;
        secondaryVertex_y[tau_index] = -999.;
        secondaryVertex_z[tau_index] = -999.;
      };

      // secondaryVertex_t[tau_index] = tausIP.value(tau_index)->secondaryVertex().t();

    }

    auto tauTable = std::make_unique<nanoaod::FlatTable>(tausIP.size(), "Tau", false, true);
    tauTable->addColumn<float>("dxy", dxy, "tau transverse impact parameter", precision_);
    tauTable->addColumn<float>("dxy_error", dxy_error, " dxy_error ", precision_);
    tauTable->addColumn<float>("dxy_Sig", dxy_Sig, " dxy_Sig ", precision_);
    tauTable->addColumn<float>("ip3d", ip3d, " ip3d ", precision_);
    tauTable->addColumn<float>("ip3d_error", ip3d_error, " ip3d_error ", precision_);
    tauTable->addColumn<float>("ip3d_Sig", ip3d_Sig, " ip3d_Sig ", precision_);
    tauTable->addColumn<float>("hasSecondaryVertex", hasSecondaryVertex, " hasSecondaryVertex ", precision_);
    tauTable->addColumn<float>("flightLength_x", flightLength_x, "flightLength_x", precision_);
    tauTable->addColumn<float>("flightLength_y", flightLength_y, "flightLength_y", precision_);
    tauTable->addColumn<float>("flightLength_z", flightLength_z, "flightLength_z", precision_);
    tauTable->addColumn<float>("flightLengthSig", flightLengthSig, "flightLengthSig", precision_);
    tauTable->addColumn<float>("secondaryVertex_x", secondaryVertex_x, "secondaryVertex_x", precision_);
    tauTable->addColumn<float>("secondaryVertex_y", secondaryVertex_y, "secondaryVertex_y", precision_);
    tauTable->addColumn<float>("secondaryVertex_z", secondaryVertex_z, "secondaryVertex_z", precision_);
    tauTable->addColumn<float>("secondaryVertex_t", secondaryVertex_t, "secondaryVertex_t", precision_);
    tauTable->addColumn<float>("deepTauVSe", deepTauVSe, "tau vs electron discriminator", precision_);
    tauTable->addColumn<float>("deepTauVSmu", deepTauVSmu, "tau vs muon discriminator", precision_);
    tauTable->addColumn<float>("deepTauVSjet", deepTauVSjet, "tau vs jet discriminator", precision_);

    event.put(std::move(tauTable), "Tau");
  }

private:
  const edm::EDGetTokenT<TauCollection> tauToken_;
  const edm::EDGetTokenT<TauIPVector> tauIPToken_;
  const edm::EDGetTokenT<TauDiscrMap> deepTauVSeToken_, deepTauVSmuToken_, deepTauVSjetToken_;
  const unsigned int precision_;
};

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(TauTableProducerHLT);