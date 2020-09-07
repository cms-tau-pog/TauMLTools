// -*- C++ -*-
//
// Package:    TauMLTools/SlimmedElectronMerger
// Class:      SlimmedElectronMerger
//
/**\class SlimmedElectronMerger SlimmedElectronMerger.cc TauMLTools/SlimmedElectronMerger/plugins/SlimmedElectronMerger.cc

 Description: [one line class summary]

 Implementation:
     [Notes on implementation]
*/
//
// Original Author:  Artur Gottmann
//         Created:  Tue, 18 Aug 2020 14:06:07 GMT
//
//

// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/StreamID.h"

#include "FWCore/Utilities/interface/transform.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "DataFormats/Common/interface/CloneTrait.h"

#include "DataFormats/PatCandidates/interface/Electron.h"
#include "CommonTools/Utils/interface/PtComparator.h"

//
// class declaration
//

class SlimmedElectronMerger : public edm::stream::EDProducer<> {
public:
  SlimmedElectronMerger(const edm::ParameterSet&);
  ~SlimmedElectronMerger();

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  void produce(edm::Event&, const edm::EventSetup&) override;

  // ----------member data ---------------------------
  std::vector<edm::EDGetTokenT<pat::ElectronCollection>> electrons_token_vector;
  const GreaterByPt<pat::Electron> pTComparator_;
  
};

//
// constants, enums and typedefs
//


//
// static data member definitions
//

//
// constructors and destructor
//
SlimmedElectronMerger::SlimmedElectronMerger(const edm::ParameterSet& iConfig) :
  electrons_token_vector(edm::vector_transform(iConfig.getParameter<std::vector<edm::InputTag>>("src"), [this](edm::InputTag const& tag) { return consumes<pat::ElectronCollection>(tag); })),
  pTComparator_()
{
  //register products
  produces<pat::ElectronCollection>();
  //now do what ever other initialization is needed
}

SlimmedElectronMerger::~SlimmedElectronMerger() {
  // do anything here that needs to be done at destruction time
  // (e.g. close files, deallocate resources etc.)
  //
  // please remove this method altogether if it would be left empty
}

//
// member functions
//

// ------------ method called to produce the data  ------------
void SlimmedElectronMerger::produce(edm::Event& iEvent, const edm::EventSetup& iSetup) {
  //std::cout << "produce start " << std::endl;
  using namespace edm;
  auto mergedElectrons = std::make_unique<pat::ElectronCollection>();
  unsigned int electron_size = 0;
  for (auto  source = electrons_token_vector.begin(); source != electrons_token_vector.end(); ++source)
  {
    edm::Handle<pat::ElectronCollection> handle;
    iEvent.getByToken(*source, handle);
    //std::cout << "\tSize of handle: " << handle->size() << std::endl;
    electron_size +=  handle->size();
  }
  //std::cout << "Reserving space for " << electron_size << " electrons" << std::endl;
  mergedElectrons->reserve(electron_size);
  for (auto  source = electrons_token_vector.begin(); source != electrons_token_vector.end(); ++source)
  {
    edm::Handle<pat::ElectronCollection> handle;
    iEvent.getByToken(*source, handle);
    for (auto electron = handle->begin(); electron != handle->end(); ++electron)
    {
      //std::cout << "\t\t" << (*electron).pt() <<","<< (*electron).eta()<< std::endl;
      mergedElectrons->push_back((*electron));
    }
  }
  // sort electrons in pt
  std::sort(mergedElectrons->begin(), mergedElectrons->end(), pTComparator_);
  /*std::cout << "Size of merged electrons: " << mergedElectrons->size() << std::endl;
  //for (auto electron = mergedElectrons->begin(); electron != mergedElectrons->end(); ++electron)
  //{
  //  std::cout << "\t" << (*electron).pt() <<","<< (*electron).eta()<< std::endl;
  }*/
  iEvent.put(std::move(mergedElectrons));
  //std::cout << "produce end" << std::endl;
}

// ------------ method fills 'descriptions' with the allowed parameters for the module  ------------
void SlimmedElectronMerger::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  //The following says we do not know what parameters are allowed so do no validation
  // Please change this to state exactly what you do use, even if it is no parameters
  edm::ParameterSetDescription desc;
  desc.setUnknown();
  descriptions.addDefault(desc);
}

//define this as a plug-in
DEFINE_FWK_MODULE(SlimmedElectronMerger);
