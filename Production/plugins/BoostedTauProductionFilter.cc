// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/stream/EDFilter.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/StreamID.h"
#include "DataFormats/PatCandidates/interface/Tau.h"

//
// class declaration
//

class BoostedTauProductionFilter : public edm::stream::EDFilter<> {
   public:
      explicit BoostedTauProductionFilter(const edm::ParameterSet&);
      ~BoostedTauProductionFilter();

      static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

   private:
      virtual void beginStream(edm::StreamID) override;
      virtual bool filter(edm::Event&, const edm::EventSetup&) override;
      virtual void endStream() override;

      //virtual void beginRun(edm::Run const&, edm::EventSetup const&) override;
      //virtual void endRun(edm::Run const&, edm::EventSetup const&) override;
      //virtual void beginLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&) override;
      //virtual void endLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&) override;

      // ----------member data ---------------------------
  edm::EDGetTokenT< std::vector<pat::Tau> > boostedTauCollection;
  bool verboseDebug;
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
BoostedTauProductionFilter::BoostedTauProductionFilter(const edm::ParameterSet& iConfig):
  boostedTauCollection(consumes< std::vector<pat::Tau> >(iConfig.getParameter< edm::InputTag >("boostedTauCollection") ) )
{
   //now do what ever initialization is needed
  verboseDebug = iConfig.exists("verboseDebug") ? iConfig.getParameter<bool>("verboseDebug"): false;
  if (verboseDebug) std::cout<<"Constructing boosted tau filter..."<<std::endl;
}


BoostedTauProductionFilter::~BoostedTauProductionFilter()
{
 
   // do anything here that needs to be done at destruction time
   // (e.g. close files, deallocate resources etc.)

}


//
// member functions
//

// ------------ method called on each new Event  ------------
bool
BoostedTauProductionFilter::filter(edm::Event& iEvent, const edm::EventSetup& iSetup)
{
   using namespace edm;
   /*
#ifdef THIS_IS_AN_EVENT_EXAMPLE
   Handle<ExampleData> pIn;
   iEvent.getByLabel("example",pIn);
#endif

#ifdef THIS_IS_AN_EVENTSETUP_EXAMPLE
   ESHandle<SetupData> pSetup;
   iSetup.get<SetupRecord>().get(pSetup);
#endif
   return true;
   */

   edm::Handle< std::vector<pat::Tau> > boostedTauHandle;
   iEvent.getByToken(boostedTauCollection, boostedTauHandle);

   //Our filter really only asks one very simple question. Is there a boosted tau in the event?
   if (verboseDebug) std::cout<<"boosted taus empty? "<<boostedTauHandle->empty()<<std::endl;
   if (boostedTauHandle->empty()) return false;
   return true;
   
   
}

// ------------ method called once each stream before processing any runs, lumis or events  ------------
void
BoostedTauProductionFilter::beginStream(edm::StreamID)
{
}

// ------------ method called once each stream after processing all runs, lumis and events  ------------
void
BoostedTauProductionFilter::endStream() {
}

// ------------ method called when starting to processes a run  ------------
/*
void
BoostedTauProductionFilter::beginRun(edm::Run const&, edm::EventSetup const&)
{ 
}
*/
 
// ------------ method called when ending the processing of a run  ------------
/*
void
BoostedTauProductionFilter::endRun(edm::Run const&, edm::EventSetup const&)
{
}
*/
 
// ------------ method called when starting to processes a luminosity block  ------------
/*
void
BoostedTauProductionFilter::beginLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&)
{
}
*/
 
// ------------ method called when ending the processing of a luminosity block  ------------
/*
void
BoostedTauProductionFilter::endLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&)
{
}
*/
 
// ------------ method fills 'descriptions' with the allowed parameters for the module  ------------
void
BoostedTauProductionFilter::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  //The following says we do not know what parameters are allowed so do no validation
  // Please change this to state exactly what you do use, even if it is no parameters
  edm::ParameterSetDescription desc;
  desc.setUnknown();
  descriptions.addDefault(desc);
}
//define this as a plug-in
DEFINE_FWK_MODULE(BoostedTauProductionFilter);
