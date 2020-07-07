/*! Creates tuple for tau analysis.
*/

#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "CommonTools/UtilAlgos/interface/TFileService.h"

#include "SimDataFormats/GeneratorProducts/interface/GenEventInfoProduct.h"
#include "DataFormats/PatCandidates/interface/PackedCandidate.h"
#include "DataFormats/PatCandidates/interface/Tau.h"


#include "TauMLTools/Production/interface/GenTruthTools.h"
#include "TauMLTools/Analysis/interface/TauIdResultTuple.h"
#include "TauMLTools/Analysis/interface/TauTuple.h"


namespace tau_analysis {

class DeepTauTest : public edm::EDAnalyzer {
public:
    DeepTauTest(const edm::ParameterSet& cfg) :
        isMC(cfg.getParameter<bool>("isMC")),
        genEvent_token(mayConsume<GenEventInfoProduct>(cfg.getParameter<edm::InputTag>("genEvent"))),
        genParticles_token(consumes<std::vector<reco::GenParticle>>(cfg.getParameter<edm::InputTag>("genParticles"))),
        taus_token(mayConsume<pat::TauCollection>(cfg.getParameter<edm::InputTag>("taus"))),
        output_tuple("taus", &edm::Service<TFileService>()->file(), false)
    {
    }

    virtual ~DeepTauTest() override {}

private:
    virtual void analyze(const edm::Event& event, const edm::EventSetup&) override
    {
        output_tuple().run  = event.id().run();
        output_tuple().lumi = event.id().luminosityBlock();
        output_tuple().evt  = event.id().event();

        if(isMC) {
            edm::Handle<GenEventInfoProduct> genEvent;
            event.getByToken(genEvent_token, genEvent);
            output_tuple().genEventWeight = genEvent->weight();
        } else {
            output_tuple().genEventWeight = 1;
        }

        edm::Handle<pat::TauCollection> taus;
        event.getByToken(taus_token, taus);

        unsigned tau_index = 0;
        for(const pat::Tau& tau : *taus) {
            static const bool id_names_printed = PrintTauIdNames(tau);
            (void)id_names_printed;

            static constexpr float default_value = tau_tuple::DefaultFillValue<float>();
            auto leadChargedHadrCand = dynamic_cast<const pat::PackedCandidate*>(tau.leadChargedHadrCand().get());

            output_tuple().tau_index = tau_index++;
            output_tuple().pt = static_cast<float>(tau.p4().pt());
            output_tuple().eta = static_cast<float>(tau.p4().eta());
            output_tuple().phi = static_cast<float>(tau.p4().phi());
            output_tuple().decayModeFinding = tau.tauID("decayModeFinding") > 0.5;
            output_tuple().decayMode = tau.decayMode();
            output_tuple().dxy = tau.dxy();
            output_tuple().dz = leadChargedHadrCand ? leadChargedHadrCand->dz() : default_value;
            output_tuple().dxy_pca_x = tau.dxy_PCA().x();
            output_tuple().dxy_pca_y = tau.dxy_PCA().y();
            output_tuple().dxy_pca_z = tau.dxy_PCA().z();

            if(isMC) {
                edm::Handle<std::vector<reco::GenParticle>> genParticles;
                event.getByToken(genParticles_token, genParticles);
                const auto match = gen_truth::LeptonGenMatch(tau.polarP4(), *genParticles);
                const auto gen_match = match.match;

                output_tuple().gen_e = gen_match == GenLeptonMatch::Electron
                        || gen_match == GenLeptonMatch::TauElectron;
                output_tuple().gen_mu = gen_match == GenLeptonMatch::Muon
                        || gen_match == GenLeptonMatch::TauMuon;
                output_tuple().gen_tau = gen_match == GenLeptonMatch::Tau;
                output_tuple().gen_jet = gen_match == GenLeptonMatch::NoMatch;
            }

            output_tuple().byDeepTau2017v2VSeraw = tau.tauID("byDeepTau2017v2VSeraw");
            output_tuple().byDeepTau2017v2VSmuraw = tau.tauID("byDeepTau2017v2VSmuraw");
            output_tuple().byDeepTau2017v2VSjetraw = tau.tauID("byDeepTau2017v2VSjetraw");

            output_tuple.Fill();
        }
    }

    virtual void endJob() override
    {
        output_tuple.Write();
    }

private:
    static bool PrintTauIdNames(const pat::Tau& tau)
    {
        static const std::string header(40, '-');

        std::set<std::string> tauId_names;
        for(const auto& id : tau.tauIDs())
            tauId_names.insert(id.first);
        std::cout << "Tau IDs:\n" << header << "\n";
        for(const std::string& name : tauId_names)
            std::cout << name << "\n";
        std::cout << header << std::endl;

        return true;
    }

private:
    const bool isMC;

    edm::EDGetTokenT<GenEventInfoProduct> genEvent_token;
    edm::EDGetTokenT<std::vector<reco::GenParticle>> genParticles_token;
    edm::EDGetTokenT<pat::TauCollection> taus_token;

    tau_tuple::TauIdResultTuple output_tuple;
};

} // namespace tau_analysis

#include "FWCore/Framework/interface/MakerMacros.h"
using DeepTauTest = tau_analysis::DeepTauTest;
DEFINE_FWK_MODULE(DeepTauTest);
