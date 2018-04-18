/*! Creates tuple for tau analysis.
*/

#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "CommonTools/UtilAlgos/interface/TFileService.h"
#include "DataFormats/PatCandidates/interface/Tau.h"
#include "SimDataFormats/GeneratorProducts/interface/GenEventInfoProduct.h"
#include "AnalysisDataFormats/TopObjects/interface/TtGenEvent.h"
#include "DataFormats/PatCandidates/interface/PackedCandidate.h"

#include "AnalysisTools/Core/include/Tools.h"
#include "AnalysisTools/Core/include/TextIO.h"
#include "TauML/Analysis/include/TauTuple.h"
#include "TauML/Production/include/GenTruthTools.h"
#include "TauML/Analysis/include/TauIdResults.h"



class TauTupleProducer : public edm::EDAnalyzer {
public:
    TauTupleProducer(const edm::ParameterSet& cfg) :
        isMC(cfg.getParameter<bool>("isMC")),
        saveGenTopInfo(cfg.getParameter<bool>("saveGenTopInfo")),
        genEvent_token(mayConsume<GenEventInfoProduct>(cfg.getParameter<edm::InputTag>("genEvent"))),
        topGenEvent_token(mayConsume<TtGenEvent>(cfg.getParameter<edm::InputTag>("topGenEvent"))),
        genParticles_token(consumes<std::vector<reco::GenParticle>>(cfg.getParameter<edm::InputTag>("genParticles"))),
        puInfo_token(mayConsume<std::vector<PileupSummaryInfo>>(cfg.getParameter<edm::InputTag>("puInfo"))),
        vertices_token(mayConsume<std::vector<reco::Vertex> >(cfg.getParameter<edm::InputTag>("vertices"))),
        rho_token(consumes<double>(cfg.getParameter<edm::InputTag>("rho"))),
        taus_token(mayConsume<std::vector<pat::Tau>>(cfg.getParameter<edm::InputTag>("taus"))),
        tauTuple("taus", &edm::Service<TFileService>()->file(), false)
    {
    }

private:
    virtual void analyze(const edm::Event& event, const edm::EventSetup&) override
    {
        tauTuple().run  = event.id().run();
        tauTuple().lumi = event.id().luminosityBlock();
        tauTuple().evt  = event.id().event();

        edm::Handle<std::vector<reco::Vertex>> vertices;
        event.getByToken(vertices_token, vertices);
        tauTuple().npv = vertices->size();
        edm::Handle<double> rho;
        event.getByToken(rho_token, rho);
        tauTuple().rho = *rho;

        if(isMC) {
            edm::Handle<GenEventInfoProduct> genEvent;
            event.getByToken(genEvent_token, genEvent);

            edm::Handle<std::vector<PileupSummaryInfo>> puInfo;
            event.getByToken(puInfo_token, puInfo);
            tauTuple().npu = analysis::gen_truth::GetNumberOfPileUpInteractions(puInfo);
            tauTuple().genEventWeight = genEvent->weight();

            if(saveGenTopInfo) {
                edm::Handle<TtGenEvent> topGenEvent;
                event.getByToken(topGenEvent_token, topGenEvent);
                if(topGenEvent.isValid()) {
                    analysis::GenEventType genEventType = analysis::GenEventType::Other;
                    if(topGenEvent->isFullHadronic())
                        genEventType = analysis::GenEventType::TTbar_Hadronic;
                    else if(topGenEvent->isSemiLeptonic())
                        genEventType = analysis::GenEventType::TTbar_SemiLeptonic;
                    else if(topGenEvent->isFullLeptonic())
                        genEventType = analysis::GenEventType::TTbar_Leptonic;
                    tauTuple().genEventType = static_cast<int>(genEventType);
                }
            }
        }

        edm::Handle<std::vector<pat::Tau>> taus;
        event.getByToken(taus_token, taus);
        for(const pat::Tau& tau : *taus) {
            static const bool id_names_printed = PrintTauIdNames(tau);
            (void)id_names_printed;

            tauTuple().pt = tau.p4().pt();
            tauTuple().eta = tau.p4().eta();
            tauTuple().phi = tau.p4().phi();
            tauTuple().mass = tau.p4().mass();
            tauTuple().charge = tau.charge();

            const auto packedLeadTauCand = dynamic_cast<const pat::PackedCandidate*>(tau.leadChargedHadrCand().get());
            tauTuple().dxy = packedLeadTauCand->dxy();
            tauTuple().dz = packedLeadTauCand->dz();

            if(isMC) {
                edm::Handle<std::vector<reco::GenParticle>> genParticles;
                event.getByToken(genParticles_token, genParticles);
                const auto match = analysis::gen_truth::LeptonGenMatch(tau.p4(), *genParticles);
                tauTuple().gen_match = static_cast<int>(match.first);
                const auto matched_p4 = match.second ? match.second->p4() : analysis::LorentzVectorXYZ();
                tauTuple().gen_pt = matched_p4.pt();
                tauTuple().gen_eta = matched_p4.eta();
                tauTuple().gen_phi = matched_p4.phi();
                tauTuple().gen_mass = matched_p4.mass();
            }

            tauTuple().decayMode = tau.decayMode();
            tauTuple().id_flags = CreateTauIdResults(tau).GetResultBits();
            FillRawTauIds(tau);

            tauTuple.Fill();
        }
    }

    virtual void endJob() override
    {
        tauTuple.Write();
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

    static analysis::TauIdResults CreateTauIdResults(const pat::Tau& tau)
    {
        analysis::TauIdResults results;
        const auto& descs = analysis::TauIdResults::GetResultDescriptors();
        for(size_t n = 0; n < descs.size(); ++n)
            results.SetResult(n, tau.tauID(descs.at(n).ToString()) > .5f);
        return results;
    }

    void FillRawTauIds(const pat::Tau& tau)
    {
#define VAR(type, name) tauTuple().name = tau.tauID(#name);
        RAW_TAU_IDS()
#undef VAR
    }


private:
    const bool isMC, saveGenTopInfo;

    edm::EDGetTokenT<GenEventInfoProduct> genEvent_token;
    edm::EDGetTokenT<TtGenEvent> topGenEvent_token;
    edm::EDGetTokenT<std::vector<reco::GenParticle>> genParticles_token;
    edm::EDGetTokenT<std::vector<PileupSummaryInfo>> puInfo_token;
    edm::EDGetTokenT<std::vector<reco::Vertex>> vertices_token;
    edm::EDGetTokenT<double> rho_token;
    edm::EDGetTokenT<std::vector<pat::Tau>> taus_token;

    tau_tuple::TauTuple tauTuple;
};

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(TauTupleProducer);
