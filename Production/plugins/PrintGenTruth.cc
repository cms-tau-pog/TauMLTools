/*! Print gen truth.
Based on https://github.com/cms-sw/cmssw/blob/CMSSW_8_0_X/PhysicsTools/HepMCCandAlgos/plugins/ParticleTreeDrawer.cc.
Original author: Luca Lista, INFN.
This file is part of https://github.com/hh-italian-group/AnalysisTools. */

#include <numeric>
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "SimGeneral/HepPDTRecord/interface/ParticleDataTable.h"
#include "DataFormats/Common/interface/View.h"
#include "DataFormats/Candidate/interface/Candidate.h"
#include "DataFormats/HepMCCandidate/interface/GenParticle.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/Event.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Utilities/interface/EDMException.h"
#include "SimDataFormats/GeneratorProducts/interface/LHEEventProduct.h"

class PrintGenTruth : public edm::EDAnalyzer {
public:
    using GenParticle = reco::GenParticle;
    using GenParticleVector = std::vector<GenParticle>;

    PrintGenTruth(const edm::ParameterSet& cfg) :
        particles_token(consumes<GenParticleVector>(cfg.getParameter<edm::InputTag>("genParticles"))),
        lheEventProduct_token(mayConsume<LHEEventProduct>(cfg.getParameter<edm::InputTag>("lheEventProduct"))),
        printTree(cfg.getUntrackedParameter<bool>("printTree", true)),
        printLHE(cfg.getUntrackedParameter<bool>("printLHE", true)),
        printPDT(cfg.getUntrackedParameter<bool>("printPDT", false))
    {
        const auto v_status = cfg.getUntrackedParameter<std::vector<int>>( "status", std::vector<int>());
        statusToAccept.insert(v_status.begin(), v_status.end());
    }

private:
    virtual void beginJob() override
    {
        PrintStatusBitsLegend(std::cout);
    }

    virtual void analyze(const edm::Event& event, const edm::EventSetup& es) override
    {
        std::ostream& os = std::cout;
        os << std::boolalpha;
        es.getData(pdt);


        if(printTree) {
            edm::Handle<GenParticleVector> particles_handle;
            event.getByToken(particles_token, particles_handle);
            particles = &*particles_handle;
            for(const auto& p : *particles) {
                if(!Accept(p) || p.mother() != 0) continue;
                os << "Gen particles decay tree:" << std::endl;
                PrintDecay(p, "", os);
                os << std::endl;
            }
        }

        if(printLHE) {
            edm::Handle<LHEEventProduct> lheEventProduct;
            if(event.getByToken(lheEventProduct_token, lheEventProduct)) {
                os << "Les Houches table:" << std::endl;
                PrintLesHouches(lheEventProduct->hepeup(), os);
                os << std::endl;
            }
        }

        if(printPDT) {
            os << "Particle data table:" << std::endl;
            PrintParticleTable(os);
            os << std::endl;
        }
    }

    std::string GetParticleName(int id) const
    {
        const ParticleData* pd = pdt->particle(id);
        return pd ? pd->name() : "";
    }

    void PrintDecay(const GenParticle& particle, const std::string& pre, std::ostream& os) const
    {
        os << GetParticleName(particle.pdgId()) << " <" << particle.pdgId() << ">";
        PrintInfo(particle, os);
        os << std::endl;

        for(auto d_iter = particle.begin(); d_iter != particle.end(); ++d_iter) {
            const GenParticle& daughter = *dynamic_cast<const GenParticle*>(&(*d_iter));
            if(!Accept(daughter)) continue;
            os << pre << "+-> ";
            const char pre_first = std::next(d_iter) == particle.end() ? ' ' : '|';
            const std::string pre_d = pre + pre_first + "   ";
            PrintDecay(daughter, pre_d, os);
        }
    }

    void PrintInfo(const GenParticle& p, std::ostream& os) const
    {
        os << " pt=" << p.pt() << " eta=" << p.eta() << " phi=" << p.phi() << " E=" << p.energy() << " m=" << p.mass()
           << " vx=" << p.vx() << " vy=" << p.vy() << " vz=" << p.vz() << " index=" << GetIndex(p)
           << " status=" << p.status() << " statusFlags=" << p.statusFlags().flags_;
    }

    bool Accept(const GenParticle& particle) const
    {
        return statusToAccept.size() == 0 || statusToAccept.count(particle.status());
    }

    size_t GetIndex(const GenParticle& particle) const
    {
        const auto iter = std::find_if(particles->begin(), particles->end(),
                                       [&](const GenParticle& p) { return &p == &particle; });
        if(iter == particles->end())
            throw std::runtime_error("Particle index not found.");
         return iter - particles->begin();
    }

    void PrintStatusBitsLegend(std::ostream& os) const
    {
        static const char v_sep = '|', h_sep = '-';
        using StatusBits = reco::GenStatusFlags::StatusBits;
        using NameMap = std::map<StatusBits, std::string>;

        static const NameMap statusBits {
            { StatusBits::kIsPrompt, "IsPrompt" },
            { StatusBits::kIsDecayedLeptonHadron, "IsDecayedLeptonHadron" },
            { StatusBits::kIsTauDecayProduct, "IsTauDecayProduct" },
            { StatusBits::kIsPromptTauDecayProduct, "IsPromptTauDecayProduct" },
            { StatusBits::kIsDirectTauDecayProduct, "IsDirectTauDecayProduct" },
            { StatusBits::kIsDirectPromptTauDecayProduct, "IsDirectPromptTauDecayProduct" },
            { StatusBits::kIsDirectHadronDecayProduct, "IsDirectHadronDecayProduct" },
            { StatusBits::kIsHardProcess, "IsHardProcess" },
            { StatusBits::kFromHardProcess, "FromHardProcess" },
            { StatusBits::kIsHardProcessTauDecayProduct , "IsHardProcessTauDecayProduct" },
            { StatusBits::kIsDirectHardProcessTauDecayProduct, "IsDirectHardProcessTauDecayProduct" },
            { StatusBits::kFromHardProcessBeforeFSR, "FromHardProcessBeforeFSR" },
            { StatusBits::kIsFirstCopy, "IsFirstCopy" },
            { StatusBits::kIsLastCopy, "IsLastCopy" },
            { StatusBits::kIsLastCopyBeforeFSR, "IsLastCopyBeforeFSR" }
        };

        std::vector<size_t> lengths(statusBits.size());
        std::transform(statusBits.begin(), statusBits.end(), lengths.begin(),
                       [](const NameMap::value_type& pair) { return pair.second.size(); });
        const size_t total_length = std::accumulate(lengths.begin(), lengths.end(), 0) + lengths.size() * 3 + 1;
        const std::string h_line(total_length, h_sep);
        os << "Status flag bits legend:" << std::endl << h_line << std::endl;
        for(size_t n = 0; n < statusBits.size(); ++n) {
            const size_t k = statusBits.size() - n - 1;
            os << v_sep << std::setw(lengths.at(k) + 1) << k << " ";
        }
        os << v_sep << std::endl << h_line << std::endl;

        size_t k = statusBits.size() - 1;
        for(auto iter = statusBits.rbegin(); iter != statusBits.rend(); ++iter, --k)
            os << v_sep << std::setw(lengths.at(k) + 1) << iter->second << " ";
        os << v_sep << std::endl << h_line << std::endl << std::endl;
    }

    void PrintLesHouches(const lhef::HEPEUP& lheEvent, std::ostream& os) const
    {
        static const std::vector<std::pair<std::string, size_t>> columns = {
            { "index", 10 }, { "name", 10 }, { "pdgId", 10 }, { "status", 10 }, { "pt", 12 }, { "eta", 12 },
            { "phi", 12 }, { "E", 12 }, { "mass", 12 }, { "mother_1", 10 }, { "mother_2", 10 }
        };

        const std::vector<lhef::HEPEUP::FiveVector>& lheParticles = lheEvent.PUP;

        os << std::left;
        for(const auto& column : columns)
            os << std::setw(column.second) << column.first;
        os << std::endl;

        for(size_t n = 0; n < lheParticles.size(); ++n) {
            const math::XYZTLorentzVector p4(lheParticles.at(n)[0], lheParticles.at(n)[1], lheParticles.at(n)[2],
                                             lheParticles.at(n)[3]);
            size_t k = 0;
            const auto w = [&](std::ostream& s) -> std::ostream& { s << std::setw(columns.at(k++).second); return s; };

            const int pdgId = lheEvent.IDUP.at(n);
            w(os) << n;
            w(os) << GetParticleName(pdgId);
            w(os) << pdgId;
            w(os) << lheEvent.ISTUP.at(n);
            w(os) << p4.Pt();
            w(os) << p4.Eta();
            w(os) << p4.Phi();
            w(os) << p4.E();
            w(os) << p4.M();
            w(os) << lheEvent.MOTHUP.at(n).first - 1;
            w(os) << lheEvent.MOTHUP.at(n).second - 1;
            os << std::endl;
        }
    }

    void PrintParticleTable(std::ostream& os) const
    {
        static const std::vector<std::pair<std::string, size_t>> columns = {
            { "pdgId", 12 }, { "name", 20 }, { "charge", 10 }, { "color", 10 }, { "tot_spin", 10 }, { "mass", 10 },
            { "width", 12 }, { "lifetime", 12 }, { "Type", 10 }
        };

        os << std::left << std::setprecision(3);
        for(const auto& column : columns)
            os << std::setw(column.second) << column.first;
        os << std::endl;

        static const auto type = [](const HepPDT::ParticleData& d) -> std::string {
            if(d.isMeson()) return "meson";
            if(d.isBaryon()) return "baryon";
            if(d.isDiQuark()) return "diquark";
            if(d.isHadron()) return "hadron";
            if(d.isLepton()) return "lepton";
            if(d.isNucleus()) return "nucleus";
            return "";
        };

        for(const auto& entry : *pdt) {
            const auto& data = entry.second;

            size_t k = 0;
            const auto w = [&](std::ostream& s) -> std::ostream& { s << std::setw(columns.at(k++).second); return s; };

            w(os) << data.pid();
            w(os) << data.name();
            w(os) << data.charge();
            w(os) << data.color();
            w(os) << data.spin().totalSpin();
            w(os) << data.mass();
            w(os) << data.totalWidth();
            w(os) << data.lifetime();
            w(os) << type(data);
            os << std::endl;
        }
    }

private:
    edm::EDGetTokenT<GenParticleVector> particles_token;
    edm::EDGetTokenT<LHEEventProduct> lheEventProduct_token;
    edm::EDGetTokenT<bool> printParticleTable_token;
    edm::ESHandle<ParticleDataTable> pdt;
    std::set<int> statusToAccept;
    const bool printTree, printLHE, printPDT;
    const GenParticleVector* particles;
};

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(PrintGenTruth);
