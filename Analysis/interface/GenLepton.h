/*! Definition of lepton (ele, mu or tau) at the generator level.
Author: Konstantin Androsov
*/

#pragma once

#include <bitset>

#include <Math/LorentzVector.h>
#include <Math/PtEtaPhiM4D.h>
#include <Math/Point3D.h>
#include <Math/GenVector/Cartesian3D.h>


namespace reco_tau {
namespace gen_truth {

using LorentzVectorXYZ = ROOT::Math::LorentzVector<ROOT::Math::PxPyPzE4D<double>>;
using LorentzVectorM = ROOT::Math::LorentzVector<ROOT::Math::PtEtaPhiM4D<double>>;
using Point3D = ROOT::Math::PositionVector3D<ROOT::Math::Cartesian3D<double>>;

class GenParticle {
public:
    enum class PdgId {
        electron = 11, electron_neutrino = 12, muon = 13, muon_neutrino = 14, tau = 15, tau_neutrino = 16,
        photon = 22, pi0 = 111, pi = 211, K0_L = 130, K0_S = 310, K0 = 311, K = 321,
    };

    static const std::set<PdgId>& ChargedLeptons() {
        static const std::set<PdgId> s = { PdgId::electron, PdgId::muon, PdgId::tau };
        return s;
    }

    static const std::set<PdgId>& NeutralLeptons() {
        static const std::set<PdgId> s = { PdgId::electron_neutrino, PdgId::muon_neutrino, PdgId::tau_neutrino };
        return s;
    }

    static const std::set<PdgId>& ChargedHadrons() {
        static const std::set<PdgId> s = { PdgId::pi, PdgId::K };
        return s;
    }

    static const std::set<PdgId>& NeutralHadrons() {
        static const std::set<PdgId> s = { PdgId::pi0, PdgId::K0_L, PdgId::K0_S, PdgId::K0 };
        return s;
    }

    int pdgId{0};
    int charge{0};
    int status{0};
    bool isFirstCopy{false}, isLastCopy{false};
    LorentzVectorM p4;
    Point3D vertex;
    std::set<const GenParticle*> mothers;
    std::set<const GenParticle*> daughters;

    PdgId pdgCode() const { return static_cast<PdgId>(std::abs(pdgId)); }
};

inline std::ostream& operator<<(std::ostream& os, const GenParticle& p)
{
    os << "pdgId=" << p.pdgId << " status=" << p.status  << " pt=" << p.p4.pt() << " eta=" << p.p4.eta() << " phi=" << p.p4.phi()
       << " E=" << p.p4.energy() << " m=" << p.p4.mass()
       << " vx=" << p.vertex.x() << " vy=" << p.vertex.y() << " vz=" << p.vertex.z() << " vrho=" << p.vertex.rho()
       << " vr=" << p.vertex.r() << " q=" << p.charge;
    return os;
}

class GenLepton {
public:
    enum class Kind { PromptElectron = 1, PromptMuon = 2, TauDecayedToElectron = 3, TauDecayedToMuon = 4,
                      TauDecayedToHadrons = 5, Other = 6 };

    static constexpr size_t MaxNumberOfParticles = 10000;

    template<typename GenParticleT>
    static std::vector<GenLepton> fromGenParticleCollection(const std::vector<GenParticleT>& gen_particles, bool isGeantPlusSim_)
    {
        std::vector<GenLepton> leptons;
        std::map<const GenParticleT*, int> processed_particles;
        for(const auto& particle : gen_particles) {
            if(processed_particles.count(&particle)) continue;
            const int abs_pdg = std::abs(particle.pdgId());
            if(!((particle.statusFlags().isPrompt() && particle.statusFlags().isFirstCopy())
                || (isGeantPlusSim_ && particle.status()==8 && abs_pdg==15))) continue;
            if(!GenParticle::ChargedLeptons().count(static_cast<GenParticle::PdgId>(abs_pdg)))
                continue;

            GenLepton lepton;
            if(isGeantPlusSim_) lepton.enableGenPlusSimParticleMode();
            FillImpl<GenParticleT> fillImpl(lepton, processed_particles);
            fillImpl.FillAll(&particle);
            lepton.initialize();

            leptons.push_back(lepton);
        }
        return leptons;
    }

    template<typename IntVector, typename LongVector, typename FloatVector>
    static GenLepton fromRootTuple(const bool isGeantPlusSim_,
                                   int lastMotherIndex,
                                   const IntVector& genParticle_pdgId,
                                   const IntVector& genParticle_status,
                                   const LongVector& genParticle_mother,
                                   const IntVector& genParticle_charge,
                                   const IntVector& genParticle_isFirstCopy,
                                   const IntVector& genParticle_isLastCopy,
                                   const FloatVector& genParticle_pt,
                                   const FloatVector& genParticle_eta,
                                   const FloatVector& genParticle_phi,
                                   const FloatVector& genParticle_mass,
                                   const FloatVector& genParticle_vtx_x,
                                   const FloatVector& genParticle_vtx_y,
                                   const FloatVector& genParticle_vtx_z)
    {
        try {

        const size_t N = genParticle_pdgId.size();
        assert(N <= MaxNumberOfParticles);
        assert(genParticle_mother.size() == N);
        assert(genParticle_charge.size() == N);
        assert(genParticle_isFirstCopy.size() == N);
        assert(genParticle_isLastCopy.size() == N);
        assert(genParticle_pt.size() == N);
        assert(genParticle_eta.size() == N);
        assert(genParticle_phi.size() == N);
        assert(genParticle_mass.size() == N);
        assert(genParticle_vtx_x.size() == N);
        assert(genParticle_vtx_y.size() == N);
        assert(genParticle_vtx_z.size() == N);
        assert(lastMotherIndex >= -1);

        GenLepton lepton;
        if(isGeantPlusSim_){
            assert(genParticle_status.size() == N);
            lepton.enableGenPlusSimParticleMode();
        }
        lepton.particles_->resize(N);
        lepton.firstCopy_ = &lepton.particles_->at(lastMotherIndex + 1);
        for(size_t n = 0; n < N; ++n) {
            GenParticle& p = lepton.particles_->at(n);
            p.pdgId = genParticle_pdgId.at(n);
            if(isGeantPlusSim_){
                p.status = genParticle_status.at(n);
            } else p.status = -1;
            p.charge = genParticle_charge.at(n);
            p.isFirstCopy = genParticle_isFirstCopy.at(n);
            p.isLastCopy = genParticle_isLastCopy.at(n);
            p.p4 = LorentzVectorM(genParticle_pt.at(n), genParticle_eta.at(n),
                                  genParticle_phi.at(n), genParticle_mass.at(n));
            p.vertex = Point3D(genParticle_vtx_x.at(n), genParticle_vtx_y.at(n), genParticle_vtx_z.at(n));
            std::set<size_t> mothers;
            Long64_t mother_encoded = genParticle_mother.at(n);
            if(mother_encoded >= 0) {
                do {
                    Long64_t mother_index = mother_encoded % static_cast<Long64_t>(MaxNumberOfParticles);
                    mother_encoded = (mother_encoded - mother_index) / static_cast<int>(MaxNumberOfParticles);
                    mothers.insert(static_cast<size_t>(mother_index));
                } while(mother_encoded > 0);
            }
            for(size_t mother_index : mothers) {
                assert(mother_index < N);
                p.mothers.insert(&lepton.particles_->at(mother_index));
                lepton.particles_->at(mother_index).daughters.insert(&p);
            }
        }
        lepton.initialize();
        return lepton;
        } catch(std::exception& e) {
            std::cerr << "ERROR: " << e.what() << std::endl;
            throw;
        }
    }

    static const GenParticle* findTerminalCopy(const GenParticle& genParticle, bool first)
    {
        const GenParticle* particle = &genParticle;
        while((first && !particle->isFirstCopy) || (!first && !particle->isLastCopy)) {
            bool nextCopyFound = false;
            const auto& ref = first ? particle->mothers : particle->daughters;
            for(const GenParticle* p : ref) {
                if(p->pdgId == particle->pdgId) {
                    particle = &(*p);
                    nextCopyFound = true;
                    break;
                }
            }
            if(!nextCopyFound)
                ThrowErrorStatic("unable to find a terminal copy.");
        }
        return particle;
    }

    const std::vector<GenParticle>& allParticles() const { return *particles_; }
    const std::set<const GenParticle*>& mothers() const { return firstCopy_->mothers; }
    const GenParticle& firstCopy() const { return *firstCopy_; }
    const GenParticle& lastCopy() const { return *lastCopy_; }
    Kind kind() const { return kind_; }
    int charge() const { return firstCopy().charge; }
    const std::set<const GenParticle*>& finalStateFromDecay() const { return finalStateFromDecay_; }
    const std::set<const GenParticle*>& finalStateFromRadiation() const { return finalStateFromRadiation_; }
    const std::set<const GenParticle*>& hadrons() const { return hadrons_; }
    // Intermediate hadrons are hadrons that decayed hadronically
    const std::set<const GenParticle*>& intermediateHadrons() const { return intermediateHadrons_; }
    const std::set<const GenParticle*>& otherParticles() const { return other_; }

    const LorentzVectorXYZ& visibleP4() const { return visibleP4_; }
    const LorentzVectorXYZ& radiatedP4() const { return radiatedP4_; }

    size_t nChargedHadrons() const { return nChargedHadrons_; }
    size_t nNeutralHadrons() const { return nNeutralHadrons_; }
    size_t nFinalStateElectrons() const { return nFinalStateElectrons_; }
    size_t nFinalStateMuons() const { return nFinalStateMuons_; }
    size_t nFinalStateNeutrinos() const { return nFinalStateNeutrinos_; }

    void PrintDecay(const GenParticle& particle, const std::string& pre, std::ostream& os) const
    {
        os << particle << std::endl;

        for(auto d_iter = particle.daughters.begin(); d_iter != particle.daughters.end(); ++d_iter) {
            const GenParticle& daughter = **d_iter;
            os << pre << "+-> ";
            const char pre_first = std::next(d_iter) == particle.daughters.end() ? ' ' : '|';
            const std::string pre_d = pre + pre_first + "   ";
            PrintDecay(daughter, pre_d, os);
        }
    }

    void PrintDecay(std::ostream& os) const
    {
        PrintDecay(firstCopy(), "", os);
    }

    // Keeping the default constructor public to stay compatible with RDataFrame
    GenLepton() : particles_(std::make_shared<std::vector<GenParticle>>()) {}

    void enableGenPlusSimParticleMode() { genPlusSimParticleMode=true; }

private:
    template<typename GenParticleT>
    struct FillImpl {
        static constexpr size_t NoneIndex = std::numeric_limits<size_t>::max();

        GenLepton& lepton_;
        std::map<const GenParticleT*, int>& processedParticles_;
        std::map<size_t, std::set<size_t>> relations_;

        FillImpl(GenLepton& lepton, std::map<const GenParticleT*, int>& processedParticles) :
            lepton_(lepton), processedParticles_(processedParticles)
        {
        }

        void FillAll(const GenParticleT* particle)
        {
            size_t last_mother_index = NoneIndex;

            if(!particle->motherRefVector().empty()) {
                for(const auto& mother : particle->motherRefVector())
                    FillDaughters(mother.get(), NoneIndex, false);
                last_mother_index = particle->motherRefVector().size() - 1;
            }

            FillDaughters(particle, last_mother_index, true);

            if(last_mother_index != NoneIndex) {
                lepton_.firstCopy_ = &lepton_.particles_->at(last_mother_index + 1);
                for(size_t mother_index = 0; mother_index <= last_mother_index; ++mother_index) {
                    lepton_.particles_->at(last_mother_index + 1).mothers.insert(&lepton_.particles_->at(mother_index));
                    lepton_.particles_->at(mother_index).daughters.insert(lepton_.firstCopy_);
                }
            } else {
                lepton_.firstCopy_ = &lepton_.particles_->at(0);
            }

            for(const auto& [mother, daughters] : relations_) {
                for(size_t daughter : daughters) {
                    lepton_.particles_->at(mother).daughters.insert(&lepton_.particles_->at(daughter));
                    lepton_.particles_->at(daughter).mothers.insert(&lepton_.particles_->at(mother));
                }
            }
        }

        void FillDaughters(const GenParticleT* p, size_t mother_index, bool fill_recursively)
        {
            if(fill_recursively) {
                if(processedParticles_.count(p)) {
                    const int proc_p_index = processedParticles_.at(p);
                    if(proc_p_index >= 0)
                        relations_[mother_index].insert(static_cast<size_t>(proc_p_index));
                    return;
                    // ThrowErrorStatic("particle has already been processed.");
                }
            }

            GenParticle output_p;
            output_p.pdgId = p->pdgId();
            output_p.charge = p->charge();
            output_p.isFirstCopy = (lepton_.genPlusSimParticleMode && p->status()==8) ? defCopyStatus(p, true) : p->statusFlags().isFirstCopy();
            output_p.isLastCopy = (lepton_.genPlusSimParticleMode && p->status()==8) ? defCopyStatus(p, false) : p->statusFlags().isLastCopy();
            output_p.p4 = p->p4();
            output_p.vertex = p->vertex();
            output_p.status =  p->status();

            size_t p_index = lepton_.particles_->size();
            if(mother_index != NoneIndex)
                relations_[mother_index].insert(p_index);

            lepton_.particles_->push_back(output_p);

            if(fill_recursively) {
                processedParticles_[p] = static_cast<int>(p_index);
                for(auto d : p->daughterRefVector())
                    FillDaughters(&*d, p_index, true);
            }
        }

        static bool defCopyStatus(const GenParticleT* p, bool first)
        {
            // According to the following description:
            // https://github.com/cms-sw/cmssw/blob/master/DataFormats/HepMCCandidate/interface/GenStatusFlags.h

            int pdgId = p->pdgId();
            bool isCopy = true;
            const auto& ref = first ? p->motherRefVector() : p->daughterRefVector();
            if(ref.empty()) return isCopy;

            for(const auto& relative : ref) {
                if(relative->pdgId() == pdgId) {
                    isCopy = false;
                    break;
                }
            }
            return isCopy;
        }
    };

    void initialize()
    {
        if(particles_->empty())
            ThrowError("unable to initalize from an empty particle tree.");

        lastCopy_ = findTerminalCopy(*firstCopy_, false);
        std::set<const GenParticle*> processed;
        if(genPlusSimParticleMode && lastCopy_->status == 8) {
            fillParticleCollectionsGeantMode(*firstCopy_);

            kind_ = determineKindGeantMode();
        } else {
            fillParticleCollections(*firstCopy_, false, processed);

            kind_ = determineKind();
        }
    }

    void fillParticleCollections(const GenParticle& particle, bool fromLastCopy,
                                 std::set<const GenParticle*>& processed)
    {
        // Hard fix not to demage the standart definition.
        // standart genParticle collection will also include sim-based childrens
        // in genParticlePlusGeant collection with status = 8.
        if(genPlusSimParticleMode && particle.status == 8) return;
        if(processed.count(&particle)) return;
        processed.insert(&particle);
        fromLastCopy = fromLastCopy || &particle == lastCopy_;
        bool isFinalState = genPlusSimParticleMode ? !hasNonGeantDaughter(particle) : particle.daughters.empty();
        if(isFinalState && !particle.isLastCopy) {
            std::cerr << "Inconsistent particle: " << particle << std::endl;
            ThrowError("last copy flag is not set for a final state particle.");
        }
        if(particle.isLastCopy) {
            const bool isChargedHadron = GenParticle::ChargedHadrons().count(particle.pdgCode());
            const bool isNeutralHadron = GenParticle::NeutralHadrons().count(particle.pdgCode());
            const bool isOther = !(isFinalState || isChargedHadron || isNeutralHadron);
            if(isFinalState) {
                auto& finalStateSet = fromLastCopy ? finalStateFromDecay_ : finalStateFromRadiation_;
                finalStateSet.insert(&particle);

                if(fromLastCopy) {
                    if(GenParticle::NeutralLeptons().count(particle.pdgCode())) {
                        ++nFinalStateNeutrinos_;
                    } else {
                        if(particle.pdgCode() == GenParticle::PdgId::electron)
                            ++nFinalStateElectrons_;
                        if(particle.pdgCode() == GenParticle::PdgId::muon)
                            ++nFinalStateMuons_;
                        visibleP4_ += particle.p4;
                    }
                } else {
                    radiatedP4_ += particle.p4;
                }
            }
            if(fromLastCopy && (isChargedHadron || isNeutralHadron)) {
                hadrons_.insert(&particle);
                bool isIntermediate = false;
                for(auto d : particle.daughters) {
                    if(genPlusSimParticleMode && d->status == 8) continue;
                    if(!GenParticle::ChargedLeptons().count(d->pdgCode())
                            && !GenParticle::NeutralLeptons().count(d->pdgCode())
                            && d->pdgCode() != GenParticle::PdgId::photon) {
                        isIntermediate = true;
                        break;
                    }
                }
                if(isIntermediate) {
                    intermediateHadrons_.insert(&particle);
                } else {
                    size_t& nHad = isChargedHadron ? nChargedHadrons_ : nNeutralHadrons_;
                    ++nHad;
                }
            }
            if(isOther)
                other_.insert(&particle);
        }

        for(const GenParticle* daughter : particle.daughters)
            fillParticleCollections(*daughter, fromLastCopy, processed);
    }

    const bool hasNonGeantDaughter (const GenParticle& particle)
    {
        for(const GenParticle* daughter : particle.daughters)
            if(daughter->status!=8) return true;
        return false;
    }

    void fillParticleCollectionsGeantMode(const GenParticle& particle)
    {
        // Only first level daughters are considered for the Kind definition
        // collections:
        // finalStateFromDecay_, finalStateFromRadiation_, hadrons_,
        // intermediateHadrons_, other_ -> are not defined

        if(!(particle.status == 8 &&
             particle.pdgCode() == GenParticle::PdgId::tau &&
             particle.isLastCopy)) {
            std::cerr << "Inconsistent particle: " << particle << std::endl;
            ThrowError("fillParticleCollectionsGeantMode used for wrong particle");
        }

        for(const GenParticle* daughter : particle.daughters) {
            // Hard fix of bug:
            // pdgId=15 status=8 pt=127.094 eta=1.1743 phi=-1.15058 E=225.275 m=1.77686 vx=350.016 vy=-332.128 vz=824.172 vrho=482.515 vr=955.029 q=-1
            // +-> pdgId=11 status=8 pt=0.00434613 eta=1.18129 phi=-1.79642 E=0.00776478 m=0.000510999 vx=350.107 vy=-332.333 vz=824.5 vrho=482.722 vr=955.416 q=-1
            // +-> pdgId=13 status=8 pt=25.8348 eta=1.18915 phi=-1.15298 E=46.3575 m=0.105658 vx=350.236 vy=-332.62 vz=824.96 vrho=483.012 vr=955.96 q=-1
            if(daughter->p4.pt()<0.01) continue;

            if(daughter->pdgCode() == GenParticle::PdgId::pi)
                ++nChargedHadrons_;
            else if(daughter->pdgCode() == GenParticle::PdgId::pi0)
                ++nFinalStateNeutrinos_;
            else if(daughter->pdgCode() == GenParticle::PdgId::electron)
                ++nFinalStateElectrons_;
            else if(daughter->pdgCode() == GenParticle::PdgId::muon)
                ++nFinalStateMuons_;
            else if(daughter->pdgCode() == GenParticle::PdgId::photon)
                std::cout << ".";
            else {
               std::cerr << "Undefined first level tau daughter: " << daughter << std::endl;
               ThrowError("Only ChargedHadrons, NeutralHadrons, e or mu can be daughters");
            }
            visibleP4_ += particle.p4;
        }
    }

    Kind determineKind() const
    {
        const auto pdg = lastCopy_->pdgCode();
        if(pdg == GenParticle::PdgId::electron)
            return Kind::PromptElectron;
        if(pdg == GenParticle::PdgId::muon)
            return Kind::PromptMuon;
        if(pdg != GenParticle::PdgId::tau)
            ThrowError("unknown lepton type");
        if(nChargedHadrons_ == 0 && nNeutralHadrons_ != 0)
            ThrowError("invalid hadron counts");
        if(nChargedHadrons_ != 0)
            return Kind::TauDecayedToHadrons;
        if(nFinalStateElectrons_ == 1 && nFinalStateNeutrinos_ == 2)
            return Kind::TauDecayedToElectron;
        if(nFinalStateMuons_ == 1 && nFinalStateNeutrinos_ == 2)
            return Kind::TauDecayedToMuon;
        ThrowError("unable to determine gen lepton kind.");
    }

    Kind determineKindGeantMode() const
    {
        const auto pdg = lastCopy_->pdgCode();
        if(nChargedHadrons_ == 0 && nNeutralHadrons_ != 0)
            ThrowError("invalid hadron counts");
        if(pdg == GenParticle::PdgId::tau && nChargedHadrons_ != 0)
            return Kind::TauDecayedToHadrons;
        if(nFinalStateElectrons_ == 1 && nFinalStateMuons_ == 0 && 
           nNeutralHadrons_ == 0 && nChargedHadrons_ == 0)
            return Kind::TauDecayedToElectron;
        // Hard fix of bug:
        // pdgId=-15 status=8
        // +-> pdgId=11 status=8 <- pt<0.001
        // +-> pdgId=11 status=8 <- pt<0.001
        // +-> pdgId=-11 status=8
        if(nFinalStateElectrons_ >= 2 && nFinalStateMuons_ == 0 && 
           nNeutralHadrons_ == 0 && nChargedHadrons_ == 0)
            return Kind::TauDecayedToElectron;
        if(nFinalStateElectrons_ == 0 && nFinalStateMuons_ == 1 && 
           nNeutralHadrons_ == 0 && nChargedHadrons_ == 0)
            return Kind::TauDecayedToMuon;
        // Hard fix of bug:
        // pdgId=-15 status=8
        // +-> pdgId=11 status=8 <- pt<0.001
        // +-> pdgId=11 status=8 <- pt<0.001
        // +-> pdgId=-13 status=8
        if(nFinalStateElectrons_ >= 1 && nFinalStateMuons_ == 1 &&
           nNeutralHadrons_ == 0 && nChargedHadrons_ == 0)
            return Kind::TauDecayedToMuon;
        if(pdg == GenParticle::PdgId::tau && lastCopy_->daughters.empty())
            return Kind::Other;
        ThrowError("unable to determine gen lepton kind in GeantMode");
    }

    [[noreturn]] void ThrowError(const std::string& message) const
    {
        if(particles_->size())
            PrintDecay(std::cerr);
        ThrowErrorStatic(message);
    }

    [[noreturn]] static void ThrowErrorStatic(const std::string& message)
    {
        throw std::runtime_error("GenLepton: " + message);
    }

private:
    std::shared_ptr<std::vector<GenParticle>> particles_;
    const GenParticle *firstCopy_{nullptr}, *lastCopy_{nullptr};
    Kind kind_{Kind::Other};
    std::set<const GenParticle*> finalStateFromDecay_, finalStateFromRadiation_, hadrons_, intermediateHadrons_, other_;
    LorentzVectorXYZ visibleP4_, radiatedP4_;
    size_t nChargedHadrons_{0}, nNeutralHadrons_{0}, nFinalStateElectrons_{0}, nFinalStateMuons_{0},
           nFinalStateNeutrinos_{0};
    bool genPlusSimParticleMode{false};
};

} // namespace gen_truth
} // namespace reco_tau
