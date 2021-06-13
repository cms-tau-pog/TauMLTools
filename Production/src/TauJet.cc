/* Tau jet candidate.
*/

#include "TauMLTools/Production/interface/TauJet.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

namespace tau_analysis {

TauJetBuilderSetup TauJetBuilderSetup::fromPSet(const edm::ParameterSet& builderParams)
{
    TauJetBuilderSetup builderSetup;

    const std::map<std::string, double*> builderParamNames = {
        { "genLepton_genJet_dR", &builderSetup.genLepton_genJet_dR },
        { "genLepton_tau_dR", &builderSetup.genLepton_tau_dR },
        { "genLepton_boostedTau_dR", &builderSetup.genLepton_boostedTau_dR },
        { "genLepton_jet_dR", &builderSetup.genLepton_jet_dR },
        { "genLepton_fatJet_dR", &builderSetup.genLepton_fatJet_dR },
        { "genLepton_l1Tau_dR", &builderSetup.genLepton_l1Tau_dR },
        { "genJet_tau_dR", &builderSetup.genJet_tau_dR },
        { "genJet_boostedTau_dR", &builderSetup.genJet_boostedTau_dR },
        { "genJet_jet_dR", &builderSetup.genJet_jet_dR },
        { "genJet_fatJet_dR", &builderSetup.genJet_fatJet_dR },
        { "genJet_l1Tau_dR", &builderSetup.genJet_l1Tau_dR },
        { "tau_boostedTau_dR", &builderSetup.tau_boostedTau_dR },
        { "tau_jet_dR", &builderSetup.tau_jet_dR },
        { "tau_fatJet_dR", &builderSetup.tau_fatJet_dR },
        { "tau_l1Tau_dR", &builderSetup.tau_l1Tau_dR },
        { "boostedTau_jet_dR", &builderSetup.boostedTau_jet_dR },
        { "boostedTau_fatJet_dR", &builderSetup.boostedTau_fatJet_dR },
        { "boostedTau_l1Tau_dR", &builderSetup.boostedTau_l1Tau_dR },
        { "jet_fatJet_dR", &builderSetup.jet_fatJet_dR },
        { "jet_l1Tau_dR", &builderSetup.jet_l1Tau_dR },
        { "fatJet_l1Tau_dR", &builderSetup.fatJet_l1Tau_dR },
        { "jet_maxAbsEta", &builderSetup.jet_maxAbsEta },
        { "fatJet_maxAbsEta", &builderSetup.fatJet_maxAbsEta },
        { "genLepton_cone", &builderSetup.genLepton_cone },
        { "genJet_cone", &builderSetup.genJet_cone },
        { "tau_cone", &builderSetup.tau_cone },
        { "boostedTau_cone", &builderSetup.boostedTau_cone },
        { "jet_cone", &builderSetup.jet_cone },
        { "fatJet_cone", &builderSetup.fatJet_cone },
        { "l1Tau_cone", &builderSetup.fatJet_cone },
    };

    for(const auto& paramName : builderParams.getParameterNames()) {
        auto iter = builderParamNames.find(paramName);
        if(iter == builderParamNames.end())
            throw cms::Exception("TauJetBuilderSetup") << "Unknown parameter '" << paramName <<
                                                        "' in tauJetBuilderSetup.";
        *iter->second = builderParams.getParameter<double>(paramName);
    }

    return builderSetup;
}

bool TauJetMatchResult::HasMatch() const
{
    return dR_genLepton < inf || dR_genJet < inf || dR_tau < inf || dR_boostedTau || dR_jet < inf;
}

void TauJetMatchResult::SetDeltaR_genLepton(size_t index_in, double dR_in, double dR_thr)
{
    SetDeltaR(index_in, dR_in, dR_thr, dR_genLepton);
}

void TauJetMatchResult::SetDeltaR_genJet(size_t index_in, double dR_in, double dR_thr)
{
    SetDeltaR(index_in, dR_in, dR_thr, dR_genJet);
}

void TauJetMatchResult::SetDeltaR_tau(size_t index_in, double dR_in, double dR_thr)
{
    SetDeltaR(index_in, dR_in, dR_thr, dR_tau);
}

void TauJetMatchResult::SetDeltaR_boostedTau(size_t index_in, double dR_in, double dR_thr)
{
    SetDeltaR(index_in, dR_in, dR_thr, dR_boostedTau);
}

void TauJetMatchResult::SetDeltaR_jet(size_t index_in, double dR_in, double dR_thr)
{
    SetDeltaR(index_in, dR_in, dR_thr, dR_jet);
}

void TauJetMatchResult::SetDeltaR_fatJet(size_t index_in, double dR_in, double dR_thr)
{
    SetDeltaR(index_in, dR_in, dR_thr, dR_fatJet);
}

void TauJetMatchResult::SetDeltaR_l1Tau(size_t index_in, double dR_in, double dR_thr)
{
    SetDeltaR(index_in, dR_in, dR_thr, dR_l1Tau);
}

void TauJetMatchResult::SetDeltaR(size_t index_in, double dR_in, double dR_thr, double& dR_out)
{
    if(dR_in < dR_thr && dR_in < dR_out) {
        dR_out = dR_in;
        index = static_cast<int>(index_in);
    }
}

bool TauJetMatchResult::operator <(const TauJetMatchResult& other) const
{
    if(dR_genLepton != other.dR_genLepton) return dR_genLepton < other.dR_genLepton;
    if(dR_genJet != other.dR_genJet) return dR_genJet < other.dR_genJet;
    if(dR_tau != other.dR_tau) return dR_tau < other.dR_tau;
    if(dR_boostedTau != other.dR_boostedTau) return dR_boostedTau < other.dR_boostedTau;
    if(dR_jet != other.dR_jet) return dR_jet < other.dR_jet;
    if(dR_fatJet != other.dR_fatJet) return dR_fatJet < other.dR_fatJet;
    if(dR_l1Tau != other.dR_l1Tau) return dR_l1Tau < other.dR_l1Tau;
    return index < other.index;
}

std::ostream& operator<<(std::ostream& os, const TauJetMatchResult& match)
{
    const auto print_dR = [&](double dR, const std::string& name) {
        if(dR < TauJetMatchResult::inf)
            os << ", " << name << " = " << dR;
    };
    if(match.index) {
        os << "index = " << *match.index;
        print_dR(match.dR_genLepton, "dR_genLepton");
        print_dR(match.dR_genJet, "dR_genJet");
        print_dR(match.dR_tau, "dR_tau");
        print_dR(match.dR_boostedTau, "dR_boostedTau");
        print_dR(match.dR_jet, "dR_jet");
        print_dR(match.dR_l1Tau, "dR_l1Tau");
    } else {
        os << "no_match";
    }
    return os;
}

} // namespace tau_analysis
