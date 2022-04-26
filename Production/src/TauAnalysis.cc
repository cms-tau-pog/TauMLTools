/* Common methods used in tau analysis.
*/

#include "TauMLTools/Production/interface/TauAnalysis.h"

namespace tau_analysis {

bool CalculateGottfriedJacksonAngleDifference(const pat::Tau& tau, double& gj_diff)
{
    if(tau.hasSecondaryVertex()) {
        static constexpr double mTau = 1.77682;
        const double mAOne = tau.p4().M();
        const double pAOneMag = tau.p();
        const double argumentThetaGJmax = (std::pow(mTau,2) - std::pow(mAOne,2) ) / ( 2 * mTau * pAOneMag );
        const double argumentThetaGJmeasured = tau.p4().Vect().Dot(tau.flightLength())
                / ( pAOneMag * tau.flightLength().R() );
        if ( std::abs(argumentThetaGJmax) <= 1. && std::abs(argumentThetaGJmeasured) <= 1. ) {
            double thetaGJmax = std::asin( argumentThetaGJmax );
            double thetaGJmeasured = std::acos( argumentThetaGJmeasured );
            gj_diff = thetaGJmeasured - thetaGJmax;
            return true;
        }
    }
    return false;
}

double CalculateDeltaEtaCrack(double eta)
{
    // IN: define locations of the eta-cracks
    static constexpr double cracks[5] = { 0., 4.44747e-01, 7.92824e-01, 1.14090e+00, 1.47464e+00 };

    double retVal = 99.;
    for ( int iCrack = 0; iCrack < 5 ; ++iCrack ) {
        double d = AbsMin(eta - cracks[iCrack], eta + cracks[iCrack]);
        if ( std::abs(d) < std::abs(retVal) ) {
            retVal = d;
        }
    }
    return std::abs(retVal);
}

double PFRelIsolation(const pat::Muon& muon)
{
    const double sum_neutral = muon.pfIsolationR04().sumNeutralHadronEt
                             + muon.pfIsolationR04().sumPhotonEt
                             - 0.5 * muon.pfIsolationR04().sumPUPt;
    const double abs_iso = muon.pfIsolationR04().sumChargedHadronPt + std::max(sum_neutral, 0.0);
    return abs_iso / muon.pt();
}

double PFRelIsolation(const pat::Electron& electron, float rho)
{   
    const std::vector<float> eta_upper = {1.0, 1.479, 2.0, 2.2, 2.3, 2.4, 5.0}; // upper bounds of eta ranges
    const std::vector<float> ea_values = {0.1440, 0.1562, 0.1032, 0.0859, 0.1116, 0.1321, 0.1654};
    auto eta_bin = std::upper_bound(eta_upper.begin(), eta_upper.end(), std::abs(electron.eta())); // iterator pointing to corresponding eta value
    auto ea_index = std::min<size_t>(std::distance(eta_upper.begin(), eta_bin), ea_values.size() - 1); // return corresponding index, or last bin if eta>=5
    float ea = ea_values.at(ea_index); // get EA at that index

    const double sum_neutral = electron.neutralHadronIso()
                             + electron.photonIso()
                             - ea* rho;
    const double abs_iso = electron.chargedHadronIso() + std::max(sum_neutral, 0.0);
    return abs_iso / electron.pt();
}

} // namespace tau_analysis
