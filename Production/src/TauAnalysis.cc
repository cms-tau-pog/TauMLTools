/* Common methods used in tau analysis.
*/

#include "TauMLTools/Production/interface/TauAnalysis.h"

namespace tau_analysis {

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

bool CalculateGottfriedJacksonAngleDifference(const pat::Tau& tau, double& gj_diff)
{
    return CalculateGottfriedJacksonAngleDifference(tau.hasSecondaryVertex(),
                                                    tau.p4(),
                                                    reco::PFTauTransverseImpactParameter::Vector(tau.flightLength()),
                                                    gj_diff);
}

bool CalculateGottfriedJacksonAngleDifference(const reco::PFTau& tau,
                                              const reco::PFTauTransverseImpactParameter& tau_transverse_ip,
                                              double& gj_diff)
{
    return CalculateGottfriedJacksonAngleDifference(tau_transverse_ip.hasSecondaryVertex(), tau.p4(),
                                                    tau_transverse_ip.flightLength(), gj_diff);
}

bool CalculateGottfriedJacksonAngleDifference(bool hasSecondaryVertex,
                                              const reco::Candidate::LorentzVector& tau_p4,
                                              const reco::PFTauTransverseImpactParameter::Vector& tau_flightLength,
                                              double& gj_diff)
{
    if(hasSecondaryVertex) {
        static constexpr double mTau = 1.77682;
        const double mAOne = tau_p4.M();
        const double pAOneMag = tau_p4.P();
        const double argumentThetaGJmax = (std::pow(mTau,2) - std::pow(mAOne,2) ) / ( 2 * mTau * pAOneMag );
        const double argumentThetaGJmeasured = tau_p4.Vect().Dot(tau_flightLength)
                / ( pAOneMag * tau_flightLength.R() );
        if ( std::abs(argumentThetaGJmax) <= 1. && std::abs(argumentThetaGJmeasured) <= 1. ) {
            double thetaGJmax = std::asin( argumentThetaGJmax );
            double thetaGJmeasured = std::acos( argumentThetaGJmeasured );
            gj_diff = thetaGJmeasured - thetaGJmax;
            return true;
        }
    }
    return false;
}


} // namespace tau_analysis
