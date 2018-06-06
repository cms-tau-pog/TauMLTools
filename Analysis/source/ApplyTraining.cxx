/*! Shuffle input tuples into one.
*/

#include <fstream>
#include <boost/algorithm/string.hpp>
#include <boost/preprocessor/seq.hpp>
#include <boost/preprocessor/variadic.hpp>
#include "AnalysisTools/Run/include/program_main.h"
#include "AnalysisTools/Core/include/RootExt.h"
#include "TauML/Analysis/include/AnalysisTypes.h"
#include "TauML/Analysis/include/TauTuple.h"
#include "TauML/Analysis/include/TauIdResults.h"
#include "TauML/Analysis/include/TauIdResultTuple.h"
#include "PhysicsTools/TensorFlow/interface/TensorFlow.h"

struct Arguments {
    REQ_ARG(std::string, input_dir);
    REQ_ARG(std::string, input_list);
    REQ_ARG(std::string, output);
    REQ_ARG(std::string, model);
    OPT_ARG(std::string, tree_name, "taus");
};

namespace analysis {

#define CREATE_VAR(r, type, name) VAR(name)
#define VAR_LIST(...) BOOST_PP_SEQ_FOR_EACH(CREATE_VAR, type, BOOST_PP_VARIADIC_TO_SEQ(__VA_ARGS__))

#define INPUT_VARS() \
    VAR_LIST( pt, eta, mass, decayMode, chargedIsoPtSum, neutralIsoPtSum, neutralIsoPtSumWeight, \
              photonPtSumOutsideSignalCone, puCorrPtSum, \
              dxy, dxy_sig, dz, ip3d, ip3d_sig, \
              hasSecondaryVertex, flightLength_r, flightLength_dEta, flightLength_dPhi, \
              flightLength_sig, leadChargedHadrCand_pt, leadChargedHadrCand_dEta, \
              leadChargedHadrCand_dPhi, leadChargedHadrCand_mass, pt_weighted_deta_strip, \
              pt_weighted_dphi_strip, pt_weighted_dr_signal, pt_weighted_dr_iso, \
              leadingTrackNormChi2, e_ratio, gj_angle_diff, n_photons, emFraction, \
              has_gsf_track, inside_ecal_crack, \
              gsf_ele_matched, gsf_ele_pt, gsf_ele_dEta, gsf_ele_dPhi, gsf_ele_mass, gsf_ele_Ee, \
              gsf_ele_Egamma, gsf_ele_Pin, gsf_ele_Pout, gsf_ele_EtotOverPin, gsf_ele_Eecal, \
              gsf_ele_dEta_SeedClusterTrackAtCalo, gsf_ele_dPhi_SeedClusterTrackAtCalo, gsf_ele_mvaIn_sigmaEtaEta, \
              gsf_ele_mvaIn_hadEnergy, \
              gsf_ele_mvaIn_deltaEta, gsf_ele_Chi2NormGSF, gsf_ele_GSFNumHits, gsf_ele_GSFTrackResol, \
              gsf_ele_GSFTracklnPt, gsf_ele_Chi2NormKF, gsf_ele_KFNumHits, \
              leadChargedCand_etaAtEcalEntrance, leadChargedCand_pt, leadChargedHadrCand_HoP, \
              leadChargedHadrCand_EoP, tau_visMass_innerSigCone) \
    VAR_LIST( n_matched_muons, muon_pt, muon_dEta, muon_dPhi, \
              muon_n_matches_DT_1, muon_n_matches_DT_2, muon_n_matches_DT_3, muon_n_matches_DT_4, \
              muon_n_matches_CSC_1, muon_n_matches_CSC_2, muon_n_matches_CSC_3, muon_n_matches_CSC_4, \
              muon_n_hits_DT_2, muon_n_hits_DT_3, muon_n_hits_DT_4, \
              muon_n_hits_CSC_2, muon_n_hits_CSC_3, muon_n_hits_CSC_4, \
              muon_n_hits_RPC_2, muon_n_hits_RPC_3, muon_n_hits_RPC_4, \
              muon_n_stations_with_matches_03, muon_n_stations_with_hits_23) \
    VAR_LIST( signalChargedHadrCands_sum_innerSigCone_pt, signalChargedHadrCands_sum_innerSigCone_dEta, \
              signalChargedHadrCands_sum_innerSigCone_dPhi, signalChargedHadrCands_sum_innerSigCone_mass, \
              signalChargedHadrCands_sum_outerSigCone_pt, signalChargedHadrCands_sum_outerSigCone_dEta, \
              signalChargedHadrCands_sum_outerSigCone_dPhi, signalChargedHadrCands_sum_outerSigCone_mass, \
              signalChargedHadrCands_nTotal_innerSigCone, signalChargedHadrCands_nTotal_outerSigCone, \
              signalNeutrHadrCands_sum_innerSigCone_pt, signalNeutrHadrCands_sum_innerSigCone_dEta, \
              signalNeutrHadrCands_sum_innerSigCone_dPhi, signalNeutrHadrCands_sum_innerSigCone_mass, \
              signalNeutrHadrCands_sum_outerSigCone_pt, signalNeutrHadrCands_sum_outerSigCone_dEta, \
              signalNeutrHadrCands_sum_outerSigCone_dPhi, signalNeutrHadrCands_sum_outerSigCone_mass, \
              signalNeutrHadrCands_nTotal_innerSigCone, signalNeutrHadrCands_nTotal_outerSigCone, \
              signalGammaCands_sum_innerSigCone_pt, signalGammaCands_sum_innerSigCone_dEta, \
              signalGammaCands_sum_innerSigCone_dPhi, signalGammaCands_sum_innerSigCone_mass, \
              signalGammaCands_sum_outerSigCone_pt, signalGammaCands_sum_outerSigCone_dEta, \
              signalGammaCands_sum_outerSigCone_dPhi, signalGammaCands_sum_outerSigCone_mass, \
              signalGammaCands_nTotal_innerSigCone, signalGammaCands_nTotal_outerSigCone, \
              isolationChargedHadrCands_sum_pt, isolationChargedHadrCands_sum_dEta, \
              isolationChargedHadrCands_sum_dPhi, isolationChargedHadrCands_sum_mass, \
              isolationChargedHadrCands_nTotal, \
              isolationNeutrHadrCands_sum_pt, isolationNeutrHadrCands_sum_dEta, \
              isolationNeutrHadrCands_sum_dPhi, isolationNeutrHadrCands_sum_mass, \
              isolationNeutrHadrCands_nTotal, \
              isolationGammaCands_sum_pt, isolationGammaCands_sum_dEta, \
              isolationGammaCands_sum_dPhi, isolationGammaCands_sum_mass, \
              isolationGammaCands_nTotal) \
    /**/

namespace dnn_input {
#define VAR(name) name,
    enum vars { INPUT_VARS() NumberOfInputs };
#undef VAR
}

class ApplyTraining {
public:
    using Tau = tau_tuple::Tau;
    using TauTuple = tau_tuple::TauTuple;
    using TauIdResult = tau_tuple::TauIdResult;
    using TauIdResultTuple = tau_tuple::TauIdResultTuple;
    using GraphPtr = std::shared_ptr<tensorflow::GraphDef>;
    using SessionPtr = std::shared_ptr<tensorflow::Session>;

    ApplyTraining(const Arguments& _args) :
        args(_args), graph(tensorflow::loadGraphDef(args.model())), session(tensorflow::createSession(graph.get())),
        x(tensorflow::DT_FLOAT, {1, dnn_input::NumberOfInputs})
    {
        std::cout << "Total number of inputs: " << static_cast<int>(dnn_input::NumberOfInputs) << std::endl;
        std::ifstream cfg(args.input_list());
        if(cfg.fail())
            throw exception("Failed to open config '%1%'.") % args.input_list();

        while(cfg.good()) {
            std::string line;
            std::getline(cfg, line);
            boost::trim_if(line, boost::is_any_of(" \t"));
            if(line.empty() || line.at(0) == '#') continue;
            input_files.push_back(line);
        }
    }

    ~ApplyTraining()
    {
        tensorflow::closeSession(session);
    }

    void Run()
    {
        auto output_file = root_ext::CreateRootFile(args.output(), ROOT::kLZ4, 5);
        TauIdResultTuple output_tuple(args.tree_name(), output_file.get(), false);
        for(const auto& input_name : input_files) {
            std::cout << "Processing '" << input_name << "'..." << std::endl;
            auto file = root_ext::OpenRootFile(args.input_dir() + "/" + input_name);
            TauTuple tuple(args.tree_name(), file.get(), true);
            Long64_t n = 0, n_total = tuple.GetEntries();

            for(const Tau& tau : tuple) {
                const GenMatch gen_match = static_cast<GenMatch>(tau.gen_match);
                const TauIdResults ref_id_result(tau.id_flags);

                output_tuple().run = tau.run;
                output_tuple().lumi = tau.lumi;
                output_tuple().evt = tau.evt;
                output_tuple().tau_index = tau.tau_index;
                output_tuple().pt = tau.pt;
                output_tuple().eta = tau.eta;
                output_tuple().phi = tau.phi;
                output_tuple().decayMode = tau.decayMode;
                output_tuple().gen_e = gen_match == GenMatch::Electron || gen_match == GenMatch::TauElectron;
                output_tuple().gen_mu = gen_match == GenMatch::Muon || gen_match == GenMatch::TauMuon;
                output_tuple().gen_tau = gen_match == GenMatch::Tau;
                output_tuple().gen_jet = gen_match == GenMatch::NoMatch;
                output_tuple().refId_e = tau.againstElectronMVA6Raw;
                output_tuple().refId_mu_loose = ref_id_result.Result(TauIdDiscriminator::againstMuon3,
                                                                     DiscriminatorWP::Loose);
                output_tuple().refId_mu_tight = ref_id_result.Result(TauIdDiscriminator::againstMuon3,
                                                                     DiscriminatorWP::Tight);
                output_tuple().refId_jet = tau.byIsolationMVArun2017v2DBoldDMwLTraw2017;

                const auto& pred = RunGraph(tau);
                output_tuple().deepId_e = pred.matrix<float>()(0, 0);
                output_tuple().deepId_mu = pred.matrix<float>()(0, 1);
                output_tuple().deepId_tau = pred.matrix<float>()(0, 2);
                output_tuple().deepId_jet = pred.matrix<float>()(0, 3);

                output_tuple.Fill();
                if(++n % 10000 == 0)
                    std::cout << "\tProgress: " << n << "/" << n_total << std::endl;
            }
        }

        output_tuple.Write();
        std::cout << "All files are processed." << std::endl;
    }

private:

#define VAR(name) x.matrix<float>()(0, dnn_input::name) = tau.name;

    tensorflow::Tensor RunGraph(const Tau& tau)
    {
        INPUT_VARS()

        std::vector<tensorflow::Tensor> pred;
        tensorflow::run(session, { { "dense_94_input:0", x } }, { "output_node0:0"}, &pred);
        return pred.at(0);
    }

#undef VAR

private:
    Arguments args;
    std::vector<std::string> input_files;
    GraphPtr graph;
    tensorflow::Session* session;
    tensorflow::Tensor x;
};

} // namespace analysis

#undef INPUT_VARS
#undef VAR_LIST
#undef CREATE_VAR

PROGRAM_MAIN(analysis::ApplyTraining, Arguments)
