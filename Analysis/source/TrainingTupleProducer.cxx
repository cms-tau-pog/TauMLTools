/*! Produce training tuple from tau tuple.
*/

#include <boost/preprocessor/seq.hpp>
#include <boost/preprocessor/variadic.hpp>

#include "AnalysisTools/Run/include/program_main.h"
#include "AnalysisTools/Core/include/AnalysisMath.h"
#include "AnalysisTools/Core/include/RootExt.h"
#include "TauML/Analysis/include/TauTuple.h"
#include "TauML/Analysis/include/TrainingTuple.h"
#include "AnalysisTools/Core/include/ProgressReporter.h"

#define CP_BR_EX(r, placeholder, name) CP_BR(name)
#define CP_BRANCHES(...) \
    BOOST_PP_SEQ_FOR_EACH(CP_BR_EX, placeholder, BOOST_PP_VARIADIC_TO_SEQ(__VA_ARGS__))

struct Arguments {
    run::Argument<std::string> input{"input", "input root file with tau tuple"};
    run::Argument<std::string> output{"output", "output root file with training tuple"};
    run::Argument<unsigned> n_cells{"n-cells", "number of cells in eta and phi", 61};
    run::Argument<double> cell_size{"cell-size", "size of the cell in eta and phi", 0.01};
    run::Argument<unsigned> n_threads{"n-threads", "number of threads", 1};
    run::Argument<Long64_t> start_entry{"start-entry", "start entry", 0};
    run::Argument<Long64_t> end_entry{"end-entry", "end entry", std::numeric_limits<Long64_t>::max()};
};

namespace analysis {

enum class CellObjectType { PfCandidate, Electron, Muon };
using Cell = std::map<CellObjectType, std::set<size_t>>;
struct CellIndex { int eta, phi; };

class CellGrid {
public:
    CellGrid(unsigned _nCellsEta, unsigned _nCellsPhi, double _cellSizeEta, double _cellSizePhi) :
        nCellsEta(_nCellsEta), nCellsPhi(_nCellsPhi), nTotal(nCellsEta * nCellsPhi),
        cellSizeEta(_cellSizeEta), cellSizePhi(_cellSizePhi), cells(nTotal)
    {
        if(nCellsEta % 2 != 1 || nCellsEta < 1)
            throw exception("Invalid number of eta cells.");
        if(nCellsPhi % 2 != 1 || nCellsPhi < 1)
            throw exception("Invalid number of phi cells.");
        if(cellSizeEta <= 0 || cellSizePhi <= 0)
            throw exception("Invalid cell size.");
    }

    int MaxEtaIndex() const { return static_cast<int>((nCellsEta - 1) / 2); }
    int MaxPhiIndex() const { return static_cast<int>((nCellsPhi - 1) / 2); }
    double MaxDeltaEta() const { return cellSizeEta * (0.5 + MaxEtaIndex()); }
    double MaxDeltaPhi() const { return cellSizePhi * (0.5 + MaxPhiIndex()); }

    bool TryGetCellIndex(double deltaEta, double deltaPhi, CellIndex& cellIndex) const
    {
        static auto getCellIndex = [](double x, double maxX, double size, int& index) {
            const double absX = std::abs(x);
            if(absX > maxX) return false;
            const double absIndex = std::floor(std::abs(absX / size - 0.5));
            index = static_cast<int>(std::copysign(absIndex, x));
            return true;
        };

        return getCellIndex(deltaEta, MaxDeltaEta(), cellSizeEta, cellIndex.eta)
               && getCellIndex(deltaPhi, MaxDeltaPhi(), cellSizePhi, cellIndex.phi);
    }

    Cell& at(const CellIndex& cellIndex) { return cells.at(GetFlatIndex(cellIndex)); }
    const Cell& at(const CellIndex& cellIndex) const { return cells.at(GetFlatIndex(cellIndex)); }

private:
    size_t GetFlatIndex(const CellIndex& cellIndex) const
    {
        if(std::abs(cellIndex.eta) > MaxEtaIndex() || std::abs(cellIndex.phi) > MaxPhiIndex())
            throw exception("Cell index is out of range");
        const unsigned shiftedEta = static_cast<unsigned>(cellIndex.eta + MaxEtaIndex());
        const unsigned shiftedPhi = static_cast<unsigned>(cellIndex.phi + MaxPhiIndex());
        return shiftedEta * nCellsPhi + shiftedPhi;
    }

private:
    const unsigned nCellsEta, nCellsPhi, nTotal;
    const double cellSizeEta, cellSizePhi;
    std::vector<Cell> cells;
};

class TrainingTupleProducer {
public:
    using Tau = tau_tuple::Tau;
    using TauTuple = tau_tuple::TauTuple;
    using TrainingTau = tau_tuple::TrainingTau;
    using TrainingTauTuple = tau_tuple::TrainingTauTuple;
    using TrainingCell = tau_tuple::TrainingCell;
    using TrainingCellTuple = tau_tuple::TrainingCellTuple;

    TrainingTupleProducer(const Arguments& _args) :
        args(_args), inputFile(root_ext::OpenRootFile(args.input())),
        outputFile(root_ext::CreateRootFile(args.output(), ROOT::kLZ4, 4)),
        tauTuple(inputFile.get(), true), trainingTauTuple(outputFile.get(), false),
        cellTuple(outputFile.get(), false),
        cellGridRef(args.n_cells(), args.n_cells(), args.cell_size(), args.cell_size())
    {
        if(args.n_threads() > 1)
            ROOT::EnableImplicitMT(args.n_threads());
    }

    void Run()
    {
        const Long64_t end_entry = std::min(tauTuple.GetEntries(), args.end_entry());
        size_t n_processed = 0, n_total = static_cast<size_t>(end_entry - args.start_entry());
        tools::ProgressReporter reporter(10, std::cout, "Creating training tuple...");
        reporter.SetTotalNumberOfEvents(n_total);
        for(Long64_t current_entry = args.start_entry(); current_entry < end_entry; ++current_entry) {
            tauTuple.GetEntry(current_entry);
            const auto& tau = tauTuple.data();
            FillTauBranches(tau);
            auto cellGrid = CreateCellGrid(tau);
            const int max_eta_index = cellGrid.MaxEtaIndex(), max_phi_index = cellGrid.MaxPhiIndex();
            for(int eta_index = -max_eta_index; eta_index <= max_eta_index; ++eta_index) {
                for(int phi_index = -max_phi_index; phi_index <= max_phi_index; ++phi_index) {
                    const CellIndex cellIndex{eta_index, phi_index};
                    FillCellBranches(tau, cellIndex, cellGrid.at(cellIndex));
                }
            }
            if(++n_processed % 100 == 0)
                reporter.Report(n_processed);
        }
        reporter.Report(n_processed, true);

        trainingTauTuple.Write();
        cellTuple.Write();
        std::cout << "Training tuples has been successfully stored in " << args.output() << "." << std::endl;
    }

private:
    static constexpr float default_value = tau_tuple::DefaultFillValue<float>();
    static constexpr int default_int_value = tau_tuple::DefaultFillValue<int>();

    #define CP_BR(name) trainingTauTuple().name = tau.name;
    #define TAU_ID(name, pattern, has_raw, wp_list) CP_BR(name) CP_BR(name##raw)
    void FillTauBranches(const Tau& tau)
    {
        CP_BRANCHES(run, lumi, evt, npv, rho, genEventWeight, trainingWeight, npu, pv_x, pv_y, pv_z, pv_chi2, pv_ndof)
        CP_BRANCHES(jet_index, jet_pt, jet_eta, jet_phi, jet_mass, jet_neutralHadronEnergyFraction,
                    jet_neutralEmEnergyFraction, jet_nConstituents, jet_chargedMultiplicity, jet_neutralMultiplicity,
                    jet_partonFlavour, jet_hadronFlavour, jet_has_gen_match, jet_gen_pt, jet_gen_eta, jet_gen_phi,
                    jet_gen_mass, jet_gen_n_b, jet_gen_n_c)
        CP_BRANCHES(jetTauMatch, tau_index, tau_pt, tau_eta, tau_phi, tau_mass, tau_charge, lepton_gen_match,
                    lepton_gen_charge, lepton_gen_pt, lepton_gen_eta, lepton_gen_phi, lepton_gen_mass,
                    qcd_gen_match, qcd_gen_charge, qcd_gen_pt, qcd_gen_eta, qcd_gen_phi, qcd_gen_mass,
                    tau_decayMode, tau_decayModeFinding, tau_decayModeFindingNewDMs, chargedIsoPtSum,
                    chargedIsoPtSumdR03, footprintCorrection, footprintCorrectiondR03, neutralIsoPtSum,
                    neutralIsoPtSumWeight, neutralIsoPtSumWeightdR03, neutralIsoPtSumdR03,
                    photonPtSumOutsideSignalCone, photonPtSumOutsideSignalConedR03, puCorrPtSum)

        CP_BRANCHES(tau_dxy_pca_x, tau_dxy_pca_y, tau_dxy_pca_z, tau_dxy, tau_dxy_error, tau_ip3d, tau_ip3d_error,
                    tau_dz, tau_dz_error, tau_hasSecondaryVertex, tau_sv_x, tau_sv_y, tau_sv_z,
                    tau_flightLength_x, tau_flightLength_y, tau_flightLength_z, tau_flightLength_sig)

        CP_BRANCHES(tau_pt_weighted_deta_strip, tau_pt_weighted_dphi_strip, tau_pt_weighted_dr_signal,
                    tau_pt_weighted_dr_iso, tau_leadingTrackNormChi2, tau_e_ratio, tau_gj_angle_diff, tau_n_photons,
                    tau_emFraction, tau_inside_ecal_crack, leadChargedCand_etaAtEcalEntrance)

        TAU_IDS()
        const TauType tauType = GenMatchToTauType(static_cast<GenLeptonMatch>(tau.lepton_gen_match));
        trainingTauTuple().gen_e = tauType == TauType::e;
        trainingTauTuple().gen_mu = tauType == TauType::mu;
        trainingTauTuple().gen_tau = tauType == TauType::tau;
        trainingTauTuple().gen_jet = tauType == TauType::jet;
        if(tauType != TauType::jet) {
            const auto gen_vis_sum = SumP4(tau.lepton_gen_vis_pt, tau.lepton_gen_vis_eta, tau.lepton_gen_vis_phi,
                                           tau.lepton_gen_vis_mass);
            trainingTauTuple().lepton_gen_vis_pt = static_cast<float>(gen_vis_sum.first.pt());
            trainingTauTuple().lepton_gen_vis_eta = static_cast<float>(gen_vis_sum.first.eta());
            trainingTauTuple().lepton_gen_vis_phi = static_cast<float>(gen_vis_sum.first.phi());
            trainingTauTuple().lepton_gen_vis_mass = static_cast<float>(gen_vis_sum.first.mass());
        } else {
            trainingTauTuple().lepton_gen_vis_pt = default_value;
            trainingTauTuple().lepton_gen_vis_eta = default_value;
            trainingTauTuple().lepton_gen_vis_phi = default_value;
            trainingTauTuple().lepton_gen_vis_mass = default_value;
        }

        trainingTauTuple.Fill();
    }
    #undef TAU_ID
    #undef CP_BR

    void FillCellBranches(const Tau& tau, const CellIndex& cellIndex, Cell& cell)
    {
        static boost::optional<TrainingCell> empty_training_cell;

        if(empty_training_cell)
            cellTuple() = *empty_training_cell;

        cellTuple().eta_index = cellIndex.eta;
        cellTuple().phi_index = cellIndex.phi;
        cellTuple().tau_pt = tau.tau_pt;

        const auto getPt = [&](CellObjectType type, size_t index) {
            if(type == CellObjectType::PfCandidate)
                return tau.pfCand_pt.at(index);
            if(type == CellObjectType::Electron)
                return tau.ele_pt.at(index);
            if(type == CellObjectType::Muon)
                return tau.muon_pt.at(index);
            throw exception("Unsupported CellObjectType");
        };

        const auto getBestObj = [&](CellObjectType type, size_t& n_total, size_t& best_idx) {
            const auto& index_set = cell[type];
            n_total = index_set.size();
            double max_pt = std::numeric_limits<double>::lowest();
            for(size_t index : index_set) {
                const double pt = getPt(type, index);
                if(pt > max_pt) {
                    max_pt = pt;
                    best_idx = index;
                }
            }
        };

        const auto fillMomenta = [&](const std::set<size_t>& indices, float& sum_pt, float& sum_pt_scalar,
                                     float& sum_E, const std::vector<float>& pt, const std::vector<float>& eta,
                                     const std::vector<float>& phi, const std::vector<float>& mass) {
            if(!indices.empty()) {
                const auto sum = SumP4(pt, eta, phi, mass, indices);
                sum_pt = static_cast<float>(sum.first.pt());
                sum_pt_scalar = static_cast<float>(sum.second);
                sum_E = static_cast<float>(sum.first.energy());
            } else {
                sum_pt = 0;
                sum_pt_scalar = 0;
                sum_E = 0;
            }
        };

        size_t n_pfCand, pfCand_idx;
        getBestObj(CellObjectType::PfCandidate, n_pfCand, pfCand_idx);
        if(!empty_training_cell || n_pfCand > 0) {
            cellTuple().pfCand_n_total = static_cast<int>(n_pfCand);
            cellTuple().pfCand_max_pt = n_pfCand != 0 ? tau.pfCand_pt.at(pfCand_idx) : 0;
            fillMomenta(cell.at(CellObjectType::PfCandidate), cellTuple().pfCand_sum_pt,
                                cellTuple().pfCand_sum_pt_scalar, cellTuple().pfCand_sum_E, tau.pfCand_pt,
                                tau.pfCand_eta, tau.pfCand_phi, tau.pfCand_mass);

            #define CP_BR(name) cellTuple().pfCand_##name = n_pfCand != 0 ? tau.pfCand_##name.at(pfCand_idx) : 0;
            CP_BRANCHES(jetDaughter, tauSignal, leadChargedHadrCand, tauIso, pvAssociationQuality, fromPV, puppiWeight,
                        puppiWeightNoLep, pdgId, charge, lostInnerHits, numberOfPixelHits, vertex_x, vertex_y, vertex_z,
                        hasTrackDetails, dxy, dxy_error, dz, dz_error, track_chi2, track_ndof, hcalFraction,
                        rawCaloFraction)
            #undef CP_BR
        }

        size_t n_ele, ele_idx;
        getBestObj(CellObjectType::Electron, n_ele, ele_idx);
        if(!empty_training_cell || n_ele > 0) {
            cellTuple().ele_n_total = static_cast<int>(n_ele);
            cellTuple().ele_max_pt = n_ele != 0 ? tau.ele_pt.at(ele_idx) : 0;
            fillMomenta(cell.at(CellObjectType::Electron), cellTuple().ele_sum_pt, cellTuple().ele_sum_pt_scalar,
                        cellTuple().ele_sum_E, tau.ele_pt, tau.ele_eta, tau.ele_phi, tau.ele_mass);

            #define CP_BR(name) cellTuple().ele_##name = n_ele != 0 ? tau.ele_##name.at(ele_idx) : 0;
            CP_BRANCHES(cc_ele_energy, cc_gamma_energy, cc_n_gamma, trackMomentumAtVtx, trackMomentumAtCalo,
                        trackMomentumOut, trackMomentumAtEleClus, trackMomentumAtVtxWithConstraint, ecalEnergy,
                        ecalEnergy_error, eSuperClusterOverP, eSeedClusterOverP, eSeedClusterOverPout,
                        eEleClusterOverPout, deltaEtaSuperClusterTrackAtVtx, deltaEtaSeedClusterTrackAtCalo,
                        deltaEtaEleClusterTrackAtCalo, deltaPhiEleClusterTrackAtCalo, deltaPhiSuperClusterTrackAtVtx,
                        deltaPhiSeedClusterTrackAtCalo, mvaInput_earlyBrem, mvaInput_lateBrem,
                        mvaInput_sigmaEtaEta, mvaInput_hadEnergy, mvaInput_deltaEta, gsfTrack_normalizedChi2,
                        gsfTrack_numberOfValidHits, gsfTrack_pt, gsfTrack_pt_error, closestCtfTrack_normalizedChi2,
                        closestCtfTrack_numberOfValidHits)
            #undef CP_BR
        }

        size_t n_muon, muon_idx;
        getBestObj(CellObjectType::Muon, n_muon, muon_idx);
        if(!empty_training_cell || n_muon > 0) {
            cellTuple().muon_n_total = static_cast<int>(n_muon);
            cellTuple().muon_max_pt = n_muon != 0 ? tau.muon_pt.at(muon_idx) : 0;
            fillMomenta(cell.at(CellObjectType::Muon), cellTuple().muon_sum_pt, cellTuple().muon_sum_pt_scalar,
                        cellTuple().muon_sum_E, tau.muon_pt, tau.muon_eta, tau.muon_phi, tau.muon_mass);

            #define CP_BR(name) cellTuple().muon_##name = n_muon != 0 ? tau.muon_##name.at(muon_idx) : 0;
            CP_BRANCHES(dxy, dxy_error, normalizedChi2, numberOfValidHits, segmentCompatibility, caloCompatibility,
                        pfEcalEnergy, n_matches_DT_1, n_matches_DT_2, n_matches_DT_3, n_matches_DT_4,
                        n_matches_CSC_1, n_matches_CSC_2, n_matches_CSC_3, n_matches_CSC_4,
                        n_matches_RPC_1, n_matches_RPC_2, n_matches_RPC_3, n_matches_RPC_4,
                        n_hits_DT_1, n_hits_DT_2, n_hits_DT_3, n_hits_DT_4,
                        n_hits_CSC_1, n_hits_CSC_2, n_hits_CSC_3, n_hits_CSC_4,
                        n_hits_RPC_1, n_hits_RPC_2, n_hits_RPC_3, n_hits_RPC_4)
            #undef CP_BR
        }

        if(!empty_training_cell && !n_pfCand && !n_ele && !n_muon)
            empty_training_cell = cellTuple();

        cellTuple.Fill();
    }

    CellGrid CreateCellGrid(const Tau& tau) const
    {
        CellGrid grid = cellGridRef;
        const double tau_eta = tau.tau_eta, tau_phi = tau.tau_phi;

        const auto fillGrid = [&](CellObjectType type, const std::vector<float>& eta_vec,
                                  const std::vector<float>& phi_vec) {
            if(eta_vec.size() != phi_vec.size())
                throw exception("Inconsistent cell inputs.");
            for(size_t n = 0; n < eta_vec.size(); ++n) {
                const double eta = eta_vec.at(n), phi = phi_vec.at(n);
                CellIndex cellIndex;
                if(grid.TryGetCellIndex(eta - tau_eta, phi - tau_phi, cellIndex))
                    grid.at(cellIndex)[type].insert(n);
            }
        };

        fillGrid(CellObjectType::PfCandidate, tau.pfCand_eta, tau.pfCand_phi);
        fillGrid(CellObjectType::Electron, tau.ele_eta, tau.ele_phi);
        fillGrid(CellObjectType::Muon, tau.muon_eta, tau.muon_phi);

        return grid;
    }

    static std::pair<LorentzVectorXYZ, double> SumP4(const std::vector<float>& pt, const std::vector<float>& eta,
                                                     const std::vector<float>& phi, const std::vector<float>& mass,
                                                     const std::set<size_t>& indices = {})
    {
        const size_t N = pt.size();
        if(eta.size() != N || phi.size() != N || mass.size() != N)
            throw exception("Inconsistent component sizes for p4.");
        LorentzVectorXYZ sum_p4(0, 0, 0, 0);
        double pt_scalar_sum = 0;

        const auto for_body = [&](size_t n) {
            const LorentzVectorM p4(pt.at(n), eta.at(n), phi.at(n), mass.at(n));
            sum_p4 += p4;
            pt_scalar_sum += pt.at(n);
        };

        if(indices.empty()) {
            for(size_t n = 0; n < N; ++n)
                for_body(n);
        } else {
            for(size_t n : indices)
                for_body(n);
        }
        return std::make_pair(sum_p4, pt_scalar_sum);
    }

private:
    const Arguments args;
    std::shared_ptr<TFile> inputFile, outputFile;
    TauTuple tauTuple;
    TrainingTauTuple trainingTauTuple;
    TrainingCellTuple cellTuple;
    const CellGrid cellGridRef;
};

} // namespace analysis

PROGRAM_MAIN(analysis::TrainingTupleProducer, Arguments)
