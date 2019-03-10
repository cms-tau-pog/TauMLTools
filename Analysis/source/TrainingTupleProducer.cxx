/*! Produce training tuple from tau tuple.
*/

#include <boost/preprocessor/seq.hpp>
#include <boost/preprocessor/variadic.hpp>
#include <boost/math/constants/constants.hpp>

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
    run::Argument<unsigned> n_inner_cells{"n-inner-cells", "number of inner cells in eta and phi", 21};
    run::Argument<double> inner_cell_size{"inner-cell-size", "size of the inner cell in eta and phi", 0.01};
    run::Argument<unsigned> n_outer_cells{"n-outer-cells", "number of outer cells in eta and phi", 13};
    run::Argument<double> outer_cell_size{"outer-cell-size", "size of the outer cell in eta and phi", 0.05};
    run::Argument<unsigned> n_threads{"n-threads", "number of threads", 1};
    run::Argument<Long64_t> start_entry{"start-entry", "start entry", 0};
    run::Argument<Long64_t> end_entry{"end-entry", "end entry", std::numeric_limits<Long64_t>::max()};
};

namespace analysis {

enum class CellObjectType { PfCand_electron, PfCand_muon, PfCand_chargedHadron, PfCand_neutralHadron,
                            PfCand_gamma, Electron, Muon };
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

    bool IsEmpty(const CellIndex& cellIndex) const
    {
        const Cell& cell = at(cellIndex);
        for(const auto& col : cell) {
            if(!col.second.empty())
                return false;
        }
        return true;
    }

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
        innerCellTuple("inner_cells", outputFile.get(), false), outerCellTuple("outer_cells", outputFile.get(), false),
        innerCellGridRef(args.n_inner_cells(), args.n_inner_cells(), args.inner_cell_size(), args.inner_cell_size()),
        outerCellGridRef(args.n_outer_cells(), args.n_outer_cells(), args.outer_cell_size(), args.outer_cell_size()),
        trainingWeightFactor(tauTuple.GetEntries() / 4.f)
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
            FillCellGrid(tau, innerCellGridRef, innerCellTuple, trainingTauTuple().innerCells_begin,
                         trainingTauTuple().innerCells_end);
            FillCellGrid(tau, outerCellGridRef, outerCellTuple, trainingTauTuple().outerCells_begin,
                         trainingTauTuple().outerCells_end);

            trainingTauTuple.Fill();
            if(++n_processed % 1000 == 0)
                reporter.Report(n_processed);
        }
        reporter.Report(n_processed, true);

        trainingTauTuple.Write();
        innerCellTuple.Write();
        outerCellTuple.Write();
        std::cout << "Training tuples has been successfully stored in " << args.output() << "." << std::endl;
    }

private:
    static constexpr float pi = boost::math::constants::pi<float>();

    template<typename T>
    static float GetValue(T value)
    {
        return std::isnormal(value) ? static_cast<float>(value) : 0.f;
    }

    template<typename T>
    static float GetValueLinear(T value, float min_value, float max_value, bool positive)
    {
        const float fixed_value = GetValue(value);
        const float clamped_value = std::clamp(fixed_value, min_value, max_value);
        float transformed_value = (clamped_value - min_value) / (max_value - min_value);
        if(!positive)
            transformed_value = transformed_value * 2 - 1;
        return transformed_value;
    }

    template<typename T>
    static float GetValueNorm(T value, float mean, float sigma, float n_sigmas_max = 5)
    {
        const float fixed_value = GetValue(value);
        const float norm_value = (fixed_value - mean) / sigma;
        return std::clamp(norm_value, -n_sigmas_max, n_sigmas_max);
    }

    #define CP_BR(name) trainingTauTuple().name = tau.name;
    #define TAU_ID(name, pattern, has_raw, wp_list) CP_BR(name) CP_BR(name##raw)
    void FillTauBranches(const Tau& tau)
    {
        auto& out = trainingTauTuple();
        out.run = tau.run;
        out.lumi = tau.lumi;
        out.evt = tau.evt;
        out.npv = GetValueNorm(tau.npv, 29.51, 13.31);
        out.rho = GetValueNorm(tau.rho, 21.49, 9.713);
        out.genEventWeight = tau.genEventWeight;
        out.trainingWeight = tau.trainingWeight * trainingWeightFactor;
        out.npu = tau.npu;
        out.pv_x = GetValueNorm(tau.pv_x, -0.0274, 0.001759);
        out.pv_y = GetValueNorm(tau.pv_y, 0.06932, 0.001734);
        out.pv_z = GetValueNorm(tau.pv_z, 0.8196, 3.501);
        out.pv_chi2 = GetValueNorm(tau.pv_chi2, 95.6, 45.13);
        out.pv_ndof = GetValueNorm(tau.pv_ndof, 125.2, 56.96);

        CP_BRANCHES(jet_index, jet_pt, jet_eta, jet_phi, jet_mass, jet_neutralHadronEnergyFraction,
                    jet_neutralEmEnergyFraction, jet_nConstituents, jet_chargedMultiplicity, jet_neutralMultiplicity,
                    jet_partonFlavour, jet_hadronFlavour, jet_has_gen_match, jet_gen_pt, jet_gen_eta, jet_gen_phi,
                    jet_gen_mass, jet_gen_n_b, jet_gen_n_c, jetTauMatch)

        out.tau_index = tau.tau_index;
        out.tau_pt = GetValueLinear(tau.tau_pt, 20.f, 1000.f, true);
        out.tau_eta = GetValueLinear(tau.tau_eta, -2.3f, 2.3f, false);
        out.tau_phi = GetValueLinear(tau.tau_phi, -pi, pi, false);
        out.tau_mass = GetValueNorm(tau.tau_mass, 0.6669f, 0.6553f);
        const LorentzVectorM tau_p4(tau.tau_pt, tau.tau_eta, tau.tau_phi, tau.tau_mass);
        out.tau_E_over_pt = GetValueLinear(tau_p4.energy() / tau.tau_pt, 1.f, 5.2f);
        out.tau_charge = GetValue(tau.tau_charge);
        out.tau_n_charged_prongs = GetValueLinear(tau.tau_decayMode / 5 + 1, 1, 3);
        out.tau_n_neutral_prongs = GetValueLinear(tau.tau_decayMode % 5, 0, 2);
        CP_BRANCHES(lepton_gen_match, lepton_gen_charge, lepton_gen_pt, lepton_gen_eta, lepton_gen_phi, lepton_gen_mass,
                    qcd_gen_match, qcd_gen_charge, qcd_gen_pt, qcd_gen_eta, qcd_gen_phi, qcd_gen_mass,
                    tau_decayMode, tau_decayModeFinding, tau_decayModeFindingNewDMs)


        out.chargedIsoPtSum = GetValueNorm(tau.chargedIsoPtSum, 47.78f, 123.5f);
        out.chargedIsoPtSumdR03_over_dR05 = GetValue(tau.chargedIsoPtSumdR03 / tau.chargedIsoPtSum);
        out.footprintCorrection = GetValueNorm(tau.footprintCorrection, 9.029f, 26.42f);
        out.neutralIsoPtSum = GetValueNorm(tau.neutralIsoPtSum, 57.59f, 155.3f);
        out.neutralIsoPtSumWeight_over_neutralIsoPtSum = GetValue(tau.neutralIsoPtSumWeight / tau.neutralIsoPtSum);
        out.neutralIsoPtSumWeightdR03_over_neutralIsoPtSum =
            GetValue(tau.neutralIsoPtSumWeightdR03 / tau.neutralIsoPtSum);
        out.neutralIsoPtSumdR03_over_dR05 = GetValue(tau.neutralIsoPtSumdR03 / tau.neutralIsoPtSum);
        out.photonPtSumOutsideSignalCone = GetValueNorm(tau.photonPtSumOutsideSignalCone, 1.731f, 6.846f);
        out.puCorrPtSum = GetValueNorm(tau.puCorrPtSum, 22.38f, 16.34f);

        out.tau_dxy_pca_x = GetValueNorm(tau.tau_dxy_pca_x, -0.02409f, 0.007366f);
        out.tau_dxy_pca_y = GetValueNorm(tau.tau_dxy_pca_y, 0.06747f, 0.01276f);
        out.tau_dxy_pca_z = GetValueNorm(tau.tau_dxy_pca_z, 0.7973f, 3.456f);

        const bool tau_dxy_valid = std::isnormal(tau.tau_dxy) && tau.tau_dxy > - 10
                                   && std::isnormal(tau.tau_dxy_error) && tau.tau_dxy_error > 0;
        out.tau_dxy_valid = tau_dxy_valid;
        out.tau_dxy = tau_dxy_valid ? GetValueNorm(tau.tau_dxy, 0.001813f, 0.00853f) : 0.f;
        out.tau_dxy_sig = tau_dxy_valid ? GetValueNorm(std::abs(tau.tau_dxy)/tau.tau_dxy_error, 2.26f, 4.191f) : 0.f;

        const bool tau_ip3d_valid = std::isnormal(tau.tau_ip3d) && tau.tau_ip3d > - 10
                                    && std::isnormal(tau.tau_ip3d_error) && tau.tau_ip3d_error > 0;
        out.tau_ip3d_valid = tau_ip3d_valid;
        out.tau_ip3d = tau_ip3d_valid ? GetValueNorm(tau.tau_ip3d, 0.002572f, 0.01138f) : 0.f;
        out.tau_ip3d_sig = tau_ip3d_valid
                         ? GetValueNorm(std::abs(tau.tau_ip3d) / tau.tau_ip3d_error, 2.928f, 4.466f) : 0.f;

        out.tau_dz = GetValueNorm(tau.tau_dz, 2.512e-5f, 0.01899);
        const bool tau_dz_sig_valid = std::isnormal(tau.tau_dz) && std::isnormal(tau.tau_dz_error)
                                      && tau.tau_dz_error > 0;
        out.tau_dz_sig_valid = tau_dz_sig_valid;
        out.tau_dz_sig = GetValueNorm(std::abs(tau.tau_dz) / tau.tau_dz_error, 4.717f, 11.78f);

        out.tau_flightLength_x = GetValueNorm(tau.tau_flightLength_x, -0.0002541f, 0.7362f);
        out.tau_flightLength_y = GetValueNorm(tau.tau_flightLength_y, -0.0008614f, 0.7354f);
        out.tau_flightLength_z = GetValueNorm(tau.tau_flightLength_z, -0.002195f, 1.993f);
        out.tau_flightLength_sig = GetValueNorm(out.tau_flightLength_sig, -4.78f, 9.573f);

        out.tau_pt_weighted_deta_strip = GetValueLinear(tau.tau_pt_weighted_deta_strip, 0, 1, true);
        out.tau_pt_weighted_dphi_strip = GetValueLinear(tau.tau_pt_weighted_dphi_strip, 0, 1, true);
        out.tau_pt_weighted_dr_signal = GetValueNorm(tau.tau_pt_weighted_dr_signal, 0.005201f, 0.01433f);
        out.tau_pt_weighted_dr_iso = GetValueLinear(tau.tau_pt_weighted_dr_iso, 0, 1, true);

        out.tau_leadingTrackNormChi2 = GetValueNorm(tau.tau_leadingTrackNormChi2, 1.538f, 4.401f);
        const bool tau_e_ratio_valid = std::isnormal(tau.tau_e_ratio) && tau.tau_e_ratio > 0.f;
        out.tau_e_ratio_valid = tau_e_ratio_valid;
        out.tau_e_ratio = tau_e_ratio_valid ? GetValueLinear(tau.tau_e_ratio, 0, 1, true) : 0.f;
        const bool tau_gj_angle_diff_valid = (std::isnormal(tau.tau_gj_angle_diff) || tau.tau_gj_angle_diff == 0)
            && tau.tau_gj_angle_diff >= 0;
        out.tau_gj_angle_diff_valid = tau_gj_angle_diff_valid;
        out.tau_gj_angle_diff = tau_gj_angle_diff_valid ? GetValueLinear(tau.tau_gj_angle_diff_valid, 0, pi, true);
        out.tau_n_photons = GetValueNorm(tau.tau_n_photons, 2.95f, 3.927f);
        out.tau_emFraction = GetValueLinear(tau.tau_emFraction, -1, 1, false);
        out.tau_inside_ecal_crack = GetValue(tau.tau_inside_ecal_crack);
        out.leadChargedCand_etaAtEcalEntrance_minus_tau_eta =
            GetValueNorm(tau.leadChargedCand_etaAtEcalEntrance - tau.tau_eta, 0.004058f, 0.03232f);

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
    }
    #undef TAU_ID
    #undef CP_BR

    void FillCellGrid(const Tau& tau, const CellGrid& cellGridRef, TrainingCellTuple& cellTuple, Long64_t& begin,
                      Long64_t& end, bool inner)
    {
        begin = cellTuple.GetEntries();
        auto cellGrid = CreateCellGrid(tau, cellGridRef, inner);
        const int max_eta_index = cellGrid.MaxEtaIndex(), max_phi_index = cellGrid.MaxPhiIndex();
        for(int eta_index = -max_eta_index; eta_index <= max_eta_index; ++eta_index) {
            for(int phi_index = -max_phi_index; phi_index <= max_phi_index; ++phi_index) {
                const CellIndex cellIndex{eta_index, phi_index};
                if(!cellGrid.IsEmpty(cellIndex))
                    FillCellBranches(tau, cellIndex, cellGrid.at(cellIndex), cellTuple);
            }
        }
        end = cellTuple.GetEntries();
    }

    void FillCellBranches(const Tau& tau, const CellIndex& cellIndex, Cell& cell, TrainingCellTuple& cellTuple,
                          bool inner)
    {
        auto& out = cellTuple();
        out.eta_index = cellIndex.eta;
        out.phi_index = cellIndex.phi;
        out.tau_pt = GetValueLinear(tau.tau_pt, 20.f, 1000.f, true);

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

        { // CellObjectType::PfCandidate_electron
            size_t n_pfCand, pfCand_idx;
            getBestObj(CellObjectType::PfCandidate_electron, n_pfCand, pfCand_idx);
            const bool valid = n_pfCand != 0;
            out.pfCand_ele_n_total = static_cast<int>(n_pfCand);
            out.pfCand_ele_valid = valid;
            if(inner) {
                out.pfCand_ele_pt = valid ?
                    GetValueNorm(tau.pfCand_pt.at(pfCand_idx) / tau.tau_pt, ???142.6f, ???153.9f) : 0;
            } else {
                out.pfCand_ele_pt = valid ? GetValueNorm(tau.pfCand_pt.at(pfCand_idx), 23.7f, 85.72f) : 0;
            }
            out.pfCand_ele_pt = valid ?
                GetValueNorm(tau.pfCand_pt.at(pfCand_idx), inner ? 142.6f : 23.7f, inner ? 153.9f : 85.72f) : 0;
            out.pfCand_ele_deta = valid ? GetValueLinear(tau.pfCand_eta.at(pfCand_idx) - tau.tau_eta,
                inner ? -0.1f : -0.5f, inner ? 0.1f : 0.5f, false) : 0;
            out.pfCand_ele_dphi = valid ? GetValueLinear(tau.pfCand_phi.at(pfCand_idx) - tau.tau_phi,
                inner ? -0.1f : -0.5f, inner ? 0.1f : 0.5f, false);
            out.pfCand_ele_pvAssociationQuality = valid ?
                GetValueLinear(tau.pfCand_pvAssociationQuality.at(pfCand_idx), 0, 7, true) : 0;
            out.pfCand_ele_puppiWeight = valid ? GetValue(tau.pfCand_puppiWeight.at(pfCand_idx)) : 0;
            out.pfCand_ele_charge = valid ? GetValue(tau.pfCand_charge.at(pfCand_idx)) : 0;
            out.pfCand_ele_lostInnerHits = valid ? GetValue(tau.pfCand_lostInnerHits.at(pfCand_idx)) : 0;
            out.pfCand_ele_numberOfPixelHits = valid ?
                GetValueLinear(tau.pfCand_numberOfPixelHits.at(pfCand_idx), 0, 10, true) : 0;

            const bool hasTrackDetails = valid && tau.pfCand_hasTrackDetails.at(pfCand_idx) == 1;
            out.pfCand_ele_hasTrackDetails = hasTrackDetails;
            out.pfCand_ele_dxy = hasTrackDetails ? GetValueNorm(tau.pfCand_dxy.at(pfCand_idx), 2.731e-5f, 0.171f) : 0;
            out.pfCand_ele_dxy_sig = hasTrackDetails ? GetValueNorm(std::abs(tau.pfCand_dxy.at(pfCand_idx)) /
                tau.pfCand_dxy_error.at(pfCand_idx), 1.634f, 6.45f) : 0;
            out.pfCand_ele_dz = hasTrackDetails ? GetValueNorm(tau.pfCand_dz.at(pfCand_idx), 0.001f, 1.02f) : 0;
            out.pfCand_ele_dz_sig = hasTrackDetails ? GetValueNorm(std::abs(tau.pfCand_dz.at(pfCand_idx)) /
                tau.pfCand_dz_error.at(pfCand_idx), 24.56f, 210.4f) : 0;
            out.pfCand_ele_track_chi2_ndof = hasTrackDetails ?
                GetValueNorm(tau.pfCand_track_chi2.at(pfCand_idx), ???30.04f, ???101.7f) : 0;
            out.pfCand_ele_track_ndof = hasTrackDetails ?
                GetValueNorm(tau.pfCand_track_ndof.at(pfCand_idx), 15.18f, 3.203f) : 0;
        }

        { // CellObjectType::PfCandidate_muon
            size_t n_pfCand, pfCand_idx;
            getBestObj(CellObjectType::PfCandidate_muon, n_pfCand, pfCand_idx);
            const bool valid = n_pfCand != 0;
            out.pfCand_muon_n_total = static_cast<int>(n_pfCand);
            out.pfCand_muon_valid = valid;
            if(inner) {
                out.pfCand_muon_pt = valid ?
                    GetValueNorm(tau.pfCand_pt.at(pfCand_idx) / tau.tau_pt, 0.9509f, 0.4294f) : 0;
            } else {
                out.pfCand_muon_pt = valid ? GetValueNorm(tau.pfCand_pt.at(pfCand_idx), ???0.9509f, ???0.4294f) : 0;
            }
        }

        cellTuple().pfCand_max_pt = n_pfCand != 0 ? tau.pfCand_pt.at(pfCand_idx) : 0;
        fillMomenta(cell.at(CellObjectType::PfCandidate), cellTuple().pfCand_sum_pt,
                            cellTuple().pfCand_sum_pt_scalar, cellTuple().pfCand_sum_E, tau.pfCand_pt,
                            tau.pfCand_eta, tau.pfCand_phi, tau.pfCand_mass);

        #define CP_BR(name) cellTuple().pfCand_##name = n_pfCand != 0 ? GetValue(tau.pfCand_##name.at(pfCand_idx)) : 0;
        CP_BRANCHES(jetDaughter, tauSignal, leadChargedHadrCand, tauIso, pvAssociationQuality, fromPV, puppiWeight,
                    puppiWeightNoLep, pdgId, charge, lostInnerHits, numberOfPixelHits, vertex_x, vertex_y, vertex_z,
                    hasTrackDetails, dxy, dxy_error, dz, dz_error, track_chi2, track_ndof, hcalFraction,
                    rawCaloFraction)
        #undef CP_BR

        size_t n_ele, ele_idx;
        getBestObj(CellObjectType::Electron, n_ele, ele_idx);
        cellTuple().ele_n_total = static_cast<int>(n_ele);
        cellTuple().ele_max_pt = n_ele != 0 ? tau.ele_pt.at(ele_idx) : 0;
        fillMomenta(cell.at(CellObjectType::Electron), cellTuple().ele_sum_pt, cellTuple().ele_sum_pt_scalar,
                    cellTuple().ele_sum_E, tau.ele_pt, tau.ele_eta, tau.ele_phi, tau.ele_mass);

        #define CP_BR(name) cellTuple().ele_##name = n_ele != 0 ? GetValue(tau.ele_##name.at(ele_idx)) : 0;
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

        size_t n_muon, muon_idx;
        getBestObj(CellObjectType::Muon, n_muon, muon_idx);
        cellTuple().muon_n_total = static_cast<int>(n_muon);
        cellTuple().muon_max_pt = n_muon != 0 ? tau.muon_pt.at(muon_idx) : 0;
        fillMomenta(cell.at(CellObjectType::Muon), cellTuple().muon_sum_pt, cellTuple().muon_sum_pt_scalar,
                    cellTuple().muon_sum_E, tau.muon_pt, tau.muon_eta, tau.muon_phi, tau.muon_mass);

        #define CP_BR(name) cellTuple().muon_##name = n_muon != 0 ? GetValue(tau.muon_##name.at(muon_idx)) : 0;
        CP_BRANCHES(dxy, dxy_error, normalizedChi2, numberOfValidHits, segmentCompatibility, caloCompatibility,
                    pfEcalEnergy, n_matches_DT_1, n_matches_DT_2, n_matches_DT_3, n_matches_DT_4,
                    n_matches_CSC_1, n_matches_CSC_2, n_matches_CSC_3, n_matches_CSC_4,
                    n_matches_RPC_1, n_matches_RPC_2, n_matches_RPC_3, n_matches_RPC_4,
                    n_hits_DT_1, n_hits_DT_2, n_hits_DT_3, n_hits_DT_4,
                    n_hits_CSC_1, n_hits_CSC_2, n_hits_CSC_3, n_hits_CSC_4,
                    n_hits_RPC_1, n_hits_RPC_2, n_hits_RPC_3, n_hits_RPC_4)
        #undef CP_BR

        cellTuple.Fill();
    }

    static double getInnerSignalConeRadius(double pt)
    {
        static constexpr double min_pt = 30., min_radius = 0.05, cone_opening_coef = 3.;
        // This is equivalent of the original formula (std::max(std::min(0.1, 3.0/pt), 0.05)
        return std::max(cone_opening_coef / std::max(pt, min_pt), min_radius);
    }

    static CellObjectType GetCellObjectType(int pdgId)
    {
        static const std::map<int, CellObjectType> obj_types = {
            { 11, CellObjectType::PfCand_electron },
            { 13, CellObjectType::PfCand_muon },
            { 22, CellObjectType::PfCand_gamma },
            { 130, CellObjectType::PfCand_neutralHadron },
            { 211, CellObjectType::PfCand_chargedHadron }
        };

        auto iter = obj_types.find(std::abs(pdgId));
        if(iter == obj_types.end())
            throw exception("Unknown object pdg id = %1%.") % pdgId;
        return iter->second;
    }

    CellGrid CreateCellGrid(const Tau& tau, const CellGrid& cellGridRef, bool inner) const
    {
        static constexpr double iso_cone = 0.5;

        CellGrid grid = cellGridRef;
        const double tau_pt = tau.tau_pt, tau_eta = tau.tau_eta, tau_phi = tau.tau_phi;

        const auto fillGrid = [&](CellObjectType type, const std::vector<float>& eta_vec,
                                  const std::vector<float>& phi_vec, const std::vector<int>& pdgId = {}) {
            if(eta_vec.size() != phi_vec.size())
                throw exception("Inconsistent cell inputs.");
            for(size_t n = 0; n < eta_vec.size(); ++n) {
                if(pdgId.size() && GetCellObjectType(pdgId.at(n)) != type) continue;
                const double eta = eta_vec.at(n), phi = phi_vec.at(n);
                const double deta = eta - tau_eta, dphi = phi - tau_phi;
                const double dR = std::hypot(deta, dphi);
                const bool inside_signal_cone = dR < getInnerSignalConeRadius(tau_pt);
                const bool inside_iso_cone = dR < iso_cone;
                if(inner && !inside_signal_cone) continue;
                if(!inner && (inside_signal_cone || !inside_iso_cone)) continue;
                CellIndex cellIndex;
                if(grid.TryGetCellIndex(eta - tau_eta, phi - tau_phi, cellIndex))
                    grid.at(cellIndex)[type].insert(n);
            }
        };

        fillGrid(CellObjectType::PfCandidate_electron, tau.pfCand_eta, tau.pfCand_phi, tau.pfCand_pdgId);
        fillGrid(CellObjectType::PfCandidate_muon, tau.pfCand_eta, tau.pfCand_phi, tau.pfCand_pdgId);
        fillGrid(CellObjectType::PfCandidate_chargedHadron, tau.pfCand_eta, tau.pfCand_phi, tau.pfCand_pdgId);
        fillGrid(CellObjectType::PfCandidate_neutralHadron, tau.pfCand_eta, tau.pfCand_phi, tau.pfCand_pdgId);
        fillGrid(CellObjectType::PfCandidate_gamma, tau.pfCand_eta, tau.pfCand_phi, tau.pfCand_pdgId);
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
    TrainingCellTuple innerCellTuple, outerCellTuple;
    const CellGrid innerCellGridRef, outerCellGridRef;
    const float trainingWeightFactor;
};

} // namespace analysis

PROGRAM_MAIN(analysis::TrainingTupleProducer, Arguments)
