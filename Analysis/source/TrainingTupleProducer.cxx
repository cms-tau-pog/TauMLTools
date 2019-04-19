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
    run::Argument<unsigned> n_inner_cells{"n-inner-cells", "number of inner cells in eta and phi", 11};
    run::Argument<double> inner_cell_size{"inner-cell-size", "size of the inner cell in eta and phi", 0.02};
    run::Argument<unsigned> n_outer_cells{"n-outer-cells", "number of outer cells in eta and phi", 21};
    run::Argument<double> outer_cell_size{"outer-cell-size", "size of the outer cell in eta and phi", 0.05};
    run::Argument<unsigned> n_threads{"n-threads", "number of threads", 1};
    run::Argument<Long64_t> start_entry{"start-entry", "start entry", 0};
    run::Argument<Long64_t> end_entry{"end-entry", "end entry", std::numeric_limits<Long64_t>::max()};
};

namespace analysis {

enum class CellObjectType { PfCand_electron, PfCand_muon, PfCand_chargedHadron, PfCand_neutralHadron,
                            PfCand_gamma, Electron, Muon };
using Cell = std::map<CellObjectType, std::set<size_t>>;
struct CellIndex {
    int eta, phi;

    bool operator<(const CellIndex& other) const
    {
        if(eta != other.eta) return eta < other.eta;
        return phi < other.phi;
    }
};

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
                         trainingTauTuple().innerCells_end, true);
            FillCellGrid(tau, outerCellGridRef, outerCellTuple, trainingTauTuple().outerCells_begin,
                         trainingTauTuple().outerCells_end, false);

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

    template<typename Scalar>
    static Scalar DeltaPhi(Scalar phi1, Scalar phi2)
    {
        static constexpr Scalar pi = boost::math::constants::pi<Scalar>();
        Scalar dphi = phi1 - phi2;
        if(dphi > pi)
            dphi -= 2*pi;
        else if(dphi <= -pi)
            dphi += 2*pi;
        return dphi;
    }

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
        out.npv = GetValueNorm(tau.npv, 29.51f, 13.31f);
        out.rho = GetValueNorm(tau.rho, 21.49f, 9.713f);
        out.genEventWeight = tau.genEventWeight;
        out.trainingWeight = tau.trainingWeight * trainingWeightFactor;
        out.npu = tau.npu;
        out.pv_x = GetValueNorm(tau.pv_x, -0.0274f, 0.0018f);
        out.pv_y = GetValueNorm(tau.pv_y, 0.0693f, 0.0017f);
        out.pv_z = GetValueNorm(tau.pv_z, 0.8196f, 3.501f);
        out.pv_chi2 = GetValueNorm(tau.pv_chi2, 95.6f, 45.13f);
        out.pv_ndof = GetValueNorm(tau.pv_ndof, 125.2f, 56.96f);

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
        out.tau_E_over_pt = GetValueLinear(tau_p4.energy() / tau.tau_pt, 1.f, 5.2f, true);
        out.tau_charge = GetValue(tau.tau_charge);
        out.tau_n_charged_prongs = GetValueLinear(tau.tau_decayMode / 5 + 1, 1, 3, true);
        out.tau_n_neutral_prongs = GetValueLinear(tau.tau_decayMode % 5, 0, 2, true);
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

        out.tau_dxy_pca_x = GetValueNorm(tau.tau_dxy_pca_x, -0.0241f, 0.0074f);
        out.tau_dxy_pca_y = GetValueNorm(tau.tau_dxy_pca_y, 0.0675f, 0.0128f);
        out.tau_dxy_pca_z = GetValueNorm(tau.tau_dxy_pca_z, 0.7973f, 3.456f);

        const bool tau_dxy_valid = std::isnormal(tau.tau_dxy) && tau.tau_dxy > - 10
                                   && std::isnormal(tau.tau_dxy_error) && tau.tau_dxy_error > 0;
        out.tau_dxy_valid = tau_dxy_valid;
        out.tau_dxy = tau_dxy_valid ? GetValueNorm(tau.tau_dxy, 0.0018f, 0.0085f) : 0.f;
        out.tau_dxy_sig = tau_dxy_valid ? GetValueNorm(std::abs(tau.tau_dxy)/tau.tau_dxy_error, 2.26f, 4.191f) : 0.f;

        const bool tau_ip3d_valid = std::isnormal(tau.tau_ip3d) && tau.tau_ip3d > - 10
                                    && std::isnormal(tau.tau_ip3d_error) && tau.tau_ip3d_error > 0;
        out.tau_ip3d_valid = tau_ip3d_valid;
        out.tau_ip3d = tau_ip3d_valid ? GetValueNorm(tau.tau_ip3d, 0.0026f, 0.0114f) : 0.f;
        out.tau_ip3d_sig = tau_ip3d_valid
                         ? GetValueNorm(std::abs(tau.tau_ip3d) / tau.tau_ip3d_error, 2.928f, 4.466f) : 0.f;

        out.tau_dz = GetValueNorm(tau.tau_dz, 0.f, 0.0190f);
        const bool tau_dz_sig_valid = std::isnormal(tau.tau_dz) && std::isnormal(tau.tau_dz_error)
                                      && tau.tau_dz_error > 0;
        out.tau_dz_sig_valid = tau_dz_sig_valid;
        out.tau_dz_sig = GetValueNorm(std::abs(tau.tau_dz) / tau.tau_dz_error, 4.717f, 11.78f);

        out.tau_flightLength_x = GetValueNorm(tau.tau_flightLength_x, -0.0003f, 0.7362f);
        out.tau_flightLength_y = GetValueNorm(tau.tau_flightLength_y, -0.0009f, 0.7354f);
        out.tau_flightLength_z = GetValueNorm(tau.tau_flightLength_z, -0.0022f, 1.993f);
        out.tau_flightLength_sig = GetValueNorm(out.tau_flightLength_sig, -4.78f, 9.573f);

        out.tau_pt_weighted_deta_strip = GetValueLinear(tau.tau_pt_weighted_deta_strip, 0, 1, true);
        out.tau_pt_weighted_dphi_strip = GetValueLinear(tau.tau_pt_weighted_dphi_strip, 0, 1, true);
        out.tau_pt_weighted_dr_signal = GetValueNorm(tau.tau_pt_weighted_dr_signal, 0.0052f, 0.01433f);
        out.tau_pt_weighted_dr_iso = GetValueLinear(tau.tau_pt_weighted_dr_iso, 0, 1, true);

        out.tau_leadingTrackNormChi2 = GetValueNorm(tau.tau_leadingTrackNormChi2, 1.538f, 4.401f);
        const bool tau_e_ratio_valid = std::isnormal(tau.tau_e_ratio) && tau.tau_e_ratio > 0.f;
        out.tau_e_ratio_valid = tau_e_ratio_valid;
        out.tau_e_ratio = tau_e_ratio_valid ? GetValueLinear(tau.tau_e_ratio, 0, 1, true) : 0.f;
        const bool tau_gj_angle_diff_valid = (std::isnormal(tau.tau_gj_angle_diff) || tau.tau_gj_angle_diff == 0)
            && tau.tau_gj_angle_diff >= 0;
        out.tau_gj_angle_diff_valid = tau_gj_angle_diff_valid;
        out.tau_gj_angle_diff = tau_gj_angle_diff_valid ? GetValueLinear(tau.tau_gj_angle_diff, 0, pi, true) : 0;
        out.tau_n_photons = GetValueNorm(tau.tau_n_photons, 2.95f, 3.927f);
        out.tau_emFraction = GetValueLinear(tau.tau_emFraction, -1, 1, false);
        out.tau_inside_ecal_crack = GetValue(tau.tau_inside_ecal_crack);
        out.leadChargedCand_etaAtEcalEntrance_minus_tau_eta =
            GetValueNorm(tau.leadChargedCand_etaAtEcalEntrance - tau.tau_eta, 0.0042f, 0.0323f);

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
            trainingTauTuple().lepton_gen_vis_pt = 0;
            trainingTauTuple().lepton_gen_vis_eta = 0;
            trainingTauTuple().lepton_gen_vis_phi = 0;
            trainingTauTuple().lepton_gen_vis_mass = 0;
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
        const int max_distance = max_eta_index + max_phi_index;
        std::set<CellIndex> processed_cells;
        for(int distance = 0; distance <= max_distance; ++distance) {
            const int max_eta_d = std::min(max_eta_index, distance);
            for(int eta_index = -max_eta_d; eta_index <= max_eta_d; ++eta_index) {
                const int max_phi_d = distance - std::abs(eta_index);
                if(max_phi_d > max_phi_index) continue;
                const size_t n_max = max_phi_d ? 2 : 1;
                for(size_t n = 0; n < n_max; ++n) {
                    int phi_index = n ? max_phi_d : -max_phi_d;
                    const CellIndex cellIndex{eta_index, phi_index};
                    if(processed_cells.count(cellIndex))
                        throw exception("Duplicated cell index in FillCellGrid.");
                    processed_cells.insert(cellIndex);
                    if(!cellGrid.IsEmpty(cellIndex))
                        FillCellBranches(tau, cellIndex, cellGrid.at(cellIndex), cellTuple, inner);
                }
            }
        }
        if(processed_cells.size() != static_cast<size_t>( (2 * max_eta_index + 1) * (2 * max_phi_index + 1) ))
            throw exception("Not all cell indices are processed in FillCellGrid.");
        end = cellTuple.GetEntries();
    }

    void FillCellBranches(const Tau& tau, const CellIndex& cellIndex, Cell& cell, TrainingCellTuple& cellTuple,
                          bool inner)
    {
        auto& out = cellTuple();
        out.eta_index = cellIndex.eta;
        out.phi_index = cellIndex.phi;
        out.tau_pt = GetValueLinear(tau.tau_pt, 20.f, 1000.f, true);
        out.rho = GetValueNorm(tau.rho, 21.49f, 9.713f);

        const auto getPt = [&](CellObjectType type, size_t index) {
            if(type == CellObjectType::Electron)
                return tau.ele_pt.at(index);
            if(type == CellObjectType::Muon)
                return tau.muon_pt.at(index);
            return tau.pfCand_pt.at(index);
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

        { // CellObjectType::PfCand_electron
            size_t n_pfCand, pfCand_idx;
            getBestObj(CellObjectType::PfCand_electron, n_pfCand, pfCand_idx);
            const bool valid = n_pfCand != 0;
            out.pfCand_ele_n_total = static_cast<int>(n_pfCand);
            out.pfCand_ele_valid = valid;

            out.pfCand_ele_rel_pt = valid ? GetValueNorm(tau.pfCand_pt.at(pfCand_idx) / tau.tau_pt,
                inner ? 0.9792f : 0.304f, inner ? 0.5383f : 1.845f) : 0;
            out.pfCand_ele_deta = valid ? GetValueLinear(tau.pfCand_eta.at(pfCand_idx) - tau.tau_eta,
                inner ? -0.1f : -0.5f, inner ? 0.1f : 0.5f, false) : 0;
            out.pfCand_ele_dphi = valid ? GetValueLinear(DeltaPhi(tau.pfCand_phi.at(pfCand_idx), tau.tau_phi),
                inner ? -0.1f : -0.5f, inner ? 0.1f : 0.5f, false) : 0;
            out.pfCand_ele_tauSignal = valid ? GetValue(tau.pfCand_tauSignal.at(pfCand_idx)) : 0;
            out.pfCand_ele_tauIso = valid ? GetValue(tau.pfCand_tauIso.at(pfCand_idx)) : 0;
            out.pfCand_ele_pvAssociationQuality = valid ?
                GetValueLinear(tau.pfCand_pvAssociationQuality.at(pfCand_idx), 0, 7, true) : 0;
            out.pfCand_ele_puppiWeight = valid ? GetValue(tau.pfCand_puppiWeight.at(pfCand_idx)) : 0;
            out.pfCand_ele_charge = valid ? GetValue(tau.pfCand_charge.at(pfCand_idx)) : 0;
            out.pfCand_ele_lostInnerHits = valid ? GetValue(tau.pfCand_lostInnerHits.at(pfCand_idx)) : 0;
            out.pfCand_ele_numberOfPixelHits = valid ?
                GetValueLinear(tau.pfCand_numberOfPixelHits.at(pfCand_idx), 0, 10, true) : 0;

            out.pfCand_ele_vertex_dx = valid ?
                GetValueNorm(tau.pfCand_vertex_x.at(pfCand_idx) - tau.pv_x, 0.f, 0.1221f) : 0;
            out.pfCand_ele_vertex_dy = valid ?
                GetValueNorm(tau.pfCand_vertex_y.at(pfCand_idx) - tau.pv_y, 0.f, 0.1226f) : 0;
            out.pfCand_ele_vertex_dz = valid ?
                GetValueNorm(tau.pfCand_vertex_z.at(pfCand_idx) - tau.pv_z, 0.001f, 1.024f) : 0;
            out.pfCand_ele_vertex_dx_tauFL = valid ?
                GetValueNorm(tau.pfCand_vertex_x.at(pfCand_idx) - tau.pv_x - tau.tau_flightLength_x, 0.f, 0.3411f) : 0;
            out.pfCand_ele_vertex_dy_tauFL = valid ? GetValueNorm(tau.pfCand_vertex_y.at(pfCand_idx) - tau.pv_y -
                tau.tau_flightLength_y, 0.0003f, 0.3385f) : 0;
            out.pfCand_ele_vertex_dz_tauFL = valid ? GetValueNorm(tau.pfCand_vertex_z.at(pfCand_idx) - tau.pv_z -
                tau.tau_flightLength_z, 0.f, 1.307f) : 0;

            const bool hasTrackDetails = valid && tau.pfCand_hasTrackDetails.at(pfCand_idx) == 1;
            out.pfCand_ele_hasTrackDetails = hasTrackDetails;
            out.pfCand_ele_dxy = hasTrackDetails ? GetValueNorm(tau.pfCand_dxy.at(pfCand_idx), 0.f, 0.171f) : 0;
            out.pfCand_ele_dxy_sig = hasTrackDetails ? GetValueNorm(std::abs(tau.pfCand_dxy.at(pfCand_idx)) /
                tau.pfCand_dxy_error.at(pfCand_idx), 1.634f, 6.45f) : 0;
            out.pfCand_ele_dz = hasTrackDetails ? GetValueNorm(tau.pfCand_dz.at(pfCand_idx), 0.001f, 1.02f) : 0;
            out.pfCand_ele_dz_sig = hasTrackDetails ? GetValueNorm(std::abs(tau.pfCand_dz.at(pfCand_idx)) /
                tau.pfCand_dz_error.at(pfCand_idx), 24.56f, 210.4f) : 0;
            out.pfCand_ele_track_chi2_ndof = hasTrackDetails ? GetValueNorm(tau.pfCand_track_chi2.at(pfCand_idx) /
                tau.pfCand_track_ndof.at(pfCand_idx), 2.272f, 8.439f) : 0;
            out.pfCand_ele_track_ndof = hasTrackDetails ?
                GetValueNorm(tau.pfCand_track_ndof.at(pfCand_idx), 15.18f, 3.203f) : 0;
        }

        { // CellObjectType::PfCand_muon
            size_t n_pfCand, pfCand_idx;
            getBestObj(CellObjectType::PfCand_muon, n_pfCand, pfCand_idx);
            const bool valid = n_pfCand != 0;
            out.pfCand_muon_n_total = static_cast<int>(n_pfCand);
            out.pfCand_muon_valid = valid;

            out.pfCand_muon_rel_pt = valid ? GetValueNorm(tau.pfCand_pt.at(pfCand_idx) / tau.tau_pt,
                inner ? 0.9509f : 0.0861f, inner ? 0.4294f : 0.4065f) : 0;
            out.pfCand_muon_deta = valid ? GetValueLinear(tau.pfCand_eta.at(pfCand_idx) - tau.tau_eta,
                inner ? -0.1f : -0.5f, inner ? 0.1f : 0.5f, false) : 0;
            out.pfCand_muon_dphi = valid ? GetValueLinear(DeltaPhi(tau.pfCand_phi.at(pfCand_idx), tau.tau_phi),
                inner ? -0.1f : -0.5f, inner ? 0.1f : 0.5f, false) : 0;
            out.pfCand_muon_tauSignal = valid ? GetValue(tau.pfCand_tauSignal.at(pfCand_idx)) : 0;
            out.pfCand_muon_tauIso = valid ? GetValue(tau.pfCand_tauIso.at(pfCand_idx)) : 0;
            out.pfCand_muon_pvAssociationQuality = valid ?
                GetValueLinear(tau.pfCand_pvAssociationQuality.at(pfCand_idx), 0, 7, true) : 0;
            out.pfCand_muon_fromPV = valid ? GetValueLinear(tau.pfCand_fromPV.at(pfCand_idx), 0, 3, true) : 0;
            out.pfCand_muon_puppiWeight = valid ? GetValue(tau.pfCand_puppiWeight.at(pfCand_idx)) : 0;
            out.pfCand_muon_charge = valid ? GetValue(tau.pfCand_charge.at(pfCand_idx)) : 0;
            out.pfCand_muon_lostInnerHits = valid ? GetValue(tau.pfCand_lostInnerHits.at(pfCand_idx)) : 0;
            out.pfCand_muon_numberOfPixelHits = valid ?
                GetValueLinear(tau.pfCand_numberOfPixelHits.at(pfCand_idx), 0, 11, true) : 0;

            out.pfCand_muon_vertex_dx = valid ?
                GetValueNorm(tau.pfCand_vertex_x.at(pfCand_idx) - tau.pv_x, -0.0007f, 0.6869f) : 0;
            out.pfCand_muon_vertex_dy = valid ?
                GetValueNorm(tau.pfCand_vertex_y.at(pfCand_idx) - tau.pv_y, 0.0001f, 0.6784f) : 0;
            out.pfCand_muon_vertex_dz = valid ?
                GetValueNorm(tau.pfCand_vertex_z.at(pfCand_idx) - tau.pv_z, -0.0117f, 4.097f) : 0;
            out.pfCand_muon_vertex_dx_tauFL = valid ? GetValueNorm(tau.pfCand_vertex_x.at(pfCand_idx) - tau.pv_x -
                tau.tau_flightLength_x, -0.0001f, 0.8642f) : 0;
            out.pfCand_muon_vertex_dy_tauFL = valid ? GetValueNorm(tau.pfCand_vertex_y.at(pfCand_idx) - tau.pv_y -
                tau.tau_flightLength_y, 0.0004f, 0.8561f) : 0;
            out.pfCand_muon_vertex_dz_tauFL = valid ? GetValueNorm(tau.pfCand_vertex_z.at(pfCand_idx) - tau.pv_z -
                tau.tau_flightLength_z, -0.0118f, 4.405f) : 0;

            const bool hasTrackDetails = valid && tau.pfCand_hasTrackDetails.at(pfCand_idx) == 1;
            out.pfCand_muon_hasTrackDetails = hasTrackDetails;
            out.pfCand_muon_dxy = hasTrackDetails ?
                GetValueNorm(tau.pfCand_dxy.at(pfCand_idx), -0.0045f, 0.9655f) : 0;
            out.pfCand_muon_dxy_sig = hasTrackDetails ? GetValueNorm(std::abs(tau.pfCand_dxy.at(pfCand_idx)) /
                tau.pfCand_dxy_error.at(pfCand_idx), 4.575f, 42.36f) : 0;
            out.pfCand_muon_dz = hasTrackDetails ? GetValueNorm(tau.pfCand_dz.at(pfCand_idx), -0.0117f, 4.097f) : 0;
            out.pfCand_muon_dz_sig = hasTrackDetails ? GetValueNorm(std::abs(tau.pfCand_dz.at(pfCand_idx)) /
                tau.pfCand_dz_error.at(pfCand_idx), 80.37f, 343.3f) : 0;
            out.pfCand_muon_track_chi2_ndof = hasTrackDetails ? GetValueNorm(tau.pfCand_track_chi2.at(pfCand_idx) /
                tau.pfCand_track_ndof.at(pfCand_idx), 0.69f, 1.711f) : 0;
            out.pfCand_muon_track_ndof = hasTrackDetails ?
                GetValueNorm(tau.pfCand_track_ndof.at(pfCand_idx), 17.5f, 5.11f) : 0;
        }

        { // CellObjectType::PfCand_chargedHadron
            size_t n_pfCand, pfCand_idx;
            getBestObj(CellObjectType::PfCand_chargedHadron, n_pfCand, pfCand_idx);
            const bool valid = n_pfCand != 0;
            out.pfCand_chHad_n_total = static_cast<int>(n_pfCand);
            out.pfCand_chHad_valid = valid;

            out.pfCand_chHad_rel_pt = valid ? GetValueNorm(tau.pfCand_pt.at(pfCand_idx) / tau.tau_pt,
                inner ? 0.2564f : 0.0194f, inner ? 0.8607f : 0.1865f) : 0;
            out.pfCand_chHad_deta = valid ? GetValueLinear(tau.pfCand_eta.at(pfCand_idx) - tau.tau_eta,
                inner ? -0.1f : -0.5f, inner ? 0.1f : 0.5f, false) : 0;
            out.pfCand_chHad_dphi = valid ? GetValueLinear(DeltaPhi(tau.pfCand_phi.at(pfCand_idx), tau.tau_phi),
                inner ? -0.1f : -0.5f, inner ? 0.1f : 0.5f, false) : 0;
            out.pfCand_chHad_tauSignal = valid ? GetValue(tau.pfCand_tauSignal.at(pfCand_idx)) : 0;
            out.pfCand_chHad_leadChargedHadrCand = valid ? GetValue(tau.pfCand_leadChargedHadrCand.at(pfCand_idx)) : 0;
            out.pfCand_chHad_tauIso = valid ? GetValue(tau.pfCand_tauIso.at(pfCand_idx)) : 0;
            out.pfCand_chHad_pvAssociationQuality = valid ?
                GetValueLinear(tau.pfCand_pvAssociationQuality.at(pfCand_idx), 0, 7, true) : 0;
            out.pfCand_chHad_fromPV = valid ? GetValueLinear(tau.pfCand_fromPV.at(pfCand_idx), 0, 3, true) : 0;
            out.pfCand_chHad_puppiWeight = valid ? GetValue(tau.pfCand_puppiWeight.at(pfCand_idx)) : 0;
            out.pfCand_chHad_puppiWeightNoLep = valid ? GetValue(tau.pfCand_puppiWeightNoLep.at(pfCand_idx)) : 0;
            out.pfCand_chHad_charge = valid ? GetValue(tau.pfCand_charge.at(pfCand_idx)) : 0;
            out.pfCand_chHad_lostInnerHits = valid ? GetValue(tau.pfCand_lostInnerHits.at(pfCand_idx)) : 0;
            out.pfCand_chHad_numberOfPixelHits = valid ?
                GetValueLinear(tau.pfCand_numberOfPixelHits.at(pfCand_idx), 0, 12, true) : 0;

            out.pfCand_chHad_vertex_dx = valid ?
                GetValueNorm(tau.pfCand_vertex_x.at(pfCand_idx) - tau.pv_x, 0.0005f, 1.735f) : 0;
            out.pfCand_chHad_vertex_dy = valid ?
                GetValueNorm(tau.pfCand_vertex_y.at(pfCand_idx) - tau.pv_y, -0.0008f, 1.752f) : 0;
            out.pfCand_chHad_vertex_dz = valid ?
                GetValueNorm(tau.pfCand_vertex_z.at(pfCand_idx) - tau.pv_z, -0.0201f, 8.333f) : 0;
            out.pfCand_chHad_vertex_dx_tauFL = valid ? GetValueNorm(tau.pfCand_vertex_x.at(pfCand_idx) - tau.pv_x -
                tau.tau_flightLength_x, -0.0014f, 1.93f) : 0;
            out.pfCand_chHad_vertex_dy_tauFL = valid ? GetValueNorm(tau.pfCand_vertex_y.at(pfCand_idx) - tau.pv_y -
                tau.tau_flightLength_y, 0.0022f, 1.948f) : 0;
            out.pfCand_chHad_vertex_dz_tauFL = valid ? GetValueNorm(tau.pfCand_vertex_z.at(pfCand_idx) - tau.pv_z -
                tau.tau_flightLength_z, -0.0138f, 8.622f) : 0;

            const bool hasTrackDetails = valid && tau.pfCand_hasTrackDetails.at(pfCand_idx) == 1;
            out.pfCand_chHad_hasTrackDetails = hasTrackDetails;
            out.pfCand_chHad_dxy = hasTrackDetails ?
                GetValueNorm(tau.pfCand_dxy.at(pfCand_idx), -0.012f, 2.386f) : 0;
            out.pfCand_chHad_dxy_sig = hasTrackDetails ? GetValueNorm(std::abs(tau.pfCand_dxy.at(pfCand_idx)) /
                tau.pfCand_dxy_error.at(pfCand_idx), 6.417f, 36.28f) : 0;
            out.pfCand_chHad_dz = hasTrackDetails ? GetValueNorm(tau.pfCand_dz.at(pfCand_idx), -0.0246f, 7.618f) : 0;
            out.pfCand_chHad_dz_sig = hasTrackDetails ? GetValueNorm(std::abs(tau.pfCand_dz.at(pfCand_idx)) /
                tau.pfCand_dz_error.at(pfCand_idx), 301.3f, 491.1f) : 0;
            out.pfCand_chHad_track_chi2_ndof = hasTrackDetails && tau.pfCand_track_ndof.at(pfCand_idx) > 0 ?
                GetValueNorm(tau.pfCand_track_chi2.at(pfCand_idx) / tau.pfCand_track_ndof.at(pfCand_idx),
                0.7876f, 3.694f) : 0;
            out.pfCand_chHad_track_ndof = hasTrackDetails && tau.pfCand_track_ndof.at(pfCand_idx) > 0 ?
                GetValueNorm(tau.pfCand_track_ndof.at(pfCand_idx), 13.92f, 6.581f) : 0;

            out.pfCand_chHad_hcalFraction = valid ? GetValue(tau.pfCand_hcalFraction.at(pfCand_idx)) : 0;
            out.pfCand_chHad_rawCaloFraction = valid ?
                GetValueLinear(tau.pfCand_rawCaloFraction.at(pfCand_idx), 0.f, 2.6f, true) : 0;
        }

        { // CellObjectType::PfCand_neutralHadron
            size_t n_pfCand, pfCand_idx;
            getBestObj(CellObjectType::PfCand_neutralHadron, n_pfCand, pfCand_idx);
            const bool valid = n_pfCand != 0;
            out.pfCand_nHad_n_total = static_cast<int>(n_pfCand);
            out.pfCand_nHad_valid = valid;

            out.pfCand_nHad_rel_pt = valid ? GetValueNorm(tau.pfCand_pt.at(pfCand_idx) / tau.tau_pt,
                inner ? 0.3163f : 0.0502f, inner ? 0.2769f : 0.4266f) : 0;
            out.pfCand_nHad_deta = valid ? GetValueLinear(tau.pfCand_eta.at(pfCand_idx) - tau.tau_eta,
                inner ? -0.1f : -0.5f, inner ? 0.1f : 0.5f, false) : 0;
            out.pfCand_nHad_dphi = valid ? GetValueLinear(DeltaPhi(tau.pfCand_phi.at(pfCand_idx), tau.tau_phi),
                inner ? -0.1f : -0.5f, inner ? 0.1f : 0.5f, false) : 0;
            out.pfCand_nHad_tauSignal = valid ? GetValue(tau.pfCand_tauSignal.at(pfCand_idx)) : 0;
            out.pfCand_nHad_tauIso = valid ? GetValue(tau.pfCand_tauIso.at(pfCand_idx)) : 0;
            out.pfCand_nHad_puppiWeight = valid ? GetValue(tau.pfCand_puppiWeight.at(pfCand_idx)) : 0;
            out.pfCand_nHad_puppiWeightNoLep = valid ? GetValue(tau.pfCand_puppiWeightNoLep.at(pfCand_idx)) : 0;
            out.pfCand_nHad_hcalFraction = valid ? GetValue(tau.pfCand_hcalFraction.at(pfCand_idx)) : 0;
        }

        { // CellObjectType::PfCand_gamma
            size_t n_pfCand, pfCand_idx;
            getBestObj(CellObjectType::PfCand_gamma, n_pfCand, pfCand_idx);
            const bool valid = n_pfCand != 0;
            out.pfCand_gamma_n_total = static_cast<int>(n_pfCand);
            out.pfCand_gamma_valid = valid;

            out.pfCand_gamma_rel_pt = valid ? GetValueNorm(tau.pfCand_pt.at(pfCand_idx) / tau.tau_pt,
                inner ? 0.6048f : 0.02576f, inner ? 1.669f : 0.3833f) : 0;
            out.pfCand_gamma_deta = valid ? GetValueLinear(tau.pfCand_eta.at(pfCand_idx) - tau.tau_eta,
                inner ? -0.1f : -0.5f, inner ? 0.1f : 0.5f, false) : 0;
            out.pfCand_gamma_dphi = valid ? GetValueLinear(DeltaPhi(tau.pfCand_phi.at(pfCand_idx), tau.tau_phi),
                inner ? -0.1f : -0.5f, inner ? 0.1f : 0.5f, false) : 0;
            out.pfCand_gamma_tauSignal = valid ? GetValue(tau.pfCand_tauSignal.at(pfCand_idx)) : 0;
            out.pfCand_gamma_tauIso = valid ? GetValue(tau.pfCand_tauIso.at(pfCand_idx)) : 0;
            out.pfCand_gamma_pvAssociationQuality = valid ?
                GetValueLinear(tau.pfCand_pvAssociationQuality.at(pfCand_idx), 0, 7, true) : 0;
            out.pfCand_gamma_fromPV = valid ? GetValueLinear(tau.pfCand_fromPV.at(pfCand_idx), 0, 3, true) : 0;
            out.pfCand_gamma_puppiWeight = valid ? GetValue(tau.pfCand_puppiWeight.at(pfCand_idx)) : 0;
            out.pfCand_gamma_puppiWeightNoLep = valid ? GetValue(tau.pfCand_puppiWeightNoLep.at(pfCand_idx)) : 0;
            out.pfCand_gamma_lostInnerHits = valid ? GetValue(tau.pfCand_lostInnerHits.at(pfCand_idx)) : 0;
            out.pfCand_gamma_numberOfPixelHits = valid ?
                GetValueLinear(tau.pfCand_numberOfPixelHits.at(pfCand_idx), 0, 7, true) : 0;

            out.pfCand_gamma_vertex_dx = valid ?
                GetValueNorm(tau.pfCand_vertex_x.at(pfCand_idx) - tau.pv_x, 0.f, 0.0067f) : 0;
            out.pfCand_gamma_vertex_dy = valid ?
                GetValueNorm(tau.pfCand_vertex_y.at(pfCand_idx) - tau.pv_y, 0.f, 0.0069f) : 0;
            out.pfCand_gamma_vertex_dz = valid ?
                GetValueNorm(tau.pfCand_vertex_z.at(pfCand_idx) - tau.pv_z, 0.f, 0.0578f) : 0;
            out.pfCand_gamma_vertex_dx_tauFL = valid ? GetValueNorm(tau.pfCand_vertex_x.at(pfCand_idx) - tau.pv_x -
                tau.tau_flightLength_x, 0.001f, 0.9565f) : 0;
            out.pfCand_gamma_vertex_dy_tauFL = valid ? GetValueNorm(tau.pfCand_vertex_y.at(pfCand_idx) - tau.pv_y -
                tau.tau_flightLength_y, 0.0008f, 0.9592f) : 0;
            out.pfCand_gamma_vertex_dz_tauFL = valid ? GetValueNorm(tau.pfCand_vertex_z.at(pfCand_idx) - tau.pv_z -
                tau.tau_flightLength_z, 0.0038f, 2.154f) : 0;

            const bool hasTrackDetails = valid && tau.pfCand_hasTrackDetails.at(pfCand_idx) == 1;
            out.pfCand_gamma_hasTrackDetails = hasTrackDetails;
            out.pfCand_gamma_dxy = hasTrackDetails ?
                GetValueNorm(tau.pfCand_dxy.at(pfCand_idx), 0.0004f, 0.882f) : 0;
            out.pfCand_gamma_dxy_sig = hasTrackDetails ? GetValueNorm(std::abs(tau.pfCand_dxy.at(pfCand_idx)) /
                tau.pfCand_dxy_error.at(pfCand_idx), 4.271f, 63.78f) : 0;
            out.pfCand_gamma_dz = hasTrackDetails ? GetValueNorm(tau.pfCand_dz.at(pfCand_idx), 0.0071f, 5.285f) : 0;
            out.pfCand_gamma_dz_sig = hasTrackDetails ? GetValueNorm(std::abs(tau.pfCand_dz.at(pfCand_idx)) /
                tau.pfCand_dz_error.at(pfCand_idx), 162.1f, 622.4f) : 0;
            out.pfCand_gamma_track_chi2_ndof = hasTrackDetails && tau.pfCand_track_ndof.at(pfCand_idx) > 0 ?
                GetValueNorm(tau.pfCand_track_chi2.at(pfCand_idx) / tau.pfCand_track_ndof.at(pfCand_idx),
                4.268f, 15.47f) : 0;
            out.pfCand_gamma_track_ndof = hasTrackDetails && tau.pfCand_track_ndof.at(pfCand_idx) > 0 ?
                GetValueNorm(tau.pfCand_track_ndof.at(pfCand_idx), 12.25f, 4.774f) : 0;
        }

        { // PAT electron
            size_t n_ele, idx;
            getBestObj(CellObjectType::Electron, n_ele, idx);
            const bool valid = n_ele != 0;
            out.ele_n_total = static_cast<int>(n_ele);
            out.ele_valid = valid;

            out.ele_rel_pt = valid ? GetValueNorm(tau.ele_pt.at(idx) / tau.tau_pt,
                inner ? 1.067f : 0.5111f, inner ? 1.521f : 2.765f) : 0;
            out.ele_deta = valid ? GetValueLinear(tau.ele_eta.at(idx) - tau.tau_eta,
                inner ? -0.1f : -0.5f, inner ? 0.1f : 0.5f, false) : 0;
            out.ele_dphi = valid ? GetValueLinear(DeltaPhi(tau.ele_phi.at(idx), tau.tau_phi),
                inner ? -0.1f : -0.5f, inner ? 0.1f : 0.5f, false) : 0;

            const bool cc_valid = valid && tau.ele_cc_ele_energy.at(idx) >= 0;
            out.ele_cc_valid = cc_valid;
            out.ele_cc_ele_rel_energy = cc_valid ? GetValueNorm(tau.ele_cc_ele_energy.at(idx) / tau.ele_pt.at(idx),
                1.729f, 1.644f) : 0;
            out.ele_cc_gamma_rel_energy = cc_valid ? GetValueNorm(tau.ele_cc_gamma_energy.at(idx) /
                tau.ele_cc_ele_energy.at(idx), 0.1439f, 0.3284f) : 0;
            out.ele_cc_n_gamma = cc_valid ? GetValueNorm(tau.ele_cc_n_gamma.at(idx), 1.794f, 2.079f) : 0;
            out.ele_rel_trackMomentumAtVtx = valid ? GetValueNorm(tau.ele_trackMomentumAtVtx.at(idx) /
                tau.ele_pt.at(idx), 1.531f, 1.424f) : 0;
            out.ele_rel_trackMomentumAtCalo = valid ? GetValueNorm(tau.ele_trackMomentumAtCalo.at(idx) /
                tau.ele_pt.at(idx), 1.531f, 1.424f) : 0;
            out.ele_rel_trackMomentumOut = valid ? GetValueNorm(tau.ele_trackMomentumOut.at(idx) /
                tau.ele_pt.at(idx), 0.7735f, 0.935f) : 0;
            out.ele_rel_trackMomentumAtEleClus = valid ? GetValueNorm(tau.ele_trackMomentumAtEleClus.at(idx) /
                tau.ele_pt.at(idx), 0.7735f, 0.935f) : 0;
            out.ele_rel_trackMomentumAtVtxWithConstraint = valid ?
                GetValueNorm(tau.ele_trackMomentumAtVtxWithConstraint.at(idx) / tau.ele_pt.at(idx), 1.625f, 1.581f) : 0;
            out.ele_rel_ecalEnergy = valid ? GetValueNorm(tau.ele_ecalEnergy.at(idx) /
                tau.ele_pt.at(idx), 1.993f, 1.308f) : 0;
            out.ele_ecalEnergy_sig = valid ? GetValueNorm(tau.ele_ecalEnergy.at(idx) /
                tau.ele_ecalEnergy_error.at(idx), 70.25f, 58.16f) : 0;
            out.ele_eSuperClusterOverP = valid ? GetValueNorm(tau.ele_eSuperClusterOverP.at(idx), 2.432f, 15.13f) : 0;
            out.ele_eSeedClusterOverP = valid ? GetValueNorm(tau.ele_eSeedClusterOverP.at(idx), 2.034f, 13.96f) : 0;
            out.ele_eSeedClusterOverPout = valid ? GetValueNorm(tau.ele_eSeedClusterOverPout.at(idx), 6.64f, 36.8f) : 0;
            out.ele_eEleClusterOverPout = valid ? GetValueNorm(tau.ele_eEleClusterOverPout.at(idx), 4.183f, 20.63f) : 0;
            out.ele_deltaEtaSuperClusterTrackAtVtx = valid ?
                GetValueNorm(tau.ele_deltaEtaSuperClusterTrackAtVtx.at(idx), 0.f, 0.0363f) : 0;
            out.ele_deltaEtaSeedClusterTrackAtCalo = valid ?
                GetValueNorm(tau.ele_deltaEtaSeedClusterTrackAtCalo.at(idx), -0.0001f, 0.0512f) : 0;
            out.ele_deltaEtaEleClusterTrackAtCalo = valid ? GetValueNorm(tau.ele_deltaEtaEleClusterTrackAtCalo.at(idx),
                -0.0001f, 0.0541f) : 0;
            out.ele_deltaPhiEleClusterTrackAtCalo = valid ? GetValueNorm(tau.ele_deltaPhiEleClusterTrackAtCalo.at(idx),
                0.0002f, 0.0553f) : 0;
            out.ele_deltaPhiSuperClusterTrackAtVtx = valid ?
                GetValueNorm(tau.ele_deltaPhiSuperClusterTrackAtVtx.at(idx), 0.0001f, 0.0523f) : 0;
            out.ele_deltaPhiSeedClusterTrackAtCalo = valid ?
                GetValueNorm(tau.ele_deltaPhiSeedClusterTrackAtCalo.at(idx), 0.0004f, 0.0777f) : 0;
            out.ele_mvaInput_earlyBrem = valid ? GetValue(tau.ele_mvaInput_earlyBrem.at(idx)) : 0;
            out.ele_mvaInput_lateBrem = valid ? GetValue(tau.ele_mvaInput_lateBrem.at(idx)) : 0;
            out.ele_mvaInput_sigmaEtaEta = valid ? GetValueNorm(tau.ele_mvaInput_sigmaEtaEta.at(idx),
                0.0008f, 0.0052f) : 0;
            out.ele_mvaInput_hadEnergy = valid ? GetValueNorm(tau.ele_mvaInput_hadEnergy.at(idx), 14.04f, 69.48f) : 0;
            out.ele_mvaInput_deltaEta = valid ? GetValueNorm(tau.ele_mvaInput_deltaEta.at(idx), 0.0099f, 0.0851f) : 0;
            out.ele_gsfTrack_normalizedChi2 = valid ? GetValueNorm(tau.ele_gsfTrack_normalizedChi2.at(idx),
                3.049f, 10.39f) : 0;
            out.ele_gsfTrack_numberOfValidHits = valid ? GetValueNorm(tau.ele_gsfTrack_numberOfValidHits.at(idx),
                16.52f, 2.806f) : 0;
            out.ele_rel_gsfTrack_pt = valid ? GetValueNorm(tau.ele_gsfTrack_pt.at(idx) / tau.ele_pt.at(idx),
                1.355f, 16.81f) : 0;
            out.ele_gsfTrack_pt_sig = valid ? GetValueNorm(tau.ele_gsfTrack_pt.at(idx) /
                tau.ele_gsfTrack_pt_error.at(idx), 5.046f, 3.119f) : 0;
            const bool has_closestCtfTrack = valid && tau.ele_closestCtfTrack_normalizedChi2.at(idx) >= 0;
            out.ele_has_closestCtfTrack = has_closestCtfTrack;
            out.ele_closestCtfTrack_normalizedChi2 = has_closestCtfTrack ?
                GetValueNorm(tau.ele_closestCtfTrack_normalizedChi2.at(idx), 2.411f, 6.98f) : 0;
            out.ele_closestCtfTrack_numberOfValidHits = has_closestCtfTrack ?
                GetValueNorm(tau.ele_closestCtfTrack_numberOfValidHits.at(idx), 15.16f, 5.26f) : 0;
        }

        { // PAT muon
            size_t n_muon, idx;
            getBestObj(CellObjectType::Muon, n_muon, idx);
            const bool valid = n_muon != 0;
            out.muon_n_total = static_cast<int>(n_muon);
            out.muon_valid = valid;

            out.muon_rel_pt = valid ? GetValueNorm(tau.muon_pt.at(idx) / tau.tau_pt,
                inner ? 0.7966f : 0.2678f, inner ? 3.402f : 3.592f) : 0;
            out.muon_deta = valid ? GetValueLinear(tau.muon_eta.at(idx) - tau.tau_eta,
                inner ? -0.1f : -0.5f, inner ? 0.1f : 0.5f, false) : 0;
            out.muon_dphi = valid ? GetValueLinear(DeltaPhi(tau.muon_phi.at(idx), tau.tau_phi),
                inner ? -0.1f : -0.5f, inner ? 0.1f : 0.5f, false) : 0;

            out.muon_dxy = valid ? GetValueNorm(tau.muon_dxy.at(idx), 0.0019f, 1.039f) : 0;
            out.muon_dxy_sig = valid ? GetValueNorm(std::abs(tau.muon_dxy.at(idx)) / tau.muon_dxy_error.at(idx),
                8.98f, 71.17f) : 0;
            const bool normalizedChi2_valid = valid && tau.muon_normalizedChi2.at(idx) >= 0;
            out.muon_normalizedChi2_valid = normalizedChi2_valid;
            out.muon_normalizedChi2 = normalizedChi2_valid ? GetValueNorm(tau.muon_normalizedChi2.at(idx),
                21.52f, 265.8f) : 0;
            out.muon_numberOfValidHits = normalizedChi2_valid ? GetValueNorm(tau.muon_numberOfValidHits.at(idx),
                21.84f, 10.59f) : 0;
            out.muon_segmentCompatibility = valid ? GetValue(tau.muon_segmentCompatibility.at(idx)) : 0;
            out.muon_caloCompatibility = valid ? GetValue(tau.muon_caloCompatibility.at(idx)) : 0;
            const bool pfEcalEnergy_valid = valid && tau.muon_pfEcalEnergy.at(idx) >= 0;
            out.muon_pfEcalEnergy_valid = pfEcalEnergy_valid;
            out.muon_rel_pfEcalEnergy = pfEcalEnergy_valid ? GetValueNorm(tau.muon_pfEcalEnergy.at(idx) /
                tau.muon_pt.at(idx), 0.2273f, 0.4865f) : 0;
            out.muon_n_matches_DT_1 = valid ? GetValueLinear(tau.muon_n_matches_DT_1.at(idx), 0, 2, true) : 0;
            out.muon_n_matches_DT_2 = valid ? GetValueLinear(tau.muon_n_matches_DT_2.at(idx), 0, 2, true) : 0;
            out.muon_n_matches_DT_3 = valid ? GetValueLinear(tau.muon_n_matches_DT_3.at(idx), 0, 2, true) : 0;
            out.muon_n_matches_DT_4 = valid ? GetValueLinear(tau.muon_n_matches_DT_4.at(idx), 0, 2, true) : 0;
            out.muon_n_matches_CSC_1 = valid ? GetValueLinear(tau.muon_n_matches_CSC_1.at(idx), 0, 6, true) : 0;
            out.muon_n_matches_CSC_2 = valid ? GetValueLinear(tau.muon_n_matches_CSC_2.at(idx), 0, 2, true) : 0;
            out.muon_n_matches_CSC_3 = valid ? GetValueLinear(tau.muon_n_matches_CSC_3.at(idx), 0, 2, true) : 0;
            out.muon_n_matches_CSC_4 = valid ? GetValueLinear(tau.muon_n_matches_CSC_4.at(idx), 0, 2, true) : 0;
            out.muon_n_matches_RPC_1 = valid ? GetValueLinear(tau.muon_n_matches_RPC_1.at(idx), 0, 7, true) : 0;
            out.muon_n_matches_RPC_2 = valid ? GetValueLinear(tau.muon_n_matches_RPC_2.at(idx), 0, 6, true) : 0;
            out.muon_n_matches_RPC_3 = valid ? GetValueLinear(tau.muon_n_matches_RPC_3.at(idx), 0, 4, true) : 0;
            out.muon_n_matches_RPC_4 = valid ? GetValueLinear(tau.muon_n_matches_RPC_4.at(idx), 0, 4, true) : 0;
            out.muon_n_hits_DT_1 = valid ? GetValueLinear(tau.muon_n_hits_DT_1.at(idx), 0, 12, true) : 0;
            out.muon_n_hits_DT_2 = valid ? GetValueLinear(tau.muon_n_hits_DT_2.at(idx), 0, 12, true) : 0;
            out.muon_n_hits_DT_3 = valid ? GetValueLinear(tau.muon_n_hits_DT_3.at(idx), 0, 12, true) : 0;
            out.muon_n_hits_DT_4 = valid ? GetValueLinear(tau.muon_n_hits_DT_4.at(idx), 0, 8, true) : 0;
            out.muon_n_hits_CSC_1 = valid ? GetValueLinear(tau.muon_n_hits_CSC_1.at(idx), 0, 24, true) : 0;
            out.muon_n_hits_CSC_2 = valid ? GetValueLinear(tau.muon_n_hits_CSC_2.at(idx), 0, 12, true) : 0;
            out.muon_n_hits_CSC_3 = valid ? GetValueLinear(tau.muon_n_hits_CSC_3.at(idx), 0, 12, true) : 0;
            out.muon_n_hits_CSC_4 = valid ? GetValueLinear(tau.muon_n_hits_CSC_4.at(idx), 0, 12, true) : 0;
            out.muon_n_hits_RPC_1 = valid ? GetValueLinear(tau.muon_n_hits_RPC_1.at(idx), 0, 4, true) : 0;
            out.muon_n_hits_RPC_2 = valid ? GetValueLinear(tau.muon_n_hits_RPC_2.at(idx), 0, 4, true) : 0;
            out.muon_n_hits_RPC_3 = valid ? GetValueLinear(tau.muon_n_hits_RPC_3.at(idx), 0, 2, true) : 0;
            out.muon_n_hits_RPC_4 = valid ? GetValueLinear(tau.muon_n_hits_RPC_4.at(idx), 0, 2, true) : 0;
        }

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
                const double deta = eta - tau_eta, dphi = DeltaPhi(phi, tau_phi);
                const double dR = std::hypot(deta, dphi);
                const bool inside_signal_cone = dR < getInnerSignalConeRadius(tau_pt);
                const bool inside_iso_cone = dR < iso_cone;
                if(inner && !inside_signal_cone) continue;
                // if(!inner && (inside_signal_cone || !inside_iso_cone)) continue;
                if(!inner && !inside_iso_cone) continue;
                CellIndex cellIndex;
                if(grid.TryGetCellIndex(deta, dphi, cellIndex))
                    grid.at(cellIndex)[type].insert(n);
            }
        };

        fillGrid(CellObjectType::PfCand_electron, tau.pfCand_eta, tau.pfCand_phi, tau.pfCand_pdgId);
        fillGrid(CellObjectType::PfCand_muon, tau.pfCand_eta, tau.pfCand_phi, tau.pfCand_pdgId);
        fillGrid(CellObjectType::PfCand_chargedHadron, tau.pfCand_eta, tau.pfCand_phi, tau.pfCand_pdgId);
        fillGrid(CellObjectType::PfCand_neutralHadron, tau.pfCand_eta, tau.pfCand_phi, tau.pfCand_pdgId);
        fillGrid(CellObjectType::PfCand_gamma, tau.pfCand_eta, tau.pfCand_phi, tau.pfCand_pdgId);
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
