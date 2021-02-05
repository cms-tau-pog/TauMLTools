#include <boost/preprocessor/variadic.hpp>
#include <boost/math/constants/constants.hpp>

#include "TauMLTools/Core/interface/AnalysisMath.h"
#include "TauMLTools/Core/interface/RootExt.h"
#include "TauMLTools/Analysis/interface/TauTuple.h"

#include "TROOT.h"
#include "TLorentzVector.h"
#include "DataLoader_setup.h"

using namespace ROOT::Math;

std::shared_ptr<TFile> OpenRootFile(const std::string& file_name){
    std::shared_ptr<TFile> file(TFile::Open(file_name.c_str(), "READ"));
    if(!file || file->IsZombie())
        throw std::runtime_error("File not opened.");
    return file;
}

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
            throw std::invalid_argument("Invalid number of eta cells.");
        if(nCellsPhi % 2 != 1 || nCellsPhi < 1)
            throw std::invalid_argument("Invalid number of phi cells.");
        if(cellSizeEta <= 0 || cellSizePhi <= 0)
            throw std::invalid_argument("Invalid cell size.");
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
            const double absIndex = std::floor(absX / size + 0.5);
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

    size_t GetFlatIndex(const CellIndex& cellIndex) const
    {
        if(std::abs(cellIndex.eta) > MaxEtaIndex() || std::abs(cellIndex.phi) > MaxPhiIndex())
            throw std::runtime_error("Cell index is out of range");
        const unsigned shiftedEta = static_cast<unsigned>(cellIndex.eta + MaxEtaIndex());
        const unsigned shiftedPhi = static_cast<unsigned>(cellIndex.phi + MaxPhiIndex());
        return shiftedEta * nCellsPhi + shiftedPhi;
    }

    size_t GetnTotal() const { return nTotal; }

private:
    const unsigned nCellsEta, nCellsPhi, nTotal;
    const double cellSizeEta, cellSizePhi;
    std::vector<Cell> cells;
};

struct Data { // (n taus, taus features, n_inner_cells, n_outer_cells, n pfelectron features, n pfmuon features)
    Data(size_t n_tau, size_t tau_fn, size_t n_inner_cells,
         size_t n_outer_cells, size_t pfelectron_fn, size_t pfmuon_fn,
         size_t pfchargedhad_fn, size_t pfneutralhad_fn, size_t pfgamma_fn,
         size_t electron_fn, size_t muon_fn, size_t tau_labels) :
         x_tau(n_tau * tau_fn, 0), weight(n_tau, 1), y_onehot(n_tau * tau_labels, 0)
         {
           // y_onehot = std::vector<std::vector<float>>( n_tau, std::vector<float> (tau_labels, 0) );

          // pf electron
           x_grid[CellObjectType::PfCand_electron][0] = std::vector<float>(n_tau * n_outer_cells * n_outer_cells * pfelectron_fn,0);
           x_grid[CellObjectType::PfCand_electron][1] = std::vector<float>(n_tau * n_inner_cells * n_inner_cells * pfelectron_fn,0);

           // pf muons
           x_grid[CellObjectType::PfCand_muon][0] = std::vector<float>(n_tau * n_outer_cells * n_outer_cells * pfmuon_fn,0);
           x_grid[CellObjectType::PfCand_muon][1] = std::vector<float>(n_tau * n_inner_cells * n_inner_cells * pfmuon_fn,0);

           // pf charged hadrons
           x_grid[CellObjectType::PfCand_chargedHadron][0] = std::vector<float>(n_tau * n_outer_cells * n_outer_cells * pfchargedhad_fn,0);
           x_grid[CellObjectType::PfCand_chargedHadron][1] = std::vector<float>(n_tau * n_inner_cells * n_inner_cells * pfchargedhad_fn,0);

           // pf neutral hadrons
           x_grid[CellObjectType::PfCand_neutralHadron][0] = std::vector<float>(n_tau * n_outer_cells * n_outer_cells * pfneutralhad_fn,0);
           x_grid[CellObjectType::PfCand_neutralHadron][1] = std::vector<float>(n_tau * n_inner_cells * n_inner_cells * pfneutralhad_fn,0);

           // pf gamma
           x_grid[CellObjectType::PfCand_gamma][0] = std::vector<float>(n_tau * n_outer_cells * n_outer_cells * pfgamma_fn,0);
           x_grid[CellObjectType::PfCand_gamma][1] = std::vector<float>(n_tau * n_inner_cells * n_inner_cells * pfgamma_fn,0);

           // electrons
           x_grid[CellObjectType::Electron][0] = std::vector<float>(n_tau * n_outer_cells * n_outer_cells * electron_fn,0);
           x_grid[CellObjectType::Electron][1] = std::vector<float>(n_tau * n_inner_cells * n_inner_cells * electron_fn,0);

           // muons
           x_grid[CellObjectType::Muon][0] = std::vector<float>(n_tau * n_outer_cells * n_outer_cells * muon_fn,0);
           x_grid[CellObjectType::Muon][1] = std::vector<float>(n_tau * n_inner_cells * n_inner_cells * muon_fn,0);
         }

    std::vector<float> x_tau;
    std::map<CellObjectType, std::map<bool, std::vector<float>>> x_grid; // [enum class CellObjectType][ 0 - outer, 1 - inner]
    std::vector<float> weight;
    std::vector<int> y_onehot;
};


class DataLoader {

public:
    using Tau = tau_tuple::Tau;
    using TauTuple = tau_tuple::TauTuple;

    DataLoader() :
        n_tau(setup::n_tau), current_entry(setup::start_dataset), start_dataset(setup::start_dataset), end_dataset(setup::end_dataset),
        n_inner_cells(setup::n_inner_cells), inner_cell_size(setup::inner_cell_size), n_outer_cells(setup::n_outer_cells),
        outer_cell_size(setup::outer_cell_size), n_threads(setup::n_threads), parity(setup::parity),
        n_fe_tau(setup::n_fe_tau), n_pf_el(setup::n_pf_el) ,n_pf_mu(setup::n_pf_mu),
        n_pf_chHad(setup::n_pf_chHad), n_pf_nHad(setup::n_pf_nHad) ,n_pf_gamma(setup::n_pf_gamma),
        n_electrons(setup::n_ele), n_muons(setup::n_muon), n_labels(setup::tau_types),
        // trainingWeightFactor(tauTuple.GetEntries() / training_weight_factor,
        innerCellGridRef(n_inner_cells, n_inner_cells, inner_cell_size, inner_cell_size),
        outerCellGridRef(n_outer_cells, n_outer_cells, outer_cell_size, outer_cell_size)
    {
      if(n_threads > 1) ROOT::EnableImplicitMT(n_threads);
    }

    void Initialize(std::string file_name)
    {
        if(tauTuple)
            throw std::runtime_error("DataLoader is already initialized.");
        file = OpenRootFile(file_name);
        tauTuple = std::make_shared<tau_tuple::TauTuple>(file.get(), true);
    }

    bool HasNext() {
        return (current_entry + n_tau) < end_dataset;
    }

    std::shared_ptr<Data> LoadNext(){

        if(!tauTuple)
            throw std::runtime_error("DataLoader is not initialized.");

        auto data = std::make_shared<Data>(n_tau, n_fe_tau,
                                           n_inner_cells, n_outer_cells,
                                           n_pf_el,
                                           n_pf_mu,
                                           n_pf_chHad,
                                           n_pf_nHad,
                                           n_pf_gamma,
                                           n_electrons,
                                           n_muons,
                                           n_labels
                                          );

        const Long64_t end_entry = current_entry + n_tau;
        size_t n_processed = 0, n_total = static_cast<size_t>(end_entry - start_dataset);
        for(Long64_t tau_i = 0; tau_i < n_tau; ++tau_i, ++current_entry) {
            tauTuple->GetEntry(current_entry);
            const auto& tau = tauTuple->data();
            if(parity == -1 || tau.evt % 2 == static_cast<unsigned>(parity)) {
                FillTauBranches(tau, tau_i, data);
                FillCellGrid(tau, tau_i, innerCellGridRef, data, true);
                FillCellGrid(tau, tau_i, outerCellGridRef, data, false);
            }
        }

        std::cout << "One batch is returned" << std::endl;
        return data;
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


      void FillTauBranches(const Tau& tau, Long64_t tau_i, std::shared_ptr<Data>& data)
      {
        Long64_t start_array_index = tau_i * n_fe_tau;

        // Filling Tau Branche
        auto get_tau_branch = [&](TauFlat_f _fe) -> float& {
            size_t _fe_ind = static_cast<size_t>(_fe);
            size_t index = start_array_index + _fe_ind;
            return data->x_tau.at(index);
        };

        // filling labels one_hot vector
        Int_t tau_type = tau.tauType;
        if(tau_type<=3) data->y_onehot[ tau_i * n_labels + tau_type ] = 1;
        else if(tau_type>=6) data->y_onehot[ tau_i * n_labels + tau_type - 2 ] = 1;

        get_tau_branch(TauFlat_f::tau_pt) = tau.tau_pt;
        get_tau_branch(TauFlat_f::tau_eta) = tau.tau_eta;
        get_tau_branch(TauFlat_f::tau_phi) = tau.tau_phi;
        get_tau_branch(TauFlat_f::tau_mass) = tau.tau_mass;

        const analysis::LorentzVectorM tau_p4(tau.tau_pt, tau.tau_eta, tau.tau_phi, tau.tau_mass);
        get_tau_branch(TauFlat_f::tau_E_over_pt) = tau_p4.energy() / tau.tau_pt;
        get_tau_branch(TauFlat_f::tau_charge) = tau.tau_charge;
        get_tau_branch(TauFlat_f::tau_n_charged_prongs) = tau.tau_decayMode / 5;
        get_tau_branch(TauFlat_f::tau_n_neutral_prongs) = tau.tau_decayMode % 5;
        get_tau_branch(TauFlat_f::chargedIsoPtSum) = tau.chargedIsoPtSum;
        get_tau_branch(TauFlat_f::chargedIsoPtSumdR03_over_dR05) = tau.chargedIsoPtSumdR03 / tau.chargedIsoPtSum;
        get_tau_branch(TauFlat_f::footprintCorrection) = tau.footprintCorrection;
        get_tau_branch(TauFlat_f::neutralIsoPtSum) = tau.neutralIsoPtSum;
        get_tau_branch(TauFlat_f::neutralIsoPtSumWeight_over_neutralIsoPtSum) = tau.neutralIsoPtSumWeight / tau.neutralIsoPtSum;
        get_tau_branch(TauFlat_f::neutralIsoPtSumWeightdR03_over_neutralIsoPtSum) =tau.neutralIsoPtSumWeightdR03 / tau.neutralIsoPtSum;
        get_tau_branch(TauFlat_f::neutralIsoPtSumdR03_over_dR05) = tau.neutralIsoPtSumdR03 / tau.neutralIsoPtSum;
        get_tau_branch(TauFlat_f::photonPtSumOutsideSignalCone) = tau.photonPtSumOutsideSignalCone;
        get_tau_branch(TauFlat_f::puCorrPtSum) = tau.puCorrPtSum;

        const bool tau_dxy_valid = std::isnormal(tau.tau_dxy) && tau.tau_dxy > - 10
                                   && std::isnormal(tau.tau_dxy_error) && tau.tau_dxy_error > 0;
        get_tau_branch(TauFlat_f::tau_dxy_valid) = tau_dxy_valid;
        get_tau_branch(TauFlat_f::tau_dxy) =  tau_dxy_valid ? tau.tau_dxy : 0.f;
        get_tau_branch(TauFlat_f::tau_dxy_sig) =  tau_dxy_valid ? std::abs(tau.tau_dxy)/tau.tau_dxy_error : 0.f;

        const bool tau_ip3d_valid = std::isnormal(tau.tau_ip3d) && tau.tau_ip3d > - 10
                                    && std::isnormal(tau.tau_ip3d_error) && tau.tau_ip3d_error > 0;
        get_tau_branch(TauFlat_f::tau_ip3d_valid) =  tau_ip3d_valid;
        get_tau_branch(TauFlat_f::tau_ip3d) = tau_ip3d_valid ? tau.tau_ip3d : 0.f;
        get_tau_branch(TauFlat_f::tau_ip3d_sig) =  tau_ip3d_valid
                         ? std::abs(tau.tau_ip3d) / tau.tau_ip3d_error : 0.f;
        get_tau_branch(TauFlat_f::tau_dz) = tau.tau_dz;
        const bool tau_dz_sig_valid = std::isnormal(tau.tau_dz) && std::isnormal(tau.tau_dz_error)
                                      && tau.tau_dz_error > 0;
        get_tau_branch(TauFlat_f::tau_dz_sig_valid) = tau_dz_sig_valid;
        get_tau_branch(TauFlat_f::tau_dz_sig) =  tau_dz_sig_valid ? std::abs(tau.tau_dz) / tau.tau_dz_error : 0.f;

        get_tau_branch(TauFlat_f::tau_flightLength_x) = tau.tau_flightLength_x;
        get_tau_branch(TauFlat_f::tau_flightLength_y) = tau.tau_flightLength_y;
        get_tau_branch(TauFlat_f::tau_flightLength_z) = tau.tau_flightLength_z;
        get_tau_branch(TauFlat_f::tau_flightLength_sig) = tau.tau_flightLength_sig;

        get_tau_branch(TauFlat_f::tau_pt_weighted_deta_strip) = tau.tau_pt_weighted_deta_strip ;
        get_tau_branch(TauFlat_f::tau_pt_weighted_dphi_strip) = tau.tau_pt_weighted_dphi_strip ;
        get_tau_branch(TauFlat_f::tau_pt_weighted_dr_signal) = tau.tau_pt_weighted_dr_signal ;
        get_tau_branch(TauFlat_f::tau_pt_weighted_dr_iso) = tau.tau_pt_weighted_dr_iso;

        get_tau_branch(TauFlat_f::tau_leadingTrackNormChi2) = tau.tau_leadingTrackNormChi2;
        const bool tau_e_ratio_valid = std::isnormal(tau.tau_e_ratio) && tau.tau_e_ratio > 0.f;
        get_tau_branch(TauFlat_f::tau_e_ratio_valid) = tau_e_ratio_valid;
        get_tau_branch(TauFlat_f::tau_e_ratio) = tau_e_ratio_valid ? tau.tau_e_ratio : 0.f;

        const bool tau_gj_angle_diff_valid = (std::isnormal(tau.tau_gj_angle_diff) || tau.tau_gj_angle_diff == 0)
            && tau.tau_gj_angle_diff >= 0;
        get_tau_branch(TauFlat_f::tau_gj_angle_diff_valid) = tau_gj_angle_diff_valid;
        get_tau_branch(TauFlat_f::tau_gj_angle_diff) = tau_gj_angle_diff_valid ? tau.tau_gj_angle_diff : 0;
        get_tau_branch(TauFlat_f::tau_n_photons) = tau.tau_n_photons;
        get_tau_branch(TauFlat_f::tau_emFraction) = tau.tau_emFraction;
        get_tau_branch(TauFlat_f::tau_inside_ecal_crack) = tau.tau_inside_ecal_crack;
        get_tau_branch(TauFlat_f::leadChargedCand_etaAtEcalEntrance_minus_tau_eta) = tau.leadChargedCand_etaAtEcalEntrance;
      }

      void FillCellGrid(const Tau& tau, Long64_t tau_i,  const CellGrid& cellGridRef, std::shared_ptr<Data>& data, bool inner)
      {
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
                          throw std::runtime_error("Duplicated cell index in FillCellGrid.");
                      processed_cells.insert(cellIndex);
                      if(!cellGrid.IsEmpty(cellIndex))
                          FillCellBranches(tau, tau_i, cellGridRef, cellIndex, cellGrid.at(cellIndex), data, inner);
                  }
              }
          }
          if(processed_cells.size() != static_cast<size_t>( (2 * max_eta_index + 1) * (2 * max_phi_index + 1) ))
              throw std::runtime_error("Not all cell indices are processed in FillCellGrid.");
      }

      void FillCellBranches(const Tau& tau, Long64_t tau_i,  const CellGrid& cellGridRef, const CellIndex& cellIndex,
                            Cell& cell, std::shared_ptr<Data>& data, bool inner)
      {


        auto getIndex = [&](size_t n_features, size_t feature_index) {
          size_t index_begin = tau_i * cellGridRef.GetnTotal() * n_features;
          return index_begin + cellGridRef.GetFlatIndex(cellIndex) * n_features + feature_index;
        };


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
            typedef PfCand_electron_f Br;
            auto get_PFelectron = [&](PfCand_electron_f _fe_n) -> float& {
              size_t flat_index = getIndex(n_pf_el, static_cast<size_t>(_fe_n));
              return data->x_grid.at(CellObjectType::PfCand_electron).at(inner).at(flat_index);
            };

            size_t n_pfCand, pfCand_idx;
            getBestObj(CellObjectType::PfCand_electron, n_pfCand, pfCand_idx);

            const bool valid = n_pfCand != 0;
            // get_PFelectron(Br::pfCand_ele_n_total) = static_cast<int>(n_pfCand);
            get_PFelectron(Br::pfCand_ele_valid) = valid;
            get_PFelectron(Br::pfCand_ele_rel_pt) = valid ? tau.pfCand_pt.at(pfCand_idx) / tau.tau_pt : 0;
            get_PFelectron(Br::pfCand_ele_deta) = valid ? tau.pfCand_eta.at(pfCand_idx) - tau.tau_eta : 0;
            get_PFelectron(Br::pfCand_ele_dphi) = valid ? DeltaPhi(tau.pfCand_phi.at(pfCand_idx), tau.tau_phi) : 0;
            // get_PFelectron(Br::pfCand_ele_tauSignal) = valid ? tau.pfCand_tauSignal.at(pfCand_idx) : 0;
            // get_PFelectron(Br::pfCand_ele_tauIso) = valid ? tau.pfCand_tauIso.at(pfCand_idx) : 0;
            get_PFelectron(Br::pfCand_ele_pvAssociationQuality) = valid ? tau.pfCand_pvAssociationQuality.at(pfCand_idx) : 0;
            get_PFelectron(Br::pfCand_ele_puppiWeight) = valid ? tau.pfCand_puppiWeight.at(pfCand_idx) : 0;
            get_PFelectron(Br::pfCand_ele_charge) = valid ? tau.pfCand_charge.at(pfCand_idx) : 0;
            get_PFelectron(Br::pfCand_ele_lostInnerHits) = valid ? tau.pfCand_lostInnerHits.at(pfCand_idx) : 0;
            get_PFelectron(Br::pfCand_ele_numberOfPixelHits) = valid ? tau.pfCand_numberOfPixelHits.at(pfCand_idx) : 0;

            get_PFelectron(Br::pfCand_ele_vertex_dx) = valid ? tau.pfCand_vertex_x.at(pfCand_idx) - tau.pv_x : 0;
            get_PFelectron(Br::pfCand_ele_vertex_dy) = valid ? tau.pfCand_vertex_y.at(pfCand_idx) - tau.pv_y : 0;
            get_PFelectron(Br::pfCand_ele_vertex_dz) = valid ? tau.pfCand_vertex_z.at(pfCand_idx) - tau.pv_z : 0;
            get_PFelectron(Br::pfCand_ele_vertex_dx_tauFL) = valid ? tau.pfCand_vertex_x.at(pfCand_idx) - tau.pv_x - tau.tau_flightLength_x : 0;
            get_PFelectron(Br::pfCand_ele_vertex_dy_tauFL) = valid ? tau.pfCand_vertex_y.at(pfCand_idx) - tau.pv_y - tau.tau_flightLength_y : 0;
            get_PFelectron(Br::pfCand_ele_vertex_dz_tauFL) = valid ? tau.pfCand_vertex_z.at(pfCand_idx) - tau.pv_z - tau.tau_flightLength_z : 0;

            const bool hasTrackDetails = valid && tau.pfCand_hasTrackDetails.at(pfCand_idx) == 1;
            get_PFelectron(Br::pfCand_ele_hasTrackDetails) = hasTrackDetails;
            get_PFelectron(Br::pfCand_ele_dxy) = hasTrackDetails ? tau.pfCand_dxy.at(pfCand_idx) : 0;
            get_PFelectron(Br::pfCand_ele_dxy_sig) = hasTrackDetails ? std::abs(tau.pfCand_dxy.at(pfCand_idx)) / tau.pfCand_dxy_error.at(pfCand_idx) : 0;
            get_PFelectron(Br::pfCand_ele_dz) = hasTrackDetails ? tau.pfCand_dz.at(pfCand_idx) : 0;
            get_PFelectron(Br::pfCand_ele_dz_sig) = hasTrackDetails ? std::abs(tau.pfCand_dz.at(pfCand_idx)) / tau.pfCand_dz_error.at(pfCand_idx) : 0;
            get_PFelectron(Br::pfCand_ele_track_chi2_ndof) = hasTrackDetails && tau.pfCand_track_ndof.at(pfCand_idx) > 0 ?
                tau.pfCand_track_chi2.at(pfCand_idx) / tau.pfCand_track_ndof.at(pfCand_idx) : 0;
            get_PFelectron(Br::pfCand_ele_track_ndof) = hasTrackDetails && tau.pfCand_track_ndof.at(pfCand_idx) > 0 ?
                tau.pfCand_track_ndof.at(pfCand_idx) : 0;
        }

        { // CellObjectType::PfCand_muon
            typedef PfCand_muon_f Br;
            auto get_PFmuon = [&](PfCand_muon_f _fe_n) -> float& {
              size_t flat_index = getIndex(n_pf_mu, static_cast<size_t>(_fe_n));
              return data->x_grid.at(CellObjectType::PfCand_muon).at(inner).at(flat_index);
            };

            size_t n_pfCand, pfCand_idx;
            getBestObj(CellObjectType::PfCand_muon, n_pfCand, pfCand_idx);

            const bool valid = n_pfCand != 0;
            get_PFmuon(Br::pfCand_muon_valid) = valid;
            get_PFmuon(Br::pfCand_muon_rel_pt) = valid ? tau.pfCand_pt.at(pfCand_idx) / tau.tau_pt : 0;
            get_PFmuon(Br::pfCand_muon_deta) = valid ? tau.pfCand_eta.at(pfCand_idx) - tau.tau_eta : 0;
            get_PFmuon(Br::pfCand_muon_dphi) = valid ? DeltaPhi(tau.pfCand_phi.at(pfCand_idx), tau.tau_phi) : 0;
            get_PFmuon(Br::pfCand_muon_pvAssociationQuality) = valid ? tau.pfCand_pvAssociationQuality.at(pfCand_idx) : 0;
            get_PFmuon(Br::pfCand_muon_fromPV) = valid ? tau.pfCand_fromPV.at(pfCand_idx) : 0;
            get_PFmuon(Br::pfCand_muon_puppiWeight) = valid ? tau.pfCand_puppiWeight.at(pfCand_idx) : 0;
            get_PFmuon(Br::pfCand_muon_charge) = valid ? tau.pfCand_charge.at(pfCand_idx) : 0;
            get_PFmuon(Br::pfCand_muon_lostInnerHits) = valid ? tau.pfCand_lostInnerHits.at(pfCand_idx) : 0;
            get_PFmuon(Br::pfCand_muon_numberOfPixelHits) = valid ? tau.pfCand_numberOfPixelHits.at(pfCand_idx) : 0;

            get_PFmuon(Br::pfCand_muon_vertex_dx) = valid ? tau.pfCand_vertex_x.at(pfCand_idx) - tau.pv_x : 0;
            get_PFmuon(Br::pfCand_muon_vertex_dy) = valid ? tau.pfCand_vertex_y.at(pfCand_idx) - tau.pv_y : 0;
            get_PFmuon(Br::pfCand_muon_vertex_dz) = valid ? tau.pfCand_vertex_z.at(pfCand_idx) - tau.pv_z : 0;
            get_PFmuon(Br::pfCand_muon_vertex_dx_tauFL) = valid ? tau.pfCand_vertex_x.at(pfCand_idx) - tau.pv_x - tau.tau_flightLength_x : 0;
            get_PFmuon(Br::pfCand_muon_vertex_dy_tauFL) = valid ? tau.pfCand_vertex_y.at(pfCand_idx) - tau.pv_y - tau.tau_flightLength_y : 0;
            get_PFmuon(Br::pfCand_muon_vertex_dz_tauFL) = valid ? tau.pfCand_vertex_z.at(pfCand_idx) - tau.pv_z - tau.tau_flightLength_z : 0;

            const bool hasTrackDetails = valid && tau.pfCand_hasTrackDetails.at(pfCand_idx) == 1;
            get_PFmuon(Br::pfCand_muon_hasTrackDetails) = hasTrackDetails;
            get_PFmuon(Br::pfCand_muon_dxy) = hasTrackDetails ? tau.pfCand_dxy.at(pfCand_idx) : 0;
            get_PFmuon(Br::pfCand_muon_dxy_sig) = hasTrackDetails ? std::abs(tau.pfCand_dxy.at(pfCand_idx)) / tau.pfCand_dxy_error.at(pfCand_idx) : 0;
            get_PFmuon(Br::pfCand_muon_dz) = hasTrackDetails ? tau.pfCand_dz.at(pfCand_idx) : 0;
            get_PFmuon(Br::pfCand_muon_dz_sig) = hasTrackDetails ? std::abs(tau.pfCand_dz.at(pfCand_idx)) / tau.pfCand_dz_error.at(pfCand_idx) : 0;
            get_PFmuon(Br::pfCand_muon_track_chi2_ndof) = hasTrackDetails && tau.pfCand_track_ndof.at(pfCand_idx) > 0 ?
                tau.pfCand_track_chi2.at(pfCand_idx) / tau.pfCand_track_ndof.at(pfCand_idx) : 0;
            get_PFmuon(Br::pfCand_muon_track_ndof) = hasTrackDetails && tau.pfCand_track_ndof.at(pfCand_idx) > 0 ?
                tau.pfCand_track_ndof.at(pfCand_idx) : 0;
        }

        { // CellObjectType::PfCand_chargedHadron
          typedef PfCand_chargedHadron_f Br;
          auto get_PFchHad = [&](PfCand_chargedHadron_f _fe_n) -> float& {
            size_t flat_index = getIndex(n_pf_chHad, static_cast<size_t>(_fe_n));
            return data->x_grid.at(CellObjectType::PfCand_chargedHadron).at(inner).at(flat_index);
          };

          size_t n_pfCand, pfCand_idx;
          getBestObj(CellObjectType::PfCand_chargedHadron, n_pfCand, pfCand_idx);
          const bool valid = n_pfCand != 0;

          get_PFchHad(Br::pfCand_chHad_valid) = valid;
          get_PFchHad(Br::pfCand_chHad_rel_pt) = valid ? tau.pfCand_pt.at(pfCand_idx) / tau.tau_pt : 0;
          get_PFchHad(Br::pfCand_chHad_deta) = valid ? tau.pfCand_eta.at(pfCand_idx) - tau.tau_eta : 0;
          get_PFchHad(Br::pfCand_chHad_dphi) = valid ? DeltaPhi(tau.pfCand_phi.at(pfCand_idx), tau.tau_phi) : 0;
          get_PFchHad(Br::pfCand_chHad_leadChargedHadrCand) = valid ? tau.pfCand_leadChargedHadrCand.at(pfCand_idx) : 0;
          get_PFchHad(Br::pfCand_chHad_pvAssociationQuality) = valid ? tau.pfCand_pvAssociationQuality.at(pfCand_idx) : 0;
          get_PFchHad(Br::pfCand_chHad_fromPV) = valid ? tau.pfCand_fromPV.at(pfCand_idx) : 0;
          get_PFchHad(Br::pfCand_chHad_puppiWeight) = valid ? tau.pfCand_puppiWeight.at(pfCand_idx) : 0;
          get_PFchHad(Br::pfCand_chHad_puppiWeightNoLep) = valid ? tau.pfCand_puppiWeightNoLep.at(pfCand_idx) : 0;
          get_PFchHad(Br::pfCand_chHad_charge) = valid ? tau.pfCand_charge.at(pfCand_idx) : 0;
          get_PFchHad(Br::pfCand_chHad_lostInnerHits) = valid ? tau.pfCand_lostInnerHits.at(pfCand_idx) : 0;
          get_PFchHad(Br::pfCand_chHad_numberOfPixelHits) = valid ? tau.pfCand_numberOfPixelHits.at(pfCand_idx) : 0;

          get_PFchHad(Br::pfCand_chHad_vertex_dx) = valid ? tau.pfCand_vertex_x.at(pfCand_idx) - tau.pv_x : 0;
          get_PFchHad(Br::pfCand_chHad_vertex_dy) = valid ? tau.pfCand_vertex_y.at(pfCand_idx) - tau.pv_y : 0;
          get_PFchHad(Br::pfCand_chHad_vertex_dz) = valid ? tau.pfCand_vertex_z.at(pfCand_idx) - tau.pv_z : 0;
          get_PFchHad(Br::pfCand_chHad_vertex_dx_tauFL) = valid ? tau.pfCand_vertex_x.at(pfCand_idx) - tau.pv_x -
                                                          tau.tau_flightLength_x : 0;
          get_PFchHad(Br::pfCand_chHad_vertex_dy_tauFL) = valid ? tau.pfCand_vertex_y.at(pfCand_idx) - tau.pv_y -
                                                          tau.tau_flightLength_y : 0;
          get_PFchHad(Br::pfCand_chHad_vertex_dz_tauFL) = valid ? tau.pfCand_vertex_z.at(pfCand_idx) - tau.pv_z -
                                                          tau.tau_flightLength_z : 0;

          const bool hasTrackDetails = valid && tau.pfCand_hasTrackDetails.at(pfCand_idx) == 1;
          get_PFchHad(Br::pfCand_chHad_hasTrackDetails) = hasTrackDetails;
          get_PFchHad(Br::pfCand_chHad_dxy) = hasTrackDetails ? tau.pfCand_dxy.at(pfCand_idx) : 0;
          get_PFchHad(Br::pfCand_chHad_dxy_sig) = hasTrackDetails ? std::abs(tau.pfCand_dxy.at(pfCand_idx)) /
                                                  tau.pfCand_dxy_error.at(pfCand_idx) : 0;
          get_PFchHad(Br::pfCand_chHad_dz) = hasTrackDetails ? tau.pfCand_dz.at(pfCand_idx) : 0;
          get_PFchHad(Br::pfCand_chHad_dz_sig) = hasTrackDetails ? std::abs(tau.pfCand_dz.at(pfCand_idx)) /
                                                 tau.pfCand_dz_error.at(pfCand_idx) : 0;
          get_PFchHad(Br::pfCand_chHad_track_chi2_ndof) = hasTrackDetails && tau.pfCand_track_ndof.at(pfCand_idx) > 0 ?
                                                          tau.pfCand_track_chi2.at(pfCand_idx) / tau.pfCand_track_ndof.at(pfCand_idx) : 0;
          get_PFchHad(Br::pfCand_chHad_track_ndof) = hasTrackDetails && tau.pfCand_track_ndof.at(pfCand_idx) > 0 ?
                                                     tau.pfCand_track_ndof.at(pfCand_idx) : 0;

          get_PFchHad(Br::pfCand_chHad_hcalFraction) = valid ? tau.pfCand_hcalFraction.at(pfCand_idx) : 0;
          get_PFchHad(Br::pfCand_chHad_rawCaloFraction) = valid ? tau.pfCand_rawCaloFraction.at(pfCand_idx) : 0;
        }

        { // CellObjectType::PfCand_neutralHadron
          typedef PfCand_neutralHadron_f Br;
          auto get_PFchHad = [&](PfCand_neutralHadron_f _fe_n) -> float& {
            size_t flat_index = getIndex(n_pf_nHad, static_cast<size_t>(_fe_n));
            return data->x_grid.at(CellObjectType::PfCand_neutralHadron).at(inner).at(flat_index);
          };

          size_t n_pfCand, pfCand_idx;
          getBestObj(CellObjectType::PfCand_neutralHadron, n_pfCand, pfCand_idx);
          const bool valid = n_pfCand != 0;

          get_PFchHad(Br::pfCand_nHad_valid) = valid;
          get_PFchHad(Br::pfCand_nHad_rel_pt) = valid ? tau.pfCand_pt.at(pfCand_idx) / tau.tau_pt : 0;
          get_PFchHad(Br::pfCand_nHad_deta) = valid ? tau.pfCand_eta.at(pfCand_idx) - tau.tau_eta : 0;
          get_PFchHad(Br::pfCand_nHad_dphi) = valid ? DeltaPhi(tau.pfCand_phi.at(pfCand_idx), tau.tau_phi) : 0;
          get_PFchHad(Br::pfCand_nHad_puppiWeight) = valid ? tau.pfCand_puppiWeight.at(pfCand_idx) : 0;
          get_PFchHad(Br::pfCand_nHad_puppiWeightNoLep) = valid ? tau.pfCand_puppiWeightNoLep.at(pfCand_idx) : 0;
          get_PFchHad(Br::pfCand_nHad_hcalFraction) = valid ? tau.pfCand_hcalFraction.at(pfCand_idx) : 0;
        }

        { // CellObjectType::PfCand_gamma
          typedef pfCand_gamma_f Br;
          auto get_PFgamma= [&](pfCand_gamma_f _fe_n) -> float& {
            size_t flat_index = getIndex(n_pf_gamma, static_cast<size_t>(_fe_n));
            return data->x_grid.at(CellObjectType::PfCand_gamma).at(inner).at(flat_index);
          };

          size_t n_pfCand, pfCand_idx;
          getBestObj(CellObjectType::PfCand_gamma, n_pfCand, pfCand_idx);
          const bool valid = n_pfCand != 0;

          get_PFgamma(Br::pfCand_gamma_valid) = valid;

          get_PFgamma(Br::pfCand_gamma_rel_pt) = valid ? tau.pfCand_pt.at(pfCand_idx) / tau.tau_pt : 0;
          get_PFgamma(Br::pfCand_gamma_deta) = valid ? tau.pfCand_eta.at(pfCand_idx) - tau.tau_eta : 0;
          get_PFgamma(Br::pfCand_gamma_dphi) = valid ? DeltaPhi(tau.pfCand_phi.at(pfCand_idx), tau.tau_phi) : 0;
          get_PFgamma(Br::pfCand_gamma_pvAssociationQuality) = valid ? tau.pfCand_pvAssociationQuality.at(pfCand_idx) : 0;
          get_PFgamma(Br::pfCand_gamma_fromPV) = valid ? tau.pfCand_fromPV.at(pfCand_idx) : 0;
          get_PFgamma(Br::pfCand_gamma_puppiWeight) = valid ? tau.pfCand_puppiWeight.at(pfCand_idx) : 0;
          get_PFgamma(Br::pfCand_gamma_puppiWeightNoLep) = valid ? tau.pfCand_puppiWeightNoLep.at(pfCand_idx) : 0;
          get_PFgamma(Br::pfCand_gamma_lostInnerHits) = valid ? tau.pfCand_lostInnerHits.at(pfCand_idx) : 0;
          get_PFgamma(Br::pfCand_gamma_numberOfPixelHits) = valid ? tau.pfCand_numberOfPixelHits.at(pfCand_idx) : 0;

          get_PFgamma(Br::pfCand_gamma_vertex_dx) = valid ? tau.pfCand_vertex_x.at(pfCand_idx) - tau.pv_x : 0;
          get_PFgamma(Br::pfCand_gamma_vertex_dy) = valid ? tau.pfCand_vertex_y.at(pfCand_idx) - tau.pv_y : 0;
          get_PFgamma(Br::pfCand_gamma_vertex_dz) = valid ? tau.pfCand_vertex_z.at(pfCand_idx) - tau.pv_z : 0;
          get_PFgamma(Br::pfCand_gamma_vertex_dx_tauFL) = valid ? tau.pfCand_vertex_x.at(pfCand_idx) - tau.pv_x -
                                                          tau.tau_flightLength_x : 0;
          get_PFgamma(Br::pfCand_gamma_vertex_dy_tauFL) = valid ? tau.pfCand_vertex_y.at(pfCand_idx) - tau.pv_y -
                                                          tau.tau_flightLength_y : 0;
          get_PFgamma(Br::pfCand_gamma_vertex_dz_tauFL) = valid ? tau.pfCand_vertex_z.at(pfCand_idx) - tau.pv_z -
                                                          tau.tau_flightLength_z : 0;

          const bool hasTrackDetails = valid && tau.pfCand_hasTrackDetails.at(pfCand_idx) == 1;
          get_PFgamma(Br::pfCand_gamma_hasTrackDetails) = hasTrackDetails;
          get_PFgamma(Br::pfCand_gamma_dxy) = hasTrackDetails ? tau.pfCand_dxy.at(pfCand_idx) : 0;
          get_PFgamma(Br::pfCand_gamma_dxy_sig) = hasTrackDetails ? std::abs(tau.pfCand_dxy.at(pfCand_idx)) /
                                                  tau.pfCand_dxy_error.at(pfCand_idx) : 0;
          get_PFgamma(Br::pfCand_gamma_dz) = hasTrackDetails ? tau.pfCand_dz.at(pfCand_idx) : 0;
          get_PFgamma(Br::pfCand_gamma_dz_sig) = hasTrackDetails ? std::abs(tau.pfCand_dz.at(pfCand_idx)) /
                                                 tau.pfCand_dz_error.at(pfCand_idx) : 0;
          get_PFgamma(Br::pfCand_gamma_track_chi2_ndof) = hasTrackDetails && tau.pfCand_track_ndof.at(pfCand_idx) > 0 ?
                                                          tau.pfCand_track_chi2.at(pfCand_idx) / tau.pfCand_track_ndof.at(pfCand_idx) : 0;
          get_PFgamma(Br::pfCand_gamma_track_ndof) = hasTrackDetails && tau.pfCand_track_ndof.at(pfCand_idx) > 0 ?
                                                     tau.pfCand_track_ndof.at(pfCand_idx) : 0;
        }

        { // PAT electron
          typedef Electron_f Br;
          auto get_PATele = [&](Electron_f _fe_n) -> float& {
            size_t flat_index = getIndex(n_electrons, static_cast<size_t>(_fe_n));
            return data->x_grid.at(CellObjectType::Electron).at(inner).at(flat_index);
          };

          size_t n_ele, idx;
          getBestObj(CellObjectType::Electron, n_ele, idx);
          const bool valid = n_ele != 0;

          get_PATele(Br::ele_valid) = valid;
          get_PATele(Br::ele_rel_pt) = valid ? tau.ele_pt.at(idx) / tau.tau_pt : 0;
          get_PATele(Br::ele_deta) = valid ? tau.ele_eta.at(idx) - tau.tau_eta : 0;
          get_PATele(Br::ele_dphi) = valid ? DeltaPhi(tau.ele_phi.at(idx), tau.tau_phi) : 0;

          const bool cc_valid = valid && tau.ele_cc_ele_energy.at(idx) >= 0;
          get_PATele(Br::ele_cc_valid) = cc_valid;
          get_PATele(Br::ele_cc_ele_rel_energy) = cc_valid ? tau.ele_cc_ele_energy.at(idx) / tau.ele_pt.at(idx) : 0;
          get_PATele(Br::ele_cc_gamma_rel_energy) = cc_valid ? tau.ele_cc_gamma_energy.at(idx) /
              tau.ele_cc_ele_energy.at(idx) : 0;
          get_PATele(Br::ele_cc_n_gamma) = cc_valid ? tau.ele_cc_n_gamma.at(idx) : 0;
          get_PATele(Br::ele_rel_trackMomentumAtVtx) = valid ? tau.ele_trackMomentumAtVtx.at(idx) /
              tau.ele_pt.at(idx) : 0;
          get_PATele(Br::ele_rel_trackMomentumAtCalo) = valid ? tau.ele_trackMomentumAtCalo.at(idx) /
              tau.ele_pt.at(idx) : 0;
          get_PATele(Br::ele_rel_trackMomentumOut) = valid ? tau.ele_trackMomentumOut.at(idx) /
              tau.ele_pt.at(idx) : 0;
          get_PATele(Br::ele_rel_trackMomentumAtEleClus) = valid ? tau.ele_trackMomentumAtEleClus.at(idx) /
              tau.ele_pt.at(idx) : 0;
          get_PATele(Br::ele_rel_trackMomentumAtVtxWithConstraint) = valid ?
              tau.ele_trackMomentumAtVtxWithConstraint.at(idx) / tau.ele_pt.at(idx) : 0;
          get_PATele(Br::ele_rel_ecalEnergy) = valid ? tau.ele_ecalEnergy.at(idx) /
              tau.ele_pt.at(idx) : 0;
          get_PATele(Br::ele_ecalEnergy_sig) = valid ? tau.ele_ecalEnergy.at(idx) /
              tau.ele_ecalEnergy_error.at(idx) : 0;
          get_PATele(Br::ele_eSuperClusterOverP) = valid ? tau.ele_eSuperClusterOverP.at(idx) : 0;
          get_PATele(Br::ele_eSeedClusterOverP) = valid ? tau.ele_eSeedClusterOverP.at(idx) : 0;
          get_PATele(Br::ele_eSeedClusterOverPout) = valid ? tau.ele_eSeedClusterOverPout.at(idx) : 0;
          get_PATele(Br::ele_eEleClusterOverPout) = valid ? tau.ele_eEleClusterOverPout.at(idx) : 0;
          get_PATele(Br::ele_deltaEtaSuperClusterTrackAtVtx) = valid ?
              tau.ele_deltaEtaSuperClusterTrackAtVtx.at(idx) : 0;
          get_PATele(Br::ele_deltaEtaSeedClusterTrackAtCalo) = valid ?
              tau.ele_deltaEtaSeedClusterTrackAtCalo.at(idx): 0;
          get_PATele(Br::ele_deltaEtaEleClusterTrackAtCalo) = valid ? tau.ele_deltaEtaEleClusterTrackAtCalo.at(idx) : 0;
          get_PATele(Br::ele_deltaPhiEleClusterTrackAtCalo) = valid ? tau.ele_deltaPhiEleClusterTrackAtCalo.at(idx) : 0;
          get_PATele(Br::ele_deltaPhiSuperClusterTrackAtVtx) = valid ? tau.ele_deltaPhiSuperClusterTrackAtVtx.at(idx) : 0;
          get_PATele(Br::ele_deltaPhiSeedClusterTrackAtCalo) = valid ? tau.ele_deltaPhiSeedClusterTrackAtCalo.at(idx) : 0;
          get_PATele(Br::ele_mvaInput_earlyBrem) = valid ? tau.ele_mvaInput_earlyBrem.at(idx) : 0;
          get_PATele(Br::ele_mvaInput_lateBrem) = valid ? tau.ele_mvaInput_lateBrem.at(idx) : 0;
          get_PATele(Br::ele_mvaInput_sigmaEtaEta) = valid ? tau.ele_mvaInput_sigmaEtaEta.at(idx) : 0;
          get_PATele(Br::ele_mvaInput_hadEnergy) = valid ? tau.ele_mvaInput_hadEnergy.at(idx) : 0;
          get_PATele(Br::ele_mvaInput_deltaEta) = valid ? tau.ele_mvaInput_deltaEta.at(idx) : 0;
          get_PATele(Br::ele_gsfTrack_normalizedChi2) = valid ? tau.ele_gsfTrack_normalizedChi2.at(idx) : 0;
          get_PATele(Br::ele_gsfTrack_numberOfValidHits) = valid ? tau.ele_gsfTrack_numberOfValidHits.at(idx) : 0;
          get_PATele(Br::ele_rel_gsfTrack_pt) = valid ? tau.ele_gsfTrack_pt.at(idx) / tau.ele_pt.at(idx) : 0;
          get_PATele(Br::ele_gsfTrack_pt_sig) = valid ? tau.ele_gsfTrack_pt.at(idx) / tau.ele_gsfTrack_pt_error.at(idx) : 0;
          const bool has_closestCtfTrack = valid && tau.ele_closestCtfTrack_normalizedChi2.at(idx) >= 0;
          get_PATele(Br::ele_has_closestCtfTrack) = has_closestCtfTrack;
          get_PATele(Br::ele_closestCtfTrack_normalizedChi2) = has_closestCtfTrack ?
              tau.ele_closestCtfTrack_normalizedChi2.at(idx) : 0;
          get_PATele(Br::ele_closestCtfTrack_numberOfValidHits) = has_closestCtfTrack ?
              tau.ele_closestCtfTrack_numberOfValidHits.at(idx) : 0;
        }

        { // PAT muon
          typedef Muon_f Br;
          auto get_PATmuon = [&](Muon_f _fe_n) -> float& {
            size_t flat_index = getIndex(n_muons, static_cast<size_t>(_fe_n));
            return data->x_grid.at(CellObjectType::Muon).at(inner).at(flat_index);
          };

            size_t n_muon, idx;
            getBestObj(CellObjectType::Muon, n_muon, idx);
            const bool valid = n_muon != 0;

            get_PATmuon(Br::muon_valid) = valid;
            get_PATmuon(Br::muon_rel_pt) = valid ? tau.muon_pt.at(idx) / tau.tau_pt : 0;
            get_PATmuon(Br::muon_deta) = valid ? tau.muon_eta.at(idx) - tau.tau_eta : 0;
            get_PATmuon(Br::muon_dphi) = valid ? DeltaPhi(tau.muon_phi.at(idx), tau.tau_phi) : 0;

            get_PATmuon(Br::muon_dxy) = valid ? tau.muon_dxy.at(idx), 0.0019f, 1.039f : 0;
            get_PATmuon(Br::muon_dxy_sig) = valid ? std::abs(tau.muon_dxy.at(idx)) / tau.muon_dxy_error.at(idx) : 0;
            const bool normalizedChi2_valid = valid && tau.muon_normalizedChi2.at(idx) >= 0;
            get_PATmuon(Br::muon_normalizedChi2_valid) = normalizedChi2_valid;
            get_PATmuon(Br::muon_normalizedChi2) = normalizedChi2_valid ? tau.muon_normalizedChi2.at(idx) : 0;
            get_PATmuon(Br::muon_numberOfValidHits) = normalizedChi2_valid ? tau.muon_numberOfValidHits.at(idx) : 0;
            get_PATmuon(Br::muon_segmentCompatibility) = valid ? tau.muon_segmentCompatibility.at(idx) : 0;
            get_PATmuon(Br::muon_caloCompatibility) = valid ? tau.muon_caloCompatibility.at(idx) : 0;
            const bool pfEcalEnergy_valid = valid && tau.muon_pfEcalEnergy.at(idx) >= 0;
            get_PATmuon(Br::muon_pfEcalEnergy_valid) = pfEcalEnergy_valid;
            get_PATmuon(Br::muon_rel_pfEcalEnergy) = pfEcalEnergy_valid ? tau.muon_pfEcalEnergy.at(idx) /
                                                     tau.muon_pt.at(idx) : 0;
            get_PATmuon(Br::muon_n_matches_DT_1) = valid ? tau.muon_n_matches_DT_1.at(idx) : 0;
            get_PATmuon(Br::muon_n_matches_DT_2) = valid ? tau.muon_n_matches_DT_2.at(idx) : 0;
            get_PATmuon(Br::muon_n_matches_DT_3) = valid ? tau.muon_n_matches_DT_3.at(idx) : 0;
            get_PATmuon(Br::muon_n_matches_DT_4) = valid ? tau.muon_n_matches_DT_4.at(idx) : 0;
            get_PATmuon(Br::muon_n_matches_CSC_1) = valid ? tau.muon_n_matches_CSC_1.at(idx) : 0;
            get_PATmuon(Br::muon_n_matches_CSC_2) = valid ? tau.muon_n_matches_CSC_2.at(idx) : 0;
            get_PATmuon(Br::muon_n_matches_CSC_3) = valid ? tau.muon_n_matches_CSC_3.at(idx) : 0;
            get_PATmuon(Br::muon_n_matches_CSC_4) = valid ? tau.muon_n_matches_CSC_4.at(idx) : 0;
            get_PATmuon(Br::muon_n_matches_RPC_1) = valid ? tau.muon_n_matches_RPC_1.at(idx) : 0;
            get_PATmuon(Br::muon_n_matches_RPC_2) = valid ? tau.muon_n_matches_RPC_2.at(idx) : 0;
            get_PATmuon(Br::muon_n_matches_RPC_3) = valid ? tau.muon_n_matches_RPC_3.at(idx) : 0;
            get_PATmuon(Br::muon_n_matches_RPC_4) = valid ? tau.muon_n_matches_RPC_4.at(idx) : 0;
            get_PATmuon(Br::muon_n_hits_DT_1) = valid ? tau.muon_n_hits_DT_1.at(idx) : 0;
            get_PATmuon(Br::muon_n_hits_DT_2) = valid ? tau.muon_n_hits_DT_2.at(idx) : 0;
            get_PATmuon(Br::muon_n_hits_DT_3) = valid ? tau.muon_n_hits_DT_3.at(idx) : 0;
            get_PATmuon(Br::muon_n_hits_DT_4) = valid ? tau.muon_n_hits_DT_4.at(idx) : 0;
            get_PATmuon(Br::muon_n_hits_CSC_1) = valid ? tau.muon_n_hits_CSC_1.at(idx) : 0;
            get_PATmuon(Br::muon_n_hits_CSC_2) = valid ? tau.muon_n_hits_CSC_2.at(idx) : 0;
            get_PATmuon(Br::muon_n_hits_CSC_3) = valid ? tau.muon_n_hits_CSC_3.at(idx) : 0;
            get_PATmuon(Br::muon_n_hits_CSC_4) = valid ? tau.muon_n_hits_CSC_4.at(idx) : 0;
            get_PATmuon(Br::muon_n_hits_RPC_1) = valid ? tau.muon_n_hits_RPC_1.at(idx) : 0;
            get_PATmuon(Br::muon_n_hits_RPC_2) = valid ? tau.muon_n_hits_RPC_2.at(idx) : 0;
            get_PATmuon(Br::muon_n_hits_RPC_3) = valid ? tau.muon_n_hits_RPC_3.at(idx) : 0;
            get_PATmuon(Br::muon_n_hits_RPC_4) = valid ? tau.muon_n_hits_RPC_4.at(idx) : 0;
        }
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
              throw std::runtime_error("Unknown object pdg id = "+std::to_string(pdgId));
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
                  throw std::runtime_error("Inconsistent cell inputs.");
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

private:
  const size_t n_tau; // number of events(=taus)
  Long64_t start_dataset;
  Long64_t end_dataset;
  Long64_t current_entry; // number of the current entry
  const size_t n_inner_cells;
  const double inner_cell_size;
  const size_t n_outer_cells;
  const double outer_cell_size;
  const int n_threads;
  const size_t parity;
  const size_t n_fe_tau;
  const size_t n_pf_el;
  const size_t n_pf_mu;
  const size_t n_pf_chHad;
  const size_t n_pf_nHad;
  const size_t n_pf_gamma;
  const size_t n_electrons;
  const size_t n_muons;
  const int n_labels;
  const CellGrid innerCellGridRef, outerCellGridRef;
  // const float trainingWeightFactor;


  std::shared_ptr<TFile> file;
  std::shared_ptr<tau_tuple::TauTuple> tauTuple; // tuple is the tree

};
