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
         size_t n_outer_cells, size_t pfelectron_fn, size_t pfmuon_fn) :
         x_tau(n_tau * tau_fn, 0), weight(n_tau, 1), y_labels(n_tau, 0)
         {
           x_grid[CellObjectType::PfCand_electron][0] = std::vector<float>(n_tau * n_outer_cells * n_outer_cells * pfelectron_fn,0);
           x_grid[CellObjectType::PfCand_electron][1] = std::vector<float>(n_tau * n_inner_cells * n_inner_cells * pfelectron_fn,0);
           x_grid[CellObjectType::PfCand_muon][0] = std::vector<float>(n_tau * n_outer_cells * n_outer_cells * pfmuon_fn,0);
           x_grid[CellObjectType::PfCand_muon][1] = std::vector<float>(n_tau * n_inner_cells * n_inner_cells * pfmuon_fn,0);
         }

    std::vector<float> x_tau;
    std::map<CellObjectType, std::map<bool, std::vector<float>>> x_grid; // [enum class CellObjectType][ 0 - outer, 1 - inner]
    std::vector<float> weight;
    std::vector<float> y_labels;
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
                                           n_pf_el, n_pf_mu);

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

        get_tau_branch(TauFlat_f::tau_index) = tau.tau_index;
        get_tau_branch(TauFlat_f::tau_pt) = tau.tau_pt;
        get_tau_branch(TauFlat_f::tau_eta) = tau.tau_eta;
        get_tau_branch(TauFlat_f::tau_phi) = tau.tau_phi;
        get_tau_branch(TauFlat_f::tau_mass) = tau.tau_mass;

        const analysis::LorentzVectorM tau_p4(tau.tau_pt, tau.tau_eta, tau.tau_phi, tau.tau_mass);
        get_tau_branch(TauFlat_f::tau_E_over_pt) = tau_p4.energy() / tau.tau_pt;
        get_tau_branch(TauFlat_f::tau_charge) = tau.tau_charge;
        get_tau_branch(TauFlat_f::tau_n_charged_prongs) = tau.tau_decayMode / 5;
        get_tau_branch(TauFlat_f::tau_n_neutral_prongs) = tau.tau_decayMode % 5;

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
            get_PFelectron(Br::pfCand_ele_n_total) = static_cast<int>(n_pfCand);
            get_PFelectron(Br::pfCand_ele_valid) = valid;

            get_PFelectron(Br::pfCand_ele_rel_pt) = valid ? tau.pfCand_pt.at(pfCand_idx) / tau.tau_pt : 0;
            get_PFelectron(Br::pfCand_ele_deta) = valid ? tau.pfCand_eta.at(pfCand_idx) - tau.tau_eta : 0;
            get_PFelectron(Br::pfCand_ele_dphi) = valid ? DeltaPhi(tau.pfCand_phi.at(pfCand_idx), tau.tau_phi) : 0;
            get_PFelectron(Br::pfCand_ele_tauSignal) = valid ? tau.pfCand_tauSignal.at(pfCand_idx) : 0;
            get_PFelectron(Br::pfCand_ele_tauIso) = valid ? tau.pfCand_tauIso.at(pfCand_idx) : 0;
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
            get_PFmuon(Br::pfCand_muon_n_total) = static_cast<int>(n_pfCand);
            get_PFmuon(Br::pfCand_muon_valid) = valid;

            get_PFmuon(Br::pfCand_muon_rel_pt) = valid ? tau.pfCand_pt.at(pfCand_idx) / tau.tau_pt : 0;
            get_PFmuon(Br::pfCand_muon_deta) = valid ? tau.pfCand_eta.at(pfCand_idx) - tau.tau_eta : 0;
            get_PFmuon(Br::pfCand_muon_dphi) = valid ? DeltaPhi(tau.pfCand_phi.at(pfCand_idx), tau.tau_phi) : 0;
            get_PFmuon(Br::pfCand_muon_tauSignal) = valid ? tau.pfCand_tauSignal.at(pfCand_idx) : 0;
            get_PFmuon(Br::pfCand_muon_tauIso) = valid ? tau.pfCand_tauIso.at(pfCand_idx) : 0;
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

      // static std::pair<LorentzVectorXYZ, double> SumP4(const std::vector<float>& pt, const std::vector<float>& eta,
      //                                                  const std::vector<float>& phi, const std::vector<float>& mass,
      //                                                  const std::set<size_t>& indices = {})
      // {
      //     const size_t N = pt.size();
      //     if(eta.size() != N || phi.size() != N || mass.size() != N)
      //         throw analysis::exception("Inconsistent component sizes for p4.");
      //     LorentzVectorXYZ sum_p4(0, 0, 0, 0);
      //     double pt_scalar_sum = 0;
      //
      //     const auto for_body = [&](size_t n) {
      //         const LorentzVectorM p4(pt.at(n), eta.at(n), phi.at(n), mass.at(n));
      //         sum_p4 += p4;
      //         pt_scalar_sum += pt.at(n);
      //     };
      //
      //     if(indices.empty()) {
      //         for(size_t n = 0; n < N; ++n)
      //             for_body(n);
      //     } else {
      //         for(size_t n : indices)
      //             for_body(n);
      //     }
      //     return std::make_pair(sum_p4, pt_scalar_sum);
      // }

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
  const int parity;
  const int n_fe_tau;
  const int n_pf_el;
  const int n_pf_mu;
  const CellGrid innerCellGridRef, outerCellGridRef;
  // const float trainingWeightFactor;


  std::shared_ptr<TFile> file;
  std::shared_ptr<tau_tuple::TauTuple> tauTuple; // tuple is the tree

};
