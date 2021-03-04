#include "TauMLTools/Core/interface/AnalysisMath.h"
#include "TauMLTools/Analysis/interface/TauTuple.h"
#include "TauMLTools/Training/interface/DataLoader_tools.h"

#include "TROOT.h"
#include "TLorentzVector.h"

using namespace std;

enum class CellObjectType {
  PfCand_electron,
  PfCand_muon,
  PfCand_chargedHadron,
  PfCand_neutralHadron,
  PfCand_gamma,
  Electron,
  Muon
};

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


struct Data {
    typedef std::unordered_map<CellObjectType, std::unordered_map<bool, std::vector<float>>> GridMap;

    Data(size_t n_tau, size_t tau_fn, size_t n_inner_cells,
         size_t n_outer_cells, size_t pfelectron_fn, size_t pfmuon_fn,
         size_t pfchargedhad_fn, size_t pfneutralhad_fn, size_t pfgamma_fn,
         size_t electron_fn, size_t muon_fn, size_t tau_labels) :
         x_tau(n_tau * tau_fn, 0), weight(n_tau, 0), y_onehot(n_tau * tau_labels, 0)
         {
          // pf electron
           // x_grid[CellObjectType::PfCand_electron][0] = std::vector<float>(n_tau * n_outer_cells * n_outer_cells * pfelectron_fn,0);
           x_grid[CellObjectType::PfCand_electron][0].resize(n_tau * n_outer_cells * n_outer_cells * pfelectron_fn,0);
           x_grid[CellObjectType::PfCand_electron][1].resize(n_tau * n_inner_cells * n_inner_cells * pfelectron_fn,0);
           // pf muons
           x_grid[CellObjectType::PfCand_muon][0].resize(n_tau * n_outer_cells * n_outer_cells * pfmuon_fn,0);
           x_grid[CellObjectType::PfCand_muon][1].resize(n_tau * n_inner_cells * n_inner_cells * pfmuon_fn,0);
           // pf charged hadrons
           x_grid[CellObjectType::PfCand_chargedHadron][0].resize(n_tau * n_outer_cells * n_outer_cells * pfchargedhad_fn,0);
           x_grid[CellObjectType::PfCand_chargedHadron][1].resize(n_tau * n_inner_cells * n_inner_cells * pfchargedhad_fn,0);
           // pf neutral hadrons
           x_grid[CellObjectType::PfCand_neutralHadron][0].resize(n_tau * n_outer_cells * n_outer_cells * pfneutralhad_fn,0);
           x_grid[CellObjectType::PfCand_neutralHadron][1].resize(n_tau * n_inner_cells * n_inner_cells * pfneutralhad_fn,0);
           // pf gamma
           x_grid[CellObjectType::PfCand_gamma][0].resize(n_tau * n_outer_cells * n_outer_cells * pfgamma_fn,0);
           x_grid[CellObjectType::PfCand_gamma][1].resize(n_tau * n_inner_cells * n_inner_cells * pfgamma_fn,0);
           // electrons
           x_grid[CellObjectType::Electron][0].resize(n_tau * n_outer_cells * n_outer_cells * electron_fn,0);
           x_grid[CellObjectType::Electron][1].resize(n_tau * n_inner_cells * n_inner_cells * electron_fn,0);
           // muons
           x_grid[CellObjectType::Muon][0].resize(n_tau * n_outer_cells * n_outer_cells * muon_fn,0);
           x_grid[CellObjectType::Muon][1].resize(n_tau * n_inner_cells * n_inner_cells * muon_fn,0);
         }

    std::vector<float> x_tau;
    GridMap x_grid; // [enum class CellObjectType][ 0 - outer, 1 - inner]
    std::vector<float> weight;
    std::vector<float> y_onehot;
};


using namespace Setup;
class DataLoader {

public:
    using Tau = tau_tuple::Tau;
    using TauTuple = tau_tuple::TauTuple;

    DataLoader() : current_entry(start_dataset),
        innerCellGridRef(n_inner_cells, n_inner_cells, inner_cell_size, inner_cell_size),
        outerCellGridRef(n_outer_cells, n_outer_cells, outer_cell_size, outer_cell_size),
        input_files(FindInputFiles(input_dirs,file_name_pattern,
                                   exclude_list, exclude_dir_list)),
        hasData(false)
    {
      if(n_threads > 1) ROOT::EnableImplicitMT(n_threads);

      // file = OpenRootFile(file_name);
      // tauTuple = std::make_shared<tau_tuple::TauTuple>(file.get(), true);

      std::cout << "Number of files to process: " << input_files.size() << std::endl;
      tauTuple = std::make_shared<TauTuple>("taus", input_files);
      end_entry = std::min((long long)end_dataset, tauTuple->GetEntries());

      // histogram to calculate weights
      // TH1::AddDirectory(kFALSE);
      TFile* file_input = new TFile(input_spectrum.c_str());
      TFile* file_target = new TFile(target_spectrum.c_str());
      TH2D* target_hist = (TH2D*)file_target->Get("eta_pt_hist_tau");

      for(int type = 0; type < 4; ++type) {
        hist_weights.push_back(shared_ptr<TH2D>(new TH2D(("w_1_"+tau_types_names[type]).c_str(),
                                                         ("w_1_"+tau_types_names[type]).c_str(),
                                                         n_eta_bins, eta_min, eta_max,
                                                         n_pt_bins, pt_min, pt_max)));
        hist_weights[type]->SetDirectory(0); // disabling the file referencing

        TH2D* after_rebin_input_hist = new TH2D("input_hist", "input_hist",
                                                n_eta_bins, eta_min, eta_max,
                                                n_pt_bins, pt_min, pt_max);
        TH2D* input_hist = (TH2D*)file_input->Get(("eta_pt_hist_"+tau_types_names[type]).c_str());

        RebinAndFill(*hist_weights[type], *target_hist);
        RebinAndFill(*after_rebin_input_hist, *input_hist);
        hist_weights[type]->Divide(after_rebin_input_hist);
        delete after_rebin_input_hist;
      }
      file_input->Close();
      file_target->Close();
      MaxDisbCheck(hist_weights, weight_thr);
    }

    bool MoveNext() {
        if(!tauTuple)
          throw std::runtime_error("DataLoader is not initialized.");

        data = std::make_shared<Data>(n_tau, n_fe_tau, n_inner_cells, n_outer_cells,
                                      n_pf_el, n_pf_mu, n_pf_chHad, n_pf_nHad,
                                      n_pf_gamma, n_ele, n_muon, tau_types
                                      );

        const Long64_t end_point = current_entry + n_tau;
        size_t n_processed = 0, n_total = static_cast<size_t>(end_point - current_entry);
        for(Long64_t tau_i = 0; tau_i < n_tau; ++current_entry) {
          if(current_entry == end_entry) return false;
          tauTuple->GetEntry(current_entry);
          const auto& tau = tauTuple->data();
          // skip event if it is not tau_e, tau_mu, tau_jet or tau_h
          if(tau.tauType < 0 || tau.tauType > 3) continue;
          else {
            data->y_onehot[ tau_i * tau_types + tau.tauType ] = 1.0; // filling labels
            data->weight.at(tau_i) = GetWeight(tau.tauType, tau.tau_pt, std::abs(tau.tau_eta)); // filling weights
            FillTauBranches(tau, tau_i, data);
            FillCellGrid(tau, tau_i, innerCellGridRef, data, true);
            FillCellGrid(tau, tau_i, outerCellGridRef, data, false);
            ++tau_i;
          }
        }
        hasData = true;
        return true;
    }

    size_t GetEntries () { return end_entry - current_entry; }

    std::shared_ptr<Data> LoadData() {
      if(!hasData)
        throw std::runtime_error("Data was not loaded with MoveNext()");
      hasData = false;
      return data;
    }

    static void MaxDisbCheck(std::vector<std::shared_ptr<TH2D>> hists, Double_t max_thr) {
      double min_weight = std::numeric_limits<double>::max();
      double max_weight = std::numeric_limits<double>::lowest();
      for(auto h: hists) {
        min_weight = std::min(h->GetMinimum(), min_weight);
        max_weight = std::max(h->GetMaximum(), max_weight);
      }
      std::cout << "Weights imbalance: " << max_weight / min_weight
                << ", imbalance threshold: " <<  max_thr << std::endl;
      if(max_weight / min_weight > max_thr)
        throw std::runtime_error("The imbalance in the weights exceeds the threshold.");
    }

  private:

      static constexpr float pi = boost::math::constants::pi<float>();

      const double GetWeight(const int type_id, const double pt, const double eta) const
      {
        // if(eta <= eta_min || eta >= eta_max || pt<=pt_min || pt>=pt_max) return 0;
        return hist_weights.at(type_id)->GetBinContent(
               hist_weights.at(type_id)->GetXaxis()->FindBin(eta),
               hist_weights.at(type_id)->GetYaxis()->FindBin(pt));

      }

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
        float foo;
        auto get_tau_branch = [&](TauFlat_Features _fe) -> float& {
            if(static_cast<int>(_fe) < 0) return foo;
            size_t _fe_ind = static_cast<size_t>(_fe);
            size_t index = start_array_index + _fe_ind;
            return data->x_tau.at(index);
        };

        get_tau_branch(TauFlat_Features::tau_pt) = tau.tau_pt;
        get_tau_branch(TauFlat_Features::tau_eta) = tau.tau_eta;
        get_tau_branch(TauFlat_Features::tau_phi) = tau.tau_phi;
        get_tau_branch(TauFlat_Features::tau_mass) = tau.tau_mass;

        const analysis::LorentzVectorM tau_p4(tau.tau_pt, tau.tau_eta, tau.tau_phi, tau.tau_mass);
        get_tau_branch(TauFlat_Features::tau_E_over_pt) = tau_p4.energy() / tau.tau_pt;
        get_tau_branch(TauFlat_Features::tau_charge) = tau.tau_charge;
        get_tau_branch(TauFlat_Features::tau_n_charged_prongs) = tau.tau_decayMode / 5;
        get_tau_branch(TauFlat_Features::tau_n_neutral_prongs) = tau.tau_decayMode % 5;
        get_tau_branch(TauFlat_Features::chargedIsoPtSum) = tau.chargedIsoPtSum;
        get_tau_branch(TauFlat_Features::chargedIsoPtSumdR03_over_dR05) = tau.chargedIsoPtSumdR03 / tau.chargedIsoPtSum;
        get_tau_branch(TauFlat_Features::footprintCorrection) = tau.footprintCorrection;
        get_tau_branch(TauFlat_Features::neutralIsoPtSum) = tau.neutralIsoPtSum;
        get_tau_branch(TauFlat_Features::neutralIsoPtSumWeight_over_neutralIsoPtSum) = tau.neutralIsoPtSumWeight / tau.neutralIsoPtSum;
        get_tau_branch(TauFlat_Features::neutralIsoPtSumWeightdR03_over_neutralIsoPtSum) =tau.neutralIsoPtSumWeightdR03 / tau.neutralIsoPtSum;
        get_tau_branch(TauFlat_Features::neutralIsoPtSumdR03_over_dR05) = tau.neutralIsoPtSumdR03 / tau.neutralIsoPtSum;
        get_tau_branch(TauFlat_Features::photonPtSumOutsideSignalCone) = tau.photonPtSumOutsideSignalCone;
        get_tau_branch(TauFlat_Features::puCorrPtSum) = tau.puCorrPtSum;

        const bool tau_dxy_valid = std::isnormal(tau.tau_dxy) && tau.tau_dxy > - 10
                                   && std::isnormal(tau.tau_dxy_error) && tau.tau_dxy_error > 0;
        get_tau_branch(TauFlat_Features::tau_dxy_valid) = tau_dxy_valid;
        if(tau_dxy_valid) {
          get_tau_branch(TauFlat_Features::tau_dxy) =  tau.tau_dxy;
          get_tau_branch(TauFlat_Features::tau_dxy_sig) =  std::abs(tau.tau_dxy)/tau.tau_dxy_error;
        }

        const bool tau_ip3d_valid = std::isnormal(tau.tau_ip3d) && tau.tau_ip3d > - 10
                                    && std::isnormal(tau.tau_ip3d_error) && tau.tau_ip3d_error > 0;
        get_tau_branch(TauFlat_Features::tau_ip3d_valid) =  tau_ip3d_valid;
        if(tau_ip3d_valid) {
          get_tau_branch(TauFlat_Features::tau_ip3d) = tau.tau_ip3d;
          get_tau_branch(TauFlat_Features::tau_ip3d_sig) = std::abs(tau.tau_ip3d) / tau.tau_ip3d_error;
        }
        get_tau_branch(TauFlat_Features::tau_dz) = tau.tau_dz;

        const bool tau_dz_sig_valid = std::isnormal(tau.tau_dz) && std::isnormal(tau.tau_dz_error)
                                      && tau.tau_dz_error > 0;
        get_tau_branch(TauFlat_Features::tau_dz_sig_valid) = tau_dz_sig_valid;
        if(tau_dz_sig_valid)
          get_tau_branch(TauFlat_Features::tau_dz_sig) = std::abs(tau.tau_dz) / tau.tau_dz_error;

        get_tau_branch(TauFlat_Features::tau_flightLength_x) = tau.tau_flightLength_x;
        get_tau_branch(TauFlat_Features::tau_flightLength_y) = tau.tau_flightLength_y;
        get_tau_branch(TauFlat_Features::tau_flightLength_z) = tau.tau_flightLength_z;
        get_tau_branch(TauFlat_Features::tau_flightLength_sig) = tau.tau_flightLength_sig;

        get_tau_branch(TauFlat_Features::tau_pt_weighted_deta_strip) = tau.tau_pt_weighted_deta_strip ;
        get_tau_branch(TauFlat_Features::tau_pt_weighted_dphi_strip) = tau.tau_pt_weighted_dphi_strip ;
        get_tau_branch(TauFlat_Features::tau_pt_weighted_dr_signal) = tau.tau_pt_weighted_dr_signal ;
        get_tau_branch(TauFlat_Features::tau_pt_weighted_dr_iso) = tau.tau_pt_weighted_dr_iso;

        get_tau_branch(TauFlat_Features::tau_leadingTrackNormChi2) = tau.tau_leadingTrackNormChi2;
        const bool tau_e_ratio_valid = std::isnormal(tau.tau_e_ratio) && tau.tau_e_ratio > 0.f;
        get_tau_branch(TauFlat_Features::tau_e_ratio_valid) = tau_e_ratio_valid;
        if(tau_e_ratio_valid)
          get_tau_branch(TauFlat_Features::tau_e_ratio) = tau.tau_e_ratio;

        const bool tau_gj_angle_diff_valid = (std::isnormal(tau.tau_gj_angle_diff) || tau.tau_gj_angle_diff == 0)
            && tau.tau_gj_angle_diff >= 0;
        get_tau_branch(TauFlat_Features::tau_gj_angle_diff_valid) = tau_gj_angle_diff_valid;

        if(tau_gj_angle_diff_valid)
          get_tau_branch(TauFlat_Features::tau_gj_angle_diff) = tau.tau_gj_angle_diff;

        get_tau_branch(TauFlat_Features::tau_n_photons) = tau.tau_n_photons;
        get_tau_branch(TauFlat_Features::tau_emFraction) = tau.tau_emFraction;
        get_tau_branch(TauFlat_Features::tau_inside_ecal_crack) = tau.tau_inside_ecal_crack;
        get_tau_branch(TauFlat_Features::leadChargedCand_etaAtEcalEntrance_minus_tau_eta) = tau.leadChargedCand_etaAtEcalEntrance;
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

        float foo;

        { // CellObjectType::PfCand_electron
            typedef PfCand_electron_Features Br;
            auto get_PFelectron = [&](PfCand_electron_Features _fe_n) -> float& {
              if(static_cast<int>(_fe_n) < 0) return foo;
              size_t flat_index = getIndex(n_pf_el, static_cast<size_t>(_fe_n));
              return data->x_grid.at(CellObjectType::PfCand_electron).at(inner).at(flat_index);
            };

            size_t n_pfCand, pfCand_idx;
            getBestObj(CellObjectType::PfCand_electron, n_pfCand, pfCand_idx);

            const bool valid = n_pfCand != 0;

            get_PFelectron(Br::pfCand_ele_valid) = valid;
            if(valid) {
              get_PFelectron(Br::pfCand_ele_rel_pt) = tau.pfCand_pt.at(pfCand_idx) / tau.tau_pt;
              get_PFelectron(Br::pfCand_ele_deta) = tau.pfCand_eta.at(pfCand_idx) - tau.tau_eta;
              get_PFelectron(Br::pfCand_ele_dphi) = DeltaPhi(tau.pfCand_phi.at(pfCand_idx), tau.tau_phi);
              // get_PFelectron(Br::pfCand_ele_tauSignal) = valid ? tau.pfCand_tauSignal.at(pfCand_idx) : 0;
              // get_PFelectron(Br::pfCand_ele_tauIso) = valid ? tau.pfCand_tauIso.at(pfCand_idx) : 0;
              get_PFelectron(Br::pfCand_ele_pvAssociationQuality) = tau.pfCand_pvAssociationQuality.at(pfCand_idx);
              get_PFelectron(Br::pfCand_ele_puppiWeight) = tau.pfCand_puppiWeight.at(pfCand_idx);
              get_PFelectron(Br::pfCand_ele_charge) = tau.pfCand_charge.at(pfCand_idx);
              get_PFelectron(Br::pfCand_ele_lostInnerHits) = tau.pfCand_lostInnerHits.at(pfCand_idx);
              get_PFelectron(Br::pfCand_ele_numberOfPixelHits) = tau.pfCand_numberOfPixelHits.at(pfCand_idx);

              get_PFelectron(Br::pfCand_ele_vertex_dx) = tau.pfCand_vertex_x.at(pfCand_idx) - tau.pv_x;
              get_PFelectron(Br::pfCand_ele_vertex_dy) = tau.pfCand_vertex_y.at(pfCand_idx) - tau.pv_y;
              get_PFelectron(Br::pfCand_ele_vertex_dz) = tau.pfCand_vertex_z.at(pfCand_idx) - tau.pv_z;
              get_PFelectron(Br::pfCand_ele_vertex_dx_tauFL) = tau.pfCand_vertex_x.at(pfCand_idx) - tau.pv_x - tau.tau_flightLength_x;
              get_PFelectron(Br::pfCand_ele_vertex_dy_tauFL) = tau.pfCand_vertex_y.at(pfCand_idx) - tau.pv_y - tau.tau_flightLength_y;
              get_PFelectron(Br::pfCand_ele_vertex_dz_tauFL) = tau.pfCand_vertex_z.at(pfCand_idx) - tau.pv_z - tau.tau_flightLength_z;
            }

            const bool hasTrackDetails = valid && tau.pfCand_hasTrackDetails.at(pfCand_idx) == 1;
            get_PFelectron(Br::pfCand_ele_hasTrackDetails) = hasTrackDetails;

            if(hasTrackDetails) {
              get_PFelectron(Br::pfCand_ele_dxy) = tau.pfCand_dxy.at(pfCand_idx);
              get_PFelectron(Br::pfCand_ele_dxy_sig) = std::abs(tau.pfCand_dxy.at(pfCand_idx)) / tau.pfCand_dxy_error.at(pfCand_idx);
              get_PFelectron(Br::pfCand_ele_dz) = tau.pfCand_dz.at(pfCand_idx);
              get_PFelectron(Br::pfCand_ele_dz_sig) = std::abs(tau.pfCand_dz.at(pfCand_idx)) / tau.pfCand_dz_error.at(pfCand_idx);

              if(tau.pfCand_track_ndof.at(pfCand_idx) > 0) {
                get_PFelectron(Br::pfCand_ele_track_chi2_ndof) = tau.pfCand_track_chi2.at(pfCand_idx) / tau.pfCand_track_ndof.at(pfCand_idx);
                get_PFelectron(Br::pfCand_ele_track_ndof) = tau.pfCand_track_ndof.at(pfCand_idx);
              }
            }
        }

        { // CellObjectType::PfCand_muon
            typedef PfCand_muon_Features Br;
            auto get_PFmuon = [&](PfCand_muon_Features _fe_n) -> float& {
              if(static_cast<int>(_fe_n) < 0) return foo;
              size_t flat_index = getIndex(n_pf_mu, static_cast<size_t>(_fe_n));
              return data->x_grid.at(CellObjectType::PfCand_muon).at(inner).at(flat_index);
            };

            size_t n_pfCand, pfCand_idx;
            getBestObj(CellObjectType::PfCand_muon, n_pfCand, pfCand_idx);

            const bool valid = n_pfCand != 0;
            get_PFmuon(Br::pfCand_muon_valid) = valid;

            if(valid){
              get_PFmuon(Br::pfCand_muon_rel_pt) = tau.pfCand_pt.at(pfCand_idx) / tau.tau_pt;
              get_PFmuon(Br::pfCand_muon_deta) = tau.pfCand_eta.at(pfCand_idx) - tau.tau_eta;
              get_PFmuon(Br::pfCand_muon_dphi) = DeltaPhi(tau.pfCand_phi.at(pfCand_idx), tau.tau_phi);
              get_PFmuon(Br::pfCand_muon_pvAssociationQuality) = tau.pfCand_pvAssociationQuality.at(pfCand_idx);
              get_PFmuon(Br::pfCand_muon_fromPV) = tau.pfCand_fromPV.at(pfCand_idx);
              get_PFmuon(Br::pfCand_muon_puppiWeight) = tau.pfCand_puppiWeight.at(pfCand_idx);
              get_PFmuon(Br::pfCand_muon_charge) = tau.pfCand_charge.at(pfCand_idx);
              get_PFmuon(Br::pfCand_muon_lostInnerHits) = tau.pfCand_lostInnerHits.at(pfCand_idx);
              get_PFmuon(Br::pfCand_muon_numberOfPixelHits) = tau.pfCand_numberOfPixelHits.at(pfCand_idx);

              get_PFmuon(Br::pfCand_muon_vertex_dx) = tau.pfCand_vertex_x.at(pfCand_idx) - tau.pv_x;
              get_PFmuon(Br::pfCand_muon_vertex_dy) = tau.pfCand_vertex_y.at(pfCand_idx) - tau.pv_y;
              get_PFmuon(Br::pfCand_muon_vertex_dz) = tau.pfCand_vertex_z.at(pfCand_idx) - tau.pv_z;
              get_PFmuon(Br::pfCand_muon_vertex_dx_tauFL) = tau.pfCand_vertex_x.at(pfCand_idx) - tau.pv_x - tau.tau_flightLength_x;
              get_PFmuon(Br::pfCand_muon_vertex_dy_tauFL) = tau.pfCand_vertex_y.at(pfCand_idx) - tau.pv_y - tau.tau_flightLength_y;
              get_PFmuon(Br::pfCand_muon_vertex_dz_tauFL) = tau.pfCand_vertex_z.at(pfCand_idx) - tau.pv_z - tau.tau_flightLength_z;

              const bool hasTrackDetails = valid && tau.pfCand_hasTrackDetails.at(pfCand_idx) == 1;
              get_PFmuon(Br::pfCand_muon_hasTrackDetails) = hasTrackDetails;

              if(hasTrackDetails){

              get_PFmuon(Br::pfCand_muon_dxy) = tau.pfCand_dxy.at(pfCand_idx);
              get_PFmuon(Br::pfCand_muon_dxy_sig) = std::abs(tau.pfCand_dxy.at(pfCand_idx)) / tau.pfCand_dxy_error.at(pfCand_idx);
              get_PFmuon(Br::pfCand_muon_dz) = tau.pfCand_dz.at(pfCand_idx);
              get_PFmuon(Br::pfCand_muon_dz_sig) = std::abs(tau.pfCand_dz.at(pfCand_idx)) / tau.pfCand_dz_error.at(pfCand_idx);

              if(tau.pfCand_track_ndof.at(pfCand_idx) > 0) {
                get_PFmuon(Br::pfCand_muon_track_chi2_ndof) = tau.pfCand_track_chi2.at(pfCand_idx) / tau.pfCand_track_ndof.at(pfCand_idx);
                get_PFmuon(Br::pfCand_muon_track_ndof) = tau.pfCand_track_ndof.at(pfCand_idx);
              }
              }
            }
        }

        { // CellObjectType::PfCand_chargedHadron
          typedef PfCand_chHad_Features Br;
          auto get_PFchHad = [&](PfCand_chHad_Features _fe_n) -> float& {
            if(static_cast<int>(_fe_n) < 0) return foo;
            size_t flat_index = getIndex(n_pf_chHad, static_cast<size_t>(_fe_n));
            return data->x_grid.at(CellObjectType::PfCand_chargedHadron).at(inner).at(flat_index);
          };

          size_t n_pfCand, pfCand_idx;
          getBestObj(CellObjectType::PfCand_chargedHadron, n_pfCand, pfCand_idx);
          const bool valid = n_pfCand != 0;
          get_PFchHad(Br::pfCand_chHad_valid) = valid;

          if(valid) {
            get_PFchHad(Br::pfCand_chHad_rel_pt) = tau.pfCand_pt.at(pfCand_idx) / tau.tau_pt;
            get_PFchHad(Br::pfCand_chHad_deta) = tau.pfCand_eta.at(pfCand_idx) - tau.tau_eta ;
            get_PFchHad(Br::pfCand_chHad_dphi) = DeltaPhi(tau.pfCand_phi.at(pfCand_idx), tau.tau_phi);
            get_PFchHad(Br::pfCand_chHad_leadChargedHadrCand) = tau.pfCand_leadChargedHadrCand.at(pfCand_idx);
            get_PFchHad(Br::pfCand_chHad_pvAssociationQuality) = tau.pfCand_pvAssociationQuality.at(pfCand_idx);
            get_PFchHad(Br::pfCand_chHad_fromPV) = tau.pfCand_fromPV.at(pfCand_idx);
            get_PFchHad(Br::pfCand_chHad_puppiWeight) = tau.pfCand_puppiWeight.at(pfCand_idx);
            get_PFchHad(Br::pfCand_chHad_puppiWeightNoLep) = tau.pfCand_puppiWeightNoLep.at(pfCand_idx);
            get_PFchHad(Br::pfCand_chHad_charge) = tau.pfCand_charge.at(pfCand_idx);
            get_PFchHad(Br::pfCand_chHad_lostInnerHits) = tau.pfCand_lostInnerHits.at(pfCand_idx);
            get_PFchHad(Br::pfCand_chHad_numberOfPixelHits) = tau.pfCand_numberOfPixelHits.at(pfCand_idx);

            get_PFchHad(Br::pfCand_chHad_vertex_dx) = tau.pfCand_vertex_x.at(pfCand_idx) - tau.pv_x;
            get_PFchHad(Br::pfCand_chHad_vertex_dy) = tau.pfCand_vertex_y.at(pfCand_idx) - tau.pv_y;
            get_PFchHad(Br::pfCand_chHad_vertex_dz) = tau.pfCand_vertex_z.at(pfCand_idx) - tau.pv_z;
            get_PFchHad(Br::pfCand_chHad_vertex_dx_tauFL) = tau.pfCand_vertex_x.at(pfCand_idx) - tau.pv_x - tau.tau_flightLength_x;
            get_PFchHad(Br::pfCand_chHad_vertex_dy_tauFL) =  tau.pfCand_vertex_y.at(pfCand_idx) - tau.pv_y - tau.tau_flightLength_y;
            get_PFchHad(Br::pfCand_chHad_vertex_dz_tauFL) = tau.pfCand_vertex_z.at(pfCand_idx) - tau.pv_z - tau.tau_flightLength_z;

            const bool hasTrackDetails = tau.pfCand_hasTrackDetails.at(pfCand_idx) == 1;
            get_PFchHad(Br::pfCand_chHad_hasTrackDetails) = hasTrackDetails;
            if(hasTrackDetails) {
              get_PFchHad(Br::pfCand_chHad_dxy) = tau.pfCand_dxy.at(pfCand_idx);
              get_PFchHad(Br::pfCand_chHad_dxy_sig) = std::abs(tau.pfCand_dxy.at(pfCand_idx)) /
                                                      tau.pfCand_dxy_error.at(pfCand_idx);
              get_PFchHad(Br::pfCand_chHad_dz) = tau.pfCand_dz.at(pfCand_idx);
              get_PFchHad(Br::pfCand_chHad_dz_sig) = std::abs(tau.pfCand_dz.at(pfCand_idx)) /
                                                     tau.pfCand_dz_error.at(pfCand_idx);
              get_PFchHad(Br::pfCand_chHad_track_chi2_ndof) = tau.pfCand_track_chi2.at(pfCand_idx) / tau.pfCand_track_ndof.at(pfCand_idx);
              get_PFchHad(Br::pfCand_chHad_track_ndof) = tau.pfCand_track_ndof.at(pfCand_idx);
            }

            get_PFchHad(Br::pfCand_chHad_hcalFraction) = tau.pfCand_hcalFraction.at(pfCand_idx);
            get_PFchHad(Br::pfCand_chHad_rawCaloFraction) = tau.pfCand_rawCaloFraction.at(pfCand_idx);
          }
        }

        { // CellObjectType::PfCand_neutralHadron
          typedef PfCand_nHad_Features Br;
          auto get_PFchHad = [&](PfCand_nHad_Features _fe_n) -> float& {
            if(static_cast<int>(_fe_n) < 0) return foo;
            size_t flat_index = getIndex(n_pf_nHad, static_cast<size_t>(_fe_n));
            return data->x_grid.at(CellObjectType::PfCand_neutralHadron).at(inner).at(flat_index);
          };

          size_t n_pfCand, pfCand_idx;
          getBestObj(CellObjectType::PfCand_neutralHadron, n_pfCand, pfCand_idx);
          const bool valid = n_pfCand != 0;
          get_PFchHad(Br::pfCand_nHad_valid) = valid;

          if(valid) {
            get_PFchHad(Br::pfCand_nHad_rel_pt) = tau.pfCand_pt.at(pfCand_idx) / tau.tau_pt;
            get_PFchHad(Br::pfCand_nHad_deta) = tau.pfCand_eta.at(pfCand_idx) - tau.tau_eta;
            get_PFchHad(Br::pfCand_nHad_dphi) = DeltaPhi(tau.pfCand_phi.at(pfCand_idx), tau.tau_phi);
            get_PFchHad(Br::pfCand_nHad_puppiWeight) = tau.pfCand_puppiWeight.at(pfCand_idx);
            get_PFchHad(Br::pfCand_nHad_puppiWeightNoLep) = tau.pfCand_puppiWeightNoLep.at(pfCand_idx);
            get_PFchHad(Br::pfCand_nHad_hcalFraction) = tau.pfCand_hcalFraction.at(pfCand_idx);
          }
        }

        { // CellObjectType::PfCand_gamma
          typedef pfCand_gamma_Features Br;
          auto get_PFgamma= [&](pfCand_gamma_Features _fe_n) -> float& {
            if(static_cast<int>(_fe_n) < 0) return foo;
            size_t flat_index = getIndex(n_pf_gamma, static_cast<size_t>(_fe_n));
            return data->x_grid.at(CellObjectType::PfCand_gamma).at(inner).at(flat_index);
          };

          size_t n_pfCand, pfCand_idx;
          getBestObj(CellObjectType::PfCand_gamma, n_pfCand, pfCand_idx);
          const bool valid = n_pfCand != 0;
          get_PFgamma(Br::pfCand_gamma_valid) = valid;

          if(valid) {
            get_PFgamma(Br::pfCand_gamma_rel_pt) = tau.pfCand_pt.at(pfCand_idx) / tau.tau_pt;
            get_PFgamma(Br::pfCand_gamma_deta) = tau.pfCand_eta.at(pfCand_idx) - tau.tau_eta;
            get_PFgamma(Br::pfCand_gamma_dphi) = DeltaPhi(tau.pfCand_phi.at(pfCand_idx), tau.tau_phi);
            get_PFgamma(Br::pfCand_gamma_pvAssociationQuality) = tau.pfCand_pvAssociationQuality.at(pfCand_idx);
            get_PFgamma(Br::pfCand_gamma_fromPV) = tau.pfCand_fromPV.at(pfCand_idx);
            get_PFgamma(Br::pfCand_gamma_puppiWeight) = tau.pfCand_puppiWeight.at(pfCand_idx);
            get_PFgamma(Br::pfCand_gamma_puppiWeightNoLep) = tau.pfCand_puppiWeightNoLep.at(pfCand_idx);
            get_PFgamma(Br::pfCand_gamma_lostInnerHits) = tau.pfCand_lostInnerHits.at(pfCand_idx);
            get_PFgamma(Br::pfCand_gamma_numberOfPixelHits) = tau.pfCand_numberOfPixelHits.at(pfCand_idx);

            get_PFgamma(Br::pfCand_gamma_vertex_dx) = tau.pfCand_vertex_x.at(pfCand_idx) - tau.pv_x;
            get_PFgamma(Br::pfCand_gamma_vertex_dy) = tau.pfCand_vertex_y.at(pfCand_idx) - tau.pv_y;
            get_PFgamma(Br::pfCand_gamma_vertex_dz) = tau.pfCand_vertex_z.at(pfCand_idx) - tau.pv_z;
            get_PFgamma(Br::pfCand_gamma_vertex_dx_tauFL) = tau.pfCand_vertex_x.at(pfCand_idx) - tau.pv_x -
                                                            tau.tau_flightLength_x;
            get_PFgamma(Br::pfCand_gamma_vertex_dy_tauFL) = tau.pfCand_vertex_y.at(pfCand_idx) - tau.pv_y -
                                                            tau.tau_flightLength_y;
            get_PFgamma(Br::pfCand_gamma_vertex_dz_tauFL) = tau.pfCand_vertex_z.at(pfCand_idx) - tau.pv_z -
                                                            tau.tau_flightLength_z;

            const bool hasTrackDetails = tau.pfCand_hasTrackDetails.at(pfCand_idx) == 1;
            get_PFgamma(Br::pfCand_gamma_hasTrackDetails) = hasTrackDetails;

            if(hasTrackDetails){
              get_PFgamma(Br::pfCand_gamma_dxy) = hasTrackDetails ? tau.pfCand_dxy.at(pfCand_idx) : 0;
              get_PFgamma(Br::pfCand_gamma_dxy_sig) = hasTrackDetails ? std::abs(tau.pfCand_dxy.at(pfCand_idx)) /
                                                      tau.pfCand_dxy_error.at(pfCand_idx) : 0;
              get_PFgamma(Br::pfCand_gamma_dz) = hasTrackDetails ? tau.pfCand_dz.at(pfCand_idx) : 0;
              get_PFgamma(Br::pfCand_gamma_dz_sig) = hasTrackDetails ? std::abs(tau.pfCand_dz.at(pfCand_idx)) /
                                                     tau.pfCand_dz_error.at(pfCand_idx) : 0;
              if(tau.pfCand_track_ndof.at(pfCand_idx) > 0) {
                get_PFgamma(Br::pfCand_gamma_track_chi2_ndof) = tau.pfCand_track_chi2.at(pfCand_idx) / tau.pfCand_track_ndof.at(pfCand_idx);
                get_PFgamma(Br::pfCand_gamma_track_ndof) = tau.pfCand_track_ndof.at(pfCand_idx);
              }
            }
          }
        }

        { // PAT electron
          typedef Electron_Features Br;
          auto get_PATele = [&](Electron_Features _fe_n) -> float& {
            if(static_cast<int>(_fe_n) < 0) return foo;
            size_t flat_index = getIndex(n_ele, static_cast<size_t>(_fe_n));
            return data->x_grid.at(CellObjectType::Electron).at(inner).at(flat_index);
          };

          size_t n_ele, idx;
          getBestObj(CellObjectType::Electron, n_ele, idx);
          const bool valid = n_ele != 0;

          get_PATele(Br::ele_valid) = valid;

          if(valid) {
            get_PATele(Br::ele_rel_pt) = tau.ele_pt.at(idx) / tau.tau_pt;
            get_PATele(Br::ele_deta) = tau.ele_eta.at(idx) - tau.tau_eta;
            get_PATele(Br::ele_dphi) = DeltaPhi(tau.ele_phi.at(idx), tau.tau_phi);

            const bool cc_valid = tau.ele_cc_ele_energy.at(idx) >= 0;
            get_PATele(Br::ele_cc_valid) = cc_valid;

            if(cc_valid) {
              get_PATele(Br::ele_cc_ele_rel_energy) = tau.ele_cc_ele_energy.at(idx) / tau.ele_pt.at(idx);
              get_PATele(Br::ele_cc_gamma_rel_energy) = tau.ele_cc_gamma_energy.at(idx) /
                                                        tau.ele_cc_ele_energy.at(idx);
              get_PATele(Br::ele_cc_n_gamma) = tau.ele_cc_n_gamma.at(idx);
            }
            get_PATele(Br::ele_rel_trackMomentumAtVtx) = tau.ele_trackMomentumAtVtx.at(idx) / tau.ele_pt.at(idx);
            get_PATele(Br::ele_rel_trackMomentumAtCalo) = tau.ele_trackMomentumAtCalo.at(idx) / tau.ele_pt.at(idx);
            get_PATele(Br::ele_rel_trackMomentumOut) = tau.ele_trackMomentumOut.at(idx) / tau.ele_pt.at(idx);
            get_PATele(Br::ele_rel_trackMomentumAtEleClus) = tau.ele_trackMomentumAtEleClus.at(idx) / tau.ele_pt.at(idx);
            get_PATele(Br::ele_rel_trackMomentumAtVtxWithConstraint) = tau.ele_trackMomentumAtVtxWithConstraint.at(idx) / tau.ele_pt.at(idx);
            get_PATele(Br::ele_rel_ecalEnergy) = tau.ele_ecalEnergy.at(idx) / tau.ele_pt.at(idx);
            get_PATele(Br::ele_ecalEnergy_sig) = tau.ele_ecalEnergy.at(idx) / tau.ele_ecalEnergy_error.at(idx);
            get_PATele(Br::ele_eSuperClusterOverP) = tau.ele_eSuperClusterOverP.at(idx);
            get_PATele(Br::ele_eSeedClusterOverP) = tau.ele_eSeedClusterOverP.at(idx);
            get_PATele(Br::ele_eSeedClusterOverPout) = tau.ele_eSeedClusterOverPout.at(idx);
            get_PATele(Br::ele_eEleClusterOverPout) = tau.ele_eEleClusterOverPout.at(idx);
            get_PATele(Br::ele_deltaEtaSuperClusterTrackAtVtx) = tau.ele_deltaEtaSuperClusterTrackAtVtx.at(idx);
            get_PATele(Br::ele_deltaEtaSeedClusterTrackAtCalo) = tau.ele_deltaEtaSeedClusterTrackAtCalo.at(idx);
            get_PATele(Br::ele_deltaEtaEleClusterTrackAtCalo) = tau.ele_deltaEtaEleClusterTrackAtCalo.at(idx);
            get_PATele(Br::ele_deltaPhiEleClusterTrackAtCalo) = tau.ele_deltaPhiEleClusterTrackAtCalo.at(idx);
            get_PATele(Br::ele_deltaPhiSuperClusterTrackAtVtx) = tau.ele_deltaPhiSuperClusterTrackAtVtx.at(idx);
            get_PATele(Br::ele_deltaPhiSeedClusterTrackAtCalo) = tau.ele_deltaPhiSeedClusterTrackAtCalo.at(idx);
            get_PATele(Br::ele_mvaInput_earlyBrem) = tau.ele_mvaInput_earlyBrem.at(idx);
            get_PATele(Br::ele_mvaInput_lateBrem) = tau.ele_mvaInput_lateBrem.at(idx);
            get_PATele(Br::ele_mvaInput_sigmaEtaEta) = tau.ele_mvaInput_sigmaEtaEta.at(idx);
            get_PATele(Br::ele_mvaInput_hadEnergy) = tau.ele_mvaInput_hadEnergy.at(idx);
            get_PATele(Br::ele_mvaInput_deltaEta) = tau.ele_mvaInput_deltaEta.at(idx);
            get_PATele(Br::ele_gsfTrack_normalizedChi2) = tau.ele_gsfTrack_normalizedChi2.at(idx);
            get_PATele(Br::ele_gsfTrack_numberOfValidHits) = tau.ele_gsfTrack_numberOfValidHits.at(idx);
            get_PATele(Br::ele_rel_gsfTrack_pt) = tau.ele_gsfTrack_pt.at(idx) / tau.ele_pt.at(idx);
            get_PATele(Br::ele_gsfTrack_pt_sig) = tau.ele_gsfTrack_pt.at(idx) / tau.ele_gsfTrack_pt_error.at(idx);

            const bool has_closestCtfTrack = tau.ele_closestCtfTrack_normalizedChi2.at(idx) >= 0;
            get_PATele(Br::ele_has_closestCtfTrack) = has_closestCtfTrack;

            if(has_closestCtfTrack) {
              get_PATele(Br::ele_closestCtfTrack_normalizedChi2) = tau.ele_closestCtfTrack_normalizedChi2.at(idx);
              get_PATele(Br::ele_closestCtfTrack_numberOfValidHits) = tau.ele_closestCtfTrack_numberOfValidHits.at(idx);
            }
          }
        }

        { // PAT muon
          typedef Muon_Features Br;
          auto get_PATmuon = [&](Muon_Features _fe_n) -> float& {
            if(static_cast<int>(_fe_n) < 0) return foo;
            size_t flat_index = getIndex(n_muon, static_cast<size_t>(_fe_n));
            return data->x_grid.at(CellObjectType::Muon).at(inner).at(flat_index);
          };

            size_t n_muon, idx;
            getBestObj(CellObjectType::Muon, n_muon, idx);
            const bool valid = n_muon != 0;
            get_PATmuon(Br::muon_valid) = valid;

            if(valid) {
              get_PATmuon(Br::muon_rel_pt) = tau.muon_pt.at(idx) / tau.tau_pt;
              get_PATmuon(Br::muon_deta) = tau.muon_eta.at(idx) - tau.tau_eta;
              get_PATmuon(Br::muon_dphi) = DeltaPhi(tau.muon_phi.at(idx), tau.tau_phi);

              get_PATmuon(Br::muon_dxy) = tau.muon_dxy.at(idx);
              get_PATmuon(Br::muon_dxy_sig) = std::abs(tau.muon_dxy.at(idx)) / tau.muon_dxy_error.at(idx);

              const bool normalizedChi2_valid = tau.muon_normalizedChi2.at(idx) >= 0;
              get_PATmuon(Br::muon_normalizedChi2_valid) = normalizedChi2_valid;

              if(normalizedChi2_valid){
                get_PATmuon(Br::muon_normalizedChi2) = normalizedChi2_valid ? tau.muon_normalizedChi2.at(idx) : 0;
                get_PATmuon(Br::muon_numberOfValidHits) = normalizedChi2_valid ? tau.muon_numberOfValidHits.at(idx) : 0;
              }

              get_PATmuon(Br::muon_segmentCompatibility) = tau.muon_segmentCompatibility.at(idx);
              get_PATmuon(Br::muon_caloCompatibility) = tau.muon_caloCompatibility.at(idx);

              const bool pfEcalEnergy_valid = valid && tau.muon_pfEcalEnergy.at(idx) >= 0;
              get_PATmuon(Br::muon_pfEcalEnergy_valid) = pfEcalEnergy_valid;
              if(pfEcalEnergy_valid)
                get_PATmuon(Br::muon_rel_pfEcalEnergy) = tau.muon_pfEcalEnergy.at(idx) / tau.muon_pt.at(idx);

              get_PATmuon(Br::muon_n_matches_DT_1) = tau.muon_n_matches_DT_1.at(idx);
              get_PATmuon(Br::muon_n_matches_DT_2) = tau.muon_n_matches_DT_2.at(idx);
              get_PATmuon(Br::muon_n_matches_DT_3) = tau.muon_n_matches_DT_3.at(idx);
              get_PATmuon(Br::muon_n_matches_DT_4) = tau.muon_n_matches_DT_4.at(idx);
              get_PATmuon(Br::muon_n_matches_CSC_1) = tau.muon_n_matches_CSC_1.at(idx);
              get_PATmuon(Br::muon_n_matches_CSC_2) = tau.muon_n_matches_CSC_2.at(idx);
              get_PATmuon(Br::muon_n_matches_CSC_3) = tau.muon_n_matches_CSC_3.at(idx);
              get_PATmuon(Br::muon_n_matches_CSC_4) = tau.muon_n_matches_CSC_4.at(idx);
              get_PATmuon(Br::muon_n_matches_RPC_1) = tau.muon_n_matches_RPC_1.at(idx);
              get_PATmuon(Br::muon_n_matches_RPC_2) = tau.muon_n_matches_RPC_2.at(idx);
              get_PATmuon(Br::muon_n_matches_RPC_3) = tau.muon_n_matches_RPC_3.at(idx);
              get_PATmuon(Br::muon_n_matches_RPC_4) = tau.muon_n_matches_RPC_4.at(idx);
              get_PATmuon(Br::muon_n_hits_DT_1) = tau.muon_n_hits_DT_1.at(idx);
              get_PATmuon(Br::muon_n_hits_DT_2) = tau.muon_n_hits_DT_2.at(idx);
              get_PATmuon(Br::muon_n_hits_DT_3) = tau.muon_n_hits_DT_3.at(idx);
              get_PATmuon(Br::muon_n_hits_DT_4) = tau.muon_n_hits_DT_4.at(idx);
              get_PATmuon(Br::muon_n_hits_CSC_1) = tau.muon_n_hits_CSC_1.at(idx);
              get_PATmuon(Br::muon_n_hits_CSC_2) = tau.muon_n_hits_CSC_2.at(idx);
              get_PATmuon(Br::muon_n_hits_CSC_3) = tau.muon_n_hits_CSC_3.at(idx);
              get_PATmuon(Br::muon_n_hits_CSC_4) = tau.muon_n_hits_CSC_4.at(idx);
              get_PATmuon(Br::muon_n_hits_RPC_1) = tau.muon_n_hits_RPC_1.at(idx);
              get_PATmuon(Br::muon_n_hits_RPC_2) = tau.muon_n_hits_RPC_2.at(idx);
              get_PATmuon(Br::muon_n_hits_RPC_3) = tau.muon_n_hits_RPC_3.at(idx);
              get_PATmuon(Br::muon_n_hits_RPC_4) = tau.muon_n_hits_RPC_4.at(idx);
            }
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

  Long64_t end_entry;
  Long64_t current_entry; // number of the current entry
  const CellGrid innerCellGridRef, outerCellGridRef;
  const std::vector<std::string> input_files;

  bool hasData;

  // std::shared_ptr<TFile> file; // to open with one file
  std::shared_ptr<TauTuple> tauTuple;
  std::shared_ptr<Data> data;
  std::vector<std::shared_ptr<TH2D>> hist_weights;

};
