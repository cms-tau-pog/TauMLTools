#include "TauMLTools/Training/interface/DataLoader_tools.h"
#include "TauMLTools/Training/interface/histogram2d.h"

#include "TROOT.h"
#include "TLorentzVector.h"

#include "TauMLTools/Analysis/interface/TauSelection.h"
#include "TauMLTools/Analysis/interface/AnalysisTypes.h"
#include "TauMLTools/Core/interface/RootExt.h"

template <typename T, typename Tuple>
struct ElementIndex;

template <typename T, typename... Args>
struct ElementIndex<T, std::tuple<T, Args...>> {
    static constexpr std::size_t value = 0;
};

template <typename T, typename U, typename... Args>
struct ElementIndex<T, std::tuple<U, Args...>> {
    static constexpr std::size_t value = 1 + ElementIndex<T, std::tuple<Args...>>::value;
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
        if(nCellsEta < 1) //(nCellsEta % 2 != 1 || nCellsEta < 1)
            throw std::invalid_argument("Invalid number of eta cells.");
        if(nCellsPhi < 1) //(nCellsPhi % 2 != 1 || nCellsPhi < 1)
            throw std::invalid_argument("Invalid number of phi cells.");
        if(cellSizeEta <= 0 || cellSizePhi <= 0)
            throw std::invalid_argument("Invalid cell size.");
    }

    int MaxEtaIndex() const { return static_cast<int>(nCellsEta / 2); } //{ return static_cast<int>((nCellsEta - 1) / 2); }
    int MaxPhiIndex() const { return static_cast<int>(nCellsPhi / 2); } //{ return static_cast<int>((nCellsPhi - 1) / 2); }
    double MaxDeltaEta() const { return cellSizeEta * (0.5 * nCellsEta); } //{ return cellSizeEta * (0.5 + MaxEtaIndex()); }
    double MaxDeltaPhi() const { return cellSizePhi * (0.5 * nCellsPhi); } //{ return cellSizePhi * (0.5 + MaxPhiIndex()); }
    bool evengridsize() const { return nCellsEta % 2 != 1 || nCellsPhi % 2 != 1; }

    bool TryGetCellIndex(double deltaEta, double deltaPhi, CellIndex& cellIndex) const
    {
        static auto getCellIndex = [](double x, double maxX, double size, int& index, bool evengridsize) {
            const double absX = std::abs(x);
            if(absX > maxX) return false;
            if(evengridsize){
                const double absIndex = std::floor(x / size);
                index = static_cast<int>(absIndex);
            }else{
                const double absIndex = std::floor(absX / size + 0.5);
                index = static_cast<int>(std::copysign(absIndex, x));
            }
            return true;
        };

        return getCellIndex(deltaEta, MaxDeltaEta(), cellSizeEta, cellIndex.eta, evengridsize())
               && getCellIndex(deltaPhi, MaxDeltaPhi(), cellSizePhi, cellIndex.phi, evengridsize());
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
         size_t n_outer_cells, size_t globalgrid_fn, size_t pfelectron_fn, size_t pfmuon_fn,
         size_t pfchargedhad_fn, size_t pfneutralhad_fn, size_t pfgamma_fn,
         size_t electron_fn, size_t muon_fn, size_t tau_labels) :
         tau_i(0), x_tau(n_tau * tau_fn, 0), weight(n_tau, 0), y_onehot(n_tau * tau_labels, 0),
         uncompress_index(n_tau, 0), uncompress_size(0)
         {
           x_grid[CellObjectType::GridGlobal][0].resize(n_tau * n_outer_cells * n_outer_cells * globalgrid_fn,0);
           x_grid[CellObjectType::GridGlobal][1].resize(n_tau * n_inner_cells * n_inner_cells * globalgrid_fn,0);
           // pf electron
           x_grid[CellObjectType::PfCand_electron][0].resize(n_tau * n_outer_cells * n_outer_cells * pfelectron_fn,0);
           x_grid[CellObjectType::PfCand_electron][1].resize(n_tau * n_inner_cells * n_inner_cells * pfelectron_fn,0);
           // pf muons
           x_grid[CellObjectType::PfCand_muon][0].resize(n_tau * n_outer_cells * n_outer_cells * pfmuon_fn,0);
           x_grid[CellObjectType::PfCand_muon][1].resize(n_tau * n_inner_cells * n_inner_cells * pfmuon_fn,0);
           // pf charged hadrons
           x_grid[CellObjectType::PfCand_chHad][0].resize(n_tau * n_outer_cells * n_outer_cells * pfchargedhad_fn,0);
           x_grid[CellObjectType::PfCand_chHad][1].resize(n_tau * n_inner_cells * n_inner_cells * pfchargedhad_fn,0);
           // pf neutral hadrons
           x_grid[CellObjectType::PfCand_nHad][0].resize(n_tau * n_outer_cells * n_outer_cells * pfneutralhad_fn,0);
           x_grid[CellObjectType::PfCand_nHad][1].resize(n_tau * n_inner_cells * n_inner_cells * pfneutralhad_fn,0);
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
    Long64_t tau_i; // the number of taus filled in the tensor filled_tau <= n_tau;
    std::vector<unsigned long> uncompress_index; // index of the tau when events are dropped;
    Long64_t uncompress_size;

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
    using LorentzVectorM = ROOT::Math::LorentzVector<ROOT::Math::PtEtaPhiM4D<double>>;

    DataLoader() :
        // current_entry(start_dataset),
        innerCellGridRef(n_inner_cells, n_inner_cells, inner_cell_size, inner_cell_size),
        outerCellGridRef(n_outer_cells, n_outer_cells, outer_cell_size, outer_cell_size),
        hasData(false), fullData(false), hasFile(false)
    {
      ROOT::EnableThreadSafety();
      if(n_threads > 1) ROOT::EnableImplicitMT(n_threads);
      if (yaxis.size() != (xaxis_list.size() + 1)){
        throw std::invalid_argument("Y binning list does not match X binning length");
      }

      // file = OpenRootFile(file_name);
      // tauTuple = std::make_shared<tau_tuple::TauTuple>(file.get(), true);

      // std::cout << "Number of files to process: " << input_files.size() << std::endl;
      // tauTuple = std::make_shared<TauTuple>("taus", input_files);
      // end_entry = std::min((long long)end_dataset, tauTuple->GetEntries());

      // histogram to calculate weights

      auto file_input = std::make_shared<TFile>(input_spectrum.c_str());
      auto file_target = std::make_shared<TFile>(target_spectrum.c_str());

      Histogram_2D target_histogram("target", yaxis, xmin, xmax);
      Histogram_2D input_histogram ("input" , yaxis, xmin, xmax);
      for (int i = 0; i < xaxis_list.size(); i++){
          target_histogram.add_x_binning_by_index(i, xaxis_list[i]);
          input_histogram .add_x_binning_by_index(i, xaxis_list[i]);
      }

      std::shared_ptr<TH2D> target_th2d = std::shared_ptr<TH2D>(dynamic_cast<TH2D*>(file_target->Get("eta_pt_hist_tau")));
      if (!target_th2d) throw std::runtime_error("Target histogram could not be loaded");

      for( auto const& [tau_type, tau_name] : tau_types_names)
      {
        std::shared_ptr<TH2D> input_th2d  = std::shared_ptr<TH2D>(dynamic_cast<TH2D*>(file_input ->Get(("eta_pt_hist_"+tau_name).c_str())));
        if (!input_th2d) throw std::runtime_error("Input histogram could not be loaded for tau type "+tau_name);
        target_histogram.th2d_add(*(target_th2d.get()));
        input_histogram .th2d_add(*(input_th2d .get()));

        target_histogram.divide(input_histogram);
        hist_weights[tau_type] = target_histogram.get_weights_th2d(
            ("w_1_"+tau_name).c_str(),
            ("w_1_"+tau_name).c_str()
        );
        if (debug) hist_weights[tau_type]->SaveAs(("Temp_"+tau_name+".root").c_str()); // It's required that all bins are filled in these histograms; save them to check incase binning is too fine and some bins are empty

        target_histogram.reset();
        input_histogram .reset();
      }
      MaxDisbCheck(hist_weights, weight_thr);
    }

    DataLoader(const DataLoader&) = delete;
    DataLoader& operator=(const DataLoader&) = delete;

    void ReadFile(std::string file_name, Long64_t start_file, Long64_t end_file) { // put end_file=-1 to read all events from file
        tauTuple.reset();
        file = std::make_unique<TFile>(file_name.c_str());
        tauTuple = std::make_unique<tau_tuple::TauTuple>(file.get(), true);
        current_entry = start_file;
        end_entry = tauTuple->GetEntries();
        if(end_file!=-1) end_entry = std::min(end_file, end_entry);
        hasFile = true;
    }

    bool MoveNext() {
        if(!hasFile)
          throw std::runtime_error("File should be loaded with DataLoaderWorker::ReadFile()");

        if(!tauTuple)
          throw std::runtime_error("TauTuple is not loaded!");

        if(!hasData) {
          data = std::make_unique<Data>(n_tau, n_TauFlat, n_inner_cells, n_outer_cells, n_GridGlobal,
                                        n_PfCand_electron, n_PfCand_muon, n_PfCand_chHad, n_PfCand_nHad,
                                        n_PfCand_gamma, n_Electron, n_Muon, tau_types_names.size()
                                        );
          data->tau_i = 0;
          data->uncompress_size = 0;
          hasData = true;
        }
        while(data->tau_i < n_tau) {
          if(current_entry == end_entry) {
            hasFile = false;
            return false;
          }
          tauTuple->GetEntry(current_entry);
          auto& tau = const_cast<tau_tuple::Tau&>(tauTuple->data());

          const auto gen_match = analysis::GetGenLeptonMatch(static_cast<reco_tau::gen_truth::GenLepton::Kind>(tau.genLepton_kind), tau.genLepton_index, tau.tau_pt, tau.tau_eta, tau.tau_phi,
                                                            tau.tau_mass, tau.genLepton_vis_pt, tau.genLepton_vis_eta, tau.genLepton_vis_phi, 
                                                            tau.genLepton_vis_mass, tau.genJet_index);
          const auto sample_type = static_cast<analysis::SampleType>(tau.sampleType);

          if (gen_match && tau.tau_byDeepTau2017v2p1VSjetraw > DeepTauVSjet_cut) {
            if (recompute_tautype){
              tau.tauType = static_cast<Int_t> (GenMatchToTauType(*gen_match, sample_type));
            }
            // skip event if it is not tau_e, tau_mu, tau_jet or tau_h
            if ( tau_types_names.find(tau.tauType) != tau_types_names.end() ) {
              data->y_onehot[ data->tau_i * tau_types_names.size() + tau.tauType ] = 1.0; // filling labels
              data->weight.at(data->tau_i) = GetWeight(tau.tauType, tau.tau_pt, std::abs(tau.tau_eta)); // filling weights
              FillTauBranches(tau, data->tau_i);
              FillCellGrid(tau, data->tau_i, innerCellGridRef, true);
              FillCellGrid(tau, data->tau_i, outerCellGridRef, false);
              data->uncompress_index[data->tau_i] = data->uncompress_size;
              ++(data->tau_i);
            }
          } else if (!gen_match && include_mismatched && tau.tau_index >= 0) {
              FillTauBranches(tau, data->tau_i);
              FillCellGrid(tau, data->tau_i, innerCellGridRef, true);
              FillCellGrid(tau, data->tau_i, outerCellGridRef, false);
              data->uncompress_index[data->tau_i] = data->uncompress_size;
              ++(data->tau_i);
          }
          ++(data->uncompress_size);
          ++current_entry;
        }
        fullData = true;
        return true;
    }

    bool hasAnyData() {return hasData;}

    const Data* LoadData(bool needFull) {
      if(!fullData && needFull)
        throw std::runtime_error("Data was not loaded with MoveNext() or array was not fully filled");
      fullData = false;
      hasData = false;
      return data.get();
    }


    static void MaxDisbCheck(const std::unordered_map<int ,std::shared_ptr<TH2D>>& hists,
                             Double_t max_thr)
    {
      double min_weight = std::numeric_limits<double>::max();
      double max_weight = std::numeric_limits<double>::lowest();
      for(auto const& [tau_type, hist_] : hists) {
        min_weight = std::min(hist_->GetMinimum(), min_weight);
        max_weight = std::max(hist_->GetMaximum(), max_weight);
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
               hist_weights.at(type_id)->GetXaxis()->FindFixBin(eta),
               hist_weights.at(type_id)->GetYaxis()->FindFixBin(pt));

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

      template <typename FeatureT>
      const float Scale(const int idx, const float value, const bool inner)
      {
        return std::clamp((value - FeatureT::mean.at(idx).at(inner)) / FeatureT::std.at(idx).at(inner),
                          FeatureT::lim_min.at(idx).at(inner), FeatureT::lim_max.at(idx).at(inner));
      }

      void FillTauBranches(const Tau& tau, Long64_t tau_i)
      {
        Long64_t start_array_index = tau_i * n_TauFlat;

        // Filling Tau Branch
        auto fill_tau = [&](TauFlat_Features _fe, float value) -> void {
            if(static_cast<int>(_fe) < 0) return;
            size_t _fe_ind = static_cast<size_t>(_fe);
            size_t index = start_array_index + _fe_ind;
            data->x_tau.at(index) = Scale<Scaling::TauFlat>(_fe_ind, value, false);
        };

        fill_tau(TauFlat_Features::rho, tau.rho);
        fill_tau(TauFlat_Features::tau_pt, tau.tau_pt);
        fill_tau(TauFlat_Features::tau_eta, tau.tau_eta);
        fill_tau(TauFlat_Features::tau_phi, tau.tau_phi);
        fill_tau(TauFlat_Features::tau_mass, tau.tau_mass);

        const LorentzVectorM tau_p4(tau.tau_pt, tau.tau_eta, tau.tau_phi, tau.tau_mass);
        fill_tau(TauFlat_Features::tau_E_over_pt, tau_p4.energy() / tau.tau_pt);
        fill_tau(TauFlat_Features::tau_charge, tau.tau_charge);
        fill_tau(TauFlat_Features::tau_n_charged_prongs, tau.tau_decayMode / 5);
        fill_tau(TauFlat_Features::tau_n_neutral_prongs, tau.tau_decayMode % 5);
        fill_tau(TauFlat_Features::tau_chargedIsoPtSum, tau.tau_chargedIsoPtSum);
        if(tau.tau_chargedIsoPtSum!=0)
          fill_tau(TauFlat_Features::tau_chargedIsoPtSumdR03_over_dR05, tau.tau_chargedIsoPtSumdR03 / tau.tau_chargedIsoPtSum);
        fill_tau(TauFlat_Features::tau_footprintCorrection, tau.tau_footprintCorrection);
        fill_tau(TauFlat_Features::tau_neutralIsoPtSum, tau.tau_neutralIsoPtSum);
        if(tau.tau_neutralIsoPtSum!=0) {
          fill_tau(TauFlat_Features::tau_neutralIsoPtSumWeight_over_neutralIsoPtSum, tau.tau_neutralIsoPtSumWeight / tau.tau_neutralIsoPtSum);
          fill_tau(TauFlat_Features::tau_neutralIsoPtSumWeightdR03_over_neutralIsoPtSum,tau.tau_neutralIsoPtSumWeightdR03 / tau.tau_neutralIsoPtSum);
          fill_tau(TauFlat_Features::tau_neutralIsoPtSumdR03_over_dR05, tau.tau_neutralIsoPtSumdR03 / tau.tau_neutralIsoPtSum);
        }
        fill_tau(TauFlat_Features::tau_photonPtSumOutsideSignalCone, tau.tau_photonPtSumOutsideSignalCone);
        fill_tau(TauFlat_Features::tau_puCorrPtSum, tau.tau_puCorrPtSum);

        const bool tau_dxy_valid = std::isnormal(tau.tau_dxy) && tau.tau_dxy > - 10
                                   && std::isnormal(tau.tau_dxy_error) && tau.tau_dxy_error > 0;
        fill_tau(TauFlat_Features::tau_dxy_valid, static_cast<float>(tau_dxy_valid));
        if(tau_dxy_valid) {
          fill_tau(TauFlat_Features::tau_dxy, tau.tau_dxy);
          fill_tau(TauFlat_Features::tau_dxy_sig, std::abs(tau.tau_dxy)/tau.tau_dxy_error);
        }

        const bool tau_ip3d_valid = std::isnormal(tau.tau_ip3d) && tau.tau_ip3d > - 10
                                    && std::isnormal(tau.tau_ip3d_error) && tau.tau_ip3d_error > 0;
        fill_tau(TauFlat_Features::tau_ip3d_valid, static_cast<float>(tau_ip3d_valid));
        if(tau_ip3d_valid) {
          fill_tau(TauFlat_Features::tau_ip3d, tau.tau_ip3d);
          fill_tau(TauFlat_Features::tau_ip3d_sig, std::abs(tau.tau_ip3d) / tau.tau_ip3d_error);
        }
        fill_tau(TauFlat_Features::tau_dz, tau.tau_dz);

        const bool tau_dz_sig_valid = std::isnormal(tau.tau_dz) && std::isnormal(tau.tau_dz_error)
                                      && tau.tau_dz_error > 0;
        fill_tau(TauFlat_Features::tau_dz_sig_valid, tau_dz_sig_valid);
        if(tau_dz_sig_valid)
          fill_tau(TauFlat_Features::tau_dz_sig, std::abs(tau.tau_dz) / tau.tau_dz_error);

        fill_tau(TauFlat_Features::tau_flightLength_x, tau.tau_flightLength_x);
        fill_tau(TauFlat_Features::tau_flightLength_y, tau.tau_flightLength_y);
        fill_tau(TauFlat_Features::tau_flightLength_z, tau.tau_flightLength_z);
        fill_tau(TauFlat_Features::tau_flightLength_sig, tau.tau_flightLength_sig);

        fill_tau(TauFlat_Features::tau_pt_weighted_deta_strip, tau.tau_pt_weighted_deta_strip);
        fill_tau(TauFlat_Features::tau_pt_weighted_dphi_strip, tau.tau_pt_weighted_dphi_strip);
        fill_tau(TauFlat_Features::tau_pt_weighted_dr_signal, tau.tau_pt_weighted_dr_signal);
        fill_tau(TauFlat_Features::tau_pt_weighted_dr_iso, tau.tau_pt_weighted_dr_iso);

        fill_tau(TauFlat_Features::tau_leadingTrackNormChi2, tau.tau_leadingTrackNormChi2);
        const bool tau_e_ratio_valid = std::isnormal(tau.tau_e_ratio) && tau.tau_e_ratio > 0.f;
        fill_tau(TauFlat_Features::tau_e_ratio_valid, static_cast<float>(tau_e_ratio_valid));
        if(tau_e_ratio_valid)
          fill_tau(TauFlat_Features::tau_e_ratio, tau.tau_e_ratio);

        const bool tau_gj_angle_diff_valid = (std::isnormal(tau.tau_gj_angle_diff) || tau.tau_gj_angle_diff == 0)
            && tau.tau_gj_angle_diff >= 0;
        fill_tau(TauFlat_Features::tau_gj_angle_diff_valid, static_cast<float>(tau_gj_angle_diff_valid));

        if(tau_gj_angle_diff_valid)
          fill_tau(TauFlat_Features::tau_gj_angle_diff, tau.tau_gj_angle_diff);

        fill_tau(TauFlat_Features::tau_n_photons, tau.tau_n_photons);
        fill_tau(TauFlat_Features::tau_emFraction, tau.tau_emFraction);
        fill_tau(TauFlat_Features::tau_inside_ecal_crack, tau.tau_inside_ecal_crack);
        fill_tau(TauFlat_Features::tau_leadChargedCand_etaAtEcalEntrance_minus_tau_eta, tau.tau_leadChargedCand_etaAtEcalEntrance - tau.tau_eta);

      }

      void FillCellGrid(const Tau& tau, Long64_t tau_i,  const CellGrid& cellGridRef, bool inner)
      {
          auto cellGrid = CreateCellGrid(tau, cellGridRef, inner);
          const int max_eta_index = cellGrid.MaxEtaIndex(), max_phi_index = cellGrid.MaxPhiIndex();
          const int max_distance = max_eta_index + max_phi_index;
          std::set<CellIndex> processed_cells;
          for(int distance = 0; distance <= max_distance; ++distance) {
              const int max_eta_d = std::min(max_eta_index, distance);
              for(int eta_index = -max_eta_d; eta_index <= max_eta_d; ++eta_index) {
                  if(cellGrid.evengridsize() && eta_index == cellGrid.MaxEtaIndex()) continue;
                  const int max_phi_d = distance - std::abs(eta_index);
                  if(max_phi_d > max_phi_index) continue;
                  const size_t n_max = max_phi_d ? 2 : 1;
                  for(size_t n = 0; n < n_max; ++n) {
                      int phi_index = n ? max_phi_d : -max_phi_d;
                      if(cellGrid.evengridsize() && phi_index == cellGrid.MaxPhiIndex()) continue;
                      const CellIndex cellIndex{eta_index, phi_index};
                      if(processed_cells.count(cellIndex))
                          throw std::runtime_error("Duplicated cell index in FillCellGrid.");
                      processed_cells.insert(cellIndex);
                      if(!cellGrid.IsEmpty(cellIndex))
                          FillCellBranches(tau, tau_i, cellGridRef, cellIndex, cellGrid.at(cellIndex), inner);
                  }
              }
          }
          if( (processed_cells.size() != static_cast<size_t>( (2 * max_eta_index + 1) * (2 * max_phi_index + 1) )) && (cellGrid.evengridsize() && processed_cells.size() != static_cast<size_t>( (2 * max_eta_index) * (2 * max_phi_index) )))
              throw std::runtime_error("Not all cell indices are processed in FillCellGrid.");
      }

      template<size_t... I>
      std::vector<size_t> CreateStartIndices(const CellGrid& cellGridRef, const CellIndex& cellIndex, size_t tau_i, std::index_sequence<I...> idx_seq)
      {
         auto getStartIndex = [&](size_t n_total) {
              return tau_i * cellGridRef.GetnTotal() * n_total
                     + cellGridRef.GetFlatIndex(cellIndex) * n_total;
              };
        std::vector<size_t> start(idx_seq.size());
        ((start[I] = getStartIndex(FeaturesHelper<std::tuple_element_t<I, FeatureTuple>>::size)), ...);
        return start;
      }

      void FillCellBranches(const Tau& tau, Long64_t tau_i,  const CellGrid& cellGridRef, const CellIndex& cellIndex,
                            Cell& cell, bool inner)
      {
        static constexpr size_t nFeaturesTypes = std::tuple_size_v<FeatureTuple>;
        const auto start_indices = CreateStartIndices(cellGridRef, cellIndex, tau_i,
                                    std::make_index_sequence<nFeaturesTypes>{});

        auto fillGrid = [&](auto _feature_idx, float value) {
          if(static_cast<int>(_feature_idx) < 0) return;
          const CellObjectType obj_type = FeaturesHelper<decltype(_feature_idx)>::object_type;
          const size_t start = start_indices.at(ElementIndex<decltype(_feature_idx), FeatureTuple>::value);
          data->x_grid.at(obj_type).at(inner).at(start + static_cast<int>(_feature_idx))
                  = Scale<typename  FeaturesHelper<decltype(_feature_idx)>::scaler_type>(static_cast<int> (_feature_idx), value, inner);
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
        { // CellObjectType::GridGlobal
            typedef GridGlobal_Features Br;
            fillGrid(Br::rho, tau.rho);
            fillGrid(Br::tau_pt, tau.tau_pt);
            fillGrid(Br::tau_eta, tau.tau_eta);
            fillGrid(Br::tau_inside_ecal_crack, tau.tau_inside_ecal_crack);
        }

        { // CellObjectType::PfCand_electron

            typedef PfCand_electron_Features Br;

            size_t n_pfCand, pfCand_idx;
            getBestObj(CellObjectType::PfCand_electron, n_pfCand, pfCand_idx);

            const bool valid = n_pfCand != 0;

            fillGrid(Br::pfCand_ele_valid, static_cast<float>(valid));
            if(valid) {
              fillGrid(Br::pfCand_ele_rel_pt, tau.pfCand_pt.at(pfCand_idx) / tau.tau_pt);
              fillGrid(Br::pfCand_ele_deta, tau.pfCand_eta.at(pfCand_idx) - tau.tau_eta);
              fillGrid(Br::pfCand_ele_dphi, DeltaPhi(tau.pfCand_phi.at(pfCand_idx), tau.tau_phi));
              fillGrid(Br::pfCand_ele_pvAssociationQuality, tau.pfCand_pvAssociationQuality.at(pfCand_idx));
              fillGrid(Br::pfCand_ele_puppiWeight, tau.pfCand_puppiWeight.at(pfCand_idx));
              fillGrid(Br::pfCand_ele_charge, tau.pfCand_charge.at(pfCand_idx));
              fillGrid(Br::pfCand_ele_lostInnerHits, tau.pfCand_lostInnerHits.at(pfCand_idx));
              fillGrid(Br::pfCand_ele_nPixelHits, tau.pfCand_nPixelHits.at(pfCand_idx));

              fillGrid(Br::pfCand_ele_vertex_dx, tau.pfCand_vertex_x.at(pfCand_idx) - tau.pv_x);
              fillGrid(Br::pfCand_ele_vertex_dy, tau.pfCand_vertex_y.at(pfCand_idx) - tau.pv_y);
              fillGrid(Br::pfCand_ele_vertex_dz, tau.pfCand_vertex_z.at(pfCand_idx) - tau.pv_z);
              fillGrid(Br::pfCand_ele_vertex_dt, tau.pfCand_vertex_t.at(pfCand_idx) - tau.pv_t);
              fillGrid(Br::pfCand_ele_vertex_dx_tauFL, tau.pfCand_vertex_x.at(pfCand_idx) - tau.pv_x - tau.tau_flightLength_x);
              fillGrid(Br::pfCand_ele_vertex_dy_tauFL, tau.pfCand_vertex_y.at(pfCand_idx) - tau.pv_y - tau.tau_flightLength_y);
              fillGrid(Br::pfCand_ele_vertex_dz_tauFL, tau.pfCand_vertex_z.at(pfCand_idx) - tau.pv_z - tau.tau_flightLength_z);
            }

            const bool hasTrackDetails = valid && tau.pfCand_hasTrackDetails.at(pfCand_idx) == 1;
            fillGrid(Br::pfCand_ele_hasTrackDetails, static_cast<float>(hasTrackDetails));

            if(hasTrackDetails) {
              fillGrid(Br::pfCand_ele_dxy, tau.pfCand_dxy.at(pfCand_idx));
              if(tau.pfCand_dxy_error.at(pfCand_idx)!=0)
                fillGrid(Br::pfCand_ele_dxy_sig, std::abs(tau.pfCand_dxy.at(pfCand_idx)) / tau.pfCand_dxy_error.at(pfCand_idx));
              fillGrid(Br::pfCand_ele_dz, tau.pfCand_dz.at(pfCand_idx));
              if(tau.pfCand_dz_error.at(pfCand_idx)!=0)
                fillGrid(Br::pfCand_ele_dz_sig, std::abs(tau.pfCand_dz.at(pfCand_idx)) / tau.pfCand_dz_error.at(pfCand_idx));
              fillGrid(Br::pfCand_ele_time, tau.pfCand_time.at(pfCand_idx));
              if(tau.pfCand_timeError.at(pfCand_idx)!=0)
                fillGrid(Br::pfCand_ele_time_sig, std::abs(tau.pfCand_time.at(pfCand_idx)) / tau.pfCand_timeError.at(pfCand_idx));

              if(tau.pfCand_track_ndof.at(pfCand_idx) > 0) {
                fillGrid(Br::pfCand_ele_track_chi2_ndof, tau.pfCand_track_chi2.at(pfCand_idx) / tau.pfCand_track_ndof.at(pfCand_idx));
                fillGrid(Br::pfCand_ele_track_ndof, tau.pfCand_track_ndof.at(pfCand_idx));
              }
            }
        }

        { // CellObjectType::PfCand_muon

            typedef PfCand_muon_Features Br;

            size_t n_pfCand, pfCand_idx;
            getBestObj(CellObjectType::PfCand_muon, n_pfCand, pfCand_idx);

            const bool valid = n_pfCand != 0;
            fillGrid(Br::pfCand_muon_valid, static_cast<float>(valid));

            if(valid){
              fillGrid(Br::pfCand_muon_rel_pt, tau.pfCand_pt.at(pfCand_idx) / tau.tau_pt);
              fillGrid(Br::pfCand_muon_deta, tau.pfCand_eta.at(pfCand_idx) - tau.tau_eta);
              fillGrid(Br::pfCand_muon_dphi, DeltaPhi(tau.pfCand_phi.at(pfCand_idx), tau.tau_phi));
              fillGrid(Br::pfCand_muon_pvAssociationQuality, tau.pfCand_pvAssociationQuality.at(pfCand_idx));
              fillGrid(Br::pfCand_muon_fromPV, tau.pfCand_fromPV.at(pfCand_idx));
              fillGrid(Br::pfCand_muon_puppiWeight, tau.pfCand_puppiWeight.at(pfCand_idx));
              fillGrid(Br::pfCand_muon_charge, tau.pfCand_charge.at(pfCand_idx));
              fillGrid(Br::pfCand_muon_lostInnerHits, tau.pfCand_lostInnerHits.at(pfCand_idx));
              fillGrid(Br::pfCand_muon_nPixelHits, tau.pfCand_nPixelHits.at(pfCand_idx));

              fillGrid(Br::pfCand_muon_vertex_dx,  tau.pfCand_vertex_x.at(pfCand_idx) - tau.pv_x);
              fillGrid(Br::pfCand_muon_vertex_dy, tau.pfCand_vertex_y.at(pfCand_idx) - tau.pv_y);
              fillGrid(Br::pfCand_muon_vertex_dz, tau.pfCand_vertex_z.at(pfCand_idx) - tau.pv_z);
              fillGrid(Br::pfCand_muon_vertex_dt, tau.pfCand_vertex_t.at(pfCand_idx) - tau.pv_t);
              fillGrid(Br::pfCand_muon_vertex_dx_tauFL, tau.pfCand_vertex_x.at(pfCand_idx) - tau.pv_x - tau.tau_flightLength_x);
              fillGrid(Br::pfCand_muon_vertex_dy_tauFL, tau.pfCand_vertex_y.at(pfCand_idx) - tau.pv_y - tau.tau_flightLength_y);
              fillGrid(Br::pfCand_muon_vertex_dz_tauFL, tau.pfCand_vertex_z.at(pfCand_idx) - tau.pv_z - tau.tau_flightLength_z);

              const bool hasTrackDetails = valid && tau.pfCand_hasTrackDetails.at(pfCand_idx) == 1;
              fillGrid(Br::pfCand_muon_hasTrackDetails, static_cast<float>(hasTrackDetails));

              if(hasTrackDetails){

              fillGrid(Br::pfCand_muon_dxy, tau.pfCand_dxy.at(pfCand_idx));
              if(tau.pfCand_dxy_error.at(pfCand_idx)!=0)
                fillGrid(Br::pfCand_muon_dxy_sig, std::abs(tau.pfCand_dxy.at(pfCand_idx)) / tau.pfCand_dxy_error.at(pfCand_idx));
              fillGrid(Br::pfCand_muon_dz, tau.pfCand_dz.at(pfCand_idx));
              if(tau.pfCand_dz_error.at(pfCand_idx)!=0)
                fillGrid(Br::pfCand_muon_dz_sig, std::abs(tau.pfCand_dz.at(pfCand_idx)) / tau.pfCand_dz_error.at(pfCand_idx));
              fillGrid(Br::pfCand_muon_time, tau.pfCand_time.at(pfCand_idx));
              if(tau.pfCand_timeError.at(pfCand_idx)!=0)
                fillGrid(Br::pfCand_muon_time_sig, std::abs(tau.pfCand_time.at(pfCand_idx)) / tau.pfCand_timeError.at(pfCand_idx));

              if(tau.pfCand_track_ndof.at(pfCand_idx) > 0) {
                fillGrid(Br::pfCand_muon_track_chi2_ndof, tau.pfCand_track_chi2.at(pfCand_idx) / tau.pfCand_track_ndof.at(pfCand_idx));
                fillGrid(Br::pfCand_muon_track_ndof, tau.pfCand_track_ndof.at(pfCand_idx));
              }
              }
            }
        }

        { // CellObjectType::PfCand_chHad

          typedef PfCand_chHad_Features Br;

          size_t n_pfCand, pfCand_idx;
          getBestObj(CellObjectType::PfCand_chHad, n_pfCand, pfCand_idx);
          const bool valid = n_pfCand != 0;
          fillGrid(Br::pfCand_chHad_valid, static_cast<float>(valid));

          if(valid) {
            fillGrid(Br::pfCand_chHad_rel_pt, tau.pfCand_pt.at(pfCand_idx) / tau.tau_pt);
            fillGrid(Br::pfCand_chHad_deta, tau.pfCand_eta.at(pfCand_idx) - tau.tau_eta );
            fillGrid(Br::pfCand_chHad_dphi, DeltaPhi(tau.pfCand_phi.at(pfCand_idx), tau.tau_phi));
            fillGrid(Br::pfCand_chHad_tauLeadChargedHadrCand, tau.pfCand_tauLeadChargedHadrCand.at(pfCand_idx));
            fillGrid(Br::pfCand_chHad_pvAssociationQuality, tau.pfCand_pvAssociationQuality.at(pfCand_idx));
            fillGrid(Br::pfCand_chHad_fromPV, tau.pfCand_fromPV.at(pfCand_idx));
            fillGrid(Br::pfCand_chHad_puppiWeight, tau.pfCand_puppiWeight.at(pfCand_idx));
            fillGrid(Br::pfCand_chHad_puppiWeightNoLep, tau.pfCand_puppiWeightNoLep.at(pfCand_idx));
            fillGrid(Br::pfCand_chHad_charge, tau.pfCand_charge.at(pfCand_idx));
            fillGrid(Br::pfCand_chHad_lostInnerHits, tau.pfCand_lostInnerHits.at(pfCand_idx));
            fillGrid(Br::pfCand_chHad_nPixelHits, tau.pfCand_nPixelHits.at(pfCand_idx));

            fillGrid(Br::pfCand_chHad_vertex_dx, tau.pfCand_vertex_x.at(pfCand_idx) - tau.pv_x);
            fillGrid(Br::pfCand_chHad_vertex_dy, tau.pfCand_vertex_y.at(pfCand_idx) - tau.pv_y);
            if(std::isfinite(tau.pfCand_vertex_z.at(pfCand_idx) - tau.pv_z))
              fillGrid(Br::pfCand_chHad_vertex_dz, tau.pfCand_vertex_z.at(pfCand_idx) - tau.pv_z);
              fillGrid(Br::pfCand_chHad_vertex_dt, tau.pfCand_vertex_t.at(pfCand_idx) - tau.pv_t);
            fillGrid(Br::pfCand_chHad_vertex_dx_tauFL, tau.pfCand_vertex_x.at(pfCand_idx) - tau.pv_x - tau.tau_flightLength_x);
            fillGrid(Br::pfCand_chHad_vertex_dy_tauFL,  tau.pfCand_vertex_y.at(pfCand_idx) - tau.pv_y - tau.tau_flightLength_y);
            if(std::isfinite(tau.pfCand_vertex_z.at(pfCand_idx) - tau.pv_z - tau.tau_flightLength_z))
              fillGrid(Br::pfCand_chHad_vertex_dz_tauFL, tau.pfCand_vertex_z.at(pfCand_idx) - tau.pv_z - tau.tau_flightLength_z);

            const bool hasTrackDetails = tau.pfCand_hasTrackDetails.at(pfCand_idx) == 1;
            fillGrid(Br::pfCand_chHad_hasTrackDetails, static_cast<float>(hasTrackDetails));
            if(hasTrackDetails) {
              fillGrid(Br::pfCand_chHad_dxy, tau.pfCand_dxy.at(pfCand_idx));
              if(tau.pfCand_dxy_error.at(pfCand_idx)!=0)
                fillGrid(Br::pfCand_chHad_dxy_sig, std::abs(tau.pfCand_dxy.at(pfCand_idx)) / tau.pfCand_dxy_error.at(pfCand_idx));
              if(std::isfinite(tau.pfCand_dz.at(pfCand_idx))){
                fillGrid(Br::pfCand_chHad_dz, tau.pfCand_dz.at(pfCand_idx));
                if(tau.pfCand_dz_error.at(pfCand_idx)!=0)
                  fillGrid(Br::pfCand_chHad_dz_sig, std::abs(tau.pfCand_dz.at(pfCand_idx)) / tau.pfCand_dz_error.at(pfCand_idx));
                fillGrid(Br::pfCand_chHad_time, tau.pfCand_time.at(pfCand_idx));
                if(tau.pfCand_timeError.at(pfCand_idx)!=0)
                  fillGrid(Br::pfCand_chHad_time_sig, std::abs(tau.pfCand_time.at(pfCand_idx)) / tau.pfCand_timeError.at(pfCand_idx));
              }
              if(tau.pfCand_track_ndof.at(pfCand_idx)>0){
                fillGrid(Br::pfCand_chHad_track_chi2_ndof, tau.pfCand_track_chi2.at(pfCand_idx) / tau.pfCand_track_ndof.at(pfCand_idx));
                fillGrid(Br::pfCand_chHad_track_ndof, tau.pfCand_track_ndof.at(pfCand_idx));
              }
            }

            fillGrid(Br::pfCand_chHad_hcalFraction, tau.pfCand_hcalFraction.at(pfCand_idx));
            fillGrid(Br::pfCand_chHad_rawCaloFraction, tau.pfCand_rawCaloFraction.at(pfCand_idx));
          }
        }

        { // CellObjectType::PfCand_nHad

          typedef PfCand_nHad_Features Br;

          size_t n_pfCand, pfCand_idx;
          getBestObj(CellObjectType::PfCand_nHad, n_pfCand, pfCand_idx);
          const bool valid = n_pfCand != 0;
          fillGrid(Br::pfCand_nHad_valid, static_cast<float>(valid));

          if(valid) {
            fillGrid(Br::pfCand_nHad_rel_pt, tau.pfCand_pt.at(pfCand_idx) / tau.tau_pt);
            fillGrid(Br::pfCand_nHad_deta, tau.pfCand_eta.at(pfCand_idx) - tau.tau_eta);
            fillGrid(Br::pfCand_nHad_dphi, DeltaPhi(tau.pfCand_phi.at(pfCand_idx), tau.tau_phi));
            fillGrid(Br::pfCand_nHad_puppiWeight, tau.pfCand_puppiWeight.at(pfCand_idx));
            fillGrid(Br::pfCand_nHad_puppiWeightNoLep, tau.pfCand_puppiWeightNoLep.at(pfCand_idx));
            fillGrid(Br::pfCand_nHad_hcalFraction, tau.pfCand_hcalFraction.at(pfCand_idx));
          }
        }

        { // CellObjectType::PfCand_gamma

          typedef PfCand_gamma_Features Br;

          size_t n_pfCand, pfCand_idx;
          getBestObj(CellObjectType::PfCand_gamma, n_pfCand, pfCand_idx);
          const bool valid = n_pfCand != 0;
          fillGrid(Br::pfCand_gamma_valid, valid);

          if(valid) {
            fillGrid(Br::pfCand_gamma_rel_pt, tau.pfCand_pt.at(pfCand_idx) / tau.tau_pt);
            fillGrid(Br::pfCand_gamma_deta, tau.pfCand_eta.at(pfCand_idx) - tau.tau_eta);
            fillGrid(Br::pfCand_gamma_dphi, DeltaPhi(tau.pfCand_phi.at(pfCand_idx), tau.tau_phi));
            fillGrid(Br::pfCand_gamma_pvAssociationQuality, tau.pfCand_pvAssociationQuality.at(pfCand_idx));
            fillGrid(Br::pfCand_gamma_fromPV, tau.pfCand_fromPV.at(pfCand_idx));
            fillGrid(Br::pfCand_gamma_puppiWeight, tau.pfCand_puppiWeight.at(pfCand_idx));
            fillGrid(Br::pfCand_gamma_puppiWeightNoLep, tau.pfCand_puppiWeightNoLep.at(pfCand_idx));
            fillGrid(Br::pfCand_gamma_lostInnerHits, tau.pfCand_lostInnerHits.at(pfCand_idx));
            fillGrid(Br::pfCand_gamma_nPixelHits, tau.pfCand_nPixelHits.at(pfCand_idx));

            fillGrid(Br::pfCand_gamma_vertex_dx, tau.pfCand_vertex_x.at(pfCand_idx) - tau.pv_x);
            fillGrid(Br::pfCand_gamma_vertex_dy, tau.pfCand_vertex_y.at(pfCand_idx) - tau.pv_y);
            fillGrid(Br::pfCand_gamma_vertex_dz, tau.pfCand_vertex_z.at(pfCand_idx) - tau.pv_z);
            fillGrid(Br::pfCand_gamma_vertex_dt, tau.pfCand_vertex_t.at(pfCand_idx) - tau.pv_t);
            fillGrid(Br::pfCand_gamma_vertex_dx_tauFL, tau.pfCand_vertex_x.at(pfCand_idx) - tau.pv_x -
                                                            tau.tau_flightLength_x);
            fillGrid(Br::pfCand_gamma_vertex_dy_tauFL, tau.pfCand_vertex_y.at(pfCand_idx) - tau.pv_y -
                                                            tau.tau_flightLength_y);
            fillGrid(Br::pfCand_gamma_vertex_dz_tauFL, tau.pfCand_vertex_z.at(pfCand_idx) - tau.pv_z -
                                                            tau.tau_flightLength_z);

            const bool hasTrackDetails = tau.pfCand_hasTrackDetails.at(pfCand_idx) == 1;
            fillGrid(Br::pfCand_gamma_hasTrackDetails, static_cast<float>(hasTrackDetails));

            if(hasTrackDetails){
              fillGrid(Br::pfCand_gamma_dxy, tau.pfCand_dxy.at(pfCand_idx));
              if(tau.pfCand_dxy_error.at(pfCand_idx)!=0)
                fillGrid(Br::pfCand_gamma_dxy_sig, std::abs(tau.pfCand_dxy.at(pfCand_idx)) / tau.pfCand_dxy_error.at(pfCand_idx));
              fillGrid(Br::pfCand_gamma_dz, tau.pfCand_dz.at(pfCand_idx));
              if(tau.pfCand_dz_error.at(pfCand_idx)!=0)
                fillGrid(Br::pfCand_gamma_dz_sig, std::abs(tau.pfCand_dz.at(pfCand_idx)) / tau.pfCand_dz_error.at(pfCand_idx));
              fillGrid(Br::pfCand_gamma_time, tau.pfCand_time.at(pfCand_idx));
              if(tau.pfCand_timeError.at(pfCand_idx)!=0)
                fillGrid(Br::pfCand_gamma_time_sig, std::abs(tau.pfCand_time.at(pfCand_idx)) / tau.pfCand_timeError.at(pfCand_idx));
              if(tau.pfCand_track_ndof.at(pfCand_idx) > 0) {
                fillGrid(Br::pfCand_gamma_track_chi2_ndof, tau.pfCand_track_chi2.at(pfCand_idx) / tau.pfCand_track_ndof.at(pfCand_idx));
                fillGrid(Br::pfCand_gamma_track_ndof, tau.pfCand_track_ndof.at(pfCand_idx));
              }
            }
          }
        }

        { // PAT electron

          typedef Electron_Features Br;

          size_t n_particles, idx;
          getBestObj(CellObjectType::Electron, n_particles, idx);
          const bool valid = n_particles != 0;

          fillGrid(Br::ele_valid, static_cast<float>(valid));

          if(valid) {
            fillGrid(Br::ele_rel_pt, tau.ele_pt.at(idx) / tau.tau_pt);
            fillGrid(Br::ele_deta, tau.ele_eta.at(idx) - tau.tau_eta);
            fillGrid(Br::ele_dphi, DeltaPhi(tau.ele_phi.at(idx), tau.tau_phi));

            const bool cc_valid = tau.ele_cc_ele_energy.at(idx) >= 0;
            fillGrid(Br::ele_cc_valid, static_cast<float>(cc_valid));

            if(cc_valid) {
              fillGrid(Br::ele_cc_ele_rel_energy, tau.ele_cc_ele_energy.at(idx) / tau.ele_pt.at(idx));
              fillGrid(Br::ele_cc_gamma_rel_energy, tau.ele_cc_gamma_energy.at(idx) /
                                                        tau.ele_cc_ele_energy.at(idx));
              fillGrid(Br::ele_cc_n_gamma, tau.ele_cc_n_gamma.at(idx));
            }
            fillGrid(Br::ele_rel_trackMomentumAtVtx, tau.ele_trackMomentumAtVtx.at(idx) / tau.ele_pt.at(idx));
            fillGrid(Br::ele_rel_trackMomentumAtCalo, tau.ele_trackMomentumAtCalo.at(idx) / tau.ele_pt.at(idx));
            fillGrid(Br::ele_rel_trackMomentumOut, tau.ele_trackMomentumOut.at(idx) / tau.ele_pt.at(idx));
            fillGrid(Br::ele_rel_trackMomentumAtEleClus, tau.ele_trackMomentumAtEleClus.at(idx) / tau.ele_pt.at(idx));
            fillGrid(Br::ele_rel_trackMomentumAtVtxWithConstraint, tau.ele_trackMomentumAtVtxWithConstraint.at(idx) / tau.ele_pt.at(idx));
            fillGrid(Br::ele_rel_ecalEnergy, tau.ele_ecalEnergy.at(idx) / tau.ele_pt.at(idx));
            fillGrid(Br::ele_ecalEnergy_sig, tau.ele_ecalEnergy.at(idx) / tau.ele_ecalEnergy_error.at(idx));
            fillGrid(Br::ele_eSuperClusterOverP, tau.ele_eSuperClusterOverP.at(idx));
            fillGrid(Br::ele_eSeedClusterOverP, tau.ele_eSeedClusterOverP.at(idx));
            fillGrid(Br::ele_eSeedClusterOverPout, tau.ele_eSeedClusterOverPout.at(idx));
            fillGrid(Br::ele_eEleClusterOverPout, tau.ele_eEleClusterOverPout.at(idx));
            fillGrid(Br::ele_deltaEtaSuperClusterTrackAtVtx, tau.ele_deltaEtaSuperClusterTrackAtVtx.at(idx));
            fillGrid(Br::ele_deltaEtaSeedClusterTrackAtCalo, tau.ele_deltaEtaSeedClusterTrackAtCalo.at(idx));
            fillGrid(Br::ele_deltaEtaEleClusterTrackAtCalo, tau.ele_deltaEtaEleClusterTrackAtCalo.at(idx));
            fillGrid(Br::ele_deltaPhiEleClusterTrackAtCalo, tau.ele_deltaPhiEleClusterTrackAtCalo.at(idx));
            fillGrid(Br::ele_deltaPhiSuperClusterTrackAtVtx, tau.ele_deltaPhiSuperClusterTrackAtVtx.at(idx));
            fillGrid(Br::ele_deltaPhiSeedClusterTrackAtCalo, tau.ele_deltaPhiSeedClusterTrackAtCalo.at(idx));
            fillGrid(Br::ele_mvaInput_earlyBrem, tau.ele_mvaInput_earlyBrem.at(idx));
            fillGrid(Br::ele_mvaInput_lateBrem, tau.ele_mvaInput_lateBrem.at(idx));
            fillGrid(Br::ele_mvaInput_sigmaEtaEta, tau.ele_mvaInput_sigmaEtaEta.at(idx));
            fillGrid(Br::ele_mvaInput_hadEnergy, tau.ele_mvaInput_hadEnergy.at(idx));
            fillGrid(Br::ele_mvaInput_deltaEta, tau.ele_mvaInput_deltaEta.at(idx));
            fillGrid(Br::ele_gsfTrack_normalizedChi2, tau.ele_gsfTrack_normalizedChi2.at(idx));
            fillGrid(Br::ele_gsfTrack_numberOfValidHits, tau.ele_gsfTrack_numberOfValidHits.at(idx));
            fillGrid(Br::ele_rel_gsfTrack_pt, tau.ele_gsfTrack_pt.at(idx) / tau.ele_pt.at(idx));
            fillGrid(Br::ele_gsfTrack_pt_sig, tau.ele_gsfTrack_pt.at(idx) / tau.ele_gsfTrack_pt_error.at(idx));

            const bool has_closestCtfTrack = tau.ele_closestCtfTrack_normalizedChi2.at(idx) >= 0;
            fillGrid(Br::ele_has_closestCtfTrack, static_cast<float>(has_closestCtfTrack));

            if(has_closestCtfTrack) {
              fillGrid(Br::ele_closestCtfTrack_normalizedChi2, tau.ele_closestCtfTrack_normalizedChi2.at(idx));
              fillGrid(Br::ele_closestCtfTrack_numberOfValidHits, tau.ele_closestCtfTrack_numberOfValidHits.at(idx));
            }
          }
        }

        { // PAT muon

          typedef Muon_Features Br;

          size_t n_particles, idx;
          getBestObj(CellObjectType::Muon, n_particles, idx);
          const bool valid = n_particles != 0;
          fillGrid(Br::muon_valid, static_cast<float>(valid));

          if(valid) {
            fillGrid(Br::muon_rel_pt, tau.muon_pt.at(idx) / tau.tau_pt);
            fillGrid(Br::muon_deta, tau.muon_eta.at(idx) - tau.tau_eta);
            fillGrid(Br::muon_dphi, DeltaPhi(tau.muon_phi.at(idx), tau.tau_phi));

            fillGrid(Br::muon_dxy, tau.muon_dxy.at(idx));
            if(std::isnormal(tau.muon_dxy_error.at(idx)) && std::isnormal(tau.muon_dxy.at(idx))) 
              fillGrid(Br::muon_dxy_sig, std::abs(tau.muon_dxy.at(idx)) / tau.muon_dxy_error.at(idx));

            const bool normalizedChi2_valid = tau.muon_normalizedChi2.at(idx) >= 0;
            fillGrid(Br::muon_normalizedChi2_valid, static_cast<float>(normalizedChi2_valid));

            if(normalizedChi2_valid){
              if(std::isfinite(tau.muon_normalizedChi2.at(idx)))
                fillGrid(Br::muon_normalizedChi2, tau.muon_normalizedChi2.at(idx));
              fillGrid(Br::muon_numberOfValidHits, tau.muon_numberOfValidHits.at(idx));
            }

            fillGrid(Br::muon_segmentCompatibility, tau.muon_segmentCompatibility.at(idx));
            fillGrid(Br::muon_caloCompatibility, tau.muon_caloCompatibility.at(idx));

            const bool pfEcalEnergy_valid = valid && tau.muon_pfEcalEnergy.at(idx) >= 0;
            fillGrid(Br::muon_pfEcalEnergy_valid, static_cast<float>(pfEcalEnergy_valid));
            if(pfEcalEnergy_valid)
              fillGrid(Br::muon_rel_pfEcalEnergy, tau.muon_pfEcalEnergy.at(idx) / tau.muon_pt.at(idx));

            fillGrid(Br::muon_n_matches_DT_1, tau.muon_n_matches_DT_1.at(idx));
            fillGrid(Br::muon_n_matches_DT_2, tau.muon_n_matches_DT_2.at(idx));
            fillGrid(Br::muon_n_matches_DT_3, tau.muon_n_matches_DT_3.at(idx));
            fillGrid(Br::muon_n_matches_DT_4, tau.muon_n_matches_DT_4.at(idx));
            fillGrid(Br::muon_n_matches_CSC_1, tau.muon_n_matches_CSC_1.at(idx));
            fillGrid(Br::muon_n_matches_CSC_2, tau.muon_n_matches_CSC_2.at(idx));
            fillGrid(Br::muon_n_matches_CSC_3, tau.muon_n_matches_CSC_3.at(idx));
            fillGrid(Br::muon_n_matches_CSC_4, tau.muon_n_matches_CSC_4.at(idx));
            fillGrid(Br::muon_n_matches_RPC_1, tau.muon_n_matches_RPC_1.at(idx));
            fillGrid(Br::muon_n_matches_RPC_2, tau.muon_n_matches_RPC_2.at(idx));
            fillGrid(Br::muon_n_matches_RPC_3, tau.muon_n_matches_RPC_3.at(idx));
            fillGrid(Br::muon_n_matches_RPC_4, tau.muon_n_matches_RPC_4.at(idx));
            fillGrid(Br::muon_n_hits_DT_1, tau.muon_n_hits_DT_1.at(idx));
            fillGrid(Br::muon_n_hits_DT_2, tau.muon_n_hits_DT_2.at(idx));
            fillGrid(Br::muon_n_hits_DT_3, tau.muon_n_hits_DT_3.at(idx));
            fillGrid(Br::muon_n_hits_DT_4, tau.muon_n_hits_DT_4.at(idx));
            fillGrid(Br::muon_n_hits_CSC_1, tau.muon_n_hits_CSC_1.at(idx));
            fillGrid(Br::muon_n_hits_CSC_2, tau.muon_n_hits_CSC_2.at(idx));
            fillGrid(Br::muon_n_hits_CSC_3, tau.muon_n_hits_CSC_3.at(idx));
            fillGrid(Br::muon_n_hits_CSC_4, tau.muon_n_hits_CSC_4.at(idx));
            fillGrid(Br::muon_n_hits_RPC_1, tau.muon_n_hits_RPC_1.at(idx));
            fillGrid(Br::muon_n_hits_RPC_2, tau.muon_n_hits_RPC_2.at(idx));
            fillGrid(Br::muon_n_hits_RPC_3, tau.muon_n_hits_RPC_3.at(idx));
            fillGrid(Br::muon_n_hits_RPC_4, tau.muon_n_hits_RPC_4.at(idx));
          }
        }
      }

      static double getInnerSignalConeRadius(double pt)
      {
          static constexpr double min_pt = 30., min_radius = 0.05, cone_opening_coef = 3.;
          // This is equivalent of the original formula (std::max(std::min(0.1, 3.0/pt), 0.05)
          return std::max(cone_opening_coef / std::max(pt, min_pt), min_radius);
      }

      static bool isSameCellObjectType(int particleType, CellObjectType type)
      {
          static const std::set<int> other_types = {0, 6, 7};

          static const std::map<int, CellObjectType> obj_types = {
              { 2, CellObjectType::PfCand_electron },
              { 3, CellObjectType::PfCand_muon },
              { 4, CellObjectType::PfCand_gamma },
              { 5, CellObjectType::PfCand_nHad },
              { 1, CellObjectType::PfCand_chHad }
          };

          if(other_types.find(particleType) != other_types.end()) return false;
          auto iter = obj_types.find(particleType);
          if(iter == obj_types.end())
              throw std::runtime_error("Unknown object of particleType = "+std::to_string(particleType));
          return iter->second==type;
      }

      CellGrid CreateCellGrid(const Tau& tau, const CellGrid& cellGridRef, bool inner) const
      {

          CellGrid grid = cellGridRef;
          const double tau_pt = tau.tau_pt, tau_eta = tau.tau_eta, tau_phi = tau.tau_phi;

          const auto fillCells = [&](CellObjectType type, auto eta_vec,
                                    auto phi_vec, auto particleType) {
              if(eta_vec.size() != phi_vec.size())
                  throw std::runtime_error("Inconsistent cell inputs.");
              for(size_t n = 0; n < eta_vec.size(); ++n) {
                  if(particleType.size() && !isSameCellObjectType(particleType.at(n),type)) continue;
                  const double eta = eta_vec.at(n), phi = phi_vec.at(n);
                  const double deta = eta - tau_eta, dphi = DeltaPhi(phi, tau_phi);
                  const double dR = std::hypot(deta, dphi);
                  const bool inside_signal_cone = dR < getInnerSignalConeRadius(tau_pt);
                  const bool inside_iso_cone = dR < iso_cone;
                  const bool accept_inner = inner && inside_signal_cone;
                  const bool accept_outer = !inner && inside_iso_cone && (!rm_inner_from_outer || !inside_signal_cone);
                  if(!(accept_inner || accept_outer)) continue;
                  CellIndex cellIndex;
                  if(grid.TryGetCellIndex(deta, dphi, cellIndex))
                      grid.at(cellIndex)[type].insert(n);
              }
          };

          fillCells(CellObjectType::PfCand_electron, tau.pfCand_eta, tau.pfCand_phi, tau.pfCand_particleType);
          fillCells(CellObjectType::PfCand_muon, tau.pfCand_eta, tau.pfCand_phi, tau.pfCand_particleType);
          fillCells(CellObjectType::PfCand_chHad, tau.pfCand_eta, tau.pfCand_phi, tau.pfCand_particleType);
          fillCells(CellObjectType::PfCand_nHad, tau.pfCand_eta, tau.pfCand_phi, tau.pfCand_particleType);
          fillCells(CellObjectType::PfCand_gamma, tau.pfCand_eta, tau.pfCand_phi, tau.pfCand_particleType);
          fillCells(CellObjectType::Electron, tau.ele_eta, tau.ele_phi, std::vector<int>());
          fillCells(CellObjectType::Muon, tau.muon_eta, tau.muon_phi, std::vector<int>());

          return grid;
      }

private:

  Long64_t end_entry;
  Long64_t current_entry; // number of the current entry in the file
  Long64_t current_tau; // number of the current tau candidate
  const CellGrid innerCellGridRef, outerCellGridRef;
  // const std::vector<std::string> input_files;

  bool hasData;
  bool fullData;
  bool hasFile;

  std::unique_ptr<TFile> file; // to open with one file
  std::unique_ptr<TauTuple> tauTuple;
  std::unique_ptr<Data> data;
  std::unordered_map<int ,std::shared_ptr<TH2D>> hist_weights;

};
