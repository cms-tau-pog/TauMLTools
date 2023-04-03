// #include "TauMLTools/Analysis/interface/TauTuple.h"
#include "TauMLTools/Analysis/interface/GenLepton.h"
#include "TauMLTools/Training/interface/DataLoader_tools.h"
#include "TauMLTools/Training/interface/histogram2d.h"

#include "TROOT.h"
#include "TLorentzVector.h"
#include "TMath.h"

// #include "TauMLTools/Analysis/interface/TauSelection.h"
#include "TauMLTools/Analysis/interface/DisTauTagSelection.h"
#include "TauMLTools/Analysis/interface/AnalysisTypes.h"


struct Data {
    template<typename grid> void init_grid() {
        x[grid::object_type].resize(Setup::n_tau * grid::size * grid::length, 0);
    }
    template<typename T, T... I> Data(std::integer_sequence<T, I...> int_seq)
    : y(Setup::n_tau * Setup::output_classes, 0), tau_i(0), 
    uncompress_index(Setup::n_tau, 0), uncompress_size(0),
    weights(Setup::n_tau, 0) //, x_glob(Setup::n_tau * Setup::n_Global, 0)
    {
        ((init_grid<FeaturesHelper<std::tuple_element_t<I, FeatureTuple>>>()),...);
    }
    std::unordered_map<CellObjectType, std::vector<Float_t>> x;
    std::vector<Float_t> y;
    std::vector<Float_t> weights;
    // std::vector<Float_t> x_glob; // will not be scaled

    Long64_t tau_i; // the number of taus filled in the tensor filled_tau <= n_tau;
    std::vector<unsigned long> uncompress_index; // index of the tau when events are dropped;
    Long64_t uncompress_size;
};

// using namespace Setup;

class DataLoader {

public:
    using Tau = tau_tuple::Tau;
    using TauTuple = tau_tuple::TauTuple;
    using LorentzVectorM = ROOT::Math::LorentzVector<ROOT::Math::PtEtaPhiM4D<double>>;
    using JetType = analysis::JetType;
    static constexpr size_t nFeaturesTypes = std::tuple_size_v<FeatureTuple>;

    DataLoader() :
        hasData(false),
        fullData(false),
        hasFile(false)
    { 
        ROOT::EnableThreadSafety();
        if(Setup::n_threads > 1) ROOT::EnableImplicitMT(Setup::n_threads);

        if (Setup::yaxis.size() != (Setup::xaxis_list.size() + 1)){
            throw std::invalid_argument("Y binning list does not match X binning length");
        }

        // auto spectrum_file = std::make_shared<TFile>(Setup::spectrum_to_reweight.c_str());
        auto file_input = std::make_shared<TFile>(Setup::spectrum_to_reweight.c_str());
        auto file_target = std::make_shared<TFile>(Setup::spectrum_to_reweight.c_str());

        Histogram_2D input_histogram ("input" , Setup::yaxis, Setup::xmin, Setup::xmax);
        Histogram_2D target_histogram("target", Setup::yaxis, Setup::xmin, Setup::xmax);

        for (int i = 0; i < Setup::xaxis_list.size(); i++)
        {
            target_histogram.add_x_binning_by_index(i, Setup::xaxis_list[i]);
            input_histogram .add_x_binning_by_index(i, Setup::xaxis_list[i]);
        }

        std::shared_ptr<TH2D> target_th2d = std::shared_ptr<TH2D>(dynamic_cast<TH2D*>(file_target->Get("jet_eta_pt_tau")));
        if (!target_th2d) throw std::runtime_error("Target histogram could not be loaded");

        for( auto const& [tau_type, tau_name] : Setup::jet_types_names)
        {
            std::shared_ptr<TH2D> input_th2d  = std::shared_ptr<TH2D>(dynamic_cast<TH2D*>(file_input ->Get(("jet_eta_pt_"+tau_name).c_str())));
            if (!input_th2d) throw std::runtime_error("Input histogram could not be loaded for jet type "+tau_name);
            target_histogram.th2d_add(*(target_th2d.get()));
            input_histogram .th2d_add(*(input_th2d.get()));

            target_histogram.divide(input_histogram);
            hist_weights[tau_type] = target_histogram.get_weights_th2d(
                ("w_1_"+tau_name).c_str(),
                ("w_1_"+tau_name).c_str()
            );
            if (Setup::debug) hist_weights[tau_type]->SaveAs(("weights_"+tau_name+".root").c_str()); // It's required that all bins are filled in these histograms; save them to check incase binning is too fine and some bins are empty

            target_histogram.reset();
            input_histogram .reset();
        }

        MaxDisbCheck(hist_weights, Setup::weight_thr);
    }

    static void MaxDisbCheck(const std::unordered_map<int ,std::shared_ptr<TH2D>>& hists, Double_t max_thr)
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

    const double GetWeight(const int type_id, const double pt, const double eta) const
    {   
        return hist_weights.at(type_id)->GetBinContent(
                    hist_weights.at(type_id)->GetXaxis()->FindFixBin(eta),
                    hist_weights.at(type_id)->GetYaxis()->FindFixBin(pt)
                );
    }


    DataLoader(const DataLoader&) = delete;
    DataLoader& operator=(const DataLoader&) = delete;

    void ReadFile(std::string file_name,
                  Long64_t start_file,
                  Long64_t end_file) // put end_file=-1 to read all events from file
    { 
        tauTuple.reset();
        file = std::make_unique<TFile>(file_name.c_str());
        tauTuple = std::make_unique<tau_tuple::TauTuple>(file.get(), true);
        current_entry = start_file;
        end_entry = tauTuple->GetEntries();
        if(end_file!=-1) end_entry = std::min(end_file, end_entry);
        hasFile = true;
    } 

    bool MoveNext()
    {
        if(!hasFile)
            throw std::runtime_error("File should be loaded with DataLoaderWorker::ReadFile()");

        if(!tauTuple)
            throw std::runtime_error("TauTuple is not loaded!");

        if(!hasData)
        {
            data = std::make_unique<Data>(std::make_index_sequence<nFeaturesTypes>{});
            data->tau_i = 0;
            data->uncompress_size = 0;
            hasData = true;
        }
        while(data->tau_i < Setup::n_tau)
        {
            if(current_entry == end_entry)
            {
                hasFile = false;
                return false;
            }
            
            tauTuple->GetEntry(current_entry);
            auto& tau = const_cast<Tau&>(tauTuple->data());
            const std::optional<JetType> jet_match_type = 
                Setup::recompute_jet_type ? analysis::GetJetType(tau)
                : static_cast<JetType>(tau.tauType); 

            if (jet_match_type)
            {
                // if(Setup::to_propagate_glob) FillGlob(data->tau_i, tau, jet_match_type);
                data->y.at(data->tau_i * Setup::output_classes + static_cast<Int_t>(*jet_match_type)) = 1.0;
                data->weights.at(data->tau_i) = GetWeight(static_cast<Int_t>(*jet_match_type), tau.jet_pt, std::abs(tau.jet_eta));
                FillPfCand(data->tau_i, tau);
                data->uncompress_index[data->tau_i] = data->uncompress_size;
                ++(data->tau_i);
            }
            else if ( tau.jet_index >= 0 && Setup::include_mismatched ) {
                // if(Setup::to_propagate_glob) FillGlob(data->tau_i, tau, jet_match_type);
                FillPfCand(data->tau_i, tau);
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
    
    const Data* LoadData(bool needFull)
    {
      if(!fullData && needFull)
        throw std::runtime_error("Data was not loaded with MoveNext() or array was not fully filled");
        fullData = false;
        hasData = false;
        return data.get();
    }

  private:

    template <typename FeatureT>
    const float Scale(const Int_t idx, const Float_t value, const bool inner)
    {
        return std::clamp((value - FeatureT::mean.at(idx).at(inner)) / FeatureT::std.at(idx).at(inner),
                            FeatureT::lim_min.at(idx).at(inner), FeatureT::lim_max.at(idx).at(inner));
    }

    static constexpr Float_t pi = boost::math::constants::pi<Float_t>();

    template <typename Scalar>
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

    // void FillGlob(const Long64_t tau_i, const Tau& tau,
    //             const boost::optional<JetType> jet_match_type)
    // {
    //     auto getGlobVecRef = [&](auto _fe, Float_t value){
    //             size_t _feature_idx = static_cast<size_t>(_fe);
    //             if(_feature_idx < 0) return;
    //             const size_t index = Setup::n_Global * tau_i + _feature_idx;
    //             data->x_glob.at(index) = value;
    //         };

    //     typedef Global_Features Br;

    //     getGlobVecRef(Br::jet_pt, tau.jet_pt);
    //     getGlobVecRef(Br::jet_eta, tau.jet_eta);

        
    //     getGlobVecRef(Br::jet_index, tau.jet_index);
    //     getGlobVecRef(Br::run, tau.run);
    //     getGlobVecRef(Br::lumi, tau.lumi);
    //     getGlobVecRef(Br::evt, tau.evt);
        
    //     // Only signal features
    //     getGlobVecRef(Br::Lxy, -1);
    //     getGlobVecRef(Br::Lz, -1);
    //     getGlobVecRef(Br::Lrel, -1);
        
    //     if(jet_match_type)
    //     {
    //         if(jet_match_type == JetType::tau)
    //         {
    //             reco_tau::gen_truth::GenLepton genLeptons = 
    //                 reco_tau::gen_truth::GenLepton::fromRootTuple(
    //                         tau.genLepton_lastMotherIndex,
    //                         tau.genParticle_pdgId,
    //                         tau.genParticle_mother,
    //                         tau.genParticle_charge,
    //                         tau.genParticle_isFirstCopy,
    //                         tau.genParticle_isLastCopy,
    //                         tau.genParticle_pt,
    //                         tau.genParticle_eta,
    //                         tau.genParticle_phi,
    //                         tau.genParticle_mass,
    //                         tau.genParticle_vtx_x,
    //                         tau.genParticle_vtx_y,
    //                         tau.genParticle_vtx_z);

    //             auto vertex = genLeptons.lastCopy().vertex;
    //             if( std::abs(genLeptons.lastCopy().pdgId) != 15 )
    //                 throw std::runtime_error("Error FillGlob: last copy of genLeptons is not tau.");
    //             // get the displacement wrt to the primary vertex
    //             auto Lrel = genLeptons.lastCopy().getDisplacement();

    //             getGlobVecRef(Br::Lrel, Lrel);
    //             getGlobVecRef(Br::Lxy, std::abs(vertex.rho()));
    //             getGlobVecRef(Br::Lz, std::abs(vertex.z()));
    //         }
    //     }
    // }

    void FillPfCand(const Long64_t tau_i,
                    const Tau& tau)
    {
        
        std::vector<size_t> indices(tau.pfCand_pt.size());
        std::iota(indices.begin(), indices.end(), 0);

        // sort PfCands by pt
        std::sort(indices.begin(), indices.end(), [&](size_t a, size_t b) {
            return tau.pfCand_pt.at(a) > tau.pfCand_pt.at(b);
        });
        
        // size_t upper_index = std::min(Setup::nSeq_PfCand, indices.size());
        size_t index_size = indices.size();

        size_t pf_data_index = 0;
        size_t pf_input_index = 0;

        // for(size_t pfCand_i = 0; pfCand_i < upper_index; pfCand_i++)
        // {
        while( pf_data_index < Setup::nSeq_PfCand && pf_input_index < index_size)
        {

            size_t idx_srt = indices.at(pf_input_index);
            ++pf_input_index;

            if(!tau.pfCand_jetDaughter.at(idx_srt)) continue;
            
            auto getVecRef = [&](auto _fe, Float_t value){
                size_t _feature_idx = static_cast<size_t>(_fe);
                if(_feature_idx < 0) return;
                const size_t start_index =  FeaturesHelper<decltype(_fe)>::length * FeaturesHelper<decltype(_fe)>::size * tau_i;
                const size_t index = start_index + FeaturesHelper<decltype(_fe)>::size * pf_data_index + _feature_idx;
                data->x[FeaturesHelper<decltype(_fe)>::object_type].at(index) = 
                    Scale<typename  FeaturesHelper<decltype(_fe)>::scaler_type>(
                        _feature_idx, value, false);
            };

            {   // Categorical features
                typedef PfCandCategorical_Features Br;
                getVecRef(Br::pfCand_particleType         ,tau.pfCand_particleType.at(idx_srt));
                getVecRef(Br::pfCand_pvAssociationQuality ,tau.pfCand_pvAssociationQuality.at(idx_srt));
                getVecRef(Br::pfCand_fromPV               ,tau.pfCand_fromPV.at(idx_srt));
            }

            {   // General features
                typedef PfCand_Features Br;
                getVecRef(Br::pfCand_valid                ,1.0);
                getVecRef(Br::pfCand_pt                   ,tau.pfCand_pt.at(idx_srt));
                getVecRef(Br::pfCand_eta                  ,tau.pfCand_eta.at(idx_srt));
                getVecRef(Br::pfCand_phi                  ,tau.pfCand_phi.at(idx_srt));
                getVecRef(Br::pfCand_mass                 ,tau.pfCand_mass.at(idx_srt));
                getVecRef(Br::pfCand_charge               ,tau.pfCand_charge.at(idx_srt));
                getVecRef(Br::pfCand_puppiWeight          ,tau.pfCand_puppiWeight.at(idx_srt));
                getVecRef(Br::pfCand_puppiWeightNoLep     ,tau.pfCand_puppiWeightNoLep.at(idx_srt));
                getVecRef(Br::pfCand_lostInnerHits        ,tau.pfCand_lostInnerHits.at(idx_srt));
                getVecRef(Br::pfCand_nPixelHits           ,tau.pfCand_nPixelHits.at(idx_srt));
                getVecRef(Br::pfCand_nHits                ,tau.pfCand_nHits.at(idx_srt));
                getVecRef(Br::pfCand_hasTrackDetails      ,tau.pfCand_hasTrackDetails.at(idx_srt));
                getVecRef(Br::pfCand_caloFraction         ,tau.pfCand_caloFraction.at(idx_srt));
                getVecRef(Br::pfCand_hcalFraction         ,tau.pfCand_hcalFraction.at(idx_srt));
                getVecRef(Br::pfCand_rawCaloFraction      ,tau.pfCand_rawCaloFraction.at(idx_srt));
                getVecRef(Br::pfCand_rawHcalFraction      ,tau.pfCand_rawHcalFraction.at(idx_srt));
                
                if( tau.pfCand_hasTrackDetails.at(idx_srt) )
                {   
                    getVecRef(Br::pfCand_hasTrackDetails,tau.pfCand_hasTrackDetails.at(idx_srt));
                    
                    if(std::isfinite(tau.pfCand_dz.at(idx_srt))) getVecRef(Br::pfCand_dz, tau.pfCand_dz.at(idx_srt));
                    if(std::isfinite(tau.pfCand_dz_error.at(idx_srt))) getVecRef(Br::pfCand_dz_error, tau.pfCand_dz_error.at(idx_srt));
                    if(std::isfinite(tau.pfCand_dxy_error.at(idx_srt))) getVecRef(Br::pfCand_dxy_error, tau.pfCand_dxy_error.at(idx_srt));

                    getVecRef(Br::pfCand_dxy, tau.pfCand_dxy.at(idx_srt));
                    getVecRef(Br::pfCand_track_chi2, tau.pfCand_track_chi2.at(idx_srt));
                    getVecRef(Br::pfCand_track_ndof, tau.pfCand_track_ndof.at(idx_srt));
                }
                
                if(tau.jet_index>=0)
                {
                    Float_t jet_eta = tau.jet_eta;
                    Float_t jet_phi = tau.jet_phi;
                    // getVecRef(PfCand_Features::jet_eta)        = jet_eta;
                    // getVecRef(PfCand_Features::jet_phi)        = jet_phi;
                    getVecRef(PfCand_Features::pfCand_deta, tau.pfCand_eta.at(idx_srt) - jet_eta);
                    getVecRef(PfCand_Features::pfCand_dphi, DeltaPhi<Float_t>(tau.pfCand_phi.at(idx_srt), jet_phi));
                }
            }

            ++pf_data_index;
        }
    }

private:

  Long64_t end_entry;
  Long64_t current_entry; // number of the current entry in the file
  Long64_t current_tau; // number of the current tau candidate

  bool hasData;
  bool fullData;
  bool hasFile;

  std::unique_ptr<TFile> file; // to open with one file
  std::unique_ptr<TauTuple> tauTuple;
  std::unique_ptr<Data> data;
  std::unordered_map<int ,std::shared_ptr<TH2D>> hist_weights;

};
