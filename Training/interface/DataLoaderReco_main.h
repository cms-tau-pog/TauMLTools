#include "TauMLTools/Analysis/interface/TauTuple.h"
#include "TauMLTools/Analysis/interface/GenLepton.h"
#include "TauMLTools/Training/interface/DataLoader_tools.h"

#include "TROOT.h"
#include "TLorentzVector.h"
#include "TMath.h"

#include "TauMLTools/Analysis/interface/TauSelection.h"
#include "TauMLTools/Analysis/interface/AnalysisTypes.h"


struct Data {
    template<typename grid> void init_grid() {
        x[grid::object_type].resize(Setup::n_tau * grid::size * grid::length,0);
    }
    template<typename T, T... I> Data(std::integer_sequence<T, I...> int_seq)
    : y(Setup::n_tau * Setup::output_classes, 0) {
        ((init_grid<FeaturesHelper<std::tuple_element_t<I, FeatureTuple>>>()),...);
    }
    std::unordered_map<CellObjectType, std::vector<float>> x;
    std::vector<float> y;
};

// using namespace Setup;

class DataLoader {

public:
    using Tau = tau_tuple::Tau;
    using TauTuple = tau_tuple::TauTuple;
    using LorentzVectorM = ROOT::Math::LorentzVector<ROOT::Math::PtEtaPhiM4D<double>>;
    static constexpr size_t nFeaturesTypes = std::tuple_size_v<FeatureTuple>;

    DataLoader() :
        hasData(false),
        fullData(false),
        hasFile(false)
    { 
        ROOT::EnableThreadSafety();
        if(Setup::n_threads > 1) ROOT::EnableImplicitMT(Setup::n_threads);
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
            tau_i = 0;
            hasData = true;
        }
        while(tau_i < Setup::n_tau)
        {
            if(current_entry == end_entry) {
                hasFile = false;
                return false;
            }
            
            tauTuple->GetEntry(current_entry);
            auto& tau = const_cast<Tau&>(tauTuple->data());
            if (tau.genLepton_kind == 5 && tau.jet_index >= 0 && tau.genLepton_vis_pt > 15.0) {
                FillLabels(tau_i, tau, Setup::output_classes);
                FillPfCand(tau_i, tau);
                ++tau_i;
            }
            ++current_entry;
        }
        fullData = true;
        return true;
    }

    const Data* LoadData() {
        if(!fullData)
            throw std::runtime_error("Data was not loaded with MoveNext()");
        fullData = false;
        hasData = false;
        return data.get();
    }


  private:

      template <typename FeatureT>
      const float Scale(const int idx, const float value, const bool inner)
      {
        return std::clamp((value - FeatureT::mean.at(idx).at(inner)) / FeatureT::std.at(idx).at(inner),
                          FeatureT::lim_min.at(idx).at(inner), FeatureT::lim_max.at(idx).at(inner));
      }

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

      void FillLabels(size_t tau_i,
                      Tau& tau,
                      size_t n_classes)
      {
        auto genLeptons = reco_tau::gen_truth::GenLepton::fromRootTuple(
                            tau.genLepton_lastMotherIndex,
                            tau.genParticle_pdgId,
                            tau.genParticle_mother,
                            tau.genParticle_charge,
                            tau.genParticle_isFirstCopy,
                            tau.genParticle_isLastCopy,
                            tau.genParticle_pt,
                            tau.genParticle_eta,
                            tau.genParticle_phi,
                            tau.genParticle_mass,
                            tau.genParticle_vtx_x,
                            tau.genParticle_vtx_y,
                            tau.genParticle_vtx_z);
        auto getVecRef = [&](size_t element) ->  Float_t&{
            return data->y.at(tau_i*n_classes + element);
            };
        getVecRef(0) = genLeptons.nChargedHadrons();
        getVecRef(1) = genLeptons.nNeutralHadrons();
        getVecRef(2) = genLeptons.visibleP4().Pt();
        getVecRef(3) = TMath::Power(genLeptons.visibleP4().M(),2);
      }

      void FillPfCand(Long64_t tau_i,
                      Tau& tau)
      {
          
        std::vector<size_t> indices(tau.pfCand_pt.size());
        std::iota(indices.begin(), indices.end(), 0);

        // sort PfCands by pt
        std::sort(indices.begin(), indices.end(), [&](size_t a, size_t b) {
            return tau.pfCand_pt.at(a) > tau.pfCand_pt.at(b);
        });
        
        size_t upper_index = std::min(Setup::nSeq_PfCand, indices.size());
        for(size_t pfCand_i = 0; pfCand_i < upper_index; pfCand_i++)
        {
            auto getVecRef = [&](auto _fe, Float_t value){
                size_t _feature_idx = static_cast<size_t>(_fe);
                if(_feature_idx < 0) return;
                const size_t start_index =  FeaturesHelper<decltype(_fe)>::length * FeaturesHelper<decltype(_fe)>::size * tau_i;
                const size_t index = start_index + FeaturesHelper<decltype(_fe)>::size * pfCand_i + _feature_idx;
                data->x[FeaturesHelper<decltype(_fe)>::object_type].at(index) = 
                    Scale<typename  FeaturesHelper<decltype(_fe)>::scaler_type>(
                        _feature_idx, value, false);
            };

            size_t idx_srt = indices.at(pfCand_i);

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

                TLorentzVector v;
                v.SetPtEtaPhiM(
                    tau.pfCand_pt.at(idx_srt),
                    tau.pfCand_eta.at(idx_srt),
                    tau.pfCand_phi.at(idx_srt),
                    tau.pfCand_mass.at(idx_srt)
                );

                getVecRef(Br::pfCand_px, v.Px());
                getVecRef(Br::pfCand_py, v.Py());
                getVecRef(Br::pfCand_pz, v.Pz());
                getVecRef(Br::pfCand_E, v.E());
                
                if( tau.pfCand_hasTrackDetails.at(idx_srt) )
                {   
                    getVecRef(Br::pfCand_hasTrackDetails,tau.pfCand_hasTrackDetails.at(idx_srt));
                    
                    if(std::isnormal(tau.pfCand_dz.at(idx_srt)))
                        getVecRef(Br::pfCand_dz        ,tau.pfCand_dz.at(idx_srt));
                    if(std::isnormal( tau.pfCand_dz_error.at(idx_srt)))
                        getVecRef(Br::pfCand_dz_error  ,tau.pfCand_dz_error.at(idx_srt));
                    if(std::isnormal(tau.pfCand_dxy_error.at(idx_srt)))
                        getVecRef(Br::pfCand_dxy_error ,tau.pfCand_dxy_error.at(idx_srt));

                    getVecRef(Br::pfCand_dxy           ,tau.pfCand_dxy.at(idx_srt));
                    getVecRef(Br::pfCand_track_chi2    ,tau.pfCand_track_chi2.at(idx_srt));
                    getVecRef(Br::pfCand_track_ndof    ,tau.pfCand_track_ndof.at(idx_srt));
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
        }
      }

private:

  Long64_t end_entry;
  Long64_t current_entry; // number of the current entry in the file
  Long64_t current_tau; // number of the current tau candidate
  Long64_t tau_i;

  bool hasData;
  bool fullData;
  bool hasFile;

  std::unique_ptr<TFile> file; // to open with one file
  std::unique_ptr<TauTuple> tauTuple;
  std::unique_ptr<Data> data;

};
