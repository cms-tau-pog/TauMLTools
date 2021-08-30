#include "TauMLTools/Analysis/interface/TauTuple.h"
#include "TauMLTools/Analysis/interface/GenLepton.h"
#include "TauMLTools/Training/interface/DataLoader_tools.h"

#include "TROOT.h"
#include "TLorentzVector.h"
#include "TMath.h"

struct Data {
    Data(size_t n_tau,
         size_t pfCand_SeqSize,
         size_t pfCand_nfeatures,
         size_t output_size) :
         x(n_tau * pfCand_SeqSize * pfCand_nfeatures, 0),
         y(n_tau * output_size, 0)
         { }
    std::vector<float> x, y;
};

// using namespace Setup;

class DataLoader {

public:
    using Tau = tau_tuple::Tau;
    using TauTuple = tau_tuple::TauTuple;
    using LorentzVectorM = ROOT::Math::LorentzVector<ROOT::Math::PtEtaPhiM4D<double>>;

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
            data = std::make_unique<Data>(Setup::n_tau, Setup::nSeq_PfCand,
                                          Setup::n_PfCand, Setup::output_classes);
            tau_i = 0;
            hasData = true;
        }
        while(tau_i < Setup::n_tau)
        {
            if(current_entry == end_entry)
            {
                hasFile = false;
                return false;
            }
            tauTuple->GetEntry(current_entry);
            // only tau_h is considered for the reconstruction + requerment to have jet
            if ((*tauTuple)().genLepton_kind == 5 && (*tauTuple)().jet_index >= 0)
            // if((*tauTuple)().genLepton_kind == 5)
            {
                FillLabels(tau_i, Setup::output_classes);
                FillPfSequence<PfCand_Features>(tau_i, Setup::nSeq_PfCand, Setup::n_PfCand, "pfCand_");
                // FillPfSequence<PfCand_Features>(tau_i, Setup::nSeq_lostTrack, Setup::n_lostTrack, "lostTrack_");
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
        return std::clamp((value - FeatureT::mean[idx][inner]) / FeatureT::std[idx][inner],
                          FeatureT::lim_min[idx][inner], FeatureT::lim_max[idx][inner]);
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

      template<typename BranchType>
      const std::vector<BranchType>& getValue(const std::string& variable,
                                              const std::string& prefix)
      {
          return tauTuple->get<std::vector<BranchType>>(prefix + variable);
      }

      void FillLabels(size_t tau_i,
                      size_t n_classes)
      {
        auto& tau = const_cast<tau_tuple::Tau&>(tauTuple->data());
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

      template<typename PfType>
      void FillPfSequence(Long64_t tau_i,
                          const size_t n_sequence,
                          const size_t n_features,
                          const std::string& pref)
      {
        size_t start_index = n_sequence * n_features * tau_i;

        std::vector<size_t> indices(tauTuple->get<std::vector<Float_t>>(pref+"pt").size());
        std::iota(indices.begin(), indices.end(), 0);

        // sort PfCands by pt
        std::sort(indices.begin(), indices.end(), [&](size_t a, size_t b) {
            return tauTuple->get<std::vector<Float_t>>(pref+"pt").at(a) >
                   tauTuple->get<std::vector<Float_t>>(pref+"pt").at(b);
        });
        
        Float_t empty_float;
        size_t upper_index = std::min(n_sequence, indices.size());
        for(size_t pfCand_i = 0; pfCand_i < upper_index; pfCand_i++)
        {
            auto getVecRef = [&](PfType _fe, Float_t value){
                size_t _feature_idx = static_cast<size_t>(_fe);
                if(_feature_idx < 0) return;
                size_t index = start_index + n_features * pfCand_i + _feature_idx;
                data->x.at(index) = 
                    Scale<typename  FeaturesHelper<decltype(_fe)>::scaler_type>(
                        _feature_idx, value, false
                        );
            };

            size_t idx_srt = indices.at(pfCand_i);
            getVecRef(PfType::pfCand_valid                ,1.0);
            getVecRef(PfType::pfCand_pt                   ,getValue<Float_t>("pt",pref).at(idx_srt));
            getVecRef(PfType::pfCand_eta                  ,getValue<Float_t>("eta",pref).at(idx_srt));
            getVecRef(PfType::pfCand_phi                  ,getValue<Float_t>("phi",pref).at(idx_srt));
            getVecRef(PfType::pfCand_mass                 ,getValue<Float_t>("mass",pref).at(idx_srt));
            getVecRef(PfType::pfCand_particleType         ,getValue<Int_t>("particleType",pref).at(idx_srt));
            getVecRef(PfType::pfCand_charge               ,getValue<Int_t>("charge",pref).at(idx_srt));
            getVecRef(PfType::pfCand_pvAssociationQuality ,getValue<Int_t>("pvAssociationQuality",pref).at(idx_srt));
            getVecRef(PfType::pfCand_fromPV               ,getValue<Int_t>("fromPV",pref).at(idx_srt));
            getVecRef(PfType::pfCand_puppiWeight          ,getValue<Float_t>("puppiWeight",pref).at(idx_srt));
            getVecRef(PfType::pfCand_puppiWeightNoLep     ,getValue<Float_t>("puppiWeightNoLep",pref).at(idx_srt));
            getVecRef(PfType::pfCand_lostInnerHits        ,getValue<Int_t>("lostInnerHits",pref).at(idx_srt));
            getVecRef(PfType::pfCand_nPixelHits           ,getValue<Int_t>("nPixelHits",pref).at(idx_srt));
            getVecRef(PfType::pfCand_nHits                ,getValue<Int_t>("nHits",pref).at(idx_srt));
            getVecRef(PfType::pfCand_hasTrackDetails      ,getValue<Int_t>("hasTrackDetails",pref).at(idx_srt));
            getVecRef(PfType::pfCand_caloFraction         ,getValue<Float_t>("caloFraction",pref).at(idx_srt));
            getVecRef(PfType::pfCand_hcalFraction         ,getValue<Float_t>("hcalFraction",pref).at(idx_srt));
            getVecRef(PfType::pfCand_rawCaloFraction      ,getValue<Float_t>("rawCaloFraction",pref).at(idx_srt));
            getVecRef(PfType::pfCand_rawHcalFraction      ,getValue<Float_t>("rawHcalFraction",pref).at(idx_srt));

            TLorentzVector v;
            v.SetPtEtaPhiM(
                getValue<Float_t>("pt",pref).at(idx_srt),
                getValue<Float_t>("eta",pref).at(idx_srt),
                getValue<Float_t>("phi",pref).at(idx_srt),
                getValue<Float_t>("mass",pref).at(idx_srt)
              );

            getVecRef(PfType::pfCand_px, v.Px());
            getVecRef(PfType::pfCand_py, v.Py());
            getVecRef(PfType::pfCand_pz, v.Pz());
            getVecRef(PfType::pfCand_E, v.E());
            
            if(getValue<Int_t>("hasTrackDetails",pref).at(idx_srt))
            {
                if(std::isnormal(getValue<Float_t>("dz",pref).at(idx_srt)))
                    getVecRef(PfType::pfCand_dz        ,getValue<Float_t>("dz",pref).at(idx_srt));
                if(std::isnormal( getValue<Float_t>("dz_error", pref).at(idx_srt)))
                    getVecRef(PfType::pfCand_dz_error  ,getValue<Float_t>("dz_error", pref).at(idx_srt));
                if(std::isnormal(getValue<Float_t>("dxy_error", pref).at(idx_srt)))
                    getVecRef(PfType::pfCand_dxy_error ,getValue<Float_t>("dxy_error", pref).at(idx_srt));

                getVecRef(PfType::pfCand_dxy           ,getValue<Float_t>("dxy",pref).at(idx_srt));
                getVecRef(PfType::pfCand_track_chi2    ,getValue<Float_t>("track_chi2", pref).at(idx_srt));
                getVecRef(PfType::pfCand_track_ndof    ,getValue<Float_t>("track_ndof", pref).at(idx_srt));
            }
            
            if(tauTuple->get<Int_t>("jet_index")>=0)
            {
                Float_t jet_eta = tauTuple->get<Float_t>("jet_eta");
                Float_t jet_phi = tauTuple->get<Float_t>("jet_phi");
                // getVecRef(PfType::jet_eta)        = jet_eta;
                // getVecRef(PfType::jet_phi)        = jet_phi;
                getVecRef(PfType::pfCand_deta, getValue<Float_t>("eta",pref).at(idx_srt) - jet_eta);
                getVecRef(PfType::pfCand_dphi, DeltaPhi<Float_t>(getValue<Float_t>("phi",pref).at(idx_srt), jet_phi));
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
