#include "TFile.h"

struct DatasetDesc {
    size_t n_per_batch;
    std::string selection; // selection for mixing
    TH1D threshold; // threshold for event dropping
    std::vector<std::string> fileNames;
};

struct Dataset {
  virtual ~Dataset() {}
  virtual bool FillNext(input_tuple::TauTuple& outputTuple) = 0;
};

struct DatasetMix : Dataset {
    DatasetMix(const DatasetDesc& desc)  : desc_(desc) {
        entryIndex_ = 0;
        if(desc.fileNames.empty())
            throw std::runtime_error("Dataset is empty");
        tuple_= std::make_unique<input_tuple::TauTuple>("taus", desc.fileNames); //input tuple
    }

    virtual bool FillNext(input_tuple::TauTuple& outputTuple) override
    {   
        size_t n = 0;
        for(; n < desc_.n_per_batch && entryIndex_ < tuple_->GetEntries(); ++entryIndex_) {
            tuple_->GetEntry(entryIndex_);

            outputTuple() = tuple_->data();  
            if (outputTuple().sampleType == 0) {
                outputTuple().tauType = 8; // set tau type to data (8) for data events
            }
            else{
                outputTuple().tauType = outputTuple().mytauType; // set tau type to one computed during skimming
            }
            // Label dataset ID to be able to distinguish event types for later steps 
            if (desc_.selection == "data"){
                outputTuple().dataset_id = 0;    
            } else if (desc_.selection == "DY_taus"){
                outputTuple().dataset_id = 1;
            } else if (desc_.selection == "tt_taus"){
                outputTuple().dataset_id = 2;
            } else if (desc_.selection == "DY_muons"){
                outputTuple().dataset_id = 3;
            } else if (desc_.selection == "tt_jets"){
                outputTuple().dataset_id = 4;
            } else if (desc_.selection == "W_jets"){
                outputTuple().dataset_id = 5;
            } else if (desc_.selection == "QCD_jets"){
                outputTuple().dataset_id = 6;
            } 
            

            outputTuple.Fill();
            ++n;

        }
        return n == desc_.n_per_batch; // File not finished if true. Else we have to stop.
    }

    DatasetDesc desc_;
    std::unique_ptr<input_tuple::TauTuple> tuple_;
    Long64_t entryIndex_;
};

struct DatasetDrop : Dataset {
    DatasetDrop(const DatasetDesc& desc)  : desc_(desc) {
        entryIndex_ = 0;
        if(desc.fileNames.empty())
            throw std::runtime_error("Dataset is empty");
        tuple_= std::make_unique<input_tuple::TauTuple>("taus", desc.fileNames); //input tuple
    }

    virtual bool FillNext(input_tuple::TauTuple& outputTuple) override
    {   
        size_t n = 0;
        for(; n < desc_.n_per_batch && entryIndex_ < tuple_->GetEntries(); ++entryIndex_) {
            tuple_->GetEntry(entryIndex_);
            double v = myrndm->Rndm(); // Generate random number between 0 and 1
            outputTuple() = tuple_->data();
            if (v <= desc_.threshold.GetBinContent(desc_.threshold.FindBin(outputTuple().tau_pt))){ 
                // Keep only if random number is less than the ratio calculated for that bin
                outputTuple.Fill();
            }
            ++n;
        }
        return n == desc_.n_per_batch; // File not finished if true. Else we have to stop.
    }

    DatasetDesc desc_;
    std::unique_ptr<input_tuple::TauTuple> tuple_;
    Long64_t entryIndex_;
    TRandom3 *myrndm = new TRandom3();

};

struct OutputDesc{
    std::string fileName_;
    size_t nBatches_;
};

class DataMixer {
public:
    DataMixer(const std::vector<DatasetDesc>& datasets_desc, const std::vector<OutputDesc>& outputDescs,
                const std::string& action)
        : outputDescs_(outputDescs)
    {
        if (action == "Mix"){
        CreateDatasets<DatasetMix>(datasets_desc);
        } else if(action == "Drop") {
        CreateDatasets<DatasetDrop>(datasets_desc);
        } else {
        throw std::runtime_error("Unsupported action");
        }
    }

    void Run()
    {   
        for(const auto& outputDesc:outputDescs_){
            auto outputFile = std::make_unique<TFile>(outputDesc.fileName_.c_str(), "RECREATE", "", 404);
            if(!outputFile || outputFile->IsZombie())
                throw std::runtime_error("Can not create output file.");
            auto outputTuple = std::make_unique<input_tuple::TauTuple>(outputFile.get(), false);

            for(size_t i(0); i < outputDesc.nBatches_; ++i) {
                if (i%10 == 0){
                    cout<<"Batch "<<i<<endl;
                }
                for(auto& dataset:datasets_) {
                    if(!dataset->FillNext(*outputTuple)) {
                        throw std::runtime_error("Run out of events!");
                    }
                }
                
            }
            outputTuple->Write();
        }

    }
private:
  template<typename DS>
  void CreateDatasets(const std::vector<DatasetDesc>& datasets_desc)
  {
    for(const auto& desc : datasets_desc) {
      std::shared_ptr<Dataset> ds = std::make_shared<DS>(desc);
      datasets_.push_back(ds);
    }
  }
private:
    std::vector<OutputDesc> outputDescs_;
    std::vector<std::shared_ptr<Dataset>> datasets_;
};