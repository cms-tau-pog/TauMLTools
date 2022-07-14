#include "TFile.h"
//#include "/home/russell/AdversarialTauML/TauMLTools/Analysis/interface/TauTuple.h" 
#include "/home/russell/AdversarialTauML/TauMLTools/Analysis/interface/GenLepton.h" 


enum class Selection {data, DY_taus, tt_taus, W_jets, QCD_jets, tt_jets, DY_muons};

struct DatasetDesc {
    size_t n_per_batch;
    std::string selection;
    std::vector<std::string> fileNames;
};

struct Dataset {
    Dataset(const DatasetDesc& desc)  : desc_(desc) {
        entryIndex_ = 0;
        if(desc.fileNames.empty())
            throw std::runtime_error("Dataset is empty");
        tuple_= std::make_unique<input_tuple::TauTuple>("taus", desc.fileNames); //input tuple
    }

    bool FillNext(input_tuple::TauTuple& outputTuple)
    {   
        size_t n = 0;
        for(; n < desc_.n_per_batch && entryIndex_ < tuple_->GetEntries(); ++entryIndex_) {
            tuple_->GetEntry(entryIndex_);

            outputTuple() = tuple_->data();  
            if (outputTuple().sampleType == 0) {
                outputTuple().tauType = 8; //data
            }
            else{
                outputTuple().tauType = outputTuple().mytauType;
            }
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

struct OutputDesc{
    std::string fileName_;
    size_t nBatches_;
};

class DataMixer {
public:
    DataMixer(const std::vector<DatasetDesc>& datasets_desc, const std::vector<OutputDesc>& outputDescs)
        : outputDescs_(outputDescs)
          
    {
        for(const auto& desc : datasets_desc) { // Creating input descriptor
            datasets_.emplace_back(desc);
        }
    }

    void Run()
    {
        for(const auto& outputDesc:outputDescs_){
            auto outputFile = std::make_unique<TFile>(outputDesc.fileName_.c_str(), "RECREATE", "", 404);
            if(!outputFile || outputFile->IsZombie())
                throw std::runtime_error("Can not create output file.");
            auto outputTuple = std::make_unique<input_tuple::TauTuple>(outputFile.get(), false);

            //bool ok = true;
            for(size_t i(0); i < outputDesc.nBatches_; ++i) {
                if (i%10 == 0){
                    cout<<i<<endl;
                }
                for(auto& dataset:datasets_) {
                    if(!dataset.FillNext(*outputTuple)) {
                        //ok = false;
                        throw std::runtime_error("Run out of events!");
                    }
                }
            }
            outputTuple->Write();
        }

    }
private:
    std::vector<OutputDesc> outputDescs_;
    std::vector<Dataset> datasets_;
};