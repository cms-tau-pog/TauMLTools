#include "TFile.h"
#include "/home/russell/AdversarialTauML/TauMLTools/Analysis/interface/GenLepton.h" 


enum class Selection {qcd};

struct DatasetDesc {
    size_t n_per_batch;
    TH1D threshold;
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
            double v = myrndm->Rndm();
            //std::cout << v << std::endl;
            outputTuple() = tuple_->data();
            // std::cout<< "tau_pt" << outputTuple().tau_pt << std::endl;
            // std::cout << "in bin" <<desc_.threshold.FindBin(outputTuple().tau_pt) << std::endl;
            if (v <= desc_.threshold.GetBinContent(desc_.threshold.FindBin(outputTuple().tau_pt))){ 
                outputTuple.Fill();
                //std::cout << "passed" << std::endl;
            }
            ++n;
        }
        return n == desc_.n_per_batch; // File not finished if true. Else we have to stop.
    }

    DatasetDesc desc_;
    std::unique_ptr<input_tuple::TauTuple> tuple_;
    Long64_t entryIndex_;
    TRandom3 *myrndm = new TRandom3();

    // TFile* ICLFile = TFile::Open("/home/russell/histograms/datacard_pt_2_inclusive_mt_2018.root"," READ ");
    // TH1D* target = (TH1D*)ICLFile->Get("mt_inclusive/QCD");


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
                if (i%100 == 0){
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