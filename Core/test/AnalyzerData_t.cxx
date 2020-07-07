/*! Test AnalyzerData class.
This file is part of https://github.com/hh-italian-group/TauMLTools. */

#include "TauMLTools/Core/interface/AnalyzerData.h"
#include "TauMLTools/Core/interface/RootExt.h"
#include "TauMLTools/Core/interface/program_main.h"

struct MyAnaData : public root_ext::AnalyzerData {
//    using AnalyzerData::AnalyzerData;

    explicit MyAnaData(std::shared_ptr<TFile> _outputFile, const std::string& directoryName = "") :
        AnalyzerData(_outputFile, directoryName)
    {
        hist.Emplace("z", 3, 20, 23);
        other_hist.SetMasterHist(5, 5, 10);
    }

    const std::vector<double> bins{1, 2, 3, 4};

    TH1D_ENTRY(hist, 10, .5, 10.5)
    ANA_DATA_ENTRY(TH1D, other_hist)
    TH1D_ENTRY_CUSTOM(custom_hist, bins)
};

struct Arguments {
    REQ_ARG(std::string, output);
};

class AnalyzerData_t {
public:
    AnalyzerData_t(const Arguments& args) : output(root_ext::CreateRootFile(args.output())), anaData(output) {}

    void Run()
    {
        anaData.hist().Fill(1);
        anaData.hist(1).Fill(2);
        anaData.hist("b").Fill(3);
        anaData.hist(1).Fill(4);
        anaData.hist("z").Fill(21.5);
        anaData.other_hist(0).Fill(6);
        anaData.hist(1, "b").Fill(1.4);
        std::string f = "f";
        anaData.custom_hist().Fill(3.5);
        anaData.custom_hist(f).Fill(2.4);
        anaData.custom_hist(std::string("g")).Fill(1.5);
    }

private:
    std::shared_ptr<TFile> output;
    MyAnaData anaData;
};

PROGRAM_MAIN(AnalyzerData_t, Arguments)
