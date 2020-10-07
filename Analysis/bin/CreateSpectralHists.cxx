/*! Create histograms with pt and eta distribution for every type of tau (tau_h, tau_mu, tau_e, tau_j)
*/
#include "TauMLTools/Core/interface/program_main.h"
#include "TauMLTools/Analysis/interface/TauTuple.h"
#include "TauMLTools/Analysis/interface/SummaryTuple.h"
#include "TauMLTools/Core/interface/RootFilesMerger.h"
#include "TauMLTools/Core/interface/NumericPrimitives.h"

#include "TauMLTools/Core/interface/AnalyzerData.h"
#include "TauMLTools/Core/interface/RootExt.h"

struct Arguments {
    run::Argument<std::string> outputfile{"outputfile", "output file name"};
    run::Argument<std::vector<std::string>> input_dirs{"input-dir", "input directory"};
    run::Argument<std::string> pt_hist{"pt-hist", "pt hist setup: (number of bins, pt_min, pt_max)", "200, 0.0, 1000"};
    run::Argument<std::string> eta_hist{"eta-hist", "eta hist setup: (number of bins, abs(eta)_min, abs(eta)_max)","4, 0.0, 2.3"};
    run::Argument<std::string> file_name_pattern{"file-name-pattern", "regex expression to match file names",
                                                 "^.*\\.root$"};
    run::Argument<std::string> exclude_list{"exclude-list", "comma separated list of files to exclude", ""};
    run::Argument<std::string> exclude_dir_list{"exclude-dir-list",
                                               "comma separated list of directories to exclude", ""};
};

struct HistArgs {
  int eta_bins, pt_bins;
  double eta_min, eta_max, pt_min, pt_max;
  HistArgs(std::vector<std::string> args_pt, std::vector<std::string> args_eta)
  {
    pt_bins = analysis::Parse<int>(args_pt[0]);
    pt_min = analysis::Parse<double>(args_pt[1]);
    pt_max = analysis::Parse<double>(args_pt[2]);
    eta_bins = analysis::Parse<int>(args_eta[0]);
    eta_min = analysis::Parse<double>(args_eta[1]);
    eta_max = analysis::Parse<double>(args_eta[2]);
  }
};

class HistSpectrum : public root_ext::AnalyzerData {

public:
  HistSpectrum(std::shared_ptr<TFile> outputFile, HistArgs hargs) : root_ext::AnalyzerData(outputFile)
  {
    eta_pt_hist.SetMasterHist(hargs.eta_bins, hargs.eta_min, hargs.eta_max,
                              hargs.pt_bins, hargs.pt_min, hargs.pt_max);
  }

  TH2D_ENTRY(eta_pt_hist, 4, 0, 2.3, 200, 0, 1000)

};

namespace analysis {
class CreateSpectralHists {

private:
    std::vector<std::string> input_files;
    std::shared_ptr<TFile> outputfile;
    std::shared_ptr<HistSpectrum> hists;
    Int_t total_size=0;

public:
    using Tau = tau_tuple::Tau;
    using TauTuple = tau_tuple::TauTuple;

    CreateSpectralHists(const Arguments& args) :
      input_files(RootFilesMerger::FindInputFiles(args.input_dirs(),
                                                  args.file_name_pattern(),
                                                  args.exclude_list(),
                                                  args.exclude_dir_list())),
       outputfile(root_ext::CreateRootFile(args.outputfile()))
    {
      auto par_path = GetPathWithoutFileName(args.outputfile());
      std::cout << par_path << "\n";
      if(!boost::filesystem::exists(par_path)) boost::filesystem::create_directory(par_path);

      hists = std::make_shared<HistSpectrum>(outputfile,
             ParseHistSetup(args.pt_hist(),args.eta_hist()));
    }

    void Run()
    {
        for(const auto& file_name : input_files) {
            std::cout << "file: " << file_name << std::endl;
            auto file = root_ext::OpenRootFile(file_name);

            TauTuple input_tauTuple("taus", file.get(), true);
            for(const Tau& tau : input_tauTuple)
            {
              AddTau(tau);
              ++total_size;
            }
        }

        std::cout << "All file has been processed." << std::endl
                  << "Number of files = " << input_files.size() << std::endl
                  << "Number of processed taus = " << total_size << std::endl
                  << "Number of taus within the ranges of histogram = " << Integral()
                  << " (" << Integral()/total_size*100 << "%)" << std::endl;
    }

private:
    static HistArgs ParseHistSetup(const std::string& pt_hist, const std::string& eta_hist)
    {
        const auto& split_args_pt = SplitValueList(pt_hist, true, ",", true);
        const auto& split_args_eta = SplitValueList(eta_hist, true, ",", true);

        std::cout << "pt histogram setup (n_bins pt_min pt_max): ";
        for(const std::string& bin_str : split_args_pt) std::cout << Parse<double>(bin_str) << "  ";
        std::cout << std::endl;

        std::cout << "eta histogram setup (n_bins eta_min eta_max): ";
        for(const std::string& bin_str : split_args_eta) std::cout << Parse<double>(bin_str) << "  ";
        std::cout << std::endl;

        if(split_args_pt.size()!=3 || split_args_pt[0]<1 || split_args_pt[1]>=split_args_pt[2])
        throw exception("Invalid pt-hist arguments");

        if(split_args_eta.size()!=3 || split_args_eta[0]<1 || split_args_eta[1]>=split_args_eta[2])
        throw exception("Invalid eta-hist arguments");

        HistArgs histarg(split_args_pt, split_args_eta);

        return histarg;
    }

    void AddTau(const Tau& tau)
    {
        const auto gen_match = static_cast<analysis::GenLeptonMatch>(tau.lepton_gen_match);
        const auto sample_type = static_cast<analysis::SampleType>(tau.sampleType);
        const TauType tau_type = analysis::GenMatchToTauType(gen_match, sample_type);
        hists->eta_pt_hist(tau_type).Fill(std::abs(tau.tau_eta), tau.tau_pt);
    }

    double Integral()
    {
      return  hists->eta_pt_hist("tau").Integral()
            + hists->eta_pt_hist("e").Integral()
            + hists->eta_pt_hist("mu").Integral()
            + hists->eta_pt_hist("jet").Integral();
    }

};

} // namespace analysis

PROGRAM_MAIN(analysis::CreateSpectralHists, Arguments)
