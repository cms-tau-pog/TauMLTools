/*! Create histograms with pt and eta distribution for every type of tau (tau_h, tau_mu, tau_e, tau_j)
*/
#include "TauMLTools/Core/interface/program_main.h"
#include "TauMLTools/Analysis/interface/TauTuple.h"
#include "TauMLTools/Analysis/interface/SummaryTuple.h"
#include "TauMLTools/Core/interface/RootFilesMerger.h"
#include "TauMLTools/Core/interface/NumericPrimitives.h"

#include "TauMLTools/Core/interface/AnalyzerData.h"
#include "TauMLTools/Core/interface/RootExt.h"
#include "TauMLTools/Analysis/interface/TauSelection.h"

#include <iostream>
#include <fstream>

struct Arguments {
    run::Argument<std::string> outputfile{"outputfile", "output file name"};
    run::Argument<std::string> output_entries{"output_entries", "txt output file with filenames and number of entries"};
    run::Argument<std::vector<std::string>> input_dirs{"input-dir", "input directory"};
    run::Argument<std::string> pt_hist{"pt-hist", "pt hist setup: (number of bins, pt_min, pt_max)", "200, 0.0, 1000"};
    run::Argument<std::string> eta_hist{"eta-hist", "eta hist setup: (number of bins, abs(eta)_min, abs(eta)_max)","4, 0.0, 2.3"};
    run::Argument<std::string> file_name_pattern{"file-name-pattern", "regex expression to match file names",
                                                 "^.*\\.root$"};
    run::Argument<std::string> exclude_list{"exclude-list", "comma separated list of files to exclude", ""};
    run::Argument<std::string> exclude_dir_list{"exclude-dir-list",
                                               "comma separated list of directories to exclude", ""};
    run::Argument<unsigned> n_threads{"n-threads", "number of threads", 1};
};

struct HistArgs {
  int eta_bins, pt_bins;
  double eta_min, eta_max, pt_min, pt_max;
  HistArgs(const std::vector<std::string>& args_pt, const std::vector<std::string>& args_eta)
  {
    pt_bins = analysis::Parse<int>(args_pt.at(0));
    pt_min = analysis::Parse<double>(args_pt.at(1));
    pt_max = analysis::Parse<double>(args_pt.at(2));
    eta_bins = analysis::Parse<int>(args_eta.at(0));
    eta_min = analysis::Parse<double>(args_eta.at(1));
    eta_max = analysis::Parse<double>(args_eta.at(2));
  }
};

class HistSpectrum : public root_ext::AnalyzerData {

public:
  HistSpectrum(std::shared_ptr<TFile> outputFile, const HistArgs& hargs) : root_ext::AnalyzerData(outputFile)
  {
    eta_pt_hist.SetMasterHist(hargs.eta_bins, hargs.eta_min, hargs.eta_max,
                              hargs.pt_bins, hargs.pt_min, hargs.pt_max);
  }

  TH2D_ENTRY(eta_pt_hist, 4, 0, 2.3, 200, 0, 1000)
  TH1D_ENTRY(not_valid, 1, 0, 2)

};

namespace analysis {
class CreateSpectralHists {

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
      output_txt.open(args.output_entries(), std::ios::trunc);

      ROOT::EnableThreadSafety();
      if(args.n_threads() > 1) ROOT::EnableImplicitMT(args.n_threads());

      auto par_path = GetPathWithoutFileName(args.outputfile());
      if(!boost::filesystem::exists(par_path)) boost::filesystem::create_directory(par_path);

      hists = std::make_shared<HistSpectrum>(outputfile,
             ParseHistSetup(args.pt_hist(),args.eta_hist()));
    }

    void Run()
    {
        for(const auto& file_name : input_files) {
            std::cout << "file: " << file_name << std::endl;
            auto file = root_ext::OpenRootFile(file_name);

            TauTuple input_tauTuple("taus", file.get(), true, {},
                     {"tau_pt", "tau_eta", "sampleType", "genLepton_kind", "tau_index",
                      "genLepton_index", "genJet_index",
                      "genLepton_vis_pt", "genLepton_vis_eta", "genLepton_vis_phi", "genLepton_vis_mass",
                      "tau_pt", "tau_eta", "tau_phi", "tau_mass", "evt"});
            
            output_txt << file_name << " " << input_tauTuple.GetEntries() << "\n";

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
                  << " (" << Integral()/total_size*100 << "%)" << std::endl
                  << "Number of not Valid taus = " << hists->not_valid().GetEntries() << std::endl;
        output_txt.close();
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

        if(split_args_pt.size()!=3 || Parse<double>(split_args_pt[0])<1 || Parse<double>(split_args_pt[1])>=Parse<double>(split_args_pt[2]))
        throw exception("Invalid pt-hist arguments");

        if(split_args_eta.size()!=3 || Parse<double>(split_args_eta[0])<1 || Parse<double>(split_args_eta[1])>=Parse<double>(split_args_eta[2]))
        throw exception("Invalid eta-hist arguments");

        HistArgs histarg(split_args_pt, split_args_eta);

        return histarg;
    }

    void AddTau(const Tau& tau)
    {
        const auto gen_match = GetGenLeptonMatch(static_cast<reco_tau::gen_truth::GenLepton::Kind>(tau.genLepton_kind), tau.genLepton_index, 
                                                              tau.tau_pt, tau.tau_eta, tau.tau_phi, tau.tau_mass, tau.genLepton_vis_pt, 
                                                              tau.genLepton_vis_eta, tau.genLepton_vis_phi, tau.genLepton_vis_mass, tau.genJet_index);
        if(PassSelection(tau) && gen_match) {
            const auto sample_type = static_cast<SampleType>(tau.sampleType);
            const TauType tau_type = analysis::GenMatchToTauType(*gen_match, sample_type);
            hists->eta_pt_hist(tau_type).Fill(std::abs(tau.tau_eta), tau.tau_pt);
            ttypes[tau_type] = true;
        } else {
            hists->not_valid().Fill(1);
        }
    }

    bool PassSelection(const Tau& tau) const
    {
      return (tau.tau_index >= 0);
    }

    double Integral() const
    {
      double res = 0;
      for(auto type: ttypes) res += hists
        ->eta_pt_hist(analysis::ToString(type.first)).Integral();
      return res;
    }

  private:
      std::vector<std::string> input_files;
      std::shared_ptr<TFile> outputfile;
      std::ofstream output_txt;
      std::shared_ptr<HistSpectrum> hists;
      std::map<TauType,bool> ttypes;
      Int_t total_size=0;

};

} // namespace analysis

PROGRAM_MAIN(analysis::CreateSpectralHists, Arguments)
