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
#include "TauMLTools/Analysis/interface/DisTauTagSelection.h"

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
    run::Argument<std::string> mode{"mode",
                "eta phi of the following object will be recorded. Currently available: 1)boostedTau 2)tau 3)jet", "tau"};
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


namespace analysis {

struct HistSpectrum : public root_ext::AnalyzerData {
  using Tau = tau_tuple::Tau;
  using TauTuple = tau_tuple::TauTuple;

  HistSpectrum(std::shared_ptr<TFile>& outputFile, const HistArgs& hargs) : root_ext::AnalyzerData(outputFile) {
    eta_pt_hist.SetMasterHist(
      hargs.eta_bins, hargs.eta_min, hargs.eta_max,
      hargs.pt_bins, hargs.pt_min, hargs.pt_max);
  };

  virtual ~HistSpectrum() {};
  virtual std::set<std::string> getEnabledBranches();
  virtual void fillHist(const Tau& tau);
  static std::shared_ptr<HistSpectrum> getHist(const std::string& name,
                                               std::shared_ptr<TFile>& outputFile,
                                               const HistArgs& hargs);

  TH2D_ENTRY_EMPTY(eta_pt_hist) /* reco tau eta phi */
  TH1D_ENTRY(not_valid, 1, 0, 2) /* valid events that pass object selection */

};

std::set<std::string> HistSpectrum::getEnabledBranches() {
  return {"sampleType",
          "evt",
          "genLepton_kind",
          "genLepton_index",
          "genJet_index",
          "genLepton_vis_pt",
          "genLepton_vis_eta",
          "genLepton_vis_phi",
          "genLepton_vis_mass",
          "tau_index",
          "tau_pt",
          "tau_eta",
          "tau_phi", 
          "tau_mass"};
}

void HistSpectrum::fillHist(const Tau& tau) {
    const auto gen_match = GetGenLeptonMatch(static_cast<reco_tau::gen_truth::GenLepton::Kind>(tau.genLepton_kind), tau.genLepton_index, 
                                                          tau.tau_pt, tau.tau_eta, tau.tau_phi, tau.tau_mass, tau.genLepton_vis_pt, 
                                                          tau.genLepton_vis_eta, tau.genLepton_vis_phi, tau.genLepton_vis_mass, tau.genJet_index);
    if(tau.tau_index >= 0 && gen_match) {
        const auto sample_type = static_cast<SampleType>(tau.sampleType);
        const TauType tau_type = analysis::GenMatchToTauType(*gen_match, sample_type);
        eta_pt_hist(tau_type).Fill(std::abs(tau.tau_eta), tau.tau_pt);
    } else {
        not_valid().Fill(1);
    }
}


// Boosted Tau selector
struct BoostedHist : public HistSpectrum {

  BoostedHist(std::shared_ptr<TFile>& outputFile, const HistArgs& hargs) : HistSpectrum(outputFile, hargs) {
      boostedTau_eta_pt_hist.SetMasterHist(
          hargs.eta_bins, hargs.eta_min, hargs.eta_max,
          hargs.pt_bins, hargs.pt_min, hargs.pt_max);
  };

  std::set<std::string> getEnabledBranches() {
    return {"boostedTau_pt",
            "boostedTau_eta",
            "sampleType",
            "genLepton_kind",
            "boostedTau_index",
            "genLepton_index",
            "genJet_index",
            "genLepton_vis_pt",
            "genLepton_vis_eta",
            "genLepton_vis_phi",
            "genLepton_vis_mass",
            "boostedTau_pt",
            "boostedTau_eta",
            "boostedTau_phi",
            "boostedTau_mass",
            "evt"};
  }

  void fillHist(const Tau& tau) {
      const auto gen_match = GetGenLeptonMatch(static_cast<reco_tau::gen_truth::GenLepton::Kind>(tau.genLepton_kind), tau.genLepton_index, 
                                                            tau.tau_pt, tau.tau_eta, tau.tau_phi, tau.tau_mass, tau.genLepton_vis_pt, 
                                                            tau.genLepton_vis_eta, tau.genLepton_vis_phi, tau.genLepton_vis_mass, tau.genJet_index);
      if(tau.boostedTau_index >= 0 && gen_match) {
          const auto sample_type = static_cast<SampleType>(tau.sampleType);
          const TauType tau_type = analysis::GenMatchToTauType(*gen_match, sample_type);
          boostedTau_eta_pt_hist(tau_type).Fill(std::abs(tau.boostedTau_eta), tau.boostedTau_eta);
      } else {
          not_valid().Fill(1);
      }
  }

  TH2D_ENTRY_EMPTY(boostedTau_eta_pt_hist) /* boosted Tau eta phi*/

};

// Displaced Tau selector
struct DisTauJetHist : public HistSpectrum {

  DisTauJetHist(std::shared_ptr<TFile>& outputFile, const HistArgs& hargs) : HistSpectrum(outputFile, hargs) {
      jet_eta_pt_hist.SetMasterHist(
          hargs.eta_bins, hargs.eta_min, hargs.eta_max,
          hargs.pt_bins, hargs.pt_min, hargs.pt_max);
  };

  std::set<std::string> getEnabledBranches() {
    return { "jet_index",
            "jet_pt",
            "jet_eta",
            "jet_phi",
            "jet_mass",
            "genJet_index",
            "genJet_eta",
            "genJet_pt",
            "genLepton_kind",
            "genLepton_lastMotherIndex",
            "genParticle_pdgId",
            "genParticle_mother",
            "genParticle_charge",
            "genParticle_isFirstCopy",
            "genParticle_isLastCopy",
            "genParticle_pt",
            "genParticle_eta",
            "genParticle_phi",
            "genParticle_mass",
            "genParticle_vtx_x",
            "genParticle_vtx_y",
            "genParticle_vtx_z",
            "evt"};
}

void fillHist(const Tau& tau) {
  const auto jetType_match = GetJetType(tau);
  if(jetType_match) {
    const JetType jetType = std::move(*jetType_match);
    jet_eta_pt_hist(jetType).Fill(std::abs(tau.jet_eta), tau.jet_pt);
  } else {
    not_valid().Fill(1);
  }
}
  TH2D_ENTRY_EMPTY(jet_eta_pt_hist) /* reco jet eta phi (displaced tau mode)*/
};


std::shared_ptr<HistSpectrum> HistSpectrum::getHist(const std::string& name,
                                                    std::shared_ptr<TFile>& outputFile,
                                                    const HistArgs& hargs) {
  if(name == "tau")
    return std::make_shared<HistSpectrum>(outputFile, hargs);
  if(name == "jet")
      return std::make_shared<DisTauJetHist>(outputFile, hargs);
  if(name == "boostedTau")
      return std::make_shared<BoostedHist>(outputFile, hargs);
  throw analysis::exception("Unknown selector name = '%1%'") % name;
};

class CreateSpectralHists {

public:
    using Tau = tau_tuple::Tau;
    using TauTuple = tau_tuple::TauTuple;

    CreateSpectralHists(const Arguments& args) :
      input_files(RootFilesMerger::FindInputFiles(args.input_dirs(),
                                                  args.file_name_pattern(),
                                                  args.exclude_list(),
                                                  args.exclude_dir_list())),
      outputfile(root_ext::CreateRootFile(args.outputfile())),
      mode(args.mode())
    {
      output_txt.open(args.output_entries(), std::ios::trunc);

      ROOT::EnableThreadSafety();
      if(args.n_threads() > 1)
        ROOT::EnableImplicitMT(args.n_threads());

      auto par_path = GetPathWithoutFileName(args.outputfile());
      if(!boost::filesystem::exists(par_path))
        boost::filesystem::create_directory(par_path);

      hists = HistSpectrum::getHist(
        args.mode(), outputfile, ParseHistSetup(args.pt_hist(),args.eta_hist()));
    }

    void Run()
    {
        for(const auto& file_name : input_files) {
            std::cout << "file: " << file_name << std::endl;
            auto file = root_ext::OpenRootFile(file_name);
            TauTuple input_tauTuple("taus", file.get(), true, {}, hists->getEnabledBranches());
            output_txt << file_name << " " << input_tauTuple.GetEntries() << "\n";
            for(const Tau& tau : input_tauTuple)
            {
              hists->fillHist(tau);
              ++total_size;
            }
        }
        output_txt.close();

        // Report:
        std::cout << "All file has been processed." << std::endl
                  << "Number of files = " << input_files.size() << std::endl
                  << "Number of processed " << mode << " = " << total_size << std::endl
                  << "Number of not valid " << mode << " = " << hists->not_valid().GetEntries() << std::endl;
        
        auto hist_map = hists->GetHistogramsEx<TH2D>();
        double sum_all_types = 0.0;
        for(auto h: hist_map) {
          std::cout << "Histogram:" << h.first
                    <<" Integral:"  << h.second->Integral()
                    <<" GetEntries:" << h.second->GetEntries()
                    << std::endl;
          sum_all_types += h.second->Integral();
        }

        std::cout << "Integral over all histograms = " << sum_all_types
                  << std::setprecision(4)
                  << " (" << sum_all_types/total_size*100 << "%)" << std::endl;
                  
    }

private:
    static HistArgs ParseHistSetup(const std::string& pt_hist, const std::string& eta_hist)
    {
        const auto& split_args_pt = SplitValueList(pt_hist, true, ",", true);
        const auto& split_args_eta = SplitValueList(eta_hist, true, ",", true);

        std::cout << "pt histogram setup (n_bins pt_min pt_max): ";
        for(const std::string& bin_str : split_args_pt)
          std::cout << Parse<double>(bin_str) << "  ";
        std::cout << std::endl;

        std::cout << "eta histogram setup (n_bins eta_min eta_max): ";
        for(const std::string& bin_str : split_args_eta)
          std::cout << Parse<double>(bin_str) << "  ";
        std::cout << std::endl;

        if(split_args_pt.size() != 3 
          || Parse<double>(split_args_pt[0]) < 1
          || Parse<double>(split_args_pt[1]) >= Parse<double>(split_args_pt[2])
          ) throw exception("Invalid pt-hist arguments");

        if(split_args_eta.size()!=3
          || Parse<double>(split_args_eta[0]) < 1
          || Parse<double>(split_args_eta[1]) >= Parse<double>(split_args_eta[2])
          ) throw exception("Invalid eta-hist arguments");

        HistArgs histarg(split_args_pt, split_args_eta);
        return histarg;
    }

  private:
      std::vector<std::string> input_files;
      std::shared_ptr<TFile> outputfile;
      std::string mode;
      std::ofstream output_txt;
      std::shared_ptr<HistSpectrum> hists;
      Int_t total_size=0;

};

} // namespace analysis

PROGRAM_MAIN(analysis::CreateSpectralHists, Arguments)
