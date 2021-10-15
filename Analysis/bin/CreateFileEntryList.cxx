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
    run::Argument<std::string> output_txt{"output-txt", "txt output file with filenames and number of entries"};
    run::Argument<std::string> input_dirs{"input-dirs", "input directory"};
    run::Argument<std::string> file_name_pattern{"file-name-pattern", "regex expression to match file names",
                                                 "^.*\\.root$"};
    run::Argument<std::string> exclude_list{"exclude-list", "comma separated list of files to exclude", ""};
    run::Argument<std::string> exclude_dir_list{"exclude-dir-list",
                                               "comma separated list of directories to exclude", ""};
    run::Argument<unsigned> n_threads{"n-threads", "number of threads", 1};
};

namespace analysis {
class CreateSpectralHists {

public:
    using Tau = tau_tuple::Tau;
    using TauTuple = tau_tuple::TauTuple;

    CreateSpectralHists(const Arguments& args) :
      input_files(RootFilesMerger::FindInputFiles(ParseList(args.input_dirs()),
                                                  args.file_name_pattern(),
                                                  args.exclude_list(),
                                                  args.exclude_dir_list()))
    {
      std::cout << "Number of files: " << input_files.size() << std::endl;
      output_txt.open(args.output_txt(), std::ios::trunc);

      ROOT::EnableThreadSafety();
      if(args.n_threads() > 1) ROOT::EnableImplicitMT(args.n_threads());
    }

    void Run()
    {
        size_t processed=0;
        for(const auto& file_name : input_files) {
            // std::cout << "file: " << file_name << std::endl;
            if(processed % 100 == 0) std::cout << "proc: "
                << processed << " | " <<  input_files.size() << std::endl;
            auto file = root_ext::OpenRootFile(file_name);
            TauTuple input_tauTuple("taus", file.get(), true, {}, {"evt"});
            output_txt << file_name << " " << input_tauTuple.GetEntries() << "\n";
            processed++;
        }
        std::cout << "All file has been processed." << std::endl;
        output_txt.close();
    }

private:

    static std::vector<std::string> ParseList(const std::string& base_str)
    {
      const auto& split_strs = SplitValueList(base_str, true, ", \t", true);
      std::vector<std::string> file_list;
      for(const std::string& str : split_strs) {
        if(str.empty()) continue;
        const std::string bin = Parse<std::string>(str);
        file_list.push_back(str);
      }
      return file_list;
    }
 
    std::vector<std::string> input_files;
    std::shared_ptr<TFile> outputfile;
    std::ofstream output_txt;
};

} // namespace analysis

PROGRAM_MAIN(analysis::CreateSpectralHists, Arguments)
