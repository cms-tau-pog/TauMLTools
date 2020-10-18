/*! Sheffle and merge datasets using Histograms created with CreateSpectralHists.cxx
*/

#include <fstream>
#include <random>
#include <string>
#include <iostream>
#include <boost/algorithm/string.hpp>
#include <boost/regex.hpp>
#include <boost/foreach.hpp>

#include "TauMLTools/Core/interface/program_main.h"
#include "TauMLTools/Core/interface/RootExt.h"
#include "TauMLTools/Core/interface/PropertyConfigReader.h"
#include "TauMLTools/Core/interface/ProgressReporter.h"
#include "TauMLTools/Analysis/interface/TauTuple.h"

namespace analysis {

enum class MergeMode { MergeAll = 1, MergePerEntry = 2 };
ENUM_NAMES(MergeMode) = {
    { MergeMode::MergeAll, "MergeAll" },
    { MergeMode::MergePerEntry, "MergePerEntry" }
};
} // namespace analysis

struct Arguments {
    run::Argument<std::string> cfg{"cfg", "configuration file with the list of input sources"};
    run::Argument<std::string> input{"input", "input path with tuples for all the samples"};
    run::Argument<std::string> output{"output", "output, depending on the merging mode: MergeAll - file,"
                                                " MergePerEntry - directory."};
    run::Argument<std::string> pt_bins{"pt-bins", "pt bins"};
    run::Argument<std::string> eta_bins{"eta-bins", "eta bins"};
    run::Argument<analysis::MergeMode> mode{"mode", "merging mode: MergeAll or MergePerEntry"};
    run::Argument<bool> calc_weights{"calc-weights", "calculate training weights"};
    run::Argument<bool> ensure_uniformity{"ensure-uniformity", "stop the merge if at least one of the bins is empty"};
    run::Argument<size_t> max_bin_occupancy{"max-bin-occupancy", "maximal occupancy of a bin",
                                            std::numeric_limits<size_t>::max()};
    run::Argument<unsigned> n_threads{"n-threads", "number of threads", 1};
    run::Argument<unsigned> seed{"seed", "random seed to initialize the generator used for sampling", 1234567};
    run::Argument<std::string> disabled_branches{"disabled-branches",
                                                 "list of branches to disabled in the input tuples", ""};

    run::Argument<std::string> input_spec{"input_spec", "input path with spectrums for all the samples"};
    run::Argument<std::string> tau_type{"tau_type", "Tau type"};

};

namespace {
struct SourceDesc {
    using Tau = tau_tuple::Tau;
    using TauTuple = tau_tuple::TauTuple;
    using SampleType = analysis::SampleType;

    SourceDesc(const std::string& _name, const std::vector<std::string>& _file_names,
               double _weight, const std::set<std::string>& _disabled_branches) :
        name(_name), file_names(_file_names), disabled_branches(_disabled_branches), weight(_weight),
        current_n_processed(0), files_n_total(_file_names.size()), entries_file(666), total_n_processed(0)
    {
        if(file_names.empty())
            throw analysis::exception("Empty list of files for the source '%1%'.") % name;
        if(weight <= 0)
            throw analysis::exception("Invalid source weight for the source '%1%'.") % name;
        if(!files_n_total)
            throw analysis::exception("Empty source '%1%'.") % name;

    }

    SourceDesc(const SourceDesc&) = delete;
    SourceDesc& operator=(const SourceDesc&) = delete;

    // bool HasNextTau() const { if(current_file_index < files_n_total-1) return true;
    //                           else return current_n_processed < entries_file; }

    bool DoNextStep()
    {

      if(current_file_index == files_n_total-1 && current_n_processed >= entries_file) return false;

      while(!current_file_index || current_n_processed == entries_file) {
          if(!current_file_index)
              current_file_index = 0;
          else
              ++(*current_file_index);
          if(*current_file_index >= file_names.size())
              throw analysis::exception("The expected number of events = %1% is bigger than the actual number of"
                                        " events in source '%2%'.") % entries_file % name;
          current_n_processed = 0;
          const std::string& file_name = file_names.at(*current_file_index);
          current_tuple.reset();
          current_file = root_ext::OpenRootFile(file_name);
          current_tuple = std::make_shared<TauTuple>("taus", current_file.get(), true, disabled_branches);
          entries_file = current_tuple->GetEntries();
          if(!entries_file)
            throw analysis::exception("Root file inside %1% is empty.") % name;
      }
      ++total_n_processed;
      current_tuple->GetEntry(current_n_processed++);
      // (*current_tuple)().sampleType = static_cast<int>(sample_type);
      return true;
    }

    const Tau& GetNextTau()
    {
      return current_tuple->data();
    }

    size_t GetNumberOfEvents() const { return entries_file; }
    const std::string& GetName() const { return name; }
    const std::vector<std::string>& GetFileNames() const { return file_names; }
    double GetWeight() const { return weight; }
    // const std::string& GetBinName() const { return bin_name; }
    // void SetBinName(const std::string& _bin_name) { bin_name = _bin_name; }

  private:
      const std::string name;
      const std::vector<std::string> file_names;
      const std::set<std::string> disabled_branches;
      const double weight;
      // const SampleType sample_type;
      // std::string bin_name;
      std::shared_ptr<TFile> current_file;
      std::shared_ptr<TauTuple> current_tuple;

      boost::optional<ULong64_t> current_file_index;

      // types should be checked
      ULong64_t current_n_processed;
      ULong64_t files_n_total;;
      size_t entries_file;
      size_t total_n_processed;//, total_n_events;
  };


struct EntryDesc {
    using TauType = analysis::TauType;
    using SampleType = analysis::SampleType;

    std::string name;
    std::map< std::string, std::vector<std::string>> data_files;
    std::map< std::string, std::string> spectrum_files;
    double weight;

    EntryDesc(const analysis::PropertyConfigReader::Item& item,
              const std::string& base_input_dir,
              const std::string& base_spectrum_dir)
    {
        using boost::regex;
        using boost::regex_match;
        using boost::filesystem::path;
        using boost::make_iterator_range;
        using boost::filesystem::directory_iterator;
        using boost::filesystem::is_directory;
        using boost::filesystem::is_regular_file;

        name = item.name;
        std::cout << name << std::endl;
        const std::string dir_pattern_str = item.Get<std::string>("dir");
        const std::string file_pattern_str = item.Get<std::string>("file");
        weight = item.Has("weight") ? item.Get<double>("weight") : 1;

        const path base_input_path(base_input_dir);
        if(!is_directory(base_input_dir))
            throw analysis::exception("The base directory '%1%' (--input) does not exists.") % base_input_dir;

        const regex dir_pattern(base_input_dir + "/" + dir_pattern_str); // Patern for Big root-tuples

        bool has_dir_match = false;
        for(const auto& sample_dir_entry : make_iterator_range(directory_iterator(base_input_path))) {
            if(!is_directory(sample_dir_entry) || !regex_match(sample_dir_entry.path().string(), dir_pattern)) continue;

            std::cout << sample_dir_entry << " - dataset" << std::endl;
            const std::string dir_name = sample_dir_entry.path().filename().string();
            const path spectrum_file(base_spectrum_dir + "/" + dir_name + ".root");

            if(!is_regular_file(spectrum_file)){ //make sure SpectrumHist is found
              throw analysis::exception("No spectrum file are found for entry '%1%'") % sample_dir_entry;
              continue;
            }
            std::cout << spectrum_file << " - spectrum" << std::endl;

            has_dir_match = true;

            const regex file_pattern(sample_dir_entry.path().string() + "/" + file_pattern_str + ".root");
            bool has_file_match = false;
            for(const auto& file_entry : make_iterator_range(directory_iterator(sample_dir_entry.path()))) {
                if(is_directory(file_entry) || !regex_match(file_entry.path().string(), file_pattern)) continue;
                has_file_match = true;

                const std::string file_name = file_entry.path().string();
                // const std::string bin_name = GetBinName(file_name);
                // if(!bin_files.count(bin_name))
                //     tau_types.insert(GetTauType(bin_name));
                data_files[dir_name].push_back(file_name);
            }
            spectrum_files[dir_name] = spectrum_file.string();

            if(!has_file_match)
                throw analysis::exception("No files are found for entry '%1%' sample %2% with pattern '%3%'")
                      % name % sample_dir_entry % file_pattern_str;
        }

        if(!has_dir_match)
            throw analysis::exception("No samples are found for entry '%1%' with pattern '%2%'")
                  % name % dir_pattern_str;

        if(!(spectrum_files.size()==data_files.size()))
            throw analysis::exception("Not all spectrum hists are found");
    }
};

class SpectrumHists {
public:

    SpectrumHists(const std::vector<EntryDesc>& entries,
                  const std::vector<double>& pt_bins,
                  const std::vector<double>& eta_bins,
                  const std::string& tautype)
    {
      if(tautype=="all") types=std::vector<std::string>{"e","mu","tau","jet"};
      else types=std::vector<std::string>{tautype};

      // Collect histograms into one[tau_types]:
      for(const EntryDesc& Desc_ : entries){
      BOOST_FOREACH(auto DescEntry, Desc_.spectrum_files){
          current_file = std::make_shared<TFile>(DescEntry.second.c_str());
          for (const std::string& type :types){
            std::shared_ptr<TH2D> hist_oftype((TH2D*)current_file->Get(("eta_pt_hist_"+type).c_str()));
            //To get Number of entries in ranges
            entries_dataset_tautype[DescEntry.first][type] = hist_oftype->Integral();
            if(hists_tautype[type]) hists_tautype[type]->Add(hist_oftype.get());
            else hists_tautype[type] = hist_oftype;
          }
      }}

      // Rebin
      // for (const std::string& type :{"e","mu","tau","jet"}){
      //   Rebin(hists_tautype[type], pt_bins, eta_bins);
      // }
    }


    SpectrumHists(const SpectrumHists&) = delete;
    SpectrumHists& operator=(const SpectrumHists&) = delete;

    void CalculateProbability(const Double_t pt_threshold) // TO BE CHECKED AND DISCUSSED!
    {
      Double_t factor = 1.;

      // Right now just inverse histograms
      for (const std::string& type :types){

      Int_t n_eta = hists_tautype[type]->GetNbinsX();
      Int_t n_pt = hists_tautype[type]->GetNbinsY();
      Int_t max_n_pt = hists_tautype.at(type)->GetYaxis()->FindBin(pt_threshold);
      // std::cout<<n_eta << " " << n_pt << " " << max_n_pt << std::endl;

      // Find minimum of histogram
      Double_t minimum = hists_tautype.at(type)->Integral(); //just need any number
      // std::cout<< "min: " << minimum << std::endl;
      for(Int_t i_pt = 1; i_pt <= max_n_pt; i_pt++){
        for(Int_t i_eta = 1; i_eta <= n_eta; i_eta++){
          Double_t content = hists_tautype.at(type)->GetBinContent(i_eta,i_pt);
          if(content <= minimum && content!=0) minimum = content;
          // std::cout << "min: " << minimum << " cont: " << content << std::endl;
          }}

      if (minimum == hists_tautype.at(type)->GetBinContent(1,1))
          throw analysis::exception("Minimal bin is not found");

      // Maximal probability = 1.0 corresponds to the bin with minimal
      // number of events under the pt threshold
      // All bins bigger then pt threshold has p=1.0
      // (maybe not correct corresponding to the eta)
      for(Int_t i_pt = 1; i_pt <= n_pt; i_pt++){
        for(Int_t i_eta = 1; i_eta <= n_eta; i_eta++){

          Double_t content = hists_tautype.at(type)->GetBinContent(i_eta,i_pt);
          if(content==0)
            hists_tautype.at(type)->SetBinContent(i_eta,i_pt,factor);
          else if(i_pt<=max_n_pt)
            hists_tautype.at(type)->SetBinContent(i_eta,i_pt,
                                                  factor*minimum/content);
          else
            hists_tautype.at(type)->SetBinContent(i_eta,i_pt,factor);

          // std::cout <<i_eta << " " << i_pt << " " << hists_tautype.at(type)->GetBinContent(i_eta,i_pt) << std::endl;
      }}
    }
    }

    double GetProbability(const std::string type, const double pt, const double eta)
    {
      // efficiency ???
      return hists_tautype.at(type)->GetBinContent(
             hists_tautype.at(type)->GetXaxis()->FindBin(eta),
             hists_tautype.at(type)->GetYaxis()->FindBin(pt)
           );
    }

    const std::map<std::string, Double_t> GetSetsProbability()
    {
      std::map<std::string, Double_t> Probs;
      BOOST_FOREACH(auto entries, entries_dataset_tautype){
        Probs[entries.first] = 0.0;
        for (const std::string& type :types)
          Probs[entries.first] += entries_dataset_tautype[entries.first][type];
      }
      return Probs;
    }

    const std::map<std::string, Double_t> GetTauTypeProbability()
    {
      std::map<std::string, Double_t> Probs;
      for (const std::string& type :types){
      Probs[type] = 0.0;
      BOOST_FOREACH(auto entries, entries_dataset_tautype)
          Probs[type] += entries_dataset_tautype[entries.first][type];
      }
      auto it = std::min_element(std::begin(Probs), std::end(Probs),
                           [](const auto& l, const auto& r) { return l.second < r.second; });
      Double_t min_ = it->second;
      BOOST_FOREACH(auto tau_prob_, Probs) tau_prob_.second = min_/tau_prob_.second;
      BOOST_FOREACH(auto tau_prob_, Probs) std::cout << tau_prob_.first << " " << tau_prob_.second << "\n";
      return Probs;
    }

    const Double_t GetNumberOfEntries(const std::string &type, const std::string &data_name)
    {
      return entries_dataset_tautype.at(data_name).at(type);
    }

private:

  // void Rebin2D(std::shared_ptr<TH2D> hist, const std::vector<double>& pt_bins, const std::vector<double>& eta_bins)
  // {
  //   // ?
  // }
  std::vector<std::string> types;
  std::shared_ptr<TFile> current_file;
  std::map<std::string, std::shared_ptr<TH2D>> hists_tautype;
  std::map<std::string, std::map<std::string, Double_t>> entries_dataset_tautype;

};

class DataSetProcessor {
public:
    using TauType = analysis::TauType;
    using Tau = tau_tuple::Tau;
    using TauTuple = tau_tuple::TauTuple;
    using Generator = std::mt19937_64;
    // using Uniform = std::uniform_int_distribution<size_t>;
    using Uniform = std::uniform_real_distribution<double>;
    using Discret = std::discrete_distribution<int>;

    DataSetProcessor(const std::vector<EntryDesc>& entries, const std::vector<double>& pt_bins,
                     const std::vector<double>& eta_bins, bool calc_weights, size_t max_bin_occupancy,
                     Generator& _gen, const std::set<std::string>& disabled_branches,
                     bool verbose, const std::string& tautype) :
                     tau_mode(tautype), pt_threshold(100), // pt_threshold 100 GeV
                     gen(&_gen)
    {
    if(verbose) std::cout << "Adding sources..." << std::endl;
    SetSources(entries, calc_weights, disabled_branches);

    if(verbose) std::cout << "Reading spectrums..." << std::endl;
    spectrum = std::make_shared<SpectrumHists>(entries, pt_bins, eta_bins, tau_mode);
    spectrum->CalculateProbability(pt_threshold); //100 GeV

    // GetProbability of dataset
    // ~ Number of entries per all/one TauType
    BOOST_FOREACH(auto probab, spectrum->GetSetsProbability()){dataset_names.push_back(probab.first);
                                                               dataset_probs.push_back(probab.second);}

    // Setup randomizers
    dist_dataset = Discret(dataset_probs.begin(), dataset_probs.end());
    dist_uniform = Uniform(0.0, 1.0);
    if(tau_mode == "all") TauTypeProb = spectrum->GetTauTypeProbability();
    }

    bool DoNextStep()
    {
      current_dataset = dataset_names[dist_dataset(*gen)];
      while(sources.at(current_dataset)->DoNextStep())
      {
        const Tau& current_tuple = sources.at(current_dataset)->GetNextTau();
        if(TauSelection(current_tuple)) return true;
        current_dataset = dataset_names[dist_dataset(*gen)];
      }
      return false;
    }

    const Tau& GetNextTau(){return sources.at(current_dataset)->GetNextTau();};

private:
    bool TauSelection(const Tau& tuple)
    {
      const auto gen_match = static_cast<analysis::GenLeptonMatch>(tuple.lepton_gen_match);
      const auto sample_type = static_cast<analysis::SampleType>(tuple.sampleType);
      const std::string tau_type = analysis::ToString(analysis::GenMatchToTauType(gen_match, sample_type));
      if(tau_mode=="all" && dist_uniform(*gen) >= TauTypeProb.at(tau_type)) return false;
      else if(tau_mode!=tau_type) return false;
      Double_t pt = tuple.tau_pt;
      Double_t abs_eta = abs(tuple.tau_eta);
      if( pt<=0.0 || pt>=1000.0 || abs_eta >= 2.3) return false;
      else if(pt>=pt_threshold) return true;
      else if(dist_uniform(*gen) <= spectrum->GetProbability(tau_type, pt, abs_eta)) return true;
      return false;
    }

    void SetSources(const std::vector<EntryDesc>& entries, bool calc_weights,
                    const std::set<std::string>& disabled_branches)
    {
      for(const EntryDesc& Desc_ : entries)
      {
        BOOST_FOREACH(auto DescEntry, Desc_.data_files)
        {
          const std::string& dataname = DescEntry.first;
          std::shared_ptr<SourceDesc> source = std::make_shared<SourceDesc>(dataname,
                                  DescEntry.second,Desc_.weight, disabled_branches);
          sources[dataname] = source;
        }
      }

    }


private:
    std::string current_dataset;
    std::map<std::string, std::shared_ptr<SourceDesc>> sources;
    std::vector<Double_t> dataset_probs;
    std::vector<std::string> dataset_names;
    std::map<std::string, Double_t> TauTypeProb;
    std::shared_ptr<SpectrumHists> spectrum;
    const std::string tau_mode;
    const Double_t pt_threshold;

    Generator* gen;
    Discret dist_dataset;
    Uniform dist_uniform;

};


} // anonymous namespace

// outer development layer:
namespace analysis {
class ShuffleMergeSpectral {
public:
    using Tau = tau_tuple::Tau;
    using TauTuple = tau_tuple::TauTuple;
    using Generator = DataSetProcessor::Generator;

    ShuffleMergeSpectral(const Arguments& _args) :
        args(_args), pt_bins(ParseBins(args.pt_bins())),
        eta_bins(ParseBins(args.eta_bins())), takeTauType(args.tau_type())
    {
    // bool verbose = 1;

		if(args.n_threads() > 1)
            ROOT::EnableImplicitMT(args.n_threads());

        const auto disabled_branches_vec = SplitValueList(args.disabled_branches(), false, " ,");
        disabled_branches.insert(disabled_branches_vec.begin(), disabled_branches_vec.end());

        PrintBins("pt bins", pt_bins);
        PrintBins("eta bins", eta_bins);
        const auto all_entries = LoadEntries(args.cfg());

        // if(verbose){
        // for (unsigned int i = 0; i < all_entries.size(); i++){
        //   std::cout << "entry in all_entries-> " << std::endl;
        //   BOOST_FOREACH( auto it, all_entries[i].data_files)
        //   {
        //      std::cout << "name-> " << it.first << std::endl;
        //      for ( int j = 0; j < (int)it.second.size(); j++)
        //      std::cout << "path files-> " << it.second[j] << "\n";
        //   }
        //   BOOST_FOREACH( auto it, all_entries[i].spectrum_files)
        //   {
        //      std::cout << "name-> " << it.first << std::endl;
        //      std::cout << "spectrum files-> " << it.second << std::endl;
        //
        //   }
        // }
        // }

        if(args.mode() == MergeMode::MergeAll) {
            entries[args.output()] = all_entries;
        } else if(args.mode() == MergeMode::MergePerEntry) {
            for(const auto& entry : all_entries) {
                const std::string output_name = args.output() + "/" + entry.name + ".root";
                std::cout << output_name << std::endl;
                entries[output_name].push_back(entry);
            }
        } else {
            throw exception("Unsupported merging mode = '%1%'.") % args.mode();
        }
    }

    void Run()
    {
      //
      Generator gen(args.seed());

      for(const auto& e : entries) {
          const std::string& file_name = e.first;
          const std::vector<EntryDesc>& entry_list = e.second;
          std::cout << "Processing:";
          for(const auto& entry : entry_list)
              std::cout << ' ' << entry.name;
          std::cout << "\nOutput: " << file_name << std::endl;
          auto output_file = root_ext::CreateRootFile(file_name, ROOT::kLZ4, 4);
          auto output_tuple = std::make_shared<TauTuple>("taus", output_file.get(), false);
          std::cout << "Creating event bin map..." << std::endl;
          // EventBinMap bin_map(entry_list, pt_bins, eta_bins, args.calc_weights(), args.max_bin_occupancy(), gen,
          //                     n_events_per_file, disabled_branches, true);
          // tools::ProgressReporter reporter(10, std::cout, "Sampling taus...");
          // reporter.SetTotalNumberOfEvents(bin_map.GetNumberOfRemainingEvents());

          DataSetProcessor processor(entry_list, pt_bins, eta_bins, args.calc_weights(),
                                args.max_bin_occupancy(), gen, disabled_branches, true, takeTauType);

          // Test
          // processor.GetNextTau();

          size_t n_processed = 0;
          // bool has_empty_bins = false;

          while(processor.DoNextStep()){
            // double weight;
            // bool last_tau_in_bin;
            const auto& tau = processor.GetNextTau();
            (*output_tuple)() = tau;
            // if(args.calc_weights())
            //     (*output_tuple)().trainingWeight = static_cast<float>(weight);
            output_tuple->Fill();
            if(++n_processed % 1000 == 0)
              std::cout << n_processed << " is processed" << std::endl;
                // reporter.Report(n_processed);
          }

          // reporter.Report(n_processed, true);
          std::cout << "Writing output tuples..." << std::endl;
          output_tuple->Write();
          output_tuple.reset();
          std::cout << file_name << " has been successfully created." << std::endl;
      }

      std::cout << "All entries has been merged." << std::endl;
    }

private:
    std::vector<EntryDesc> LoadEntries(const std::string& cfg_file_name)
    {
        std::vector<EntryDesc> entries;
        PropertyConfigReader reader;
        std::cout << cfg_file_name << std::endl;
        reader.Parse(cfg_file_name);
        for(const auto& item : reader.GetItems()){
            entries.emplace_back(item.second, args.input(), args.input_spec());
        }
        return entries;
    }


    static std::vector<double> ParseBins(const std::string& bins_str)
    {
        const auto& split_bin_strs = SplitValueList(bins_str, true, ", \t", true);
        std::vector<double> bins;
        for(const std::string& bin_str : split_bin_strs) {
            if(bin_str.empty()) continue;
            const double bin = Parse<double>(bin_str);
            if(!bins.empty() && bins.back() >= bin)
                throw exception("Invalid bins order");
            bins.push_back(bin);
        }
        return bins;
    }

    static void PrintBins(const std::string& prefix, const std::vector<double>& bins)
    {
        std::cout << prefix << ": ";
        if(!bins.empty()) {
            std::cout << bins.at(0);
            for(size_t n = 1; n < bins.size(); ++n)
                std::cout << ", " << bins.at(n);
        }
        std::cout << ".\n";
    }

private:
    Arguments args;
    std::map<std::string, std::vector<EntryDesc>> entries;
    const std::vector<double> pt_bins, eta_bins;
    std::set<std::string> disabled_branches;
    const std::string takeTauType;
};

} // namespace analysis

PROGRAM_MAIN(analysis::ShuffleMergeSpectral, Arguments)
