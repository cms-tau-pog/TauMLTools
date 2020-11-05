/*! Sheffle and merge datasets using Histograms created with CreateSpectralHists.cxx
*/

#include <fstream>
#include <random>
#include <string>
#include <iostream>
#include <boost/algorithm/string.hpp>
#include <boost/regex.hpp>
#include <boost/foreach.hpp>
#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/json_parser.hpp>

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

struct Arguments {
    run::Argument<std::string> cfg{"cfg", "configuration file with the list of input sources"};
    run::Argument<std::string> input{"input", "input path with tuples for all the samples"};
    run::Argument<std::string> output{"output", "output, depending on the merging mode: MergeAll - file,"
                                                " MergePerEntry - directory."};
    run::Argument<std::string> pt_bins{"pt-bins", "pt bins"};
    run::Argument<std::string> eta_bins{"eta-bins", "eta bins"};
    run::Argument<analysis::MergeMode> mode{"mode", "merging mode: MergeAll or MergePerEntry"};
    run::Argument<size_t> max_entries{"max-entries", "maximal number of entries in output train+test tuples",
                                            std::numeric_limits<size_t>::max()};
    run::Argument<unsigned> n_threads{"n-threads", "number of threads", 1};
    run::Argument<unsigned> seed{"seed", "random seed to initialize the generator used for sampling", 1234567};
    run::Argument<std::string> disabled_branches{"disabled-branches",
                                                 "list of branches to disabled in the input tuples", ""};
    run::Argument<std::string> path_spectrum{"input-spec", "input path with spectrums for all the samples"};
    run::Argument<std::string> tau_ratio{"tau-ratio", "ratio of tau types in the final spectrum"};
    run::Argument<double> start_entry{"start-entry", "starting ratio from which file will be processed", 0};
    run::Argument<double> end_entry{"end-entry", "end ratio until which file will be processed", 1};
    run::Argument<double> test_size{"test-size", "the ralative size of the testing dataset r=N_test/N_all",0.1};
    run::Argument<double> pt_threshold{"pt-threshold", "pt threshold to take all candidates",100};
    run::Argument<double> exp_disbalance{"exp-disbalance", "maximal expected disbalance between low pt and high pt regions",0};

};

struct SourceDesc {
    using Tau = tau_tuple::Tau;
    using TauType = analysis::TauType;
    using TauTuple = tau_tuple::TauTuple;
    using SampleType = analysis::SampleType;

    SourceDesc(const std::string& _name,const std::string&  _group_name, const std::vector<std::string>& _file_names,
               const std::set<std::string>& _disabled_branches, double _begin_rel, double _end_rel,
               std::set<TauType> _tautypes) :
        name(_name), group_name(_group_name), file_names(_file_names),
        disabled_branches(_disabled_branches), entry_begin_rel(_begin_rel), entry_end_rel(_end_rel),
        tau_types(_tautypes), current_n_processed(0), files_n_total(_file_names.size()),
        entries_file(std::numeric_limits<size_t>::infinity()), total_n_processed(0), total_n_input(0)
    {
        if(file_names.empty())
          throw analysis::exception("Empty list of files for the source '%1%'.") % name;
        if(!files_n_total)
          throw analysis::exception("Empty source '%1%'.") % name;
        dataset_hash = std::hash<std::string>{}(name) % std::numeric_limits<int>::max();
        datagroup_hash = std::hash<std::string>{}(group_name) % std::numeric_limits<int>::max();
    }

    SourceDesc(const SourceDesc&) = delete;
    SourceDesc& operator=(const SourceDesc&) = delete;

    bool DoNextStep()
    {
      do {
        if(current_file_index == files_n_total-1 && current_n_processed >= entries_file) return false;
        while(!current_file_index || current_n_processed == entries_file) {
            if(!current_file_index)
                current_file_index = 0;
            else
                ++(*current_file_index);
            if(*current_file_index >= file_names.size())
                throw analysis::exception("The expected number of events = %1% is bigger than the actual number of"
                                          " events in source '%2%'.") % entries_file % name;
            const std::string& file_name = file_names.at(*current_file_index);
            current_tuple.reset();
            current_file = root_ext::OpenRootFile(file_name);
            current_tuple = std::make_shared<TauTuple>("taus", current_file.get(), true, disabled_branches);
            entries_file = current_tuple->GetEntries();
            current_n_processed =  std::floor(entry_begin_rel * entries_file);
            entries_end = std::floor(entry_end_rel * entries_file);
            if(!entries_file)
              throw analysis::exception("Root file inside %1% is empty.") % name;
        }
        current_tuple->GetEntry(current_n_processed++);
        const auto gen_match = static_cast<analysis::GenLeptonMatch>((*current_tuple)().lepton_gen_match);
        const auto sample_type = static_cast<analysis::SampleType>((*current_tuple)().sampleType);
        current_tau_type = analysis::GenMatchToTauType(gen_match, sample_type);
        ++total_n_input;
      }
      while (tau_types.find(current_tau_type) == tau_types.end());
      ++total_n_processed;
      (*current_tuple)().dataset_id = dataset_hash;
      (*current_tuple)().dataset_group_id = datagroup_hash;
      return true;
    }

    const Tau& GetNextTau() { return current_tuple->data(); }
    const size_t GetNumberOfProcessed() const { return total_n_processed; }
    const size_t GetNumberOfInput() const { return total_n_input; }
    const TauType GetType() { return current_tau_type; }
    const std::string& GetName() const { return name; }
    const std::string& GetGroupName() const { return group_name; }
    const int GetNameH() const { return dataset_hash; }
    const int GetGroupNameH() const { return datagroup_hash; }
    const std::vector<std::string>& GetFileNames() const { return file_names; }

  private:
      const std::string name;
      const std::string group_name;
      const std::vector<std::string> file_names;
      const std::set<std::string> disabled_branches;
      const double entry_begin_rel;
      const double entry_end_rel;
      const std::set<TauType> tau_types;
      std::shared_ptr<TFile> current_file;
      std::shared_ptr<TauTuple> current_tuple;
      boost::optional<ULong64_t> current_file_index;
      ULong64_t current_n_processed;
      ULong64_t files_n_total;;
      size_t entries_file;
      size_t entries_end;
      size_t total_n_processed;
      size_t total_n_input;
      TauType current_tau_type;
      int dataset_hash;
      int datagroup_hash;
  };


struct EntryDesc {
    using TauType = analysis::TauType;
    using SampleType = analysis::SampleType;

    std::string name;
    std::map< std::string, std::vector<std::string>> data_files;
    std::map< std::string, std::string> spectrum_files;
    // std::vector< std::string > tau_types;
    std::set<TauType> tau_types;
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
        using boost::split;

        name = item.name;
        std::cout << name << std::endl;
        const std::string dir_pattern_str = item.Get<std::string>("dir");
        const std::string file_pattern_str = item.Get<std::string>("file");
        weight = item.Has("weight") ? item.Get<double>("weight") : 1;
        const std::string tau_types_str = item.Get<std::string>("types");
        tau_types = SplitValueListT<TauType, std::set<TauType>>(tau_types_str, false, ",");

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

          if(!is_regular_file(spectrum_file))
            throw analysis::exception("No spectrum file are founupperd for entry '%1%'") % sample_dir_entry;
          std::cout << spectrum_file << " - spectrum" << std::endl;

          has_dir_match = true;

          const regex file_pattern(sample_dir_entry.path().string() + "/" + file_pattern_str + ".root");
          bool has_file_match = false;
          for(const auto& file_entry : make_iterator_range(directory_iterator(sample_dir_entry.path()))) {
            if(is_directory(file_entry) || !regex_match(file_entry.path().string(), file_pattern)) continue;
            has_file_match = true;
            const std::string file_name = file_entry.path().string();
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

    using TauType = analysis::TauType;

public:

    SpectrumHists(const std::string& groupname_, const std::vector<double>& pt_bins,
                  const std::vector<double>& eta_bins,
                  const Double_t exp_disbalance_,
                  const std::set<TauType>& tau_types_):
                  groupname(groupname_), exp_disbalance(exp_disbalance_), ttypes(tau_types_)
    {
      std::cout << "Initialization of group SpectrumHists..." << std::endl;
      for(TauType type: ttypes){
        const char *name = (groupname+"_"+analysis::ToString(type)).c_str();
        ttype_prob[type] = std::make_shared<TH2D>(name,name,eta_bins.size()-1,&eta_bins[0],
                                                   pt_bins.size()-2,&pt_bins[0]);
        ttype_entries[type] = std::make_shared<TH2D>(name,name,eta_bins.size()-1,&eta_bins[0],
                                                  pt_bins.size()-1,&pt_bins[0]);
      }
      // testing for the uniform distr.
      target_hist = SetTargetHist_test(1.0, pt_bins, eta_bins);
      n_entries = 0;
    }

    SpectrumHists(const SpectrumHists&) = delete;
    SpectrumHists& operator=(const SpectrumHists&) = delete;

    void AddHist(const std::string& dataset, const std::string& path_spectrum_file)
    {
      std::shared_ptr<TFile> current_file = std::make_shared<TFile>(path_spectrum_file.c_str());
      for (TauType type: ttypes){
        std::shared_ptr<TH2D> hist_ttype((TH2D*)current_file->Get(("eta_pt_hist_"+analysis::ToString(type)).c_str()));
        auto map_bins = GetBinMap(ttype_entries.at(type), hist_ttype);
        for(unsigned int ix=0; ix<=map_bins.first.size(); ix++)
          for(unsigned int iy=0; iy<=map_bins.second.size(); iy++)
            ttype_entries[type]->SetBinContent(map_bins.first[ix],map_bins.second[iy],
              ttype_entries[type]->GetBinContent(map_bins.first[ix],map_bins.second[iy])
              + hist_ttype->GetBinContent(ix,iy));
        //To get Number of entries in ranges
        entries[dataset][type] = hist_ttype->Integral();
        n_entries += hist_ttype->GetEntries(); //needed for monitoring
      }
    }

    void CalculateProbability() // TO BE CHECKED AND DISCUSSED!
    {
      for (TauType type: ttypes) {
        if(CheckZeros(ttype_entries.at(type)))
          throw analysis::exception("Empty spectrum histogram for groupname: '%1%' and tau type: '%2%'") % groupname % type;
        std::shared_ptr<TH2D> ratio_h((TH2D*)target_hist->Clone());
        for(int i_x = 1; i_x<=ratio_h->GetNbinsX(); i_x++)
          for(int i_y = 1; i_y<=ratio_h->GetNbinsY(); i_y++)
            ratio_h->SetBinContent(i_x,i_y,
              target_hist->GetBinContent(i_x,i_y)/ttype_entries.at(type)->GetBinContent(i_x,i_y)
            );
        ttype_prob.at(type) = ratio_h;
        Int_t MaxBin = ttype_prob.at(type)->GetMaximumBin();
        Int_t x,y,z;
        ttype_prob.at(type)->GetBinXYZ(MaxBin, x, y, z);
        ttype_prob.at(type)->Scale(1.0/ttype_prob.at(type)->GetBinContent(x,y));


        // double scale = 1.0;
        // if(exp_disbalance!=0.0)
        //   scale = std::min(1.0, exp_disbalance*Nhigh_pt[type]*ttype_hists.at(type)->GetBinContent(x,y));
        // if(scale!=1.0)
        //   std::cout << "WARNING: Disbalance is bigger, scale factor will be applied" << std::endl;
        // std::cout << "max pt_bin eta_bin: " << y << " " << x
        //           << " MaxBin: " << ttype_hists.at(type)->GetBinContent(x,y)
        //           << " scale factor: " << scale << "\n";
        // ttype_hists.at(type)->Scale(scale/ttype_hists.at(type)->GetBinContent(x,y));
      }
    }

    const std::map<TauType, Double_t> GetTauTypeEntriesFinal() const
    {
      std::map<TauType, Double_t> EntriesFinal;
      for (TauType type: ttypes) {
        EntriesFinal[type] = 0.0;
        std::shared_ptr<TH2D> entries_count((TH2D*)ttype_entries.at(type)->Clone());
        for(int i_x = 1; i_x<=ttype_prob.at(type)->GetNbinsX(); i_x++)
          for(int i_y = 1; i_y<=ttype_prob.at(type)->GetNbinsY(); i_y++)
            entries_count->SetBinContent(i_x,i_y,
              entries_count->GetBinContent(i_x,i_y)*ttype_prob.at(type)->GetBinContent(i_x,i_y)
            );
        EntriesFinal.at(type) = entries_count->Integral();
      }
      return EntriesFinal;
    }

    const double GetProbability(const TauType type, const double pt, const double eta) const
    {
      return ttype_prob.at(type)->GetBinContent(
             ttype_prob.at(type)->GetXaxis()->FindBin(eta),
             ttype_prob.at(type)->GetYaxis()->FindBin(pt));
    }

    const std::map<std::string, Double_t> GetDataSetsProbabilities() const
    {
      std::map<std::string, Double_t> Probs;
      for(auto entry: entries){
        Probs[entry.first] = 0.0;
        for (TauType type: ttypes)
          Probs[entry.first] += entry.second.at(type);
      }
      return Probs;
    }


    void SaveHists(const std::string& output)
    {
      if(!boost::filesystem::exists(output)) boost::filesystem::create_directory(output);
      TFile file = TFile((output+"/"+groupname+".root").c_str(),"RECREATE");
      for (TauType type: ttypes)
        ttype_prob[type]->Write((analysis::ToString(type)+"_prob").c_str());
      file.Close();
      TFile file_n = TFile((output+"/"+groupname+"_n.root").c_str(),"RECREATE");
      for (TauType type: ttypes)
        ttype_entries[type]->Write((analysis::ToString(type)+"_entries").c_str());
      file_n.Close();
    }

    const size_t GetEntries() { return n_entries; }

private:
  std::shared_ptr<TH2D> SetTargetHist_test(const double scale,const std::vector<double>& pt_bins,
                                                               const std::vector<double>& eta_bins)
  {
    std::shared_ptr<TH2D> tartget_hist = std::make_shared<TH2D>("tartget","tartget",
                                          eta_bins.size()-1,&eta_bins[0],
                                          pt_bins.size()-2,&pt_bins[0]);
    for(Int_t i_pt = 1; i_pt <= tartget_hist->GetNbinsY(); i_pt++)
      for(Int_t i_eta = 1; i_eta <= tartget_hist->GetNbinsX(); i_eta++)
        tartget_hist->SetBinContent(i_eta,i_pt,1.0);
    tartget_hist->Scale(scale/tartget_hist->Integral());
    return tartget_hist;
  }

  static std::pair<std::vector<int>,std::vector<int>> GetBinMap(const std::shared_ptr<TH2D>& newbins,
                                                                const std::shared_ptr<TH2D>& oldbins)
  {
    // check over x axis
    std::vector<int> x_map;
    for(Int_t new_x=0; new_x<=newbins->GetNbinsX(); new_x++){
      float up_x = (float)newbins->GetXaxis()->GetBinUpEdge(new_x);
      for(Int_t old_x=x_map.size(); old_x<=oldbins->GetNbinsX(); old_x++){
        x_map.push_back(new_x);
        if(up_x==(float)oldbins->GetXaxis()->GetBinUpEdge(old_x))
          break;
        else if(old_x==oldbins->GetNbinsX())
          throw analysis::exception("New histogram's bins over eta are inconsistent.");
      }
    }

    // check over y axis
    std::vector<int> y_map;
    for(Int_t new_y=0; new_y<=newbins->GetNbinsY(); new_y++){
      float up_y = newbins->GetYaxis()->GetBinUpEdge(new_y);
      for(Int_t old_y=y_map.size(); old_y<=oldbins->GetNbinsY(); old_y++){
        y_map.push_back(new_y);
        if((float)oldbins->GetYaxis()->GetBinUpEdge(old_y)==(float)up_y)
          break;
        else if(old_y==oldbins->GetNbinsY())
          throw analysis::exception("New histogram's bins over eta are inconsistent.");
      }
    }
    return std::make_pair(x_map,y_map);
  }

  static bool CheckZeros(const std::shared_ptr<TH2D>& hist)
  {
    for(Int_t i_pt = 1; i_pt <= hist->GetNbinsY(); i_pt++)
      for(Int_t i_eta = 1; i_eta <= hist->GetNbinsX(); i_eta++)
        if(hist->GetBinContent(i_eta,i_pt)==0)
          return true;
    return false;
  }

private:
  const std::string& groupname;
  std::map<TauType, std::shared_ptr<TH2D>> ttype_entries; // pair<tau_type, probability_hist>
  std::map<TauType, std::shared_ptr<TH2D>> ttype_prob; // pair<tau_type, probability_hist>
  std::map<std::string, std::map<TauType, Double_t>> entries; // pair<dataset, pair<tau type, n entries>>
  size_t n_entries; // needed for monitoring
  std::shared_ptr<TH2D> target_hist;
  Double_t pt_threshold, exp_disbalance;

public:
  const std::set<TauType> ttypes; // vector of tau types e.g: e,tau...

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
                     const std::vector<double>& eta_bins, Generator& _gen,
                     const std::set<std::string>& disabled_branches, bool verbose,
                     const std::map<TauType, Double_t> tau_ratio, const Double_t start_entry,
                     const Double_t end_entry, const double pt_threshold_, const Double_t exp_disbalance_) :
                     pt_max(pt_bins.back()), pt_min(pt_bins[0]), eta_max(eta_bins.back()),
                     pt_threshold(pt_threshold_), exp_disbalance(exp_disbalance_), gen(&_gen)
    {
      if(verbose) std::cout << "Loading Data Groups..." << std::endl;
      LoadDataGroups(entries, pt_bins, eta_bins, disabled_branches, start_entry, end_entry);

      if(verbose) std::cout << "Calculating probabilitis..." << std::endl;
      for(auto spectrum: spectrums)
        spectrum.second->CalculateProbability();

      if(verbose) std::cout << "Saving histograms..." << std::endl;
      for(auto spectrum: spectrums)
        spectrum.second->SaveHists("./out");

      if(verbose) std::cout << "Writing hash table..." << std::endl;
      WriteHashTables("./out");

      // GetProbability of dataset
      // ~ Number of entries per all/one TauType
      for(auto spectrum: spectrums)
        for(auto probab: spectrum.second->GetDataSetsProbabilities()){
          dataset_names.push_back(probab.first);
          dataset_probs.push_back(probab.second);
          set_to_group[probab.first] = spectrum.first;
        }

      // Setup randomizers
      dist_dataset = Discret(dataset_probs.begin(), dataset_probs.end());
      dist_uniform = Uniform(0.0, 1.0);
      ttype_prob = TauProb(spectrums, tau_ratio);

      all_entries = 0;
      for(auto spectrum: spectrums) all_entries+=spectrum.second->GetEntries();
    }

    bool DoNextStep()
    {
      current_dataset = dataset_names[dist_dataset(*gen)];
      while(sources.at(current_dataset)->DoNextStep()) {
        const Tau& current_tuple = sources.at(current_dataset)->GetNextTau();
        const TauType& currentType = sources.at(current_dataset)->GetType();
        if(TauSelection(current_tuple, current_dataset, currentType)) return true;
        current_dataset = dataset_names[dist_dataset(*gen)];
      }
      std::cout << "No taus at: " << current_dataset << std::endl;
      std::cout << "stop procedure..." << std::endl;
      return false;
    }

    const Tau& GetNextTau()
    {
      return sources.at(current_dataset)->GetNextTau();
    }

    void PrintStatusReport() const // time consuming
    {
      size_t input_entries=0;
      std::cout << "Status report ->";
      for(auto source: sources) input_entries+=source.second->GetNumberOfInput();
      std::cout << " init: " << all_entries;
      std::cout << " processed: " << input_entries << " (" << (float)input_entries/all_entries*100 << "%)";
      std::cout << std::endl;
    }

private:
    void WriteHashTables(const std::string& output)
    {
      if(!boost::filesystem::exists(output)) boost::filesystem::create_directory(output);
      boost::property_tree::ptree set, group;

      if(boost::filesystem::is_regular_file(output+"/dataset_hash.json"))
        boost::property_tree::read_json(output+"/dataset_hash.json", set);
      if(boost::filesystem::is_regular_file(output+"/datagroup_hash.json"))
        boost::property_tree::read_json(output+"/datagroup_hash.json", group);

      for(auto s: sources){
        set.put(s.second->GetName(), s.second->GetNameH());
        group.put(s.second->GetGroupName(), s.second->GetGroupNameH());
      }
      std::ofstream json_file (output+"/dataset_hash.json", std::ios::out);
      boost::property_tree::write_json(json_file, set);
      json_file.close();

      std::ofstream json_file2 (output+"/datagroup_hash.json", std::ios::out);
      boost::property_tree::write_json(json_file2, group);
      json_file2.close();
    }

    bool TauSelection(const Tau& tuple, const std::string& dataset, const TauType& currentType)
    {
      // check if we are taking this tau type
      if(dist_uniform(*gen) >= ttype_prob.at(currentType)) return false;
      Double_t pt = tuple.tau_pt;
      Double_t abs_eta = abs(tuple.tau_eta);
      if( pt<=pt_min || pt>=pt_max || abs_eta>=eta_max) return false;
      else if(pt>=pt_threshold) return true;
      if(dist_uniform(*gen) <= spectrums.at(set_to_group.at(dataset))
                          ->GetProbability(currentType, pt, abs_eta)) return true;
      return false;
    }

    void LoadDataGroups(const std::vector<EntryDesc>& entries,
                        const std::vector<double>& pt_bins, const std::vector<double>& eta_bins,
                        const std::set<std::string>& disabled_branches, double start_, double end_)
    {
      for(const EntryDesc& group_desc_: entries) {
        for(const std::pair<std::string,std::vector<std::string>>& file_: group_desc_.data_files) {
          const std::string& dataname = file_.first;
          std::shared_ptr<SourceDesc> source = std::make_shared<SourceDesc>(dataname, group_desc_.name,
            file_.second, disabled_branches, start_, end_, group_desc_.tau_types);
          sources[dataname] = source;
        }
        spectrums[group_desc_.name] = std::make_shared<SpectrumHists>(group_desc_.name, pt_bins,
                                                                      eta_bins, exp_disbalance,
                                                                      group_desc_.tau_types);
        for(const std::pair<std::string,std::string> spectrum: group_desc_.spectrum_files)
          spectrums[group_desc_.name]->AddHist(spectrum.first,spectrum.second);
      }
    }

    std::map<TauType, Double_t> TauProb(std::map<std::string, std::shared_ptr<SpectrumHists>> spectrums,
                                            const std::map<TauType, Double_t>& tau_ratio)
    {
      std::map<TauType, Double_t> probab;
      std::map<TauType, Double_t> accumulated_entries;
      for(auto spectrum: spectrums){
        auto entries_ = spectrum.second->GetTauTypeEntriesFinal();
        for(TauType type: spectrum.second->ttypes)
          accumulated_entries[type] += entries_.at(type);
      }
      for(auto tauR_: tau_ratio){
        // std::cout << "Number of taus: " << analysis::ToString(tauR_.first) << accumulated_entries[tauR_.first] << std::endl;
        // if(tauR_.second==0)
        //   continue;
        if(accumulated_entries.at(tauR_.first)==0)
          throw analysis::exception("No taus of the type '%1%' are found in the tuples") % tauR_.first;
        std::cout << "tau prob" << tauR_.second << " " << accumulated_entries.at(tauR_.first) << "\n";
        probab[tauR_.first] = tauR_.second/accumulated_entries.at(tauR_.first);
      }
      auto pr = std::max_element(std::begin(probab), std::end(probab),
                [] (const auto& p1, const auto& p2) {return p1.second < p2.second; });
      Double_t max_ = pr->second;
      std::cout << "max element: " << max_ << std::endl;
      for(auto tau_prob_: probab) probab.at(tau_prob_.first) = tau_prob_.second/max_;
      std::cout << "tau type probabilities:" << std::endl;
      for(auto tau_prob: probab)
        std::cout << "P(" << tau_prob.first << ")=" << tau_prob.second << " ";
      std::cout << std::endl;

      return probab;
    }


private:
    std::string current_dataset; // DataSet that was selected in DoNextStep()
    std::map<std::string, std::shared_ptr<SourceDesc>> sources; // pair<dataset_name, source>
    std::map<std::string, std::shared_ptr<SpectrumHists>> spectrums; // pair<datagroup_name, histogram>
    std::map<std::string, std::string> set_to_group; // pair<dataset_name, datagroup_name>
    std::vector<Double_t> dataset_probs; // probability of getting dataset
    std::vector<std::string> dataset_names; // corresponding dataset name
    // std::map<std::string, std::map<std::string, Double_t>> group_ttype; // pair<datagroup_name, <type, probability>>
    std::map<TauType, Double_t> ttype_prob; // pair<type, probability> (summed up per all dataset)
    const Double_t pt_max,pt_min;
    const Double_t eta_max;
    const Double_t pt_threshold;
    const Double_t exp_disbalance;
    size_t all_entries;

    Generator* gen;
    Discret dist_dataset;
    Uniform dist_uniform;

};

class ShuffleMergeSpectral {
public:
    using Tau = tau_tuple::Tau;
    using TauType = analysis::TauType;
    using TauTuple = tau_tuple::TauTuple;
    using Generator = DataSetProcessor::Generator;

    ShuffleMergeSpectral(const Arguments& _args) :
        args(_args), pt_bins(ParseBins(args.pt_bins())),
        eta_bins(ParseBins(args.eta_bins())), tau_ratio(ParseTauTypesR(args.tau_ratio())),
        max_entries(args.max_entries())
    {
      bool verbose = 1;
		  if(args.n_threads() > 1) ROOT::EnableImplicitMT(args.n_threads());

      const auto disabled_branches_vec = SplitValueList(args.disabled_branches(), false, " ,");
      disabled_branches.insert(disabled_branches_vec.begin(), disabled_branches_vec.end());

      PrintBins("pt bins", pt_bins);
      PrintBins("eta bins", eta_bins);
      const auto all_entries = LoadEntries(args.cfg());

      if(verbose) {
        for (unsigned int i = 0; i < all_entries.size(); i++) {
          std::cout << "entry in all_entries-> " << std::endl;
          for( auto it: all_entries[i].data_files) {
            std::cout << "name-> " << it.first << std::endl;
            for ( int j = 0; j < (int)it.second.size(); j++)
              std::cout << "path files-> " << it.second[j] << "\n";
          }
          for( auto it: all_entries[i].spectrum_files) {
            std::cout << "name-> " << it.first << std::endl;
            std::cout << "spectrum files-> " << it.second << std::endl;
          }
        }
      }

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
      Generator gen(args.seed());

      for(const auto& e : entries) {
        const std::string& file_name = e.first;
        const std::vector<EntryDesc>& entry_list = e.second;
        std::cout << "Processing:";
        for(const auto& entry : entry_list)
          std::cout << ' ' << entry.name;
        std::cout << "\nOutput train: " << file_name << std::endl;
        auto output_file = root_ext::CreateRootFile(file_name, ROOT::kLZ4, 4);
        auto output_tuple = std::make_shared<TauTuple>("taus", output_file.get(), false);

        const std::string& file_name_test = RemoveFileExtension(file_name)+"_test.root";
        std::cout << "Output test: " << file_name_test << std::endl;
        auto output_file_test = root_ext::CreateRootFile(file_name_test, ROOT::kLZ4, 4);
        auto output_tuple_test = std::make_shared<TauTuple>("taus", output_file_test.get(), false);

        DataSetProcessor processor(entry_list, pt_bins, eta_bins,
                                   gen, disabled_branches, true, tau_ratio,
                                   args.start_entry(),args.end_entry(),
                                   args.pt_threshold(), args.exp_disbalance());

        size_t n_processed = 0;
        int test_fill_period = (int)(1.0/args.test_size());
        while(processor.DoNextStep()){
          const auto& tau = processor.GetNextTau();
          n_processed++;
          if(n_processed % test_fill_period == 0){
            (*output_tuple_test)() = tau;
            output_tuple_test->Fill();
          } else {
            (*output_tuple)() = tau;
            output_tuple->Fill();
          }
          if(n_processed % 1000 == 0){
            std::cout << n_processed << " is selected" << std::endl;
            processor.PrintStatusReport();
          }
          if(n_processed>=max_entries){
            std::cout << "stop: number of entries exceeded max_entries" << std::endl;
            break;
          }
        }

        std::cout << "Writing output tuples..." << std::endl;
        output_tuple->Write();
        output_tuple.reset();
        output_tuple_test->Write();
        output_tuple_test.reset();
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
            entries.emplace_back(item.second, args.input(), args.path_spectrum());
        }
        return entries;
    }

    static std::map<TauType, Double_t> ParseTauTypesR(const std::string& bins_str)
    {
      std::map<TauType, Double_t> tau_ratio;
      const auto& split_strs = SplitValueList(bins_str, true, ", \t", true);
      std::cout << "ratio among tau types: " << std::endl;
      for(const std::string& str_ : split_strs) {
        if(str_.empty()) continue;
        const auto& sub_split = SplitValueList(str_, true, ":", true);
        TauType tautype = Parse<TauType>(sub_split[0]);
        Double_t ratio = Parse<double>(sub_split[1]);
        tau_ratio[tautype] = ratio;
        std::cout << tautype << ":" <<  ratio << " ";
      }
      std::cout << std::endl;
      return tau_ratio;
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
    std::map<TauType, Double_t> tau_ratio;
    std::set<std::string> disabled_branches;
    const size_t max_entries;
};

} // namespace analysis

PROGRAM_MAIN(analysis::ShuffleMergeSpectral, analysis::Arguments)
