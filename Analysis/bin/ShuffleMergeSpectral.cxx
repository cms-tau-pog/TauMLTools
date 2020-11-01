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
    run::Argument<bool> calc_weights{"calc-weights", "calculate training weights"};
    run::Argument<size_t> max_bin_occupancy{"max-bin-occupancy", "maximal occupancy of a bin",
                                            std::numeric_limits<size_t>::max()};
    run::Argument<unsigned> n_threads{"n-threads", "number of threads", 1};
    run::Argument<unsigned> seed{"seed", "random seed to initialize the generator used for sampling", 1234567};
    run::Argument<std::string> disabled_branches{"disabled-branches",
                                                 "list of branches to disabled in the input tuples", ""};
    run::Argument<std::string> path_spectrum{"input_spec", "input path with spectrums for all the samples"};
    run::Argument<std::string> tau_ratio{"tau_ratio", "ratio of tau types in the final spectrum"};
    run::Argument<double> start_entry{"start_entry", "starting ratio from which file will be processed", 0};
    run::Argument<double> end_entry{"end_entry", "end ratio until which file will be processed", 1};
};

struct SourceDesc {
    using Tau = tau_tuple::Tau;
    using TauType = analysis::TauType;
    using TauTuple = tau_tuple::TauTuple;
    using SampleType = analysis::SampleType;

    SourceDesc(const std::string& _name,const std::string&  _group_name, const std::vector<std::string>& _file_names,
               double _weight, const std::set<std::string>& _disabled_branches, double _begin_rel, double _end_rel,
               std::set<TauType> _tautypes) :
        name(_name), group_name(_group_name), file_names(_file_names),
        weight(_weight), disabled_branches(_disabled_branches), entry_begin_rel(_begin_rel), entry_end_rel(_end_rel),
        tau_types(_tautypes), current_n_processed(0), files_n_total(_file_names.size()),
        entries_file(std::numeric_limits<size_t>::infinity()), total_n_processed(0)
    {
        if(file_names.empty())
          throw analysis::exception("Empty list of files for the source '%1%'.") % name;
        if(weight <= 0)
          throw analysis::exception("Invalid source weight for the source '%1%'.") % name;
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
      }
      while (tau_types.find(current_tau_type) == tau_types.end());
      ++total_n_processed;
      (*current_tuple)().dataset_id = dataset_hash;
      (*current_tuple)().dataset_group_id = datagroup_hash;
      return true;
    }

    const Tau& GetNextTau()
    {
      return current_tuple->data();
    }

    // size_t GetNumberOfEvents() const { return entries_file; }
    const TauType GetType() { return current_tau_type; }
    const std::string& GetName() const { return name; }
    const std::string& GetGroupName() const { return group_name; }
    const int GetNameH() const { return dataset_hash; }
    const int GetGroupNameH() const { return datagroup_hash; }
    const std::vector<std::string>& GetFileNames() const { return file_names; }
    double GetWeight() const { return weight; }
    // const std::string& GetBinName() const { return bin_name; }
    // void SetBinName(const std::string& _bin_name) { bin_name = _bin_name; }

  private:
      const std::string name;
      const std::string group_name;
      const std::vector<std::string> file_names;
      const double weight;
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
            throw analysis::exception("No spectrum file are found for entry '%1%'") % sample_dir_entry;
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

    SpectrumHists(const std::string& groupname_,
                  const std::vector<double>& pt_bins,
                  const std::vector<double>& eta_bins,
                  const std::set<TauType>& tau_types_):
                  groupname(groupname_),ttypes(tau_types_)
    {
      std::cout << "Initialization of group SpectrumHists..." << std::endl;
      for(TauType type: ttypes){
        const char *name = (groupname+"_"+analysis::ToString(type)).c_str();
        ttype_hists[type] = std::make_shared<TH2D>(name,name,eta_bins.size()-1,&eta_bins[0],
                                                   pt_bins.size()-1,&pt_bins[0]);
      }
      // testing for the uniform distr.
      target_hist = SetTargetHist_test(1.0, pt_bins, eta_bins);
    }

    SpectrumHists(const SpectrumHists&) = delete;
    SpectrumHists& operator=(const SpectrumHists&) = delete;

    void AddHist(const std::string& dataset, const std::string& path_spectrum_file)
    {
      std::shared_ptr<TFile> current_file = std::make_shared<TFile>(path_spectrum_file.c_str());
      for (TauType type: ttypes){
        std::shared_ptr<TH2D> hist_ttype((TH2D*)current_file->Get(("eta_pt_hist_"+analysis::ToString(type)).c_str()));
        auto map_bins = GetBinMap(ttype_hists.at(type), hist_ttype);
        for(unsigned int ix=0; ix<=map_bins.first.size(); ix++)
          for(unsigned int iy=0; iy<=map_bins.second.size(); iy++)
            ttype_hists[type]->SetBinContent(map_bins.first[ix],map_bins.second[iy],
              ttype_hists[type]->GetBinContent(map_bins.first[ix],map_bins.second[iy])
              + hist_ttype->GetBinContent(ix,iy));

        //To get Number of entries in ranges
        entries[dataset][type] = hist_ttype->Integral();
      }
    }

    void CalculateProbability() // TO BE CHECKED AND DISCUSSED!
    {
      for (TauType type: ttypes) {
        if(CheckZeros(ttype_hists.at(type)))
          throw analysis::exception("Empty spectrum histogram for groupname: '%1%' and tau type: '%2%'") % groupname % type;
        std::shared_ptr<TH2D> new_h((TH2D*)target_hist->Clone());
        new_h->Divide(ttype_hists.at(type).get());
        ttype_hists.at(type) = new_h;
        Int_t MaxBin = ttype_hists.at(type)->GetMaximumBin();
        Int_t x,y,z;
        ttype_hists.at(type)->GetBinXYZ(MaxBin, x, y, z);
        std::cout << "max pt_bin eta_bin: " << y << " " << x
                  << " Scale: " << ttype_hists.at(type)->GetBinContent(x,y) << "\n";
        ttype_hists.at(type)->Scale(1.0/ttype_hists.at(type)->GetBinContent(x,y));
      }
    }

    const double GetProbability(const TauType type, const double pt, const double eta) const
    {
      return ttype_hists.at(type)->GetBinContent(
             ttype_hists.at(type)->GetXaxis()->FindBin(eta),
             ttype_hists.at(type)->GetYaxis()->FindBin(pt));
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

    const std::map<TauType, Double_t> GetTauTypeEntries() const
    {
      std::map<TauType, Double_t> Probs;
      for (TauType type: ttypes) {
        Probs[type] = 0.0;
        for(auto entry: entries)
            Probs[type] += entry.second.at(type);
      }
      return Probs;
    }

    void SaveHists(const std::string& output)
    {
      if(!boost::filesystem::exists(output)) boost::filesystem::create_directory(output);
      TFile file = TFile((output+"/"+groupname+".root").c_str(),"RECREATE");
      for (TauType type: ttypes)
        ttype_hists[type]->Write((analysis::ToString(type)+"_prob").c_str());
      file.Close();
    }

private:
  std::shared_ptr<TH2D> SetTargetHist_test(const double factor,const std::vector<double>& pt_bins,
                                                               const std::vector<double>& eta_bins)
  {
    std::shared_ptr<TH2D> tartget_hist = std::make_shared<TH2D>("tartget","tartget",
                                          eta_bins.size()-1,&eta_bins[0],
                                          pt_bins.size()-1,&pt_bins[0]);
    for(Int_t i_pt = 1; i_pt <= tartget_hist->GetNbinsY(); i_pt++)
      for(Int_t i_eta = 1; i_eta <= tartget_hist->GetNbinsX(); i_eta++)
        tartget_hist->SetBinContent(i_eta,i_pt,factor);
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
  std::map<TauType, std::shared_ptr<TH2D>> ttype_hists; // pair<tau_type, propability_hist>
  std::map<std::string, std::map<TauType, Double_t>> entries; // pair<dataset, pair<tau type, n entries>>
  std::shared_ptr<TH2D> target_hist;

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
                     const std::vector<double>& eta_bins, bool calc_weights, size_t max_bin_occupancy,
                     Generator& _gen, const std::set<std::string>& disabled_branches, bool verbose,
                     const std::map<TauType, Double_t> tau_ratio, Double_t start_entry, Double_t end_entry) :
                     pt_max(pt_bins.back()), pt_min(pt_bins[0]), eta_max(eta_bins.back()),
                     gen(&_gen)
    {
      if(verbose) std::cout << "Loading Data Groups..." << std::endl;
      LoadDataGroups(entries, calc_weights, pt_bins, eta_bins, disabled_branches, start_entry, end_entry);

      if(verbose) std::cout << "Calculating probabilitis..." << std::endl;
      for(auto spectrum: spectrums)
        spectrum.second->CalculateProbability();

      if(verbose) std::cout << "Saving histograms..." << std::endl;
      for(auto spectrum: spectrums)
        spectrum.second->SaveHists("./merge_out");

      if(verbose) std::cout << "Writing hash table..." << std::endl;
      WriteHashTables("./merge_out");

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
      return false;
    }

    const Tau& GetNextTau()
    {
      return sources.at(current_dataset)->GetNextTau();
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
      if( pt<=pt_min || pt>=pt_threshold || abs_eta>=eta_max) return false;
      else if(pt>=pt_max) return true;
      if(dist_uniform(*gen) <= spectrums.at(set_to_group.at(dataset))
                          ->GetProbability(currentType, pt, abs_eta)) return true;
      return false;
    }

    void LoadDataGroups(const std::vector<EntryDesc>& entries, bool calc_weights,
                        const std::vector<double>& pt_bins, const std::vector<double>& eta_bins,
                        const std::set<std::string>& disabled_branches, double start_, double end_)
    {
      for(const EntryDesc& group_desc_: entries) {
        for(const std::pair<std::string,std::vector<std::string>>& file_: group_desc_.data_files) {
          const std::string& dataname = file_.first;
          std::shared_ptr<SourceDesc> source = std::make_shared<SourceDesc>(dataname, group_desc_.name,
            file_.second, group_desc_.weight, disabled_branches, start_, end_, group_desc_.tau_types);
          sources[dataname] = source;
        }
        spectrums[group_desc_.name] = std::make_shared<SpectrumHists>(group_desc_.name,
                                                                    pt_bins,eta_bins,
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
        auto entries_ = spectrum.second->GetTauTypeEntries();
        for(TauType type: spectrum.second->ttypes)
          accumulated_entries[type] += entries_.at(type);
      }
      for(auto tauR_: tau_ratio){
        if(tauR_.second==0)
          continue;
        if(accumulated_entries.at(tauR_.first)==0)
          throw analysis::exception("No taus of the type '%1%' are found in the tuples") % tauR_.first;
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
    const Double_t pt_threshold=1000;

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
        eta_bins(ParseBins(args.eta_bins())), tau_ratio(ParseTauTypesR(args.tau_ratio()))
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
        std::cout << "\nOutput: " << file_name << std::endl;
        auto output_file = root_ext::CreateRootFile(file_name, ROOT::kLZ4, 4);
        auto output_tuple = std::make_shared<TauTuple>("taus", output_file.get(), false);
        std::cout << "Creating event bin map..." << std::endl;

        // tools::ProgressReporter reporter(10, std::cout, "Sampling taus...");
        // reporter.SetTotalNumberOfEvents(bin_map.GetNumberOfRemainingEvents());

        DataSetProcessor processor(entry_list, pt_bins, eta_bins, args.calc_weights(),
                                   args.max_bin_occupancy(), gen, disabled_branches, true, tau_ratio,
                                   args.start_entry(),args.end_entry());

        size_t n_processed = 0;

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
};

} // namespace analysis

PROGRAM_MAIN(analysis::ShuffleMergeSpectral, analysis::Arguments)
