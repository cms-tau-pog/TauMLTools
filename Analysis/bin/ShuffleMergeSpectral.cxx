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
    run::Argument<std::string> pt_bins{"pt-bins", "pt bins (last bin will be chosen as high-pt region, \
                                                   lower pt edgeof the last bin will be choosen as pt_threshold)"};
    run::Argument<std::string> eta_bins{"eta-bins", "eta bins"};
    run::Argument<analysis::MergeMode> mode{"mode", "merging mode: MergeAll or MergePerEntry"};
    run::Argument<size_t> max_entries{"max-entries", "maximal number of entries in output train+test tuples",
                                            std::numeric_limits<size_t>::max()};
    run::Argument<unsigned> n_threads{"n-threads", "number of threads", 1};
    run::Argument<unsigned> seed{"seed", "random seed to initialize the generator used for sampling", 1234567};
    run::Argument<std::string> disabled_branches{"disabled-branches",
                                                 "list of branches to disabled in the input tuples", ""};
    run::Argument<std::string> path_spectrum{"input-spec", "input path with spectrums for all the samples"};
    run::Argument<std::string> tau_ratio{"tau-ratio", "ratio of tau types in the final spectrum \
                                                      (if to take all taus of type e => e:-1 )"};
    run::Argument<double> start_entry{"start-entry", "starting ratio from which file will be processed", 0};
    run::Argument<double> end_entry{"end-entry", "end ratio until which file will be processed", 1};
    run::Argument<double> exp_disbalance{"exp-disbalance", "maximal expected disbalance between low pt and high pt regions",0};
};

struct SourceDesc {
    using Tau = tau_tuple::Tau;
    using TauType = analysis::TauType;
    using TauTuple = tau_tuple::TauTuple;
    using SampleType = analysis::SampleType;
    using Generator = std::mt19937_64;

    SourceDesc(const std::string& name_, const Int_t&  _group_hash,
               const std::vector<std::string>& _file_names, const std::vector<Int_t>& _name_hashes,
               const std::set<std::string>& _disabled_branches, const double& _begin_rel, const double& _end_rel,
               const std::set<TauType>& _tautypes, Generator& _gen) :
        name(name_),  group_hash(_group_hash),
        disabled_branches(_disabled_branches), entry_begin_rel(_begin_rel), entry_end_rel(_end_rel),
        tau_types(_tautypes), current_n_processed(0), files_n_total(_file_names.size()),
        entries_end(std::numeric_limits<size_t>::infinity()), total_n_processed(0), gen(&_gen)
    {
        if(_file_names.size()!=_name_hashes.size())
          throw analysis::exception("file_names and names vectors have different size.");
        if(_file_names.empty())
          throw analysis::exception("Empty list of files for the source '%1%'.") % name;
        if(!files_n_total)
          throw analysis::exception("Empty source '%1%'.") % name;
        ShuffleFiles(_file_names, _name_hashes);
    }

    SourceDesc(const SourceDesc&) = delete;
    SourceDesc& operator=(const SourceDesc&) = delete;

    bool DoNextStep()
    {
      do {
        if(current_file_index == files_n_total-1 && current_n_processed >= entries_end) return false;
        while(!current_file_index || current_n_processed == entries_end) {
            if(!current_file_index)
                current_file_index = 0;
            else
                ++(*current_file_index);
            if(*current_file_index >= file_names.size())
                throw analysis::exception("The expected number of events = %1% is bigger than the actual number of"
                                          " end point events in source '%2%'.") % entries_file % name;

            const std::string& file_name = file_names.at(*current_file_index);
            std::cout << "opening: " << name << " " << file_name << std::endl;
            dataset_hash = dataset_hash_arr.at(*current_file_index);
            current_tuple.reset();
            if(current_file) current_file->Close();
            current_file = root_ext::OpenRootFile(file_name);
            current_tuple = std::make_shared<TauTuple>("taus", current_file.get(), true, disabled_branches);
            entries_file = current_tuple->GetEntries();
            current_n_processed =  std::floor(entry_begin_rel * entries_file);
            entries_end = std::floor(entry_end_rel * entries_file);

            if(!entries_file)
              throw analysis::exception("Root file %1% is empty.") % file_name;
        }
        current_tuple->GetEntry(current_n_processed++);
        const auto gen_match = static_cast<analysis::GenLeptonMatch>((*current_tuple)().lepton_gen_match);
        const auto sample_type = static_cast<analysis::SampleType>((*current_tuple)().sampleType);
        current_tau_type = analysis::GenMatchToTauType(gen_match, sample_type);
      } while (tau_types.find(current_tau_type) == tau_types.end());
      ++total_n_processed;
      (*current_tuple)().tauType = static_cast<Int_t>(current_tau_type);
      (*current_tuple)().dataset_id = dataset_hash;
      (*current_tuple)().dataset_group_id = group_hash;
      return true;
    }

    const Tau& GetNextTau() { return current_tuple->data(); }
    const size_t GetNumberOfProcessed() const { return total_n_processed; }
    const TauType GetType() { return current_tau_type; }

  private:
    void ShuffleFiles(const std::vector<std::string>& _file_names,
                      const std::vector<Int_t>& _name_hashes)
    {
      std::vector<int> indexes;
      indexes.reserve(_file_names.size());
      for (UInt_t i = 0; i < _file_names.size(); ++i) indexes.push_back(i);
      std::shuffle(indexes.begin(), indexes.end(), *gen);
      for (UInt_t i = 0; i < _file_names.size(); ++i) {
        file_names.push_back(_file_names[indexes[i]]);
        dataset_hash_arr.push_back(_name_hashes[indexes[i]]);
      }
    }

    const std::string name;
    const Int_t group_hash;
    std::vector<std::string> file_names;
    std::vector<Int_t> dataset_hash_arr;
    const std::set<std::string> disabled_branches;
    const double entry_begin_rel;
    const double entry_end_rel;
    const std::set<TauType> tau_types;
    std::shared_ptr<TFile> current_file;
    std::shared_ptr<TauTuple> current_tuple;
    boost::optional<ULong64_t> current_file_index;
    size_t current_n_processed;
    ULong64_t files_n_total;;
    size_t entries_file;
    size_t entries_end;
    size_t total_n_processed;
    TauType current_tau_type;
    Int_t dataset_hash;

    Generator* gen;
  };


struct EntryDesc {
    using TauType = analysis::TauType;
    using SampleType = analysis::SampleType;

    std::string name;
    Int_t name_hash;
    std::vector<std::string> data_files;
    std::vector<std::string> data_set_names;
    std::vector<Int_t> data_set_names_hashes;
    std::vector<std::string> spectrum_files;
    std::set<TauType> tau_types;

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
        name_hash = std::hash<std::string>{}(name) % std::numeric_limits<int>::max();
        std::cout << name << std::endl;
        const std::string dir_pattern_str = item.Get<std::string>("dir");
        const std::string file_pattern_str = item.Get<std::string>("file");
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
            data_files.push_back(file_name);
            data_set_names.push_back(dir_name);
            data_set_names_hashes.push_back(std::hash<std::string>{}(dir_name) % std::numeric_limits<int>::max());
          }
          spectrum_files.push_back(spectrum_file.string());

          if(!has_file_match)
            throw analysis::exception("No files are found for entry '%1%' sample %2% with pattern '%3%'")
              % name % sample_dir_entry % file_pattern_str;
        }

        if(!has_dir_match)
            throw analysis::exception("No samples are found for entry '%1%' with pattern '%2%'")
                  % name % dir_pattern_str;
    }
};

class SpectrumHists {

    using TauType = analysis::TauType;

public:

    SpectrumHists(const std::string& groupname_, const std::vector<double>& pt_bins,
                  const std::vector<double>& eta_bins,
                  const Double_t& exp_disbalance_,
                  const std::set<TauType>& tau_types_):
                  groupname(groupname_), pt_threshold(pt_bins.end()[-2]),
                  exp_disbalance(exp_disbalance_), ttypes(tau_types_)
    {
      std::cout << "Initialization of group SpectrumHists..." << std::endl;
      for(TauType type: ttypes){
        const char *name = (groupname+"_"+analysis::ToString(type)).c_str();
        ttype_prob[type] = std::make_shared<TH2D>(name,name,eta_bins.size()-1,&eta_bins[0],
                                                   pt_bins.size()-2,&pt_bins[0]);
        ttype_entries[type] = std::make_shared<TH2D>(name,name,eta_bins.size()-1,&eta_bins[0],
                                                  pt_bins.size()-1,&pt_bins[0]);
      }

      target_hist = SetTargetHist_test(1.0, pt_bins, eta_bins);
      n_entries = 0;
    }

    SpectrumHists(const SpectrumHists&) = delete;
    SpectrumHists& operator=(const SpectrumHists&) = delete;

    void AddHist(const std::string& path_spectrum_file)
    {
      std::shared_ptr<TFile> current_file = std::make_shared<TFile>(path_spectrum_file.c_str());
      for (TauType type: ttypes){
        std::shared_ptr<TH2D> hist_ttype((TH2D*)current_file->Get(("eta_pt_hist_"+analysis::ToString(type)).c_str()));
        // if(CheckZeros(hist_ttype))
        //   throw analysis::exception("Empty spectrum histogram '%1%' for tau type '%2%'")
        //   % path_spectrum_file % analysis::ToString(type);
        auto map_bins = GetBinMap(ttype_entries.at(type), hist_ttype);
        for(UInt_t ix=0; ix<=map_bins.first.size(); ix++)
          for(UInt_t iy=0; iy<=map_bins.second.size(); iy++)
            ttype_entries[type]->SetBinContent(map_bins.first[ix],map_bins.second[iy],
              ttype_entries[type]->GetBinContent(map_bins.first[ix],map_bins.second[iy])
              + hist_ttype->GetBinContent(ix,iy));
        n_entries += hist_ttype->GetEntries();
      }
    }

    void CalculateProbability()
    {
      for (TauType type: ttypes) {
        std::shared_ptr<TH2D> ratio_h((TH2D*)target_hist->Clone());
        if(CheckZeros(ratio_h))
          throw analysis::exception("Empty histogram for tau type '%1%' in '%2%'.")
          % analysis::ToString(type) % groupname;
        for(int i_x = 1; i_x<=ratio_h->GetNbinsX(); i_x++)
          for(int i_y = 1; i_y<=ratio_h->GetNbinsY(); i_y++)
            ratio_h->SetBinContent(i_x,i_y,
              target_hist->GetBinContent(i_x,i_y)/ttype_entries.at(type)->GetBinContent(i_x,i_y)
            );
        ttype_prob.at(type) = ratio_h;
        Int_t MaxBin = ttype_prob.at(type)->GetMaximumBin();
        Int_t x,y,z;
        ttype_prob.at(type)->GetBinXYZ(MaxBin, x, y, z);

        // Adding constraints on disbalance
        // last pt bin of ttype_entries.at(type)
        // is taken as a high-Pt region
        Int_t binNx = ttype_entries.at(type)->GetNbinsX();
        Int_t binNy = ttype_entries.at(type)->GetNbinsY();
        double dis_scale = 1.0;
        if(exp_disbalance!=0)
          dis_scale = std::min(1.0, exp_disbalance*
                            ttype_entries.at(type)->Integral(0,binNx,binNy-1,binNy)*
                            ttype_prob.at(type)->GetBinContent(x,y));
        if(dis_scale!=1.0)
          std::cout << "WARNING: Disbalance for "<< analysis::ToString(type)
                    <<" is bigger, scale factor will be applied" << std::endl;
        std::cout << "max pt_bin eta_bin: " << y << " " << x
                  << " MaxBin: " << ttype_prob.at(type)->GetBinContent(x,y)
                  << " scale factor: " << dis_scale << "\n";
        ttype_prob.at(type)->Scale(dis_scale/ttype_prob.at(type)->GetBinContent(x,y));
      }
    }

    const std::map<TauType, Double_t> GetTauTypeEntriesFinal() const
    {
      std::map<TauType, Double_t> EntriesFinal;
      for (TauType type: ttypes) {
        std::shared_ptr<TH2D> entries_count((TH2D*)ttype_entries.at(type)->Clone());
        for(int i_x = 1; i_x<=ttype_prob.at(type)->GetNbinsX(); i_x++)
          for(int i_y = 1; i_y<=ttype_prob.at(type)->GetNbinsY(); i_y++)
            entries_count->SetBinContent(i_x,i_y,
              entries_count->GetBinContent(i_x,i_y)*ttype_prob.at(type)->GetBinContent(i_x,i_y)
            );
        EntriesFinal[type] = entries_count->Integral();
      }
      return EntriesFinal;
    }

    const double GetProbability(const TauType& type, const double& pt, const double& eta) const
    {
      return ttype_prob.at(type)->GetBinContent(
             ttype_prob.at(type)->GetXaxis()->FindBin(eta),
             ttype_prob.at(type)->GetYaxis()->FindBin(pt));
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
    const std::string GetGroupName() { return groupname; }

private:

  std::shared_ptr<TH2D> SetTargetHist_test(const double scale,
                                           const std::vector<double>& pt_bins,
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
          throw analysis::exception("New histogram bins over eta are inconsistent.");
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
          throw analysis::exception("New histogram bins over pt are inconsistent.");
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
    using Uniform = std::uniform_real_distribution<double>;
    using Discret = std::discrete_distribution<int>;

    DataSetProcessor(const std::vector<EntryDesc>& entries, const std::vector<double>& pt_bins,
                     const std::vector<double>& eta_bins, Generator& _gen,
                     const std::set<std::string>& disabled_branches, bool verbose,
                     const std::map<TauType, Double_t>& tau_ratio, const Double_t start_entry,
                     const Double_t end_entry, const Double_t exp_disbalance_) :
                     pt_max(pt_bins.back()), pt_min(pt_bins[0]), eta_max(eta_bins.back()),
                     pt_threshold(pt_bins.end()[-2]), exp_disbalance(exp_disbalance_), gen(&_gen)
    {
      if(verbose) std::cout << "Loading Data Groups..." << std::endl;
      LoadDataGroups(entries, pt_bins, eta_bins, disabled_branches, start_entry, end_entry);

      if(verbose) std::cout << "Calculating probabilitis..." << std::endl;
      for(auto spectrum: spectrums)
        spectrum.second->CalculateProbability();

      if(verbose) std::cout << "Saving histograms..." << std::endl;
      for(auto spectrum: spectrums) spectrum.second->SaveHists("./out");

      if(verbose) std::cout << "Writing hash table..." << std::endl;
      WriteHashTables("./out",entries);


      dist_uniform = Uniform(0.0, 1.0);
      ttype_prob = TauTypeProb(spectrums, tau_ratio);

      // Probability of data group
      // is taken proportionally number
      // of entries per considered TauTypes
      for(auto spectrum: spectrums){
        datagroup_probs.push_back((double)spectrum.second->GetEntries());
        datagroup_names.push_back(spectrum.second->GetGroupName());
      }
      dist_dataset = Discret(datagroup_probs.begin(), datagroup_probs.end());

      n_entries = 0;
      for(auto spectrum: spectrums) n_entries+=spectrum.second->GetEntries();
    }

    bool DoNextStep()
    {
      current_datagroup = datagroup_names[dist_dataset(*gen)];
      while(sources.at(current_datagroup)->DoNextStep()) {
        const Tau& current_tuple = sources.at(current_datagroup)->GetNextTau();
        const TauType& currentType = sources.at(current_datagroup)->GetType();
        if(TauSelection(current_tuple, current_datagroup, currentType)) return true;
        current_datagroup = datagroup_names[dist_dataset(*gen)];
      }
      std::cout << "No taus at: " << current_datagroup << std::endl;
      std::cout << "stop procedure..." << std::endl;
      return false;
    }

    const Tau& GetNextTau() { return sources.at(current_datagroup)->GetNextTau(); }

    void PrintStatusReport(const Double_t& read_proc) const
    {
      size_t input_entries=0;
      std::cout << "Status report ->";
      for(auto source: sources) input_entries+=source.second->GetNumberOfProcessed();
      std::cout << " init: " << (long)(read_proc*n_entries);
      std::cout << " processed: " << input_entries << " (" << (float)input_entries/(read_proc*n_entries)*100 << "%)";
      std::cout << std::endl;
    }

private:
    void WriteHashTables(const std::string& output,
                         const std::vector<EntryDesc>& entries)
    {
      if(!boost::filesystem::exists(output)) boost::filesystem::create_directory(output);
      boost::property_tree::ptree set, group;

      if(boost::filesystem::is_regular_file(output+"/dataset_hash.json"))
        boost::property_tree::read_json(output+"/dataset_hash.json", set);
      if(boost::filesystem::is_regular_file(output+"/datagroup_hash.json"))
        boost::property_tree::read_json(output+"/datagroup_hash.json", group);

      for(const EntryDesc& desc: entries){
        group.put(desc.name, desc.name_hash);
        for(UInt_t i=0; i<desc.data_set_names.size(); i++)
          set.put(desc.data_set_names[i],
                  desc.data_set_names_hashes[i]);
      }

      std::ofstream json_file (output+"/dataset_hash.json", std::ios::out);
      boost::property_tree::write_json(json_file, set);
      json_file.close();

      std::ofstream json_file2 (output+"/datagroup_hash.json", std::ios::out);
      boost::property_tree::write_json(json_file2, group);
      json_file2.close();
    }

    bool TauSelection(const Tau& tuple, const std::string& datagroup, const TauType& currentType)
    {
      // check if we are taking this tau type
      if(dist_uniform(*gen) >= ttype_prob.at(currentType)) return false;
      Double_t pt = tuple.tau_pt;
      Double_t abs_eta = abs(tuple.tau_eta);
      if( pt<=pt_min || pt>=pt_max || abs_eta>=eta_max) return false;
      else if(pt>=pt_threshold) return true;
      if(dist_uniform(*gen) <= spectrums.at(current_datagroup)
                          ->GetProbability(currentType, pt, abs_eta)) return true;
      return false;
    }

    void LoadDataGroups(const std::vector<EntryDesc>& entries,
                        const std::vector<double>& pt_bins, const std::vector<double>& eta_bins,
                        const std::set<std::string>& disabled_branches, double start_, double end_)
    {
      for(const EntryDesc& dsc: entries) {

        std::shared_ptr<SourceDesc> source = std::make_shared<SourceDesc>(dsc.name,dsc.name_hash,
                            dsc.data_files, dsc.data_set_names_hashes,disabled_branches,
                            start_, end_, dsc.tau_types, *gen);
        sources[dsc.name] = source;

        spectrums[dsc.name] = std::make_shared<SpectrumHists>(dsc.name, pt_bins,
                                                              eta_bins, exp_disbalance,
                                                              dsc.tau_types);

        for(const std::string& spectrum: dsc.spectrum_files)
          spectrums[dsc.name]->AddHist(spectrum);
      }
    }

    std::map<TauType, Double_t> TauTypeProb(const std::map<std::string, std::shared_ptr<SpectrumHists>>& spectrums,
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
        if(tauR_.second!=-1){
          if(accumulated_entries.at(tauR_.first)==0)
            throw analysis::exception("No taus of the type '%1%' are found in the tuples") % tauR_.first;
          std::cout << "tau: " << analysis::ToString(tauR_.first) << " Prob: " << tauR_.second
                    << " " << accumulated_entries.at(tauR_.first) << "\n";
          probab[tauR_.first] = tauR_.second/accumulated_entries.at(tauR_.first);
        }
      }
      auto pr = std::max_element(std::begin(probab), std::end(probab),
                [] (const auto& p1, const auto& p2) {return p1.second < p2.second; });
      Double_t max_ = pr->second;
      std::cout << "max element: " << max_ << std::endl;
      for(auto tau_prob_: probab) probab.at(tau_prob_.first) = tau_prob_.second/max_;
      for(auto tauR_: tau_ratio){
        if(tauR_.second==-1){
          if(accumulated_entries.at(tauR_.first)==0)
            throw analysis::exception("No taus of the type '%1%' are found in the tuples") % tauR_.first;
          probab[tauR_.first] = 1.0;
        }
      }
      std::cout << "tau type probabilities:" << std::endl;
      for(auto tau_prob: probab)
        std::cout << "P(" << tau_prob.first << ")=" << tau_prob.second << " ";
      std::cout << std::endl;

      return probab;
    }


private:
    std::string current_datagroup;
    std::map<std::string, std::shared_ptr<SourceDesc>> sources; // pair<datagroup_name, source>
    std::map<std::string, std::shared_ptr<SpectrumHists>> spectrums; // pair<datagroup_name, histogram>
    std::vector<Double_t> datagroup_probs; // probability of getting datagroup
    std::vector<std::string> datagroup_names; // corresponding datagroup name
    std::map<TauType, Double_t> ttype_prob; // pair<type, probability> (summed up per all dataset)

    const Double_t pt_max,pt_min;
    const Double_t eta_max;
    const Double_t pt_threshold;
    const Double_t exp_disbalance;
    size_t n_entries;

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
        for (auto dsc: all_entries) {
          std::cout << "entry group -> " << dsc.name << std::endl;
          std::cout << "files: " << dsc.data_files.size() << " "
                    << "spectrums: " << dsc.spectrum_files.size()
                    << std::endl;
        //   for(UInt_t i=0; i<dsc.data_files.size();i++)
        //     std::cout << dsc.data_files[i] << " "
        //               << dsc.data_set_names[i] << " "
        //               << dsc.data_set_names_hashes[i] << std::endl;
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

        DataSetProcessor processor(entry_list, pt_bins, eta_bins,
                                   gen, disabled_branches, true, tau_ratio,
                                   args.start_entry(),args.end_entry(),
                                   args.exp_disbalance());

        size_t n_processed = 0;
        std::cout << "starting loops:" <<std::endl;
        while(processor.DoNextStep()){
          const auto& tau = processor.GetNextTau();
          n_processed++;
          (*output_tuple)() = tau;
          output_tuple->Fill();
          if(n_processed % 1000 == 0){
            std::cout << n_processed << " is selected" << std::endl;
            processor.PrintStatusReport(args.end_entry()-args.start_entry());
          }
          if(n_processed>=max_entries){
            std::cout << "stop: number of entries exceeded max_entries" << std::endl;
            break;
          }
        }
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
    const size_t max_entries;
};

} // namespace analysis

PROGRAM_MAIN(analysis::ShuffleMergeSpectral, analysis::Arguments)
