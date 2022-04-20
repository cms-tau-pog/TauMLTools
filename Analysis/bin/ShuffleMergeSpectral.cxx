/*! Shuffle and merge datasets using Histograms created with CreateSpectralHists.cxx
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
#include "TauMLTools/Analysis/interface/TauSelection.h"

namespace analysis {

void sumBasedSplit(const std::vector<size_t>& files_entries, const size_t job_idx, const size_t n_job, const bool overflowjob,
                  std::pair<size_t, size_t>& point_entry,  std::pair<size_t, size_t>& point_exit, size_t& step)
{ 
  // sumBasedSplit splits files into n_job sub-intervals
  // the entry events for the job number job_idx are [point_entry, point_exit]...
  // if sum(files_entries) % n_job != 0 some events will be lost in datagroup

  step = std::accumulate(files_entries.begin(), files_entries.end(), 0) / n_job;
  auto find_idx = [&](const size_t index, const bool isExit) -> std::pair<size_t, size_t>{
    size_t sum_accumulate = 0;
    for(size_t f_i = 0; f_i < files_entries.size(); f_i++) {
      if(step*(index+isExit) - sum_accumulate - isExit < files_entries[f_i])
        return std::make_pair(f_i, step*(index+isExit) - sum_accumulate  - isExit);
      sum_accumulate+=files_entries[f_i];
    }
    throw exception("Sum-based splitting error!");
  };

   // .first - index of file in file list
   // .second - index of last event
  point_entry = find_idx(job_idx, false);
  point_exit = find_idx(job_idx, true);
  if(overflowjob && job_idx+1 == n_job){
    step = step + (std::accumulate(files_entries.begin(), files_entries.end(), 0) % n_job);
    point_exit = std::make_pair(files_entries.size()-1, files_entries[files_entries.size()-1]-1);
  }
};

enum class MergeMode { MergeAll = 1 };
ENUM_NAMES(MergeMode) = {
    { MergeMode::MergeAll, "MergeAll" },
};

struct Arguments {
    run::Argument<std::string> cfg{"cfg", "configuration file with the list of input sources"};
    run::Argument<std::string> input{"input", "Input file with the list of files to read. "
                                              "The --prefix argument will be placed in front of --input.", ""};
    run::Argument<std::string> prefix{"prefix", "prefix to place before the input file path read from --input. "
                                                "It can include a remote server to use with xrootd.", ""};
    run::Argument<std::string> output{"output", "output, depending on the merging mode: MergeAll - file,"
                                                " MergePerEntry - directory."};
    run::Argument<std::string> pt_bins{"pt-bins", "pt bins (last bin will be chosen as high-pt region, "
                                                  "lower pt edge of the last bin will be choosen as pt_threshold)"};
    run::Argument<std::string> eta_bins{"eta-bins", "eta bins"};
    run::Argument<MergeMode> mode{"mode", "merging mode: MergeAll is the only supported mode", MergeMode::MergeAll};
    run::Argument<size_t> max_entries{"max-entries", "maximal number of entries in output train+test tuples",
                                            std::numeric_limits<size_t>::max()};
    run::Argument<unsigned> n_threads{"n-threads", "number of threads", 1};
    run::Argument<unsigned> seed{"seed", "random seed to initialize the generator used for sampling", 1234567};
    run::Argument<std::string> disabled_branches{"disabled-branches",
                                                 "list of branches to disabled in the input tuples", ""};
    run::Argument<std::string> path_spectrum{"input-spec", "input path with spectrums for all the samples. "
                                                           "A remote server can be specified to use with xrootd."};
    run::Argument<std::string> tau_ratio{"tau-ratio", "ratio of tau types in the final spectrum "
                                                      "(if to take all taus of type e => e:-1 )"};
    run::Argument<unsigned> job_idx{"job-idx", "index of the job (starts from 0)"};
    run::Argument<unsigned> n_jobs{"n-jobs", "the number by which to divide all files"};
    run::Argument<double> lastbin_disbalance{"lastbin-disbalance", "maximal acceptable disbalance between low pt (all bins up to last one)"
                                             "and high pt region (last pt bin) " 
                                             "(option is relevant only for the case of `--lastbin-takeall True`)",1.0};
    run::Argument<bool> lastbin_takeall{"lastbin-takeall", "to take all events from the last bin up to acceptable disbalance,"
                                        "specified with `--lastbin-disbalance`",false};
    run::Argument<std::string> compression_algo{"compression-algo","ZLIB, LZMA, LZ4","LZMA"};
    run::Argument<unsigned> compression_level{"compression-level", "compression level of output file", 9};
    run::Argument<unsigned> parity{"parity","take only even:0, take only odd:1, take all entries:3", 3};
    run::Argument<bool> refill_spectrum{"refill-spectrum", "If true - spectrums of the input data will be recalculated on flight, "
                                        "only events that correspond to the current job will be considered.", false};
    run::Argument<bool> enable_emptybin{"enable-emptybin", "In case of empty pt-eta bin, the probability in this bin will be set to 0", false};
    run::Argument<bool> overflow_job{"overflow-job", "If true - include all remaining Taus from the ntuples in the last job. "
                                        "(The number of Taus included, instead of being ignored, should be less than the number of jobs.)", false};
};

struct SourceDesc {
    using Tau = tau_tuple::Tau;
    using TauTuple = tau_tuple::TauTuple;

    SourceDesc(const std::string& _name, const size_t _total_entries, const ULong64_t  _group_hash,
               const std::vector<std::string>& _file_names, const std::vector<ULong64_t>& _name_hashes,
               const std::pair<size_t, size_t> _point_entry, const std::pair<size_t, size_t> _point_exit,
               const std::set<TauType>& _tautypes, const std::set<std::string> _disabled_branches = {},
               const std::set<std::string> _enabled_branches = {}) :

        name(_name), total_entries(_total_entries), group_hash(_group_hash), file_names(_file_names), dataset_hash_arr(_name_hashes),
        disabled_branches(_disabled_branches), enabled_branches(_enabled_branches), point_entry(_point_entry), point_exit(_point_exit),
        tau_types(_tautypes), files_n_total(_file_names.size()), entries_end(std::numeric_limits<size_t>::max()),
        current_entry(0), total_n_processed(0)
    {
        if(_file_names.size()!=_name_hashes.size())
          throw exception("file_names and names vectors have different size.");
        if(_file_names.empty())
          throw exception("Empty list of files for the source '%1%'.") % name;
        if(!files_n_total)
          throw exception("Empty source '%1%'.") % name;
    }

    SourceDesc(const SourceDesc&) = delete;
    SourceDesc& operator=(const SourceDesc&) = delete;

    bool DoNextStep()
    {
      do {
        if(current_file_index == point_exit.first && current_entry > entries_end ) return false;
        while(!current_file_index || current_entry > entries_end) {
            if(!current_file_index)
                current_file_index = point_entry.first;
            else
                ++(*current_file_index);
            if(*current_file_index >= file_names.size())
                throw exception("File index: %1% is out of file_names array '%2%', DataGroup: '%3%'")
                      % current_file_index % file_names.size() % name;
            const std::string& file_name = file_names.at(*current_file_index);
            std::cout << "Opening: " << name << " " << file_name << std::endl;
            dataset_hash = dataset_hash_arr.at(*current_file_index);
            current_tuple.reset();
            if(current_file) current_file->Close();
            current_file = root_ext::OpenRootFile(file_name);
            current_tuple = std::make_shared<TauTuple>("taus", current_file.get(), true, disabled_branches, enabled_branches);
            entries_file = current_tuple->GetEntries();
            current_entry = current_file_index == point_entry.first ? point_entry.second : 0; 
            entries_end = current_file_index == point_exit.first ? point_exit.second : entries_file - 1;
            if(!entries_file)
              throw exception("Root file %1% is empty.") % file_name;
        }
        current_tuple->GetEntry(current_entry++);
        ++total_n_processed;

        current_tau_type = boost::none;
        const auto gen_match = GetGenLeptonMatch(static_cast<reco_tau::gen_truth::GenLepton::Kind>((*current_tuple)().genLepton_kind), 
                                                  (*current_tuple)().genLepton_index, (*current_tuple)().tau_pt, (*current_tuple)().tau_eta, 
                                                  (*current_tuple)().tau_phi, (*current_tuple)().tau_mass, (*current_tuple)().genLepton_vis_pt, 
                                                  (*current_tuple)().genLepton_vis_eta, (*current_tuple)().genLepton_vis_phi, 
                                                   (*current_tuple)().genLepton_vis_mass, (*current_tuple)().genJet_index);                                           
        const auto sample_type = static_cast<SampleType>((*current_tuple)().sampleType);

        if (!gen_match) continue;

        current_tau_type = GenMatchToTauType(*gen_match, sample_type);

        } while (!current_tau_type || tau_types.find(current_tau_type.get()) == tau_types.end());

      (*current_tuple)().tauType = static_cast<Int_t>(current_tau_type.get());
      (*current_tuple)().dataset_id = dataset_hash;
      (*current_tuple)().dataset_group_id = group_hash;
      return true;
    }

    const Tau& GetNextTau() { return current_tuple->data(); }
    const size_t GetNumberOfProcessed() const { return total_n_processed; }
    const size_t GetTotalEntries() const { return total_entries; }
    const TauType GetType() { return current_tau_type.get(); }

  private:
    const std::string name;
    const size_t total_entries;
    const ULong64_t group_hash;
    std::vector<std::string> file_names;
    std::vector<ULong64_t> dataset_hash_arr;
    const std::set<std::string> disabled_branches;
    const std::set<std::string> enabled_branches;
    const std::pair<size_t, size_t> point_entry;
    const std::pair<size_t, size_t> point_exit;
    const std::set<TauType> tau_types;
    std::shared_ptr<TFile> current_file;
    std::shared_ptr<TauTuple> current_tuple;
    boost::optional<size_t> current_file_index;
    size_t files_n_total;;
    size_t entries_file;
    size_t entries_end;
    size_t current_entry;
    size_t total_n_processed;
    boost::optional<TauType> current_tau_type;
    ULong64_t dataset_hash;
  };

struct EntryDesc {

    std::string name;
    ULong64_t name_hash;
    std::vector<std::string> data_files;
    std::vector<std::string> data_set_names;
    std::vector<ULong64_t> data_set_names_hashes;
    std::set<std::string> spectrum_files;
    std::set<TauType> tau_types;

    // <file idx, event> for entry and exit point
    std::pair<size_t, size_t> point_entry;
    std::pair<size_t, size_t> point_exit;
    size_t total_entries;

    EntryDesc(const PropertyConfigReader::Item& item,
              const std::string& base_spectrum_dir,
              const std::string& input_paths,
              const std::string& prefix,
              const size_t job_idx, const size_t n_jobs,
              const bool refill_spectrum, const bool overflow_job)
    {
        using boost::regex;
        using boost::regex_match;
        using boost::filesystem::is_regular_file;

        name = item.name;
        name_hash = std::hash<std::string>{}(name);

        const std::string dir_pattern_str  = item.Get<std::string>("dir");
        const std::string file_pattern_str = item.Get<std::string>("file");
        const std::string tau_types_str    = item.Get<std::string>("types");

        const regex dir_pattern (dir_pattern_str );
        const regex file_pattern(file_pattern_str);

        tau_types = SplitValueListT<TauType, std::set<TauType>>(tau_types_str, false, ",");

        std::ifstream input_files (input_paths, std::ifstream::in);
        if (!input_files){
          throw exception("The input file %1% could not be opened")
            %input_paths;
        }

        bool is_matching = false;

        std::string ifile;
        std::string dir_name, file_name, spectrum_file, file_path;
        std::vector<size_t> files_entries;

        // For every datagroup in cfg file EntryDesc iterates
        // through filelist.txt and for matched (to datagroup):
        // 1) fill an array of pathes to data_files
        // 2) fill an array of pathes spectrum_files 
        // 3) fill an array with number of entries per file
        while(std::getline(input_files, ifile)){
	      
          size_t n_entries = analysis::Parse<double>(ifile.substr(ifile.rfind(" ")));

          file_name = ifile.substr(ifile.find_last_of("/") + 1,
                                   ifile.rfind(" ")-ifile.find_last_of("/")-1);
          dir_name = ifile.substr(0,ifile.find_last_of("/"));
          dir_name  = dir_name.substr(dir_name.find_last_of("/")+1);

          if(!regex_match(dir_name , dir_pattern )) continue;
          if(!regex_match(file_name, file_pattern)) continue;

          is_matching = true; //at least one file is found

          file_path = prefix + "/" + dir_name + "/" + file_name;
          data_files.push_back(file_path);
          files_entries.push_back(n_entries);
          data_set_names.push_back(dir_name);
          data_set_names_hashes.push_back(std::hash<std::string>{}(dir_name));

          if(!refill_spectrum){
            spectrum_file = base_spectrum_dir + "/" + dir_name + ".root";
            spectrum_files.insert(spectrum_file);
          }
        }

        if(!is_matching){
          throw exception("No files are found for entry '%1%' with pattern '%2%'")
              % name % file_pattern_str;
        }

        if(!refill_spectrum){
          std::cout << name << std::endl;
          for (const auto spectrum_file : spectrum_files){
            if(!is_regular_file(spectrum_file))
              throw exception("No spectrum file are found: '%1%'") % spectrum_file;
            std::cout << spectrum_file << " - spectrum" << std::endl;
          }
        }

        if(job_idx >= n_jobs)
          throw exception("Wrong job_idx! The index should be > 0 and < n_jobs");
        
        sumBasedSplit(files_entries, job_idx, n_jobs, overflow_job, point_entry, point_exit, total_entries);
        
        std::cout <<  name << ": " <<
                     "Entry point-> " << point_entry.first << " " << point_entry.second << ", " <<
                     "Exit point-> " << point_exit.first << " " << point_exit.second << ", " <<
                     "Total entries-> " << total_entries << std::endl; 
    }
};

class SpectrumHists {

public:
    using Tau = tau_tuple::Tau;
    using TauTuple = tau_tuple::TauTuple;

    SpectrumHists(const std::string& groupname_, const std::vector<double>& pt_bins,
                  const std::vector<double>& eta_bins,
                  const Double_t exp_disbalance_,
                  const bool lastbin_takeall_,
                  const bool enable_emptybin_,
                  const std::set<TauType>& tau_types_):
                  groupname(groupname_), pt_threshold(pt_bins.end()[-2]),
                  exp_disbalance(exp_disbalance_), lastbin_takeall(lastbin_takeall_),
                  enable_emptybin(enable_emptybin_), ttypes(tau_types_)
    {
      std::cout << "Initialization of group SpectrumHists..." << std::endl;
      for(TauType type: ttypes){
        const char *name = (groupname+"_"+ToString(type)).c_str();

        // option 1: last pt bin will be taken into account for probability calculations (exp_disbalance==0)
        // option 2: all events from the last bin will be taken up to acceptable disbalance
        ttype_prob[type] = 
          std::make_shared<TH2D>(name,name,eta_bins.size()-1,&eta_bins[0], 
          lastbin_takeall==false ? pt_bins.size()-1 : pt_bins.size()-2, &pt_bins[0]);

        ttype_entries[type] = std::make_shared<TH2D>(name,name,eta_bins.size()-1,&eta_bins[0],
                                                  pt_bins.size()-1,&pt_bins[0]);
      }
      target_hist = SetTargetHistUniform(1.0, pt_bins, eta_bins);
      n_entries = 0;
    }

    SpectrumHists(const SpectrumHists&) = delete;
    SpectrumHists& operator=(const SpectrumHists&) = delete;

    void AddHist(const std::string& path_spectrum_file)
    { 
      // Following  
      std::shared_ptr<TFile> current_file = std::make_shared<TFile>(path_spectrum_file.c_str());
      for (TauType type: ttypes){
        std::shared_ptr<TH2D> hist_ttype((TH2D*)current_file->Get(("eta_pt_hist_"+ToString(type)).c_str()));
        if (!hist_ttype)
          throw exception("TauType: '%1%' is not available at '%2%'")
          % ToString(type) % path_spectrum_file;
        root_ext::RebinAndFill(*ttype_entries[type], *hist_ttype);
        n_entries += hist_ttype->GetEntries();
      }
    }

    void AddHist_refill(const std::shared_ptr<SourceDesc>& SpectrumSource)
    {
      while(SpectrumSource->DoNextStep()) {
        const Tau& tuple = SpectrumSource->GetNextTau();
        const TauType& type = SpectrumSource->GetType();
        ttype_entries.at(type)->Fill(std::abs(tuple.tau_eta), tuple.tau_pt);
        n_entries += 1;
      }
    }

    void CalculateProbability()
    {
      for (TauType type: ttypes) {
        std::shared_ptr<TH2D> ratio_h((TH2D*)target_hist->Clone());
        if(CheckZeros(ratio_h))
          throw exception("Empty histogram for tau type '%1%' in '%2%'.")
          % ToString(type) % groupname;
        for(int i_x = 1; i_x<=ratio_h->GetNbinsX(); i_x++) {
          for(int i_y = 1; i_y<=ratio_h->GetNbinsY(); i_y++) {
            if(ttype_entries.at(type)->GetBinContent(i_x,i_y)==0) {
              if(enable_emptybin) {
                std::cout << "WARNING: empty bin (pt_i:"
                          << i_y << ", eta_i:" << i_x  << ") "
                          << ToString(type) << " " << groupname << std::endl;
                ratio_h->SetBinContent(i_x,i_y,0.0);
              } else {
                throw exception("Empty bin (pt_i:'%1%', eta_i:'%2%') for tau type '%3%' in '%4%'.")
                % i_y % i_x % ToString(type) % groupname;
              }
            } else {
            ratio_h->SetBinContent(i_x,i_y,
              target_hist->GetBinContent(i_x,i_y)/ttype_entries.at(type)->GetBinContent(i_x,i_y));
            }
	  }
        }
        ttype_prob.at(type) = ratio_h;
        Int_t MaxBin = ttype_prob.at(type)->GetMaximumBin();
        Int_t x,y,z;
        ttype_prob.at(type)->GetBinXYZ(MaxBin, x, y, z);

        if(ttype_prob.at(type)->GetBinContent(x,y)==0)
          throw exception("Histogram '%1%' in '%4%' is empty.")
          % ToString(type) % groupname;

        if(lastbin_takeall==false) { // option 1: last pt bin will be taken into account for probability calculations
          ttype_prob.at(type)->Scale(1.0/ttype_prob.at(type)->GetBinContent(x,y));
        }
        else { // option 2: all events from the last bin will be taken up to acceptable disbalance
          // Adding constraints on disbalance
          // last pt bin of ttype_entries.at(type)
          // is taken as a high-pt region
          // from which all events are taken (up to acceptable disbalance)
          Int_t binNx = ttype_entries.at(type)->GetNbinsX();
          Int_t binNy = ttype_entries.at(type)->GetNbinsY();
          double dis_scale = std::min(1.0, exp_disbalance*
                             ttype_entries.at(type)->Integral(0,binNx,binNy-1,binNy)*
                             ttype_prob.at(type)->GetBinContent(x,y));
          if(dis_scale<=0.0) // As a result of ttype_entries.at(type)->Integral(0,binNx,binNy-1,binNy)==0.0
            throw exception("Disbalance scale factor is 0 for histogram '%1%' in '%2%' for bin (pt_i:'%3%', eta_i:'%4%')")
            % ToString(type) % groupname % binNy % binNx;
          if(dis_scale!=1.0)
            std::cout << "WARNING: Disbalance for "<< ToString(type)
                      <<" is bigger, scale factor will be applied" << std::endl;
          std::cout << "max pt_bin eta_bin: " << y << " " << x
                    << " MaxBin: " << ttype_prob.at(type)->GetBinContent(x,y)
                    << " scale factor: " << dis_scale << "\n";
          ttype_prob.at(type)->Scale(dis_scale/ttype_prob.at(type)->GetBinContent(x,y));
        }
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
        ttype_prob[type]->Write((ToString(type)+"_prob").c_str());
      file.Close();
      TFile file_n = TFile((output+"/"+groupname+"_n.root").c_str(),"RECREATE");
      for (TauType type: ttypes)
        ttype_entries[type]->Write((ToString(type)+"_entries").c_str());
      file_n.Close();
    }

    const size_t GetEntries() { return n_entries; }
    const std::string GetGroupName() { return groupname; }

private:

  std::shared_ptr<TH2D> SetTargetHistUniform(const double scale,
                                           const std::vector<double>& pt_bins,
                                           const std::vector<double>& eta_bins)
  {
    // option 1: last pt bin will be taken into account for probability calculations (lastbin_takeall==0)
    // option 2: all events from the last bin will be taken up to acceptable disbalance
    std::shared_ptr<TH2D> tartget_hist = 
      std::make_shared<TH2D>("tartget","tartget", eta_bins.size()-1,&eta_bins[0],
                              lastbin_takeall==false ? pt_bins.size()-1 : pt_bins.size()-2,
                              &pt_bins[0]);

    for(Int_t i_pt = 1; i_pt <= tartget_hist->GetNbinsY(); i_pt++)
      for(Int_t i_eta = 1; i_eta <= tartget_hist->GetNbinsX(); i_eta++)
        tartget_hist->SetBinContent(i_eta,i_pt,1.0);
    tartget_hist->Scale(scale/tartget_hist->Integral());
    return tartget_hist;
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
  const Double_t pt_threshold, exp_disbalance;
  const bool lastbin_takeall, enable_emptybin;

public:
  const std::set<TauType> ttypes; // vector of tau types e.g: e,tau...

};

class DataSetProcessor {
public:
    using Tau = tau_tuple::Tau;
    using TauTuple = tau_tuple::TauTuple;
    using Generator = std::mt19937_64;
    using Uniform = std::uniform_real_distribution<double>;
    using Discret = std::discrete_distribution<int>;

    DataSetProcessor(const std::vector<EntryDesc>& entries, const std::vector<double>& pt_bins,
                     const std::vector<double>& eta_bins, Generator& _gen,
                     const std::set<std::string>& disabled_branches, bool verbose,
                     const std::map<TauType, Double_t>& tau_ratio, const Double_t exp_disbalance,
                     const bool lastbin_takeall_, const bool refill_spectrum, const bool enable_emptybin) :
                     pt_max(pt_bins.back()), pt_min(pt_bins[0]), eta_max(eta_bins.back()),
                     pt_threshold(pt_bins.end()[-2]), lastbin_takeall(lastbin_takeall_), gen(&_gen)
    {
      if(verbose) std::cout << "Loading Data Groups..." << std::endl;
      LoadDataGroups(entries, pt_bins, eta_bins, disabled_branches, exp_disbalance,
                     refill_spectrum, enable_emptybin);

      if(verbose) std::cout << "Calculating probabilities..." << std::endl;
      for(auto spectrum: spectrums)
        spectrum.second->CalculateProbability();

      if(verbose) std::cout << "Saving histograms..." << std::endl;
      for(auto spectrum: spectrums) spectrum.second->SaveHists("./out");

      if(verbose) std::cout << "Writing hash table..." << std::endl;
      WriteHashTables("./out",entries);


      dist_uniform = Uniform(0.0, 1.0);
      ttype_prob = TauTypeProb(spectrums, tau_ratio);

      // Probability of data group
      // is taken proportionally to number
      // of entries per considered (for DataGroups) TauTypes
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
      PrintStatusReport();
      return false;
    }

    const Tau& GetNextTau() { return sources.at(current_datagroup)->GetNextTau(); }

    void PrintStatusReport() const
    {
      std::cout << "Status report -> ";
      for(auto source: sources) {
        size_t n_processed = source.second->GetNumberOfProcessed();
        size_t n_total = source.second->GetTotalEntries();
        std::cout << source.first << ":" << (float)n_processed/n_total*100 <<"% (" << n_total-n_processed << " left),";
      }
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
      if( lastbin_takeall==true && pt>=pt_threshold) return true;
      if(dist_uniform(*gen) <= spectrums.at(current_datagroup)
                          ->GetProbability(currentType, pt, abs_eta)) return true;
      return false;
    }

    void LoadDataGroups(const std::vector<EntryDesc>& entries,
                        const std::vector<double>& pt_bins, const std::vector<double>& eta_bins,
                        const std::set<std::string>& disabled_branches, const double exp_disbalance,
                        const bool refill_spectrum, const bool enable_emptybin)
    {
      for(const EntryDesc& dsc: entries) {

        std::shared_ptr<SourceDesc> source = std::make_shared<SourceDesc>(dsc.name,
                            dsc.total_entries, dsc.name_hash,
                            dsc.data_files, dsc.data_set_names_hashes,
                            dsc.point_entry, dsc.point_exit, dsc.tau_types,
                            disabled_branches);
        sources[dsc.name] = source;
        spectrums[dsc.name] = std::make_shared<SpectrumHists>(dsc.name, pt_bins,
                                                              eta_bins, exp_disbalance, lastbin_takeall,
                                                              enable_emptybin, dsc.tau_types);
        if(refill_spectrum) {
          std::cout << "ReFilling spectrums for - " <<  dsc.name << std::endl;
          const std::set<std::string> enabled_branches =
                                      {"tau_pt", "tau_eta", "sampleType",
                                        "genLepton_kind", "tau_index",
                                        "genLepton_index", "genJet_index",
                                        "genLepton_vis_pt", "genLepton_vis_eta",
                                        "genLepton_vis_phi", "genLepton_vis_mass",
                                        "tau_pt", "tau_eta", "tau_phi",
                                        "tau_mass", "evt"};
          std::shared_ptr<SourceDesc> spectrum_source = std::make_shared<SourceDesc>(dsc.name,
                    dsc.total_entries, dsc.name_hash,
                    dsc.data_files, dsc.data_set_names_hashes,
                    dsc.point_entry, dsc.point_exit, dsc.tau_types,
                    disabled_branches, enabled_branches);
          spectrums[dsc.name]->AddHist_refill(spectrum_source);
        } else {
          for(const std::string& spectrum: dsc.spectrum_files)
            spectrums[dsc.name]->AddHist(spectrum);
        }
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

      // Added control over the ratio of tau types
      // Step 1: for non-"-1"(take all) types:
      // <expected ration>/<entries_per_type> is calc.

      // First, check that the ratio type exists within the entries
      for(auto tauR_: tau_ratio){
        if (accumulated_entries.find(tauR_.first) == accumulated_entries.end()){
          throw exception("Tau type %1% was specified in tau ratios, but no %1% is present in the input")
            %tauR_.first; 
        }
      }

      for(auto tauR_: tau_ratio){
        if(tauR_.second!=-1){
          if(accumulated_entries.at(tauR_.first)==0)
            throw exception("No taus of the type '%1%' are found in the tuples") % tauR_.first;
          else if(tauR_.second<0)
            throw exception("Available --tau_ratio arguments should be > 0 or -1");

          std::cout << "tau type: " << ToString(tauR_.first) << ", Entries: " 
                    << accumulated_entries.at(tauR_.first) << "\n";
          probab[tauR_.first] = tauR_.second/accumulated_entries.at(tauR_.first);
        }
      }
      // Step 2: if we have for non-"-1"(take all) types
      // to take maximum possible number of taus
      // max[<expected ration>/<entries_per_type>] 
      // should be normalized to 1.0 (means take all taus of the type)
      if(!probab.empty()) {
        auto pr = std::max_element(std::begin(probab), std::end(probab),
                  [] (const auto& p1, const auto& p2) {return p1.second < p2.second; });
        Double_t max_ = pr->second;
        std::cout << "max element: " << max_ << std::endl;
        for(auto tau_prob_: probab) probab.at(tau_prob_.first) = tau_prob_.second/max_;
      }
      // Step 3: Probability to take the rest types
      // which were defined with -1 are set to 100% (1.0)
      for(auto tauR_: tau_ratio){
        if(tauR_.second==-1){
          if(accumulated_entries.at(tauR_.first)==0)
            throw exception("No taus of the type '%1%' are found in the tuples") % tauR_.first;
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
    const bool lastbin_takeall;
    size_t n_entries;

    Generator* gen;
    Discret dist_dataset;
    Uniform dist_uniform;

};

class ShuffleMergeSpectral {
public:
    using Tau = tau_tuple::Tau;
    using TauTuple = tau_tuple::TauTuple;
    using Generator = DataSetProcessor::Generator;

    ShuffleMergeSpectral(const Arguments& _args) :
        args(_args), pt_bins(ParseBins(args.pt_bins())),
        eta_bins(ParseBins(args.eta_bins())), tau_ratio(ParseTauTypesR(args.tau_ratio())),
        max_entries(args.max_entries()),parity(args.parity()), compression(compAlg(args.compression_algo()))
    {
      bool verbose = 1;
		  if(args.n_threads() > 1) ROOT::EnableImplicitMT(args.n_threads());

      const auto disabled_branches_vec = SplitValueList(args.disabled_branches(), false, " ,");
      disabled_branches.insert(disabled_branches_vec.begin(), disabled_branches_vec.end());
      // we also need to dissable *_id due to the time conversion
      // and tauType because it were not introduced in the initial tuple
      disabled_branches.insert("tauType");
      disabled_branches.insert("dataset_id");
      disabled_branches.insert("dataset_group_id");

      PrintBins("pt bins", pt_bins);
      PrintBins("eta bins", eta_bins);
      const auto all_entries = LoadEntries(args.cfg());

      if(verbose) {
        for (auto dsc: all_entries) {
          std::cout << "entry group -> " << dsc.name << std::endl;
          std::cout << "files: " << dsc.data_files.size() << " ";
          if(!args.refill_spectrum())
            std::cout << "spectrums: " << dsc.spectrum_files.size();
          std::cout << std::endl;
        }
      }

      if(args.mode() == MergeMode::MergeAll) {
        entries[args.output()] = all_entries;
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
        auto output_file = root_ext::CreateRootFile(file_name, compression, args.compression_level());
        auto output_tuple = std::make_shared<TauTuple>("taus", output_file.get(), false);

        DataSetProcessor processor(entry_list, pt_bins, eta_bins,
                                   gen, disabled_branches, true, tau_ratio,
                                   args.lastbin_disbalance(), args.lastbin_takeall(),
                                   args.refill_spectrum(), args.enable_emptybin());

        size_t n_processed = 0;
        std::cout << "starting loops:" <<std::endl;
        while(processor.DoNextStep()){
          const auto& tau = processor.GetNextTau();
          n_processed++;
          (*output_tuple)() = tau;
          if(parity == 3 || (tau.evt % 2 == parity)) output_tuple->Fill();
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
            entries.emplace_back(item.second,  args.path_spectrum(), args.input(),
                                 args.prefix(), args.job_idx(), args.n_jobs(),
                                 args.refill_spectrum(), args.overflow_job());
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

    static ROOT::ECompressionAlgorithm compAlg(const std::string& comp_string) {
      if(comp_string=="ZLIB") return ROOT::kZLIB;
      if(comp_string=="LZMA") return ROOT::kLZMA;
      if(comp_string=="LZ4") return ROOT::kLZ4;
      throw exception("Invalid Compression Algorithm!");
    }

private:
    Arguments args;
    std::map<std::string, std::vector<EntryDesc>> entries;
    const std::vector<double> pt_bins, eta_bins;
    std::map<TauType, Double_t> tau_ratio;
    std::set<std::string> disabled_branches;
    const size_t max_entries;
    const UInt_t parity;
    const ROOT::ECompressionAlgorithm compression;
};

} // namespace analysis

PROGRAM_MAIN(analysis::ShuffleMergeSpectral, analysis::Arguments)
