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
#include "TauMLTools/Core/interface/exception.h"


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


template <typename ObjectType>
struct EntryDesc {

    std::string name;
    ULong64_t name_hash;
    std::vector<std::string> data_files;
    std::vector<std::string> data_set_names;
    std::vector<ULong64_t> data_set_names_hashes;
    std::set<std::string> spectrum_files;
    std::set<ObjectType> tau_types;

    // <file idx, event> for entry and exit point
    std::pair<size_t, size_t> point_entry;
    std::pair<size_t, size_t> point_exit;
    size_t total_entries;

    double mixing_coefficient; // Used only for simple merge mix

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

        if (item.Has("mix_p"))
            mixing_coefficient = item.Get<double>("mix_p");

        const regex dir_pattern (dir_pattern_str );
        const regex file_pattern(file_pattern_str);

        tau_types = SplitValueListT<ObjectType, std::set<ObjectType>>(tau_types_str, false, ",");

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


template <typename ObjectType>
struct SourceDesc {
    using Tau = tau_tuple::Tau;
    using TauTuple = tau_tuple::TauTuple;

    SourceDesc(const std::string& _name, const size_t _total_entries, const ULong64_t  _group_hash,
               const std::vector<std::string>& _file_names, const std::vector<ULong64_t>& _name_hashes,
               const std::pair<size_t, size_t> _point_entry, const std::pair<size_t, size_t> _point_exit,
               const std::set<ObjectType>& _tautypes, const std::set<std::string> _disabled_branches = {},
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
                      % current_file_index.value() % file_names.size() % name;
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

        current_tau_type = std::nullopt;
        current_tau_type = GetObjectType((*current_tuple)());

        } while (!current_tau_type || tau_types.find(current_tau_type.value()) == tau_types.end());

      (*current_tuple)().tauType = static_cast<Int_t>(current_tau_type.value());
      (*current_tuple)().dataset_id = dataset_hash;
      (*current_tuple)().dataset_group_id = group_hash;
      return true;
    }

    virtual ~SourceDesc() {};
    
    virtual std::optional<ObjectType> GetObjectType(const Tau& tau) {return std::nullopt;}

    const Tau& GetNextTau() { return current_tuple->data(); }
    const size_t GetNumberOfProcessed() const { return total_n_processed; }
    const size_t GetTotalEntries() const { return total_entries; }
    const ObjectType GetType() { return current_tau_type.value(); }

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
    const std::set<ObjectType> tau_types;
    std::shared_ptr<TFile> current_file;
    std::shared_ptr<TauTuple> current_tuple;
    std::optional<size_t> current_file_index;
    size_t files_n_total;;
    size_t entries_file;
    size_t entries_end;
    size_t current_entry;
    size_t total_n_processed;
    std::optional<ObjectType> current_tau_type;
    ULong64_t dataset_hash;
  };

} // namespace analysis
