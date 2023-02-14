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
#include "TauMLTools/Core/interface/RootFilesMerger.h"
#include "TauMLTools/Core/interface/PropertyConfigReader.h"
#include "TauMLTools/Core/interface/ProgressReporter.h"
#include "TauMLTools/Analysis/interface/TauTuple.h"
#include "TauMLTools/Analysis/interface/GenLepton.h"
#include "TauMLTools/Analysis/interface/DisTauTagSelection.h"

namespace analysis {

struct IterIdx {
  size_t file_idx;
  size_t event_idx;
  IterIdx () {}
  IterIdx(const size_t _file_idx, 
          const size_t _event_idx) 
          : file_idx(_file_idx),
            event_idx(_event_idx) {}
};

class FilesSplit {

  public:

    IterIdx point_entry;
    IterIdx point_exit;
    size_t step;

    FilesSplit() {}
    FilesSplit(const std::vector<size_t>& files_entries,
               const size_t job_idx,
               const size_t n_job)
    {
      // FilesSplit splits files into n_job sub-intervals
      // the entry events for the job number job_idx are [point_entry, point_exit]...
      // if sum(files_entries) % n_job != 0 some events will be lost in datagroup
      if(job_idx>=n_job) throw exception("Error: job_idx should be < n_job");

      step = std::accumulate(files_entries.begin(), files_entries.end(), 0) / n_job;
      auto find_idx = [&](const size_t index, const bool isExit) -> IterIdx {
        size_t sum_accumulate = 0;
        for(size_t f_i = 0; f_i < files_entries.size(); f_i++) {
          if(step*(index+isExit) - sum_accumulate - isExit < files_entries[f_i])
            return IterIdx(f_i, step*(index+isExit) - sum_accumulate  - isExit);
          sum_accumulate+=files_entries[f_i];
        }
        throw exception("Error: sum-based splitting error!");
      };
      point_entry = find_idx(job_idx, false);
      point_exit = find_idx(job_idx, true);
    }
};

struct Arguments {
    run::Argument<std::string> cfg{"cfg", "configuration file with the list of input sources", ""};
    run::Argument<std::string> input{"input", "Input file with the list of files to read. ", ""};
    run::Argument<std::string> output{"output", "output", ""};
    // run::Argument<size_t> file_entries{"file-entries", "maximal number of entries per one file", std::numeric_limits<size_t>::max()};
    run::Argument<size_t> max_entries{"max-entries", "maximal number of entries processed", std::numeric_limits<size_t>::max()};
    run::Argument<unsigned> n_threads{"n-threads", "number of threads", 1};
    run::Argument<unsigned> seed{"seed", "random seed to initialize the generator used for sampling", 1234567};
    run::Argument<std::string> disabled_branches{"disabled-branches", "list of branches to disabled in the input tuples", ""};
    run::Argument<std::string> compression_algo{"compression-algo","ZLIB, LZMA, LZ4","LZMA"};
    run::Argument<unsigned> compression_level{"compression-level", "compression level of output file", 9};
    run::Argument<unsigned> job_idx{"job-idx", "index of the job (starts from 0)"};
    run::Argument<unsigned> n_jobs{"n-jobs", "the number by which to divide all files"};
};

struct SourceDesc {
    using Tau = tau_tuple::Tau;
    using TauTuple = tau_tuple::TauTuple;

    SourceDesc(const std::string& _name,
               const ULong64_t  _group_hash,
               const std::set<JetType> source_class,
               const std::vector<std::string>& _file_names,
               const FilesSplit _spliter,
               const std::set<std::string> _disabled_branches = {},
               const std::set<std::string> _enabled_branches = {}) :

        name(_name), group_hash(_group_hash), JetType_select(source_class), file_names(_file_names), 
        spliter(_spliter), disabled_branches(_disabled_branches), 
        enabled_branches(_enabled_branches), current_entry(0), total_n_processed(0),
        entries_end(std::numeric_limits<size_t>::max()), current_file_index(std::nullopt),
        JetType_current(std::nullopt)
    {
        if(file_names.empty())
          throw exception("Empty list of files for the source '%1%'.") % name;
    }

    SourceDesc(const SourceDesc&) = delete;
    SourceDesc& operator=(const SourceDesc&) = delete;

    bool GetNextStep()
    {
      do {

        if(current_file_index == spliter.point_exit.file_idx && current_entry > entries_end ) {
            std::cout << "No more entries in: " << name << std::endl;
            return false;
          }

        while(!current_file_index || current_entry > entries_end)
        {
            if(!current_file_index) current_file_index = spliter.point_entry.file_idx ;
            else ++(*current_file_index);
            if(*current_file_index >= file_names.size())
                throw exception("File index: %1% is out of file_names array '%2%', DataGroup: '%3%'")
                      % current_file_index.value() % file_names.size() % name;

            const std::string& file_name = file_names.at(*current_file_index);
            std::cout << "Opening: " << name << " " << file_name << std::endl;
            current_tuple.reset();
            if(current_file) current_file->Close();
            current_file = root_ext::OpenRootFile(file_name);
            current_tuple = std::make_shared<TauTuple>("taus",
                current_file.get(), true, disabled_branches, enabled_branches
                );
            entries_end = current_tuple->GetEntries();
            current_entry = current_file_index == spliter.point_entry.file_idx
                          ? spliter.point_entry.event_idx : 0; 
            entries_end = current_file_index == spliter.point_exit.file_idx
                        ? spliter.point_exit.event_idx : entries_end - 1;
            if(entries_end-current_entry<=0)
              throw exception("Error: interval is zero.");
        }

        current_tuple->GetEntry(current_entry++);
        ++total_n_processed;
        JetType_current = std::nullopt;
        JetType_current = GetJetType((*current_tuple)());

      } while (!JetType_current || JetType_select.find(JetType_current.value()) == JetType_select.end());

      (*current_tuple)().tauType = static_cast<Int_t>(JetType_current.value());
      (*current_tuple)().dataset_group_id = group_hash;
      return true;
    }

    const Tau& GetNextTau() { return current_tuple->data(); }
    const size_t GetNumberOfProcessed() const { return total_n_processed; }

  private:
    const std::string name;
    const ULong64_t group_hash;
    const std::set<JetType> JetType_select;
    const std::vector<std::string> file_names;
    const FilesSplit spliter;
    const std::set<std::string> disabled_branches;
    const std::set<std::string> enabled_branches;
    size_t current_entry;
    size_t total_n_processed;
    size_t file_index_end;
    size_t entries_end;
    std::optional<size_t> current_file_index;
    std::optional<JetType> JetType_current;
    ULong64_t dataset_hash;

    std::shared_ptr<TFile> current_file;
    std::shared_ptr<TauTuple> current_tuple;
  };


struct EntryDesc {

    std::string name;
    ULong64_t name_hash;
    std::vector<std::string> data_files;
    std::vector<std::string> data_set_names;
    std::vector<ULong64_t> data_set_names_hashes;
    std::set<JetType> tau_types;

    FilesSplit spliter;
    double mixing_coefficient;

    EntryDesc(const PropertyConfigReader::Item& item,
              const std::string& input_paths,
              const size_t job_idx, const size_t n_jobs)
    {
        using boost::regex;
        using boost::regex_match;
        using boost::filesystem::is_regular_file;
        using namespace boost::filesystem;

        name = item.name;
        name_hash = std::hash<std::string>{}(name);

        const std::string dir_pattern_str  = item.Get<std::string>("dir");
        const std::string file_pattern_str = item.Get<std::string>("file");
        const std::string tau_types_str    = item.Get<std::string>("types");
        mixing_coefficient = item.Get<double>("mix_p");

        const regex dir_pattern (dir_pattern_str );
        const regex file_pattern(file_pattern_str);
        const regex root_pattern(".*.root");

        tau_types = SplitValueListT<JetType, std::set<JetType>>(tau_types_str, false, ",");

        std::ifstream input_files (input_paths, std::ifstream::in);
        if (!input_files){
          throw exception("The input file %1% could not be opened")
            %input_paths;
        }

        bool is_matching_files = false;

        std::string ifile;
        std::string dir_name, file_name, file_path;
        std::vector<size_t> files_entries;

        while(std::getline(input_files, ifile)){

          size_t n_entries = analysis::Parse<double>(ifile.substr(ifile.rfind(" ")));
          file_name = ifile.substr(0,ifile.find_last_of(" "));

          if(!regex_match(file_name , dir_pattern )) continue;
          if(!regex_match(file_name, file_pattern)) continue;

          is_matching_files = true; //at least one file is found

          data_files.push_back(file_name);
          files_entries.push_back(n_entries);
          data_set_names.push_back(name);
          data_set_names_hashes.push_back(std::hash<std::string>{}(name));

        }

        if(!is_matching_files){
          throw exception("No files are found for entry '%1%' with pattern '%2%'")
              % name % file_pattern_str;
        }

        if(job_idx >= n_jobs)
          throw exception("Wrong job_idx! The index should be > 0 and < n_jobs");

        spliter = FilesSplit(files_entries, job_idx, n_jobs);
        std::cout <<  name << ": " 
            << "Entry point: " << spliter.point_entry.file_idx << " " << spliter.point_entry.event_idx << ", " 
            << "Exit point: " << spliter.point_exit.file_idx << " " << spliter.point_exit.event_idx << ", "
            << "Total entries: " << spliter.step << std::endl;
    }
};

class DataSetProcessor {
public:
    using Tau = tau_tuple::Tau;
    using TauTuple = tau_tuple::TauTuple;
    using Generator = std::mt19937_64;
    using Uniform = std::uniform_real_distribution<double>;
    using Discret = std::discrete_distribution<int>;

    DataSetProcessor(const std::vector<EntryDesc>& entries,
                     Generator& _gen,
                     const std::set<std::string>& disabled_branches,
                     bool verbose) :
                     gen(&_gen)
    {
      if(verbose) std::cout << "Writing hash table." << std::endl;
      WriteHashTables("./out",entries);

      for(auto entry: entries){
        sources.push_back(std::make_unique<SourceDesc>(entry.name, entry.name_hash, entry.tau_types,
                                                       entry.data_files, entry.spliter, disabled_branches));
        datagroup_probs.push_back(entry.mixing_coefficient);
      }
      dist_dataset = Discret(datagroup_probs.begin(), datagroup_probs.end());
    }

    bool DoNextStep() { 
      dg_idx=dist_dataset(*gen);
      return sources.at(dg_idx)->GetNextStep();
    }

    const Tau& GetNextTau() { return sources.at(dg_idx)->GetNextTau(); }

private:
    void WriteHashTables(const std::string& output,
                         const std::vector<EntryDesc>& entries)
    {
      if(!boost::filesystem::exists(output)) boost::filesystem::create_directory(output);
      boost::property_tree::ptree group;

      if(boost::filesystem::is_regular_file(output+"/datagroup_hash.json"))
        boost::property_tree::read_json(output+"/datagroup_hash.json", group);

      for(const EntryDesc& desc: entries){
        group.put(desc.name, desc.name_hash);
      }

      std::ofstream json_file2 (output+"/datagroup_hash.json", std::ios::out);
      boost::property_tree::write_json(json_file2, group);
      json_file2.close();
    }

private:
    std::vector<std::unique_ptr<SourceDesc>> sources;
    std::vector<Double_t> datagroup_probs;
    int dg_idx;
    // std::vector<std::string> datagroup_names;

    Generator* gen;
    Discret dist_dataset;
};


class SimpleMergeMix {
public:
    using Tau = tau_tuple::Tau;
    using TauTuple = tau_tuple::TauTuple;
    using Generator = DataSetProcessor::Generator;

    SimpleMergeMix(const Arguments& _args) :
        args(_args), disabled_branches({}),
        compression(compAlg(args.compression_algo()))
    {
      bool verbose = 1;
		  if(args.n_threads() > 1) ROOT::EnableImplicitMT(args.n_threads());

      const auto disabled_branches_vec = SplitValueList(args.disabled_branches(), false, " ,");
      disabled_branches.insert(disabled_branches_vec.begin(), disabled_branches_vec.end());
      // we also need to dissable *_id due to the time conversion
      // and tauType because it were not introduced in the initial tuple
      disabled_branches.insert("dataset_id");
      disabled_branches.insert("dataset_group_id");

      const auto all_entries = LoadEntries(args.cfg());

      if(verbose) {
        for (auto dsc: all_entries) {
          std::cout << "entry group -> " << dsc.name;
          std::cout << " files number: " << dsc.data_files.size() << "\n";
          for (auto file: dsc.data_files) std::cout << file << std::endl;
        }
      }
      global_entry = all_entries;
    }

    void Run()
    {

        Generator gen(args.seed());

        std::cout << "Output template: " << args.output() << std::endl;
        // int file_n = 0;
        std::string file_name_temp = args.output().substr(0,args.output().find_last_of("."));
        // std::string file_name = file_name_temp + "_" + std::to_string(file_n++) + ".root";
        std::string file_name = file_name_temp + ".root";
        auto output_file = root_ext::CreateRootFile(file_name, compression, args.compression_level());
        auto output_tuple = std::make_shared<TauTuple>("taus", output_file.get(), false);

        DataSetProcessor processor(global_entry, gen, disabled_branches, true);

        size_t n_processed = 0;
        std::cout << "Starting loops." <<std::endl;
        while(processor.DoNextStep())
        {
            const auto& tau = processor.GetNextTau();
            n_processed++;
            (*output_tuple)() = tau;
            output_tuple->Fill();
            if(n_processed % 10000 == 0) {
                std::cout << n_processed << " is selected" << std::endl;
            }
            if(n_processed>=args.max_entries() ){
                std::cout << "Stop: number of entries exceeded max_entries" << std::endl;
                break;
            }
            // if(n_processed>0 && (n_processed % args.file_entries() == 0)) {
            //     output_tuple->Write();
            //     output_tuple.reset();
            //     std::string file_name = file_name_temp + "_" + std::to_string(file_n++) + ".root";
            //     output_file = root_ext::CreateRootFile(file_name, compression, args.compression_level());
            //     output_tuple = std::make_shared<TauTuple>("taus", output_file.get(), false);
            // } 
        }
        output_tuple->Write();
        output_tuple.reset();
        std::cout << "DataSet has been successfully created." << std::endl;

    }

private:
    std::vector<EntryDesc> LoadEntries(const std::string& cfg_file_name)
    {
        std::vector<EntryDesc> entries;
        PropertyConfigReader reader;
        std::cout << cfg_file_name << std::endl;
        reader.Parse(cfg_file_name);
        for(const auto& item : reader.GetItems()){
            entries.emplace_back(item.second, args.input(),
                                 args.job_idx(), args.n_jobs());
        }
        return entries;
    }

    static ROOT::ECompressionAlgorithm compAlg(const std::string& comp_string) {
      if(comp_string=="ZLIB") return ROOT::kZLIB;
      if(comp_string=="LZMA") return ROOT::kLZMA;
      if(comp_string=="LZ4") return ROOT::kLZ4;
      throw exception("Invalid Compression Algorithm!");
    }

private:
    Arguments args;
    std::vector<EntryDesc> global_entry;
    std::set<std::string> disabled_branches;
    const ROOT::ECompressionAlgorithm compression;
};

} // namespace analysis

PROGRAM_MAIN(analysis::SimpleMergeMix, analysis::Arguments)