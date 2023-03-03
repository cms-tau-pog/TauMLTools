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
#include "TauMLTools/Analysis/interface/ShuffleTools.h"

namespace analysis {


struct Arguments {
    run::Argument<std::string> cfg{"cfg", "configuration file with the list of input sources", ""};
    run::Argument<std::string> input{"input", "Input file with the list of files to read. ", ""};
    run::Argument<std::string> output{"output", "output", ""};
    run::Argument<std::string> path_spectrum{"input-spec", "input path with spectrums for all the samples. "
                                                           "A remote server can be specified to use with xrootd."};
    run::Argument<std::string> prefix{"prefix", "prefix to place before the input file path read from --input. "
                                                "It can include a remote server to use with xrootd.", ""};
    run::Argument<size_t> max_entries{"max-entries", "maximal number of entries processed", std::numeric_limits<size_t>::max()};
    run::Argument<unsigned> n_threads{"n-threads", "number of threads", 1};
    run::Argument<unsigned> seed{"seed", "random seed to initialize the generator used for sampling", 1234567};
    run::Argument<std::string> disabled_branches{"disabled-branches", "list of branches to disabled in the input tuples", ""};
    run::Argument<std::string> compression_algo{"compression-algo","ZLIB, LZMA, LZ4","LZMA"};
    run::Argument<unsigned> compression_level{"compression-level", "compression level of output file", 9};
    run::Argument<unsigned> job_idx{"job-idx", "index of the job (starts from 0)"};
    run::Argument<unsigned> n_jobs{"n-jobs", "the number by which to divide all files"};
};

struct SourceDescJet : SourceDesc<JetType> {

  using Tau = tau_tuple::Tau;
  using TauTuple = tau_tuple::TauTuple;

  template<typename... Args>
  SourceDescJet(Args&&... args) : SourceDesc(std::forward<Args>(args)...) {}

  std::optional<JetType> GetObjectType(const Tau& tau) override
  {
    return GetJetType(tau);
  }

};

class DataSetProcessor {
public:
    using Tau = tau_tuple::Tau;
    using TauTuple = tau_tuple::TauTuple;
    using Generator = std::mt19937_64;
    using Uniform = std::uniform_real_distribution<double>;
    using Discret = std::discrete_distribution<int>;

    DataSetProcessor(const std::vector<EntryDesc<JetType>>& entries,
                     Generator& _gen,
                     const std::set<std::string>& disabled_branches,
                     bool verbose) :
                     gen(&_gen)
    {
      if(verbose) std::cout << "Writing hash table." << std::endl;
      WriteHashTables("./out",entries);

      for(auto entry: entries){
        sources.push_back(std::make_unique<SourceDescJet>(entry.name, entry.total_entries, entry.name_hash,
                                                          entry.data_files, entry.data_set_names_hashes, entry.point_entry, 
                                                          entry.point_exit, entry.tau_types, disabled_branches));
        datagroup_probs.push_back(entry.mixing_coefficient);
      }
      dist_dataset = Discret(datagroup_probs.begin(), datagroup_probs.end());
    }

    bool Step() { 
      dg_idx=dist_dataset(*gen);
      return sources.at(dg_idx)->DoNextStep();
    }

    const Tau& GetNextTau() { return sources.at(dg_idx)->GetNextTau(); }

private:
    void WriteHashTables(const std::string& output,
                         const std::vector<EntryDesc<JetType>>& entries)
    {
      if(!boost::filesystem::exists(output)) boost::filesystem::create_directory(output);
      boost::property_tree::ptree set, group;

      if(boost::filesystem::is_regular_file(output+"/dataset_hash.json"))
        boost::property_tree::read_json(output+"/dataset_hash.json", set);
      if(boost::filesystem::is_regular_file(output+"/datagroup_hash.json"))
        boost::property_tree::read_json(output+"/datagroup_hash.json", group);

      for(const EntryDesc<JetType>& desc: entries){
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

private:
    std::vector<std::unique_ptr<SourceDescJet>> sources;
    std::vector<Double_t> datagroup_probs;
    int dg_idx;

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
        while(processor.Step())
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
    std::vector<EntryDesc<JetType>> LoadEntries(const std::string& cfg_file_name)
    {
        std::vector<EntryDesc<JetType>> entries;
        PropertyConfigReader reader;
        std::cout << cfg_file_name << std::endl;
        reader.Parse(cfg_file_name);
        for(const auto& item : reader.GetItems()){
            entries.emplace_back(item.second, args.path_spectrum(), args.input(), 
                                 args.prefix(), args.job_idx(), args.n_jobs(), false, false);
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
    std::vector<EntryDesc<JetType>> global_entry;
    std::set<std::string> disabled_branches;
    const ROOT::ECompressionAlgorithm compression;
};

} // namespace analysis

PROGRAM_MAIN(analysis::SimpleMergeMix, analysis::Arguments)