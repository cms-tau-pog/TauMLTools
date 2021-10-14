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

namespace analysis {

enum class SignalClass {
  OtherJet = 0,
  TauJet = 1
};

boost::optional<SignalClass> GetSignalClass(const tau_tuple::Tau& tau)
{   
    using GTGenLeptonKind = reco_tau::gen_truth::GenLepton::Kind;
    const GTGenLeptonKind genLepton_kind = static_cast<GTGenLeptonKind> (tau.genLepton_kind);

    if( tau.jet_index >= 0 && tau.jet_pt >= 20 ) {
      if( tau.genLepton_index < 0 && tau.genJet_index >= 0) {
          return SignalClass::OtherJet;
      }
      else if( tau.genLepton_index >= 0 && genLepton_kind == GTGenLeptonKind::TauDecayedToHadrons ) {
          return SignalClass::TauJet;
      }
    }
    return boost::none;
}

struct Arguments {
    run::Argument<std::string> cfg{"cfg", "configuration file with the list of input sources", ""};
    run::Argument<std::string> input{"input", "Input file with the list of files to read. ", ""};
    run::Argument<std::string> output{"output", "output", ""};
    run::Argument<size_t> file_entries{"file-entries", "maximal number of entries per one file", std::numeric_limits<size_t>::max()};
    run::Argument<size_t> max_entries{"max-entries", "maximal number of entries processed", std::numeric_limits<size_t>::max()};
    run::Argument<unsigned> n_threads{"n-threads", "number of threads", 1};
    run::Argument<unsigned> seed{"seed", "random seed to initialize the generator used for sampling", 1234567};
    run::Argument<std::string> disabled_branches{"disabled-branches", "list of branches to disabled in the input tuples", ""};
    run::Argument<std::string> compression_algo{"compression-algo","ZLIB, LZMA, LZ4","LZMA"};
    run::Argument<unsigned> compression_level{"compression-level", "compression level of output file", 9};
    // run::Argument<unsigned> parity{"parity","take only even:0, take only odd:1, take all entries:3", 3};
};

struct SourceDesc {
    using Tau = tau_tuple::Tau;
    using TauTuple = tau_tuple::TauTuple;

    SourceDesc(const std::string& _name,
               const ULong64_t  _group_hash,
               const SignalClass source_class,
               const std::vector<std::string>& _file_names,
               const std::set<std::string> _disabled_branches = {},
               const std::set<std::string> _enabled_branches = {}) :

        name(_name), group_hash(_group_hash), source_class(source_class), file_names(_file_names), 
        disabled_branches(_disabled_branches), enabled_branches(_enabled_branches),
        current_entry(0), total_n_processed(0), file_index_end(file_names.size()),
        entries_end(std::numeric_limits<size_t>::max()), current_file_index(boost::none),
        signal_class(boost::none)
    {
        if(file_names.empty())
          throw exception("Empty list of files for the source '%1%'.") % name;
    }

    SourceDesc(const SourceDesc&) = delete;
    SourceDesc& operator=(const SourceDesc&) = delete;

    bool GetNextStep()
    {
      do {
        
        if(current_file_index == file_index_end && current_entry >= entries_end )
            return false;

        while(!current_file_index || current_entry >= entries_end) {
        
            if(!current_file_index) current_file_index = 0;
            else ++(*current_file_index);

            if(*current_file_index >= file_index_end)
                throw exception("File index: %1% is out of file_names array '%2%', DataGroup: '%3%'")
                      % current_file_index % file_index_end % name;

            const std::string& file_name = file_names.at(*current_file_index);
            std::cout << "Opening: " << name << " " << file_name << std::endl;
            current_tuple.reset();
            if(current_file) current_file->Close();

            current_file = root_ext::OpenRootFile(file_name);
            current_tuple = std::make_shared<TauTuple>("taus",
                current_file.get(), true, disabled_branches, enabled_branches
                );
            entries_end = current_tuple->GetEntries();

            if(entries_end==0)
              throw exception("Root file %1% is empty.") % file_name;
        }

        current_tuple->GetEntry(current_entry++);
        ++total_n_processed;

        signal_class = GetSignalClass((*current_tuple)());

      } while (!signal_class && signal_class == source_class);

      (*current_tuple)().signal_class = static_cast<Int_t>(signal_class.get());
      (*current_tuple)().dataset_group_id = group_hash;
      return true;
    }

    const Tau& GetNextTau() { return current_tuple->data(); }
    const size_t GetNumberOfProcessed() const { return total_n_processed; }

  private:
    const std::string name;
    const ULong64_t group_hash;
    const SignalClass source_class;
    const std::vector<std::string> file_names;
    const std::set<std::string> disabled_branches;
    const std::set<std::string> enabled_branches;
    size_t current_entry;
    size_t total_n_processed;
    size_t file_index_end;
    size_t entries_end;
    boost::optional<size_t> current_file_index;
    boost::optional<SignalClass> signal_class;
    ULong64_t dataset_hash;

    std::shared_ptr<TFile> current_file;
    std::shared_ptr<TauTuple> current_tuple;
  };
  

struct EntryDesc {

    std::string name;
    ULong64_t name_hash;
    std::vector<std::string> data_files{};
    SignalClass signal_class;
    double ratio;
    std::string pathtype;
    std::string path;

    EntryDesc(const PropertyConfigReader::Item& item,
              const std::string input_txt)
    {
        using boost::regex;
        using boost::regex_match;
        using boost::filesystem::is_regular_file;
        using boost::filesystem::is_directory;

        name = item.name;
        name_hash = std::hash<std::string>{}(name);

        auto isSignal = item.Get<int>("isSignal");
        signal_class = isSignal ? SignalClass::TauJet : SignalClass::OtherJet;

        pathtype = item.Get<std::string>("pathtype");
        ratio    = item.Get<float>("ratio");
        path     = item.Get<std::string>("path");

        const std::string dir_pattern_str = item.Get<std::string>("dir");
        const std::string file_pattern_str = item.Get<std::string>("file");

        const regex dir_pattern (dir_pattern_str );
        const regex file_pattern(file_pattern_str);

        std::ifstream input_files (input_txt, std::ifstream::in);
        if (!input_files){
          throw exception("The input txt file %1% could not be opened")
            %input_txt;
        }

        if( pathtype=="xrd" ) {
            std::string ifile;
            while(std::getline(input_files, ifile)){
              std::string file_name = ifile.substr(ifile.find_last_of("/") + 1,
                                       ifile.rfind(" ")-ifile.find_last_of("/")-1);
              std::string dir_name = ifile.substr(0,ifile.find_last_of("/"));
              dir_name  = dir_name.substr(dir_name.find_last_of("/")+1);
              if(!regex_match(dir_name , dir_pattern )) continue;
              if(!regex_match(path , dir_pattern )) continue;
              if(!regex_match(file_name, file_pattern)) continue;
              std::string file_path = path + "/" + file_name;
              data_files.push_back(file_path);
            }

        } else if( pathtype=="loc" ){
            if (!is_directory(path)){
              throw exception("The path %1% does not exists")
                %path;
            }
            data_files = analysis::RootFilesMerger::FindInputFiles(
              std::vector<std::string>{path},file_pattern_str,"",""
              );
        } else {
          throw exception("Undefined pathtype: %1%,'loc' or 'xrd' are available")
            % pathtype;
        }

       if(data_files.size()==0) {
         throw exception("No files available for datagroup: %1%")
            % name;
       } else {
         std::cout << "Group: " << name << " "
                   << "n_files: " << data_files.size()
                   << std::endl;
       }
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
                     Generator& _gen, const std::set<std::string>& disabled_branches,
                     bool verbose) : gen(&_gen)
    {
      if(verbose) std::cout << "Writing hash table." << std::endl;
      WriteHashTables("./out",entries);

      for(auto entry: entries){
        // std::shared_ptr<SourceDesc> source = 
        //     std::make_shared<SourceDesc>(entry.name, entry.name_hash,
        //                                  entry.data_files, disabled_branches);
        // sources[entry.name] = source;
        sources.push_back(std::make_unique<SourceDesc>(entry.name, entry.name_hash, entry.signal_class,
                                                       entry.data_files, disabled_branches));
        datagroup_probs.push_back(entry.ratio);
        // datagroup_names.push_back(entry.second->name);
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

class ShuffleMergeFlat {
public:
    using Tau = tau_tuple::Tau;
    using TauTuple = tau_tuple::TauTuple;
    using Generator = DataSetProcessor::Generator;

    ShuffleMergeFlat(const Arguments& _args) :
        args(_args), disabled_branches({}),
        compression(compAlg(args.compression_algo()))
    {
		  if(args.n_threads() > 1) ROOT::EnableImplicitMT(args.n_threads());

      const auto disabled_branches_vec = SplitValueList(args.disabled_branches(), false, " ,");
      disabled_branches.insert(disabled_branches_vec.begin(), disabled_branches_vec.end());

      entries = LoadEntries();
    }

    void Run()
    {
      Generator gen(args.seed());

      std::cout << "Output: " << args.output() << std::endl;
      auto output_file = root_ext::CreateRootFile(args.output(), compression, args.compression_level());
      auto output_tuple = std::make_shared<TauTuple>("taus", output_file.get(), false);

      DataSetProcessor processor(entries, gen, disabled_branches, true);

      size_t n_processed = 0;
      std::cout << "Starting loops." <<std::endl;
      while(processor.DoNextStep()){
        const auto& tau = processor.GetNextTau();
        n_processed++;
        (*output_tuple)() = tau;
        output_tuple->Fill();

        if(n_processed % 1000 == 0){
          std::cout << n_processed << " is selected" << std::endl;
        }

        if(n_processed>=args.max_entries()){
          std::cout << "Stop: number of entries exceeded max_entries" << std::endl;
          break;
        }
      }

      output_tuple->Write();
      output_tuple.reset();
      std::cout << args.output() << " has been successfully created." << std::endl;

    }

private:
    std::vector<EntryDesc> LoadEntries()
    {
        std::vector<EntryDesc> entries;
        PropertyConfigReader reader;
        std::cout << args.cfg() << std::endl;
        reader.Parse(args.cfg());
        for(const auto& item : reader.GetItems()){
            entries.emplace_back(item.second, args.input());
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
    std::vector<EntryDesc> entries;
    std::set<std::string> disabled_branches;
    // const UInt_t parity;
    const ROOT::ECompressionAlgorithm compression;
};

} // namespace analysis

PROGRAM_MAIN(analysis::ShuffleMergeFlat, analysis::Arguments)
