/*! Merges and shuffles input files into one.
*/

#include <fstream>
#include <random>
#include <boost/algorithm/string.hpp>
#include <boost/regex.hpp>

#include "AnalysisTools/Run/include/program_main.h"
#include "AnalysisTools/Core/include/RootExt.h"
#include "AnalysisTools/Core/include/PropertyConfigReader.h"
#include "AnalysisTools/Core/include/ProgressReporter.h"
#include "TauML/Analysis/include/TauTuple.h"

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
    run::Argument<size_t> max_bin_occupancy{"max-bin-occupancy", "maximal occupancy of a bin",
                                            std::numeric_limits<size_t>::max()};
    run::Argument<unsigned> n_threads{"n-threads", "number of threads", 1};
    run::Argument<unsigned> seed{"seed", "random seed to initialize the generator used for sampling", 1234567};
    run::Argument<std::string> disabled_branches{"disabled-branches",
                                                 "list of branches to disabled in the input tuples", ""};
};

namespace {

struct SourceDesc {
    using Tau = tau_tuple::Tau;
    using TauTuple = tau_tuple::TauTuple;

    SourceDesc(const std::string& _name, const std::vector<std::string>& _file_names, size_t _total_n_events,
               double _weight, const std::set<std::string>& _disabled_branches) :
        name(_name), file_names(_file_names), disabled_branches(_disabled_branches), weight(_weight),
        current_n_processed(0), total_n_processed(0), total_n_events(_total_n_events)
    {
        if(file_names.empty())
            throw analysis::exception("Empty list of files for the source '%1%'.") % name;
        if(weight <= 0)
            throw analysis::exception("Invalid source weight for the source '%1%'.") % name;
        if(!total_n_events)
            throw analysis::exception("Empty source '%1%'.") % name;
    }

    SourceDesc(const SourceDesc&) = delete;
    SourceDesc& operator=(const SourceDesc&) = delete;

    bool HasNextTau() const { return total_n_processed < total_n_events; }
    const Tau& GetNextTau()
    {
        if(!HasNextTau())
            throw analysis::exception("No taus are available in the source '%1%' in bin '%2%'.") % name % bin_name;

        while(!current_file_index || current_n_processed == current_tuple->GetEntries()) {
            if(!current_file_index)
                current_file_index = 0;
            else
                ++(*current_file_index);
            if(*current_file_index >= file_names.size())
                throw analysis::exception("The expected number of events = %1% is bigger than the actual number of"
                                          " events in source '%2%'.") % total_n_events % name;
            current_n_processed = 0;
            const std::string& file_name = file_names.at(*current_file_index);
            current_tuple.reset();
            current_file = root_ext::OpenRootFile(file_name);
            current_tuple = std::make_shared<TauTuple>("taus", current_file.get(), true, disabled_branches);
        }
        ++total_n_processed;
        current_tuple->GetEntry(current_n_processed++);
        return current_tuple->data();
    }

    size_t GetNumberOfEvents() const { return total_n_events; }
    const std::string& GetName() const { return name; }
    const std::vector<std::string>& GetFileNames() const { return file_names; }
    double GetWeight() const { return weight; }
    const std::string& GetBinName() const { return bin_name; }
    void SetBinName(const std::string& _bin_name) { bin_name = _bin_name; }

private:
    const std::string name;
    const std::vector<std::string> file_names;
    const std::set<std::string> disabled_branches;
    const double weight;
    std::string bin_name;
    boost::optional<size_t> current_file_index;
    std::shared_ptr<TFile> current_file;
    std::shared_ptr<TauTuple> current_tuple;
    Long64_t current_n_processed;
    size_t total_n_processed, total_n_events;
};

class EventBin {
public:
    using Tau = tau_tuple::Tau;
    using TauTuple = tau_tuple::TauTuple;
    using Generator = std::mt19937_64;
    using Uniform = std::uniform_int_distribution<size_t>;

    EventBin(const std::string& _bin_name, double _bin_size, size_t _max_n_events, Generator& _gen) :
        bin_name(_bin_name), bin_size(_bin_size), max_n_events(_max_n_events), n_events(0), n_processed(0),
        bin_weight(1), gen(&_gen)
    {
    }

    void AddSource(const std::shared_ptr<SourceDesc>& source)
    {
        if(n_processed > 0)
            throw analysis::exception("Some events in the bin has already been processed. Unable to add a new entry.");
        n_events += source->GetNumberOfEvents();
        source->SetBinName(bin_name);
        sources.push_back(source);
    }

    void CreateBalancedInputs()
    {
        if(n_processed > 0)
            throw analysis::exception("Some events in the bin has already been processed. Unable to create balanced"
                                      " inputs.");
        if(sources.empty())
            throw analysis::exception("No sources are available for bin '%1%'.") % bin_name;

        const auto sourceCmp = [&](size_t id1, size_t id2) {
            const auto& src1 = sources.at(id1);
            const auto& src2 = sources.at(id2);
            if(src1->GetWeight() != src2->GetWeight()) return src1->GetWeight() > src2->GetWeight();
            if(src1->GetNumberOfEvents() != src2->GetNumberOfEvents())
                return src1->GetNumberOfEvents() < src2->GetNumberOfEvents();
            return src1->GetName() < src2->GetName();
        };

        n_remaining_events_per_source.assign(sources.size(), 0);
        n_remaining_events = GetEffectiveNumberOfEvents();
        const size_t n_sources = sources.size();
        size_t n_non_assigned_events = n_remaining_events;
        std::vector<size_t> source_indices(n_sources);
        std::iota(source_indices.begin(), source_indices.end(), 0);
        std::sort(source_indices.begin(), source_indices.end(), sourceCmp);

        while(n_non_assigned_events > 0) {
            double total_sources_weight = 0;
            for(size_t n = 0; n < n_sources; ++n) {
                const size_t source_id = source_indices.at(n);
                const auto& source = sources.at(source_id);
                const size_t n_available_source_events = source->GetNumberOfEvents()
                                                         - n_remaining_events_per_source.at(source_id);
                if(n_available_source_events == 0) continue;
                total_sources_weight += source->GetWeight();
            }

            size_t n_newly_assigned_events = 0;
            for(size_t n = 0; n < n_sources; ++n) {
                const size_t source_id = source_indices.at(n);
                const auto& source = sources.at(source_id);
                const size_t n_available_source_events = source->GetNumberOfEvents()
                                                         - n_remaining_events_per_source.at(source_id);
                if(n_available_source_events == 0) continue;
                const size_t expected_events_per_source = static_cast<size_t>(
                    std::ceil(n_non_assigned_events * source->GetWeight() / total_sources_weight));
                const size_t events_per_source = std::min(
                    std::min(expected_events_per_source, n_available_source_events),
                    n_non_assigned_events - n_newly_assigned_events
                );
                n_remaining_events_per_source.at(source_id) += events_per_source;
                n_newly_assigned_events += events_per_source;
            }
            n_non_assigned_events -= n_newly_assigned_events;
        }
    }

    const std::string& GetName() const { return bin_name; }
    double GetBinSize() const { return bin_size; }
    size_t GetTotalNumberOfEvents() const { return n_events; }
    size_t GetEffectiveNumberOfEvents() const { return std::min(max_n_events, n_events); }

    double GetBinWeight() const { return bin_weight; }
    void SetBinWeight(double weight) { bin_weight = weight; }

    bool HasNextTau() const { return n_processed < GetEffectiveNumberOfEvents(); }
    const Tau& GetNextTau()
    {
        if(!HasNextTau())
            throw analysis::exception("No taus are available in the bin.");
        if(n_remaining_events_per_source.empty())
            throw analysis::exception("CreateBalancedInputs should be called before the first GetNextTau call.");

        Uniform distr(0, n_remaining_events - 1);
        size_t n = 0;
        for(size_t index = distr(*gen); index >= n_remaining_events_per_source.at(n);
            index -= n_remaining_events_per_source.at(n++));

        --n_remaining_events_per_source.at(n);
        --n_remaining_events;
        ++n_processed;
        return sources.at(n)->GetNextTau();
    }

    void PrintSummary() const
    {
        std::cout << bin_name << ": total n_events = " << GetEffectiveNumberOfEvents()
                  << ", bin weight = " << bin_weight << ", n_events per sample - ";
        for(size_t n = 0; n < sources.size(); ++n) {
            const auto& source = sources.at(n);
            std::cout << source->GetName() << " = " << n_remaining_events_per_source.at(n);
            if(n != sources.size() - 1)
                std::cout << ", ";
        }
        std::cout << ".\n";
    }

private:
    const std::string bin_name;
    const double bin_size;
    const size_t max_n_events;
    size_t n_events, n_processed;
    double bin_weight;
    Generator* gen;
    std::vector<std::shared_ptr<SourceDesc>> sources;
    std::vector<size_t> n_remaining_events_per_source;
    size_t n_remaining_events;
};

struct BinFiles {
    using TauType = analysis::TauType;

    std::string bin_name;
    TauType tau_type;
    std::vector<std::string> files;

    BinFiles() {}
    BinFiles(const std::string& _bin_name)
        : bin_name(_bin_name)
    {
        const auto split = analysis::SplitValueList(bin_name, true, "_", false);
        if(split.size() != 5)
            throw analysis::exception("Invalid bin name '%1%'.") % bin_name;
        tau_type = analysis::Parse<TauType>(split.at(0));
    }
};

struct EntryDesc {
    using TauType = analysis::TauType;

    std::string name;
    std::map<std::string, std::vector<std::string>> bin_files;
    std::set<TauType> tau_types;
    double weight;

    EntryDesc(const analysis::PropertyConfigReader::Item& item, const std::string& base_dir_name)
    {
        using boost::regex;
        using boost::regex_match;
        using boost::filesystem::path;
        using boost::make_iterator_range;
        using boost::filesystem::directory_iterator;
        using boost::filesystem::is_directory;

        name = item.name;
        const std::string dir_pattern_str = item.Get<std::string>("dir");
        const std::string file_pattern_str = item.Get<std::string>("file");
        weight = item.Has("weight") ? item.Get<double>("weight") : 1;

        const path base_dir_path(base_dir_name);
        if(!is_directory(base_dir_name))
            throw analysis::exception("The base directory '%1%' does not exists.") % base_dir_name;

        const regex dir_pattern(base_dir_name + "/" + dir_pattern_str);
        bool has_dir_match = false;
        for(const auto& sample_dir_entry : make_iterator_range(directory_iterator(base_dir_path))) {
            if(!is_directory(sample_dir_entry) || !regex_match(sample_dir_entry.path().string(), dir_pattern)) continue;
            has_dir_match = true;

            const regex file_pattern(sample_dir_entry.path().string() + "/" + file_pattern_str + "\\.root");
            bool has_file_match = false;
            for(const auto& file_entry : make_iterator_range(directory_iterator(sample_dir_entry.path()))) {
                if(is_directory(file_entry) || !regex_match(file_entry.path().string(), file_pattern)) continue;
                has_file_match = true;

                const std::string file_name = file_entry.path().string();
                const std::string bin_name = GetBinName(file_name);
                if(!bin_files.count(bin_name))
                    tau_types.insert(GetTauType(bin_name));
                bin_files[bin_name].push_back(file_name);
            }
            if(!has_file_match)
                throw analysis::exception("No files are found for entry '%1%' sample '%2' with pattern '%3%'")
                      % name % sample_dir_entry % file_pattern_str;
        }

        if(!has_dir_match)
            throw analysis::exception("No samples are found for entry '%1%' with pattern '%2%'")
                  % name % dir_pattern_str;
    }

    static std::string GetBinName(const std::string& file_name)
    {
        static const std::string extension = ".root";
        if(file_name.size() > extension.size()) {
            const size_t ext_pos = file_name.size() - extension.size();
            const std::string ext = file_name.substr(ext_pos);
            if(ext == extension) {
                size_t start_pos = file_name.find_last_of('/');
                if(start_pos == std::string::npos)
                    start_pos = 0;
                else
                    ++start_pos;
                if(start_pos != ext_pos)
                    return file_name.substr(start_pos, ext_pos - start_pos);
            }
        }
        throw analysis::exception("Can't extract bin name from file name '%1%'.") % file_name;
    }

    static TauType GetTauType(const std::string& bin_name)
    {
        const auto split = analysis::SplitValueList(bin_name, true, "_", false);
        if(split.size() != 5)
            throw analysis::exception("Invalid bin name '%1%'.") % bin_name;
        return analysis::Parse<TauType>(split.at(0));
    }
};

class EventBinMap {
public:
    using TauType = analysis::TauType;
    using Tau = tau_tuple::Tau;
    using TauTuple = tau_tuple::TauTuple;
    using Generator = EventBin::Generator;
    using Uniform = EventBin::Uniform;

    EventBinMap(const std::vector<EntryDesc>& entries, const std::vector<double>& pt_bins,
                const std::vector<double>& eta_bins, bool calc_weights, size_t max_bin_occupancy, Generator& _gen,
                const std::map<std::string, size_t>& n_events_per_file,
                const std::set<std::string>& disabled_branches, bool verbose) :
        gen(&_gen)
    {
        double total_area;
        if(verbose)
            std::cout << "\tCalculating bin sizes... " << std::flush;
        const std::map<std::string, double> bin_sizes = CalculateBinSizes(pt_bins, eta_bins, total_area);
        if(verbose)
            std::cout << "done.\n\tCreating bins..." << std::endl;
        CreateBins(entries, bin_sizes, max_bin_occupancy, n_events_per_file, disabled_branches, !calc_weights, verbose);
        if(calc_weights) {
            if(verbose)
                std::cout << "\tCalculating weigts..." << std::endl;
            CalcWeigts(total_area);
        }
        if(verbose)
            std::cout << "\tCreating remaining counts... " << std::flush;
        CreateRemainingCounts();
        if(verbose) {
            std::cout << "done.\n";
            PrintSummary();
        }
    }

    size_t GetNumberOfRemainingEvents() const { return n_remaining_events; }
    bool HasNextTau() const { return n_remaining_events > 0; }
    const Tau& GetNextTau(double& weight)
    {
        if(!HasNextTau())
            throw analysis::exception("No taus are available.");

        Uniform distr(0, n_remaining_events - 1);

        size_t n = 0;
        for(size_t index = distr(*gen); index >= n_remaining_events_per_bin.at(n);
            index -= n_remaining_events_per_bin.at(n++));

        --n_remaining_events_per_bin.at(n);
        --n_remaining_events;
        auto& bin = bins.at(n);
        weight = bin.GetBinWeight();
        return bin.GetNextTau();
    }

    void PrintSummary() const
    {
        std::cout << "Bins statistics:\n";
        for(const auto& bin : bins)
            bin.PrintSummary();
        std::cout << "\nTotal: n_bins = " << bins.size() << ", n_events = " << n_remaining_events << std::endl;
    }

private:
    void CreateBins(const std::vector<EntryDesc>& entries, const std::map<std::string, double>& bin_sizes,
                    size_t max_bin_occupancy, const std::map<std::string, size_t>& n_events_per_file,
                    const std::set<std::string>& disabled_branches, bool allow_empty_bins, bool verbose)
    {
        std::set<TauType> tau_types;
        for(const auto& entry : entries)
            tau_types.insert(entry.tau_types.begin(), entry.tau_types.end());

        if(verbose)
            std::cout << "\t\tCreating bins... " << std::flush;
        std::map<std::string, EventBin> bins_map;
        for(TauType tau_type : tau_types) {
            for(const auto& bin_entry : bin_sizes) {
                const std::string bin_name = analysis::ToString(tau_type) + "_" + bin_entry.first;
                const double bin_size = bin_entry.second;
                bins_map.insert({bin_name, EventBin(bin_name, bin_size, max_bin_occupancy, *gen)});
            }
        }
        if(verbose)
            std::cout << "done.\n\t\tCreating sources... " << std::flush;

        for(const auto& entry : entries) {
            for(const auto& bin_entry : entry.bin_files) {
                const std::string& bin_name = bin_entry.first;
                const auto& file_names = bin_entry.second;

                if(!bins_map.count(bin_name))
                    throw analysis::exception("Unknown bin name '%1%'.") % bin_name;

                size_t n_events_bin = 0;
                for(const auto& file_name : file_names) {
                    if(!n_events_per_file.count(file_name))
                        throw analysis::exception("Missing an information about the number of events for file '%1%'")
                              % file_name;
                    n_events_bin += n_events_per_file.at(file_name);
                }

                auto source = std::make_shared<SourceDesc>(entry.name, file_names, n_events_bin, entry.weight,
                                                           disabled_branches);
                bins_map.at(bin_name).AddSource(source);
            }
        }

        if(verbose)
            std::cout << "done.\n\t\tCreating balanced inputs... " << std::flush;
        for(auto& bin_entry : bins_map) {
            EventBin& bin_desc = bin_entry.second;
            if(bin_desc.GetTotalNumberOfEvents() > 0) {
                bin_desc.CreateBalancedInputs();
                bins.emplace_back(std::move(bin_desc));
            } else if(!allow_empty_bins) {
                throw analysis::exception("Bin '%1%' is empty.") % bin_desc.GetName();
            }
        }
        if(verbose)
            std::cout << "done." << std::endl;
    }

    void CalcWeigts(double total_area)
    {
        double min_weight = std::numeric_limits<double>::infinity();
        double max_weight = 0;
        for(auto& bin : bins) {
            const size_t n_events_bin = bin.GetEffectiveNumberOfEvents();
            if(!n_events_bin)
                throw analysis::exception("Empty bin %1%.") % bin.GetName();
            const double bin_area = bin.GetBinSize();
            const double weight = bin_area / total_area / n_events_bin;
            bin.SetBinWeight(weight);
            min_weight = std::min(min_weight, weight);
            max_weight = std::max(max_weight, weight);
        }

        const double dyn_range = max_weight / min_weight;
        std::cout << boost::format("\t\tRange of the bin weight distribution: [%1%, %2%]. max/min = %3%.")
                     % min_weight % max_weight % dyn_range << std::endl;
        if(min_weight <= 0 || dyn_range >= 1e7)
            throw analysis::exception("Range of the bin weight distribution is too large to be applied during"
                                      " the training.");
    }

    void CreateRemainingCounts()
    {
        n_remaining_events = 0;
        for(const auto& bin : bins) {
            const size_t n_events_bin = bin.GetEffectiveNumberOfEvents();
            n_remaining_events_per_bin.push_back(n_events_bin);
            n_remaining_events += n_events_bin;
        }
    }

    static std::map<std::string, double> CalculateBinSizes(const std::vector<double>& pt_bins,
                                                           const std::vector<double>& eta_bins, double& total_area)
    {
        if(pt_bins.size() < 2 || eta_bins.size() < 2)
            throw analysis::exception("Number of pt & eta bins should be >= 1.");

        const size_t n_pt_bins = pt_bins.size() - 1, n_eta_bins = eta_bins.size() - 1;
        total_area = (pt_bins.back() - pt_bins.front()) * (eta_bins.back() - eta_bins.front());
        std::map<std::string, double> bin_sizes;
        for(size_t pt_bin = 0; pt_bin < n_pt_bins; ++pt_bin) {
            const double pt_bin_edge = pt_bins.at(pt_bin);
            for(size_t eta_bin = 0; eta_bin < n_eta_bins; ++eta_bin) {
                const double eta_bin_edge = eta_bins.at(eta_bin);
                std::ostringstream ss;
                ss << "pt_" << std::fixed << std::setprecision(0) << pt_bin_edge
                   << std::setprecision(3) << "_eta_" << eta_bin_edge;
                const std::string bin_name = ss.str();
                const double bin_size = (pt_bins.at(pt_bin + 1) - pt_bin_edge)
                                        * (eta_bins.at(eta_bin + 1) - eta_bin_edge);
                bin_sizes[bin_name] = bin_size;
            }
        }
        return bin_sizes;
    }

private:
    Generator* gen;
    std::vector<EventBin> bins;
    std::vector<size_t> n_remaining_events_per_bin;
    size_t n_remaining_events;
};
} // anonymous namespace

namespace analysis {
class ShuffleMerge {
public:
    using Tau = tau_tuple::Tau;
    using TauTuple = tau_tuple::TauTuple;
    using Generator = EventBinMap::Generator;

    ShuffleMerge(const Arguments& _args) :
        args(_args), pt_bins(ParseBins(args.pt_bins())), eta_bins(ParseBins(args.eta_bins())),
        n_events_per_file(LoadNumberOfEventsPerFile(args.input() + "/size_list.txt", args.input()))
    {
		if(args.n_threads() > 1)
            ROOT::EnableImplicitMT(args.n_threads());

        const auto disabled_branches_vec = SplitValueList(args.disabled_branches(), false, " ,");
        disabled_branches.insert(disabled_branches_vec.begin(), disabled_branches_vec.end());

        PrintBins("pt bins", pt_bins);
        PrintBins("eta bins", eta_bins);

        const auto all_entries = LoadEntries(args.cfg());
        if(args.mode() == MergeMode::MergeAll) {
            entries[args.output()] = all_entries;
        } else if(args.mode() == MergeMode::MergePerEntry) {
            for(const auto& entry : all_entries) {
                const std::string output_name = args.output() + "/" + entry.name + ".root";
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
            EventBinMap bin_map(entry_list, pt_bins, eta_bins, args.calc_weights(), args.max_bin_occupancy(), gen,
                                n_events_per_file, disabled_branches, true);
            tools::ProgressReporter reporter(10, std::cout, "Sampling taus...");
            reporter.SetTotalNumberOfEvents(bin_map.GetNumberOfRemainingEvents());
            size_t n_processed = 0;
            while(bin_map.HasNextTau()) {
                double weight;
                const auto& tau = bin_map.GetNextTau(weight);
                (*output_tuple)() = tau;
                if(args.calc_weights())
                    (*output_tuple)().trainingWeight = static_cast<float>(weight);
                output_tuple->Fill();
                if(++n_processed % 1000 == 0)
                    reporter.Report(n_processed);
            }
            reporter.Report(n_processed, true);
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
        reader.Parse(cfg_file_name);
        for(const auto& item : reader.GetItems())
            entries.emplace_back(item.second, args.input());
        return entries;
    }

    static std::map<std::string, size_t> LoadNumberOfEventsPerFile(const std::string& cfg_file_name,
                                                                   const std::string& base_dir_name)
    {
        std::ifstream cfg(cfg_file_name);
        if(cfg.fail())
            throw exception("Failed to open file '%1%'.") % cfg_file_name;

        std::map<std::string, size_t> n_events_per_file;
        while(cfg.good()) {
            std::string line;
            std::getline(cfg, line);
            if(line.empty() || line.at(0) == '#') continue;
            auto split = SplitValueList(line, true, " \t", true);
            if(split.size() == 2) {
                const std::string& file_name = base_dir_name + "/" + split.at(0);
                if(n_events_per_file.count(file_name))
                    throw exception("Duplicated entry for '%1%'") % file_name;
                size_t n_events;
                if(TryParse(split.at(1), n_events)) {
                    n_events_per_file[file_name] = n_events;
                    continue;
                }
            }
            throw exception("Invalid line = '%1%' in '%2%'.") % line % cfg_file_name;
        }

        return n_events_per_file;
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
    std::map<std::string, size_t> n_events_per_file;
};

} // namespace analysis

PROGRAM_MAIN(analysis::ShuffleMerge, Arguments)
