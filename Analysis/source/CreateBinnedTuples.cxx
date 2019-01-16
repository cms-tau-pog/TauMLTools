/*! Create tuples splitted by the tau type and pt/eta bins.
*/

#include "AnalysisTools/Run/include/program_main.h"
#include "TauML/Analysis/include/TauTuple.h"
#include "TauML/Analysis/include/SummaryTuple.h"
#include "AnalysisTools/Core/include/RootFilesMerger.h"
#include "AnalysisTools/Core/include/NumericPrimitives.h"

struct Arguments {
    run::Argument<std::string> output{"output", "output directory"};
    run::Argument<std::vector<std::string>> input_dirs{"input-dir", "input directory"};
    run::Argument<std::string> pt_bins{"pt-bins", "pt bins"};
    run::Argument<std::string> eta_bins{"eta-bins", "eta bins"};
    run::Argument<std::string> file_name_pattern{"file-name-pattern", "regex expression to match file names",
                                                 "^.*\\.root$"};
    run::Argument<std::string> exclude_list{"exclude-list", "comma separated list of files to exclude", ""};
    run::Argument<std::string> exclude_dir_list{"exclude-dir-list",
                                                "comma separated list of directories to exclude", ""};
    run::Argument<unsigned> n_threads{"n-threads", "number of threads", 1};
};

namespace {
class EventBin {
public:
    using Tau = tau_tuple::Tau;
    using TauTuple = tau_tuple::TauTuple;
    using TauType = analysis::TauType;

    explicit EventBin(const std::string& _file_name) : file_name(_file_name) {}

    void AddTau(const Tau& tau)
    {
        TauTuple& tau_tuple = GetTuple();
        tau_tuple() = tau;
        tau_tuple.Fill();
    }

    void Write()
    {
        if(tuple)
            tuple->Write();
    }

    size_t GetSize() const
    {
        return tuple ? static_cast<size_t>(tuple->GetEntries()) : 0;
    }

private:
    TauTuple& GetTuple()
    {
        static constexpr Long64_t memory_limit = 10 * 1024 * 1024;
        if(!tuple) {
            file = root_ext::CreateRootFile(file_name, ROOT::kZLIB, 9);
            tuple = std::make_shared<TauTuple>("taus", file.get(), false);
            tuple->SetMaxVirtualSize(memory_limit);
            tuple->SetAutoFlush(-memory_limit);
        }
        return *tuple;
    }

private:
    std::string file_name;
    std::shared_ptr<TFile> file;
    std::shared_ptr<TauTuple> tuple;
};

class EventBinMap {
public:
    using BinRange = analysis::RangeWithStep<double>;
    using TauType = analysis::TauType;
    using Tau = tau_tuple::Tau;
    using TauTuple = tau_tuple::TauTuple;
    using ProdSummary = tau_tuple::ProdSummary;
    using SummaryTuple = tau_tuple::SummaryTuple;


    EventBinMap(const std::string& base_dir, const std::vector<double>& _pt_range,
                const std::vector<double>& _eta_range) :
        pt_range(_pt_range), eta_range(_eta_range), other(base_dir + "/other.root"),
        summary_file(root_ext::CreateRootFile(base_dir + "/summary.root")),
        summary_tuple("summary", summary_file.get(), false), total_size(0)
    {
        static const size_t n_tau_types = analysis::EnumNameMap<TauType>::GetDefault().GetEnumEntries().size();


        if(pt_range.size() <= 2 || eta_range.size() <= 2)
            throw analysis::exception("Number of pt & eta bins should be >= 1.");

        const size_t n_pt_bins = pt_range.size() - 1, n_eta_bins = eta_range.size() - 1;

        event_bins.resize(n_tau_types);
        for(size_t type_bin = 0; type_bin < n_tau_types; ++type_bin) {
            auto& type_bin_ref = event_bins.at(type_bin);
            type_bin_ref.resize(n_pt_bins);
            const TauType tau_type = static_cast<TauType>(type_bin);
            for(size_t pt_bin = 0; pt_bin < n_pt_bins; ++pt_bin) {
                auto& pt_bin_ref = type_bin_ref.at(pt_bin);
                const double pt_bin_edge = pt_range.at(pt_bin);
                for(size_t eta_bin = 0; eta_bin < n_eta_bins; ++eta_bin) {
                    const double eta_bin_edge = eta_range.at(eta_bin);
                    std::ostringstream ss;
                    ss << base_dir << "/" << tau_type << "_pt_" << std::fixed << std::setprecision(0) << pt_bin_edge
                       << std::setprecision(3) << "_eta_" << eta_bin_edge << ".root";
                    pt_bin_ref.emplace_back(ss.str());
                }
            }
        }
    }

    void AddTau(const Tau& tau)
    {
        if(PassSelection(tau)) {
            const auto gen_match = static_cast<analysis::GenLeptonMatch>(tau.lepton_gen_match);
            const TauType tau_type = analysis::GenMatchToTauType(gen_match);
            const size_t type_bin = static_cast<size_t>(tau_type);
            const size_t pt_bin = FindBin(pt_range, tau.tau_pt);
            const size_t eta_bin = FindBin(eta_range, std::abs(tau.tau_eta));
            auto& event_bin = event_bins.at(type_bin).at(pt_bin).at(eta_bin);
            event_bin.AddTau(tau);
        } else {
            other.AddTau(tau);
        }
        ++total_size;
    }

    void AddSummary(const ProdSummary& summary)
    {
        summary_tuple() = summary;
        summary_tuple.Fill();
    }

    size_t GetSize() const { return total_size; }

    void Write()
    {
        for(auto& type_bin : event_bins) {
            for(auto& pt_bin : type_bin) {
                for(auto& eta_bin : pt_bin) {
                    eta_bin.Write();
                }
            }
        }
        other.Write();
        summary_tuple.Write();
    }

    bool PassSelection(const Tau& tau) const
    {
        return tau.tau_index >= 0 && tau.tau_decayModeFindingNewDMs && std::abs(tau.tau_dz) < 0.2
                && Contains(pt_range, tau.tau_pt) && Contains(eta_range, std::abs(tau.tau_eta));
    }

    static bool Contains(const std::vector<double>& bins, double value)
    {
        return !bins.empty() && value >= bins.front() && value < bins.back();
    }

    static size_t FindBin(const std::vector<double>& bins, double value)
    {
        if(!Contains(bins, value))
            throw analysis::exception("FindBin: value is out of range.");
        size_t bin_id = 0;
        for(; bin_id < bins.size() - 2 && bins.at(bin_id) < value; ++bin_id);
        return bin_id;
    }

private:
    std::vector<double> pt_range, eta_range;
    std::vector<std::vector<std::vector<EventBin>>> event_bins;
    EventBin other;
    std::shared_ptr<TFile> summary_file;
    SummaryTuple summary_tuple;
    size_t total_size;
};
} // anonymous namespace

namespace analysis {
class CreateBinnedTuples {
public:
    using Tau = tau_tuple::Tau;
    using TauTuple = tau_tuple::TauTuple;
    using ProdSummary = tau_tuple::ProdSummary;
    using SummaryTuple = tau_tuple::SummaryTuple;
    using EntryId = tau_tuple::TauTupleEntryId;
    using EntryIdSet = std::set<EntryId>;

    CreateBinnedTuples(const Arguments& args) :
        input_files(RootFilesMerger::FindInputFiles(args.input_dirs(), args.file_name_pattern(),
                                                    args.exclude_list(), args.exclude_dir_list())),
        n_total_duplicates(0)
    {
        if(!boost::filesystem::exists(args.output()))
            boost::filesystem::create_directory(args.output());
        const auto pt_range = ParseBins(args.pt_bins());
        const auto eta_range = ParseBins(args.eta_bins());

        PrintBins("pt bins", pt_range);
        PrintBins("eta bins", eta_range);
        bin_map = std::make_shared<EventBinMap>(args.output(), pt_range, eta_range);
    }

    void Run()
    {
        for(const auto& file_name : input_files) {
            std::cout << "file: " << file_name << std::endl;
            auto file = root_ext::OpenRootFile(file_name);
            ProcessFile(file);
        }

        std::cout << "Writing output files..." << std::endl;

        bin_map->Write();

        std::cout << "All file has been merged. Number of files = " << input_files.size()
                  << ". Number of output entries = " << bin_map->GetSize()
                  << ". Total number of duplicated entires = " << n_total_duplicates << "." << std::endl;
    }

private:
    void ProcessFile(const std::shared_ptr<TFile>& file)
    {
        TauTuple input_tauTuple("taus", file.get(), true);
        size_t n_duplicates = 0;
        for(const Tau& tau : input_tauTuple) {
            const EntryId entry_id(tau);
            if(processed_entries.count(entry_id)) {
                ++n_duplicates;
                continue;
            }
            processed_entries.insert(entry_id);
            bin_map->AddTau(tau);
        }
        n_total_duplicates += n_duplicates;

        SummaryTuple input_summaryTuple("summary", file.get(), true);
        for(const ProdSummary& summary : input_summaryTuple)
            bin_map->AddSummary(summary);

        std::cout << "\tn_entries = " << input_tauTuple.GetEntries() << ", n_duplicates = " << n_duplicates << ".\n";
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
    std::vector<std::string> input_files;
    std::shared_ptr<EventBinMap> bin_map;
    EntryIdSet processed_entries;
    size_t n_total_duplicates;
};

} // namespace analysis

PROGRAM_MAIN(analysis::CreateBinnedTuples, Arguments)
