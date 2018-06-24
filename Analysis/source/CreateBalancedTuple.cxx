/*! Create tuple with balanced (pt, eta) bins.
*/

#include <fstream>
#include "AnalysisTools/Run/include/program_main.h"
#include "AnalysisTools/Core/include/RootExt.h"
#include "AnalysisTools/Core/include/NumericPrimitives.h"
#include "TauML/Analysis/include/AnalysisTypes.h"
#include "TauML/Analysis/include/TauTuple.h"

namespace analysis {

struct Arguments {
    REQ_ARG(std::string, output);
    REQ_ARG(std::string, tree_name);
    REQ_ARG(std::string, input_dir);
    REQ_ARG(std::string, input_list);
    REQ_ARG(TauType, tau_type);
    REQ_ARG(RangeWithStep<double>, pt_range);
    REQ_ARG(RangeWithStep<double>, eta_range);
    REQ_ARG(unsigned, target_entries_per_bin);
};


class EntryCountMap {
public:
    using BinId = std::pair<size_t, size_t>;
    using EtaBinnedMap = std::vector<unsigned>;
    using PtEtaBinnedMap = std::vector<EtaBinnedMap>;
    using BinRange = analysis::RangeWithStep<double>;

    EntryCountMap(const BinRange& _pt_range, const BinRange& _eta_range, unsigned _target_entries_per_bin) :
        pt_range(_pt_range), eta_range(_eta_range), target_entries_per_bin(_target_entries_per_bin), total_count(0)
    {
        if(!target_entries_per_bin)
            throw exception("Target entries per bin should be >= 1.");
        const size_t n_pt_bins = pt_range.n_bins(), n_eta_bins = eta_range.n_bins();

        if(!n_pt_bins || !n_eta_bins)
            throw exception("Number of pt & eta bins should be >= 1.");

        const EtaBinnedMap ref_eta_map(n_eta_bins, 0);
        counts.resize(n_pt_bins, ref_eta_map);
        for(size_t pt_bin = 0; pt_bin < n_pt_bins; ++pt_bin) {
            for(size_t eta_bin = 0; eta_bin < n_eta_bins; ++eta_bin) {
                incomplete_bins.insert(BinId{pt_bin, eta_bin});
                empty_bins.insert(BinId{pt_bin, eta_bin});
            }
        }
    }

    bool AddEntry(double pt, double eta)
    {
        if(!pt_range.Contains(pt) || !eta_range.Contains(eta)) return false;
        const size_t pt_bin = pt_range.find_bin(pt);
        const size_t eta_bin = eta_range.find_bin(eta);
        unsigned& count = counts.at(pt_bin).at(eta_bin);
        if(count >= target_entries_per_bin) return false;
        ++count;
        ++total_count;
        const BinId bin_id{pt_bin, eta_bin};
        empty_bins.erase(bin_id);
        if(count == target_entries_per_bin)
            incomplete_bins.erase(bin_id);
        return true;
    }

    bool IsComplete() const { return incomplete_bins.empty(); }
    bool HasEmptyBins() const { return !empty_bins.empty(); }

    const std::set<BinId>& IncompleteBins() const { return incomplete_bins; }
    const std::set<BinId>& EmptyBins() const { return empty_bins; }

    size_t NumberOfBins() const { return pt_range.n_bins() * eta_range.n_bins(); }
    size_t NumberOfCompleteBins() const { return NumberOfBins() - NumberOfIncompleteBins(); }
    size_t NumberOfIncompleteBins() const { return incomplete_bins.size(); }
    size_t NumberOfEmptyBins() const { return empty_bins.size(); }
    size_t NumberOfPartiallyFilledBins() const { return NumberOfIncompleteBins() - NumberOfEmptyBins(); }

    size_t TotalCount() const { return total_count; }


    TH2I ExportCounts() const
    {
        const size_t n_pt_bins = pt_range.n_bins(), n_eta_bins = eta_range.n_bins();
        TH2I count_hist("count_hist", "count_hist",
                        static_cast<int>(n_pt_bins), pt_range.min(), pt_range.max(),
                        static_cast<int>(n_eta_bins), eta_range.min(), eta_range.max());
        for(size_t pt_bin = 0; pt_bin < n_pt_bins; ++pt_bin) {
            for(size_t eta_bin = 0; eta_bin < n_eta_bins; ++eta_bin) {
                count_hist.SetBinContent(static_cast<int>(pt_bin + 1), static_cast<int>(eta_bin + 1),
                                         counts.at(pt_bin).at(eta_bin));
            }
        }

        return count_hist;
    }

    void ReportIncompleteBins(std::ostream& os) const { ReportBins(os, incomplete_bins, {}); }
    void ReportPartiallyFilledBins(std::ostream& os) const { ReportBins(os, incomplete_bins, empty_bins); }
    void ReportEmptyBins(std::ostream& os) const { ReportBins(os, empty_bins, {}); }

private:
    void ReportBins(std::ostream& os, const std::set<BinId>& bin_set, const std::set<BinId>& exclude_set) const
    {
        for(const auto& bin : bin_set) {
            if(exclude_set.count(bin)) continue;
            const size_t pt_bin = bin.first;
            const size_t eta_bin = bin.second;
            const double pt_min = pt_range.grid_point_value(pt_bin);
            const double pt_max = pt_range.grid_point_value(pt_bin + 1);
            const double eta_min = eta_range.grid_point_value(eta_bin);
            const double eta_max = eta_range.grid_point_value(eta_bin + 1);
            os << "Bin pt = (" << pt_min << ", " << pt_max << "), eta = (" << eta_min << ", " << eta_max
               << "), count = " << counts.at(pt_bin).at(eta_bin) << "\n";
        }
    }

private:
    BinRange pt_range, eta_range;
    unsigned target_entries_per_bin;
    PtEtaBinnedMap counts;
    std::set<BinId> incomplete_bins, empty_bins;
    size_t total_count;
};

class CreateBalancedTuple {
public:
    using Tau = tau_tuple::Tau;
    using TauTuple = tau_tuple::TauTuple;
    using BinRange = EntryCountMap::BinRange;


    CreateBalancedTuple(const Arguments& _args) :
        args(_args), count_map(args.pt_range(), args.eta_range(), args.target_entries_per_bin())
    {
        std::ifstream cfg(args.input_list());
        if(cfg.fail())
            throw exception("Failed to open config '%1%'.") % args.input_list();

        while(cfg.good()) {
            std::string line;
            std::getline(cfg, line);
            boost::trim_if(line, boost::is_any_of(" \t"));
            if(line.empty() || line.at(0) == '#') continue;
            input_files.push_back(line);
        }
    }

    void Run()
    {
        std::cout << "Creating tuple with balanced (pt, eta) bins..." << std::endl;
        auto output_file = root_ext::CreateRootFile(args.output(), ROOT::kLZ4, 5);
        TauTuple output_tuple(args.tree_name(), output_file.get(), false);

        const bool map_filled = ProcessInputs(output_tuple);

        output_tuple.Write();
        const auto hist = count_map.ExportCounts();
        root_ext::WriteObject(hist, output_file.get());

        if(map_filled) {
            std::cout << "Output is fully filled." << std::endl;
        } else {
            std::cout << "Output is partially filled." << std::endl;
        }
    }

private:
    void ReportStatus(bool report_bin_list, bool report_summary, size_t prev_count, size_t n_processed) const
    {
        static const std::string sep(20, '-');
        if(report_bin_list) {
            if(!count_map.IsComplete()) {
                std::cout << sep << "\nPartially filled bins:\n";
                count_map.ReportPartiallyFilledBins(std::cout);
                std::cout << sep << std::endl;
            }
            if(count_map.HasEmptyBins()) {
                std::cout << sep << "\nEmpty bins:\n";
                count_map.ReportEmptyBins(std::cout);
                std::cout << sep << std::endl;
            }
        }
        if(report_summary) {
            std::cout << sep << "\ntotal number of bins: " << count_map.NumberOfBins()
                      << "\nnumber of complete bins: " << count_map.NumberOfCompleteBins()
                      << "\nnumer of partially filled bins: " << count_map.NumberOfPartiallyFilledBins()
                      << "\nnumber of empty bins: " << count_map.NumberOfEmptyBins()
                      << "\nnumber of accepted taus: " << count_map.TotalCount();
            if(prev_count) {
                const size_t new_acc = count_map.TotalCount() - prev_count;
                const double eff = double(new_acc) / n_processed * 100.;
                std::cout << "\nnumber of newly accepted taus: " << new_acc << " (efficiency " << eff << "%).";
            }
            std::cout << "\n" << sep << std::endl;
        }
    }

    bool ProcessInputs(TauTuple& output_tuple)
    {
        size_t prev_count = 0;
        for(const auto& file_name : input_files) {
            std::cout << "Processing '" << file_name << "'..." << std::endl;
            const std::string input_name = args.input_dir() + "/" + file_name;
            auto file = root_ext::OpenRootFile(input_name);
            auto tuple = std::make_shared<TauTuple>(args.tree_name(), file.get(), true);
            Long64_t n_processed = 0, n_total = tuple->GetEntries();
            for(const auto& tau : *tuple) {
                if(!CheckTauType(tau)) continue;
                if(count_map.AddEntry(tau.pt, tau.eta)) {
                    output_tuple() = tau;
                    output_tuple.Fill();
                    if(count_map.IsComplete()) return true;
                }
                if(++n_processed % 100000 == 0)
                    std::cout << "\tProcessed " << n_processed << " entries out of " << n_total
                              << ". Number of accepted taus = " << count_map.TotalCount() << "." << std::endl;
            }
            ReportStatus(true, true, prev_count, static_cast<size_t>(n_total));
            prev_count = count_map.TotalCount();
        }
        return false;
    }

    bool CheckTauType(const Tau& tau) const
    {
        const GenMatch gen_match = static_cast<GenMatch>(tau.gen_match);
        const TauType tau_type = GenMatchToTauType(gen_match);
        return tau_type == args.tau_type();
    }

private:
    Arguments args;
    EntryCountMap count_map;
    std::vector<std::string> input_files;
};

} // namespace analysis

PROGRAM_MAIN(analysis::CreateBalancedTuple, analysis::Arguments)
