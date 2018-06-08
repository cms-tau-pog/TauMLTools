/*! Shuffle input tuples into one.
*/

#include <random>
#include "AnalysisTools/Run/include/program_main.h"
#include "AnalysisTools/Core/include/RootExt.h"
#include "TauML/Analysis/include/TauTuple.h"

struct Arguments {
    REQ_ARG(std::string, output);
    REQ_ARG(std::string, tree_name);
    REQ_ARG(std::vector<std::string>, input);
};

namespace analysis {

class ShuffleTuple {
public:
    using Tau = tau_tuple::Tau;
    using TauTuple = tau_tuple::TauTuple;

    ShuffleTuple(const Arguments& _args) : args(_args)
    {
    }

    void Run()
    {
        std::cout << "Starting shuffling entries..." << std::endl;
        std::vector<std::shared_ptr<TFile>> input_files;
        std::vector<std::shared_ptr<TauTuple>> input_tuples;
        std::vector<Long64_t> n_remaining_entries, current_entries;
        Long64_t n_entries_total = 0, n_processed = 0, n_total = 0;
        for(const auto& input_name : args.input()) {
            auto file = root_ext::OpenRootFile(input_name);
            auto tuple = std::make_shared<TauTuple>(args.tree_name(), file.get(), true);
            const Long64_t n_entries = tuple->GetEntries();
            n_remaining_entries.push_back(n_entries);
            current_entries.push_back(0);
            n_entries_total += n_entries;
            input_files.push_back(file);
            input_tuples.push_back(tuple);
        }

        auto output_file = root_ext::CreateRootFile(args.output(), ROOT::kLZ4, 5);
        TauTuple output_tuple(args.tree_name(), output_file.get(), false);
        std::random_device rnd;

        n_total = n_entries_total;
        while(n_entries_total > 0) {
            std::uniform_int_distribution<Long64_t> dist(1, n_entries_total);
            Long64_t pos = 0, selected_pos = dist(rnd);
            size_t tuple_index = 0;
            while(tuple_index < n_remaining_entries.size() && pos + n_remaining_entries[tuple_index] < selected_pos) {
                pos += n_remaining_entries[tuple_index++];
            }
            if(tuple_index == n_remaining_entries.size()) {
                std::cerr << "pos:" << pos << "\nremain: " << n_entries_total
                          << "\nselected pos: " << selected_pos << "\nn tuples: " << n_remaining_entries.size()
                          <<  std::endl;
                for(auto x : n_remaining_entries)
                    std::cout << "\t" << x << "\n";
                std::cout << std::endl;
                throw exception("Tuple index is out of range.");
            }
            auto input_tuple = input_tuples.at(tuple_index);
            input_tuple->GetEntry(current_entries.at(tuple_index)++);
            output_tuple() = input_tuple->data();
            output_tuple.Fill();
            --n_remaining_entries.at(tuple_index);
            --n_entries_total;
            if(++n_processed % 100000 == 0)
                std::cout << "Processed " << n_processed << " entries out of " << n_total << "." << std::endl;
        }

        output_tuple.Write();

        std::cout << "All entries are shuffled." << std::endl;
    }

private:
    Arguments args;
};

} // namespace analysis

PROGRAM_MAIN(analysis::ShuffleTuple, Arguments)
