/*! Uniformly shuffle entries of a tuple.
*/

#include <random>
#include "AnalysisTools/Run/include/program_main.h"
#include "AnalysisTools/Core/include/RootExt.h"
#include "TauML/Analysis/include/TauTuple.h"

struct Arguments {
    REQ_ARG(std::string, input);
    REQ_ARG(std::string, output);
    REQ_ARG(std::string, tree_name);
};

namespace analysis {

class ShuffleTupleEntries {
public:
    using Tau = tau_tuple::Tau;
    using TauTuple = tau_tuple::TauTuple;

    ShuffleTupleEntries(const Arguments& _args) : args(_args)
    {
    }

    void Run()
    {
        std::cout << "Starting shuffling entries..." << std::endl;
        auto input_file = root_ext::OpenRootFile(args.input());
        TauTuple input_tuple(args.tree_name(), input_file.get(), true);
        const size_t n_total = static_cast<size_t>(input_tuple.GetEntries());
        Long64_t n_processed = 0;

        std::vector<Long64_t> entries(n_total);
        std::iota(entries.begin(), entries.end(), 0);
        std::random_device rnd;
        std::mt19937_64 twister(rnd());
        std::shuffle(entries.begin(), entries.end(), twister);

        auto output_file = root_ext::CreateRootFile(args.output(), ROOT::kLZ4, 5);
        TauTuple output_tuple(args.tree_name(), output_file.get(), false);
        for(Long64_t entry : entries) {
            input_tuple.GetEntry(entry);
            output_tuple() = input_tuple.data();
            output_tuple.Fill();
            if(++n_processed % 10000 == 0)
                std::cout << "Processed " << n_processed << " entries out of " << n_total << "." << std::endl;
        }

        output_tuple.Write();

        std::cout << "All entries are shuffled." << std::endl;
    }

private:
    Arguments args;
};

} // namespace analysis

PROGRAM_MAIN(analysis::ShuffleTupleEntries, Arguments)
