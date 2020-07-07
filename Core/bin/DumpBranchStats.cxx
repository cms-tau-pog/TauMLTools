/*! Dump branch statistics.
This file is part of https://github.com/hh-italian-group/TauMLTools. */

#include <iostream>

#include <TTree.h>
#include "TauMLTools/Core/interface/RootExt.h"
#include "TauMLTools/Core/interface/program_main.h"
#include "TauMLTools/Core/interface/SmartBranch.h"
#include "TauMLTools/Core/interface/EnumNameMap.h"

struct Arguments {
    REQ_ARG(std::string, fileName);
    REQ_ARG(std::string, treeName);
    OPT_ARG(std::string, ordering, "zipsize");
};

enum class BranchOrdering { Name, Size, ZipSize, CompressionFactor };

ENUM_NAMES(BranchOrdering) = {
    { BranchOrdering::Name, "name" },
    { BranchOrdering::Size, "size" },
    { BranchOrdering::ZipSize, "zipsize" },
    { BranchOrdering::CompressionFactor, "compression" },
};

class DumpBranchStats {
public:
    using exception = analysis::exception;
    using SmartBranch = root_ext::SmartBranch;

    DumpBranchStats(const Arguments& args) :
        file(root_ext::OpenRootFile(args.fileName())),
        tree(root_ext::ReadObject<TTree>(*file, args.treeName())),
        ordering(analysis::EnumNameMap<BranchOrdering>::GetDefault().Parse(args.ordering()))
    {
    }

    void Run()
    {
        std::vector<SmartBranch> branches;
        const auto branch_names = SmartBranch::CollectBranchNames(*tree);
        for(const std::string& name : branch_names) {
            try {
                SmartBranch branch(*tree, name);
                branches.push_back(branch);
            } catch(exception& e) {
                std::cerr << e.what() << std::endl;
            }
        }

        const auto comparitor = [&](const SmartBranch& b1, const SmartBranch& b2) -> bool {
            if(ordering == BranchOrdering::Size && b1.RawSize() != b2.RawSize())
                return b1.RawSize() > b2.RawSize();
            if(ordering == BranchOrdering::ZipSize && b1.ZipSize() != b2.ZipSize())
                return b1.ZipSize() > b2.ZipSize();
            if(ordering == BranchOrdering::CompressionFactor && b1.CompressionFactor() != b2.CompressionFactor())
                return b1.CompressionFactor() > b2.CompressionFactor();
            return b1.Name() < b2.Name();
        };

        std::sort(branches.begin(), branches.end(), comparitor);
        SmartBranch::PrintStatsHeader(std::cout);
        for(const auto& branch : branches)
            branch.PrintStats(std::cout);
        const double bytes_per_entry = double(tree->GetZipBytes()) / tree->GetEntries();
        std::cout << "\nTotal: n_entries = " << tree->GetEntries() << ", zip_size = " << tree->GetZipBytes()
                  << ", bytes_per_entry = " << bytes_per_entry << "." << std::endl;
    }

private:
    std::shared_ptr<TFile> file;
    std::shared_ptr<TTree> tree;
    BranchOrdering ordering;
};

PROGRAM_MAIN(DumpBranchStats, Arguments)
