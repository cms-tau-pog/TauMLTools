/*! Dump all variables in the event.
This file is part of https://github.com/hh-italian-group/TauMLTools. */

#include <iostream>

#include <TTree.h>
#include "TauMLTools/Core/interface/RootExt.h"
#include "TauMLTools/Core/interface/program_main.h"
#include "TauMLTools/Core/interface/EventIdentifier.h"
#include "TauMLTools/Core/interface/SmartBranch.h"

struct Arguments {
    REQ_ARG(std::string, fileName);
    REQ_ARG(std::string, treeName);
    REQ_ARG(std::string, eventId);
    OPT_ARG(Long64_t, firstEntry, 0);
    OPT_ARG(Long64_t, lastEntry, std::numeric_limits<Long64_t>::max());
    OPT_ARG(size_t, maxMatches, std::numeric_limits<size_t>::max());
    OPT_ARG(std::string, eventIdBranches, "run:lumi:evt");
};

class DumpEvent {
public:
    using exception = analysis::exception;
    using EventIdentifier = analysis::EventIdentifier;
    using SmartBranch = root_ext::SmartBranch;

    DumpEvent(const Arguments& _args) :
        args(_args),
        file(root_ext::OpenRootFile(args.fileName())),
        tree(root_ext::ReadObject<TTree>(*file, args.treeName())),
        selectedEventId(args.eventId())
    {
        const auto eventIdentifierBranches = EventIdentifier::Split(args.eventIdBranches());
        for(const auto& name : eventIdentifierBranches) {
            SmartBranch branch(*tree, name);
            branch.Enable();
            idBranches.push_back(branch);
        }
    }

    void Run()
    {
        static const std::string sep(10, '-');
        std::vector<Long64_t> entries = FindEntries();
        if(!entries.size())
            throw exception("Event %1% not found.") % selectedEventId;
        const auto branch_names = SmartBranch::CollectBranchNames(*tree);
        for(Long64_t entry : entries) {
            std::cout << sep << " entry " << entry << " START " << sep << std::endl;
            for(const std::string& name : branch_names) {
                try {
                    SmartBranch branch(*tree, name);
                    branch.Enable();
                    branch->GetEntry(entry);
                    branch.PrintValue(std::cout);
                } catch(exception& e) {
                    std::cerr << e.what() << std::endl;
                }
            }
            std::cout << sep << " entry " << entry << " END " << sep << std::endl;
        }
    }

private:
    std::vector<Long64_t> FindEntries() const
    {
        std::vector<Long64_t> entries;
        const Long64_t n_max = std::min(tree->GetEntries(), args.lastEntry());
        for(Long64_t n = args.firstEntry(); n < n_max; ++n) {

            tree->GetEntry(n);
            const EventIdentifier currentEventId(idBranches.at(0).GetValue<EventIdentifier::IdType>(),
                                                 idBranches.at(1).GetValue<EventIdentifier::IdType>(),
                                                 idBranches.at(2).GetValue<EventIdentifier::IdType>());
            if(currentEventId == selectedEventId) {
                entries.push_back(n);
                if(entries.size() >= args.maxMatches()) break;
            }
        }
        return entries;
    }

private:
    Arguments args;
    std::shared_ptr<TFile> file;
    std::shared_ptr<TTree> tree;
    std::vector<SmartBranch> idBranches;
    EventIdentifier selectedEventId;
};

PROGRAM_MAIN(DumpEvent, Arguments)
