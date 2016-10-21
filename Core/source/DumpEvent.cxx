/*! Dump all variables in the event.
This file is part of https://github.com/hh-italian-group/AnalysisTools. */

#include <iostream>

#include <TTree.h>
#include "RootExt.h"
#include "AnalysisTools/Run/include/program_main.h"
#include "AnalysisTools/Core/include/EventIdentifier.h"
#include "AnalysisTools/Core/include/SmartBranch.h"

struct Arguments {
    REQ_ARG(std::string, fileName);
    REQ_ARG(std::string, treeName);
    REQ_ARG(std::string, eventId);
    OPT_ARG(std::string, eventIdBranches, "run:lumi:evt");
};

class DumpEvent {
public:
    using exception = analysis::exception;
    using EventIdentifier = analysis::EventIdentifier;
    using SmartBranch = root_ext::SmartBranch;

    DumpEvent(const Arguments& args) :
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
        const Long64_t entry = FindEntry();
        const auto branch_names = SmartBranch::CollectBranchNames(*tree);
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
    }

private:
    Long64_t FindEntry() const
    {
        for(Long64_t n = 0; n < tree->GetEntries(); ++n) {
            tree->GetEntry(n);
            const EventIdentifier currentEventId(idBranches.at(0).GetValue<EventIdentifier::IdType>(),
                                                 idBranches.at(1).GetValue<EventIdentifier::IdType>(),
                                                 idBranches.at(2).GetValue<EventIdentifier::IdType>());
            if(currentEventId == selectedEventId)
                return n;
        }
        throw exception("Event %1% not found.") % selectedEventId;
    }

private:
    std::shared_ptr<TFile> file;
    std::shared_ptr<TTree> tree;
    std::vector<SmartBranch> idBranches;
    EventIdentifier selectedEventId;
};

PROGRAM_MAIN(DumpEvent, Arguments)
