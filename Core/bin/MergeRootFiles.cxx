/*! Merge multiple root files into a single file.
This file is part of https://github.com/hh-italian-group/TauMLTools. */

#include "TauMLTools/Core/interface/RootFilesMerger.h"
#include "TauMLTools/Core/interface/program_main.h"

struct Arguments {
    run::Argument<std::string> output{"output", "output root file"};
    run::Argument<std::vector<std::string>> input_dirs{"input-dir", "input directory"};
    run::Argument<std::string> file_name_pattern{"file-name-pattern", "regex expression to match file names",
                                                 "^.*\\.root$"};
    run::Argument<std::string> exclude_list{"exclude-list", "comma separated list of files to exclude", ""};
    run::Argument<std::string> exclude_dir_list{"exclude-dir-list",
                                                "comma separated list of directories to exclude", ""};
    run::Argument<unsigned> n_threads{"n-threads", "number of threads", 1};
};

class MergeRootFiles : public analysis::RootFilesMerger {
public:
    MergeRootFiles(const Arguments& args) :
        RootFilesMerger(args.output(), args.input_dirs(), args.file_name_pattern(), args.exclude_list(),
                        args.exclude_dir_list(), args.n_threads(), ROOT::kZLIB, 9)
    {
    }

    void Run()
    {
        Process(true, true);
    }
};

PROGRAM_MAIN(MergeRootFiles, Arguments)

