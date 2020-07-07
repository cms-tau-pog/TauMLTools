/*! Rename and merge histograms.
This file is part of https://github.com/hh-italian-group/TauMLTools. */

#include <fstream>
#include <boost/algorithm/string.hpp>
#include "TauMLTools/Core/interface/RootExt.h"
#include "TauMLTools/Core/interface/TextIO.h"
#include "TauMLTools/Core/interface/program_main.h"

namespace {
struct Arguments {
    REQ_ARG(std::string, inputFile);
    REQ_ARG(std::string, outputFile);
    REQ_ARG(std::string, configFile);
};

struct Transform {
    std::vector<std::string> input_histograms;
    std::string output_histogram;

    static bool TryParse(std::string line, Transform& transform)
    {
        static const std::string separators = " \t", transition = "->";
        transform.input_histograms.clear();
        std::vector<std::string> entries;
        boost::trim_if(line, boost::is_any_of(separators));
        boost::split(entries, line, boost::is_any_of(separators), boost::token_compress_on);
        if(entries.size() < 3)
            return false;
        const auto output_hist_iter = std::prev(entries.end());
        const auto transition_iter = std::prev(output_hist_iter);
        if(*transition_iter != transition)
            return false;

        transform.input_histograms.insert(transform.input_histograms.end(), entries.begin(), transition_iter);
        transform.output_histogram = *output_hist_iter;
        return true;
    }
};

struct Config {
    Config(const std::string& file_name)
    {
        static const std::string white_spaces = " \t";
        std::ifstream cfg(file_name);

        size_t n = 0;
        while(cfg.good()) {
            std::string cfgLine;
            std::getline(cfg, cfgLine);
            ++n;
            boost::trim_if(cfgLine, boost::is_any_of(white_spaces));
            if (!cfgLine.size() || (cfgLine.size() && cfgLine.at(0) == '#')) continue;
            Transform transform;
            if(!Transform::TryParse(cfgLine, transform))
                throw analysis::exception("Invalid transform '%1%' on line %2%.") % cfgLine % n;
            transforms.push_back(transform);
        }
    }

    std::vector<Transform> transforms;
};

}

class RenameHistograms {
public:
    RenameHistograms(const Arguments& args) :
        inputFile(root_ext::OpenRootFile(args.inputFile())),
        outputFile(root_ext::CreateRootFile(args.outputFile())),
        config(args.configFile())
    {
    }

    void Run()
    {
        for(const Transform& trans : config.transforms) {
            auto iter = trans.input_histograms.begin();
            std::shared_ptr<TH1D> hist(root_ext::ReadCloneObject<TH1D>(*inputFile, *iter));
            ++iter;
            for(; iter != trans.input_histograms.end(); ++iter) {
                std::shared_ptr<TH1D> other_hist(root_ext::ReadCloneObject<TH1D>(*inputFile, *iter));
                hist->Add(other_hist.get());
            }
            std::string out_name = trans.output_histogram;
            auto dir_name_end = trans.output_histogram.find_last_of('/');
            TDirectory* out_dir = outputFile.get();
            if(dir_name_end != std::string::npos) {
                const std::string dir_name = trans.output_histogram.substr(0, dir_name_end);
                out_dir = outputFile->GetDirectory(dir_name.c_str());
                if(!out_dir)
                    out_dir = outputFile->mkdir(dir_name.c_str());
                if(!out_dir)
                    throw analysis::exception("Unable to create output directory '%1%'.") % dir_name;
                out_name = trans.output_histogram.substr(dir_name_end + 1);
            }
            root_ext::WriteObject(*hist, out_dir, out_name);
        }
    }

private:
    std::shared_ptr<TFile> inputFile, outputFile;
    Config config;
};

PROGRAM_MAIN(RenameHistograms, Arguments)
