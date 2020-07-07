/*! Compute CRC32 for an input string.
This file is part of https://github.com/hh-italian-group/TauMLTools. */

#include <boost/crc.hpp>
#include "TauMLTools/Core/interface/program_main.h"

struct Arguments {
    REQ_ARG(std::vector<std::string>, inputStrings);
};

class ComputeCRC {
public:
    ComputeCRC(const Arguments& _args) : args(_args) {}

    void Run()
    {
        for(const std::string& str : args.inputStrings()) {
            boost::crc_32_type crc;
            crc.process_bytes(str.data(), str.size());
            std::cout << crc.checksum() << std::endl;
        }
    }

private:
    Arguments args;
};

PROGRAM_MAIN(ComputeCRC, Arguments)
