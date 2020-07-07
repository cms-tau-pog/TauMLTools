/*! Common tools and definitions suitable for debug purposes.
This file is part of https://github.com/hh-italian-group/TauMLTools. */

#pragma once

#include <vector>

namespace debug {

#define PRINT_SIZEOF(name) \
    std::cout << "Sizeof " #name " = " << sizeof(name) << std::endl

inline void PrintCommonTypeSizes()
{
    PRINT_SIZEOF(Float_t);
    PRINT_SIZEOF(Double_t);
    PRINT_SIZEOF(Int_t);
    PRINT_SIZEOF(UInt_t);
    PRINT_SIZEOF(Bool_t);
    PRINT_SIZEOF(Long64_t);
    PRINT_SIZEOF(int);
    PRINT_SIZEOF(unsigned);
    PRINT_SIZEOF(float);
    PRINT_SIZEOF(double);
    PRINT_SIZEOF(bool);
    PRINT_SIZEOF(size_t);

    using std_collection_size_type = std::vector<double>::size_type;
    PRINT_SIZEOF(std_collection_size_type);
}
#undef PRINT_SIZEOF

} // debug
