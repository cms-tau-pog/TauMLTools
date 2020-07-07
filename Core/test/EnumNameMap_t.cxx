/*! Test EnumNameMap class.
This file is part of https://github.com/hh-italian-group/TauMLTools. */

#include "TauMLTools/Core/interface/EnumNameMap.h"
#include "TauMLTools/Core/interface/program_main.h"

namespace analysis {
    enum E1 { A, B, C };
    ENUM_NAMES(E1) = {
        { E1::A, "A" },
        { E1::B, "B" },
        { E1::C, "C" },
    };

    namespace test {
        using ::analysis::operator<<;
        using ::analysis::operator>>;
        enum E2 { A, B, C };
        ENUM_NAMES(E2) = {
            { E2::A, "A" },
            { E2::B, "B" },
            { E2::C, "C" },
        };
    }
}

namespace other {
    using ::analysis::operator<<;
    using ::analysis::operator>>;
    enum E3 { A, B, C };
    ENUM_NAMES(E3) = {
        { E3::A, "A" },
        { E3::B, "B" },
        { E3::C, "C" },
    };
}

struct Arguments {
};

class EnumNameMap_t {
public:
    EnumNameMap_t(const Arguments&) {}

    void Run()
    {
        const std::string name = "A";
        {
            std::istringstream is(name);
            analysis::E1 e;
            is >> e;
            std::cout << "E1: " << e << std::endl;
        }

        {
            std::istringstream is(name);
            analysis::test::E2 e;
            is >> e;
            std::cout << "E2: " << e << std::endl;
        }

        {
            std::istringstream is(name);
            other::E3 e;
            is >> e;
            std::cout << "E3: " << e << std::endl;
        }

    }
};

PROGRAM_MAIN(EnumNameMap_t, Arguments)
