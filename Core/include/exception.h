/*! Definition of the base exception class for the analysis namespace.
This file is part of https://github.com/hh-italian-group/AnalysisTools. */

#pragma once

#include <exception>
#include <string>
#include <sstream>

namespace analysis {
class exception : public std::exception {
public:
    explicit exception() noexcept {}
    explicit exception(const std::string& message) noexcept { s_msg << message; }
    exception(const exception& other) noexcept { s_msg << other.message(); }
    virtual ~exception() noexcept {}
    virtual const char* what() const noexcept { msg = s_msg.str(); return msg.c_str(); }
    const std::string& message() const noexcept { msg = s_msg.str(); return msg; }

    template<typename T>
    exception& operator << (const T& t) { s_msg << t; return *this; }

private:
    mutable std::string msg;
    std::ostringstream s_msg;
};

} // analysis
