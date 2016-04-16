/*! Definition of the base exception class for the analysis namespace.
This file is part of https://github.com/hh-italian-group/AnalysisTools. */

#pragma once

#include <boost/format.hpp>
#include <sstream>

namespace analysis {

class exception : public std::exception {
public:
    explicit exception(const std::string& message) noexcept : f_msg(message) {}
    virtual ~exception() noexcept {}
    virtual const char* what() const noexcept { return message().c_str(); }
    const std::string& message() const noexcept { msg = boost::str(f_msg); return msg; }

    template<typename T>
    exception& operator % (const T& t)
    {
        f_msg % t;
        return *this;
    }

private:
    mutable std::string msg;
    boost::format f_msg;
};

} // analysis
