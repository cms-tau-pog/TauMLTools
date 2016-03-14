/*! Definition of the base exception class for the analysis namespace.
This file is part of https://github.com/hh-italian-group/AnalysisTools. */

#pragma once

#include <boost/format.hpp>
#include <sstream>

template<typename Enum, typename = typename std::enable_if<std::is_enum<Enum>::value>::type>
std::ostream& operator<<(std::ostream& os, Enum e);

namespace analysis {

namespace detail {
template<typename T, bool is_special>
struct exceception_format;

template<typename T>
struct exceception_format<T, false> {
    static void format(boost::format& f_msg, const T& t) { f_msg % t; }
};

template<typename T>
struct exceception_format<T, true> {
    static void format(boost::format& f_msg, const T& t)
    {
        std::ostringstream ss;
        ss << t;
        f_msg % ss.str();
    }
};
}

class exception : public std::exception {
public:
    explicit exception(const std::string& message) noexcept : f_msg(message) {}
    virtual ~exception() noexcept {}
    virtual const char* what() const noexcept { return message().c_str(); }
    const std::string& message() const noexcept { msg = boost::str(f_msg); return msg; }

    template<typename T>
    exception& operator % (const T& t)
    {
        detail::exceception_format<T, std::is_enum<T>::value>::format(f_msg, t);
        return *this;
    }

private:
    mutable std::string msg;
    boost::format f_msg;
};

} // analysis
