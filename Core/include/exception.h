/*! Definition of the base exception class for the analysis namespace.
This file is part of https://github.com/hh-italian-group/AnalysisTools. */

#pragma once

#include <boost/format.hpp>
#include <sstream>
#include <memory>

namespace analysis {

class exception : public std::exception {
public:
    explicit exception(const std::string& message) noexcept : msg_valid(false), f_str(message)
    {
        try {
            f_msg = std::make_unique<boost::format>(f_str);
            f_msg->exceptions(boost::io::all_error_bits);
        } catch(boost::io::bad_format_string&) {
            msg = "bad formatted error message = '" + f_str + "'.";
            msg_valid = true;
        }
    }

    exception(const exception& e) noexcept : msg(e.msg), msg_valid(e.msg_valid), f_str(e.f_str)
    {
        if(e.f_msg)
            f_msg = std::make_unique<boost::format>(*e.f_msg);
    }

    exception(exception&& e) noexcept : msg(e.msg), msg_valid(e.msg_valid), f_msg(std::move(e.f_msg)), f_str(e.f_str) {}
    virtual ~exception() noexcept override {}
    virtual const char* what() const noexcept override { return message().c_str(); }

    const std::string& message() const noexcept
    {
        if(!msg_valid) {
            try {
                msg = boost::str(*f_msg);
            } catch(boost::io::too_few_args&) {
                msg = "too few arguments are provided to the error message = '" + f_str + "'.";
            } catch(std::exception& e) {
                process_unexpected_exception(e);
            }
            msg_valid = true;
        }
        return msg;
    }

    template<typename T>
    exception& operator % (const T& t) noexcept
    {
        try {
            if(!msg_valid && f_msg)
                *f_msg % t;
        } catch(boost::io::too_many_args&) {
            msg = "too many arguments are provided to the error message = '" + f_str + "'.";
            msg_valid = true;
        } catch(std::exception& e) {
            process_unexpected_exception(e);
        }
        return *this;
    }

private:
    void process_unexpected_exception(const std::exception& e) const
    {
        msg = "An exception has been raised while creating an error message. Error message = '" + f_str +
              "'. Exception message = '" + e.what() + "'.";
        msg_valid = true;
    }

private:
    mutable std::string msg;
    mutable bool msg_valid;
    std::unique_ptr<boost::format> f_msg;
    std::string f_str;
};

} // namespace analysis
