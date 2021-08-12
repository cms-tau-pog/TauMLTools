/*! Definition of the base exception class for the analysis namespace.
This file is part of https://github.com/hh-italian-group/TauMLTools. */

#pragma once

#include <exception>
#include <memory>
#include <sstream>
#include <boost/format.hpp>
#include <boost/stacktrace.hpp>

namespace analysis {

class exception : public std::exception {
public:
    explicit exception(const std::string& message) noexcept :
            msg(std::make_unique<std::string>()), msg_valid(std::make_unique<bool>(false)), f_str(message)
    {
        try {
            std::ostringstream ss;
            #ifndef PROJECT_VERSION
            ss << boost::stacktrace::stacktrace();
            #endif
            stack_trace = ss.str();
            f_msg = std::make_unique<boost::format>(f_str);
            f_msg->exceptions(boost::io::all_error_bits);
        } catch(boost::io::bad_format_string&) {
            *msg = "bad formatted error message = '" + f_str + "'.";
            *msg_valid = true;
        }
    }
    exception(const exception& e) noexcept :
            msg(std::make_unique<std::string>(*e.msg)), msg_valid(std::make_unique<bool>(*e.msg_valid)), f_str(e.f_str),
            stack_trace(e.stack_trace)
    {
        if(e.f_msg)
            f_msg = std::make_unique<boost::format>(*e.f_msg);
    }
    exception(exception&& e) noexcept :
        msg(std::move(e.msg)), msg_valid(std::move(e.msg_valid)), f_msg(std::move(e.f_msg)), f_str(e.f_str),
        stack_trace(e.stack_trace) {}
    virtual ~exception() noexcept override {}
    virtual const char* what() const noexcept override { return message().c_str(); }
    const std::string& message() const noexcept
    {
        if(!*msg_valid) {
            try {
                *msg = boost::str(*f_msg);
            } catch(boost::io::too_few_args&) {
                *msg = "too few arguments are provided to the error message = '" + f_str + "'.";
            } catch(std::exception& e) {
                process_unexpected_exception(e);
            }
            *msg_valid = true;
        }
        return *msg;
    }
    const std::string& stacktrace() const noexcept { return stack_trace; }

    template<typename T>
    exception& operator % (const T& t) noexcept
    {
        try {
            if(!*msg_valid && f_msg)
                *f_msg % t;
        } catch(boost::io::too_many_args&) {
            *msg = "too many arguments are provided to the error message = '" + f_str + "'.";
            *msg_valid = true;
        } catch(std::exception& e) {
            process_unexpected_exception(e);
        }
        return *this;
    }

private:
    void process_unexpected_exception(const std::exception& e) const
    {
        *msg = "An exception has been raised while creating an error message. Error message = '" + f_str +
            "'. Exception message = '" + e.what() + "'.";
        *msg_valid = true;
    }

private:
    std::unique_ptr<std::string> msg;
    std::unique_ptr<bool> msg_valid;
    std::unique_ptr<boost::format> f_msg;
    std::string f_str;
    std::string stack_trace;
};

} // namespace analysis