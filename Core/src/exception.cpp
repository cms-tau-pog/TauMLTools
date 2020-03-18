/*! Definition of the base exception class for the analysis namespace.
This file is part of https://github.com/hh-italian-group/AnalysisTools. */

#include "AnalysisTools/Core/include/exception.h"

namespace analysis {

exception::exception(const std::string& message) noexcept
    : msg(std::make_unique<std::string>()), msg_valid(std::make_unique<bool>(false)), f_str(message)
{
    try {
        f_msg = std::make_unique<boost::format>(f_str);
        f_msg->exceptions(boost::io::all_error_bits);
    } catch(boost::io::bad_format_string&) {
        *msg = "bad formatted error message = '" + f_str + "'.";
        *msg_valid = true;
    }
}

exception::exception(const exception& e) noexcept
    : msg(std::make_unique<std::string>(*e.msg)), msg_valid(std::make_unique<bool>(*e.msg_valid)), f_str(e.f_str)
{
    if(e.f_msg)
        f_msg = std::make_unique<boost::format>(*e.f_msg);
}

exception::exception(exception&& e) noexcept :
    msg(std::move(e.msg)), msg_valid(std::move(e.msg_valid)), f_msg(std::move(e.f_msg)), f_str(e.f_str) {}
const char* exception::what() const noexcept { return message().c_str(); }

const std::string& exception::message() const noexcept
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

void exception::process_unexpected_exception(const std::exception& e) const
{
    *msg = "An exception has been raised while creating an error message. Error message = '" + f_str +
           "'. Exception message = '" + e.what() + "'.";
    *msg_valid = true;
}

} // namespace analysis
