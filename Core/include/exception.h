/*! Definition of the base exception class for the analysis namespace.
This file is part of https://github.com/hh-italian-group/AnalysisTools. */

#pragma once

#include <exception>
#include <memory>
#include <boost/format.hpp>

namespace analysis {

class exception : public std::exception {
public:
    explicit exception(const std::string& message) noexcept;
    exception(const exception& e) noexcept;
    exception(exception&& e) noexcept;
    virtual ~exception() noexcept override {}
    virtual const char* what() const noexcept override;
    const std::string& message() const noexcept;

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
    void process_unexpected_exception(const std::exception& e) const;

private:
    std::unique_ptr<std::string> msg;
    std::unique_ptr<bool> msg_valid;
    std::unique_ptr<boost::format> f_msg;
    std::string f_str;
};

} // namespace analysis
