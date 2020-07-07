/*! Definition of the class to report current execution progress.
This file is part of https://github.com/hh-italian-group/TauMLTools. */

#pragma once

#include <sstream>
#include <iomanip>
#include <chrono>

namespace analysis {
namespace tools {
class ProgressReporter {
private:
    using clock = std::chrono::system_clock;

public:
    static std::string TimeStamp(const clock::time_point& time_point);

    ProgressReporter(unsigned _report_interval, std::ostream& _output,
                     const std::string& init_message = "Starting analyzer...");
    void SetTotalNumberOfEvents(size_t _total_n_events);
    void Report(size_t event_id, bool final_report = false);

private:
    clock::time_point start, block_start;
    unsigned report_interval;
    std::ostream* output;
    size_t total_n_events, last_reported_event_id;
    std::string total_n_events_str;
};
} // namespace tools
} // namespace analysis
