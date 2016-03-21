/*! Definition of the class to report current execution progress.
This file is part of https://github.com/hh-italian-group/AnalysisTools. */

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
    static std::string TimeStamp(const clock::time_point& time_point)
    {
        std::ostringstream ss;
        const std::time_t time_t_point = clock::to_time_t(time_point);
        //ss << "[" << std::put_time(std::localtime(&time_t_point), "%F %T") << "] ";
        char mbstr[100];
        if (std::strftime(mbstr,sizeof(mbstr),"%F %T",std::localtime(&time_t_point)))
            ss << "[" << mbstr << "] ";
        return ss.str();
    }

    ProgressReporter(unsigned _report_interval, std::ostream& _output)
        : start(clock::now()), block_start(start), report_interval(_report_interval), output(&_output)
    {
        *output << TimeStamp(start) << "Starting analyzer..." << std::endl;
    }

    void Report(size_t event_id, bool final_report = false)
    {
        using namespace std::chrono;

        const auto now = clock::now();
        const unsigned since_last_report = duration_cast<seconds>(now - block_start).count();
        if(!final_report && since_last_report < report_interval) return;

        const unsigned since_start = duration_cast<seconds>(now - start).count();
        const double speed = double(event_id) / since_start;
        *output << TimeStamp(now);
        if(final_report)
            *output << "Total: ";
        *output << "time = " << since_start << " seconds, events processed = " << event_id
                  << ", average speed = " << std::setprecision(1) << std::fixed << speed << " events/s" << std::endl;
        const unsigned since_start_residual = since_start % report_interval;
        block_start = now - seconds(since_start_residual);
    }

private:
    clock::time_point start, block_start;
    unsigned report_interval;
    std::ostream* output;
};
} // namespace tools
} // namespace analysis
