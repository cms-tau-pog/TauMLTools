/*! Definition of CMS event identifier.
This file is part of https://github.com/hh-italian-group/AnalysisTools. */

#pragma once

#include "exception.h"
#include <boost/algorithm/string.hpp>

namespace analysis {

struct EventIdentifier {
    using IdType = unsigned long long;

    IdType runId, lumiBlock, eventId;

    static constexpr IdType Undef_id = std::numeric_limits<IdType>::max();
    static const EventIdentifier& Undef_event() {
        static const EventIdentifier undef_event;
        return undef_event;
    }

    EventIdentifier() : runId(Undef_id), lumiBlock(Undef_id), eventId(Undef_id) {}

    EventIdentifier(IdType _runId, IdType _lumiBlock, IdType _eventId) :
        runId(_runId), lumiBlock(_lumiBlock), eventId(_eventId) {}

    explicit EventIdentifier(const std::string& id_str)
    {
        const std::vector<std::string> id_strings = Split(id_str);
        runId = Parse(id_strings.at(0), "runId");
        lumiBlock = Parse(id_strings.at(1), "lumiBlock");
        eventId = Parse(id_strings.at(2), "eventId");
    }

    bool operator == (const EventIdentifier& other) const { return !(*this != other); }

    bool operator != (const EventIdentifier& other) const
    {
        return runId != other.runId || lumiBlock != other.lumiBlock || eventId != other.eventId;
    }

    bool operator < (const EventIdentifier& other) const
    {
        if(runId != other.runId) return runId < other.runId;
        if(lumiBlock != other.lumiBlock) return lumiBlock < other.lumiBlock;
        return eventId < other.eventId;
    }

    std::string ToString() const
    {
        return boost::str(boost::format("%1%:%2%:%3%") % runId % lumiBlock % eventId);
    }

    static std::vector<std::string> Split(const std::string& id_str)
    {
        static const std::string separators = ":";
        std::vector<std::string> id_strings;
        boost::split(id_strings, id_str, boost::is_any_of(separators), boost::token_compress_on);
        if(id_strings.size() != 3)
            throw exception("Invalid event identifier '%1%'") % id_str;
        return id_strings;
    }

private:
    static IdType Parse(const std::string& str, const std::string& name)
    {
        std::istringstream ss(str);
        IdType id;
        ss >> id;
        if(ss.fail())
            throw exception("Invalid %1% = '%2%'.") % name % str;
        return id;
    }
};

std::ostream& operator <<(std::ostream& s, const EventIdentifier& event)
{
    s << event.ToString();
    return s;
}

} // namespace analysis
