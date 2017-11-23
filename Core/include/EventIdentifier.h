/*! Definition of CMS event identifier.
This file is part of https://github.com/hh-italian-group/AnalysisTools. */

#pragma once

#include <map>
#include <boost/algorithm/string.hpp>
#include "exception.h"


namespace analysis {

struct EventIdentifier {
    using IdType = unsigned long long;
    static constexpr IdType Undef_id = std::numeric_limits<IdType>::max();
    static constexpr char separator = ':';

    IdType runId{Undef_id}, lumiBlock{Undef_id}, eventId{Undef_id}, sampleId{Undef_id};

    static const EventIdentifier& Undef_event() {
        static const EventIdentifier undef_event;
        return undef_event;
    }

    static const std::vector<std::string>& Names() {
        static const std::vector<std::string> names = { "run", "lumi", "evt", "sampleId" };
        return names;
    }

    static const std::string& LegendString(size_t n) {
        static std::map<size_t, std::string> legends;
        if(!legends.count(n)) {
            std::ostringstream ss;
            for(size_t k = 0; k < n; ++k)
                ss << Names().at(k) << separator;
            std::string str = ss.str();
            if(str.size())
                str.erase(str.size() - 1);
            legends[n] = str;
        }
        return legends.at(n);
    }

    EventIdentifier() {}

    EventIdentifier(IdType _runId, IdType _lumiBlock, IdType _eventId, IdType _sampleId = Undef_id) :
        runId(_runId), lumiBlock(_lumiBlock), eventId(_eventId), sampleId(_sampleId) {}

    explicit EventIdentifier(const std::string& id_str)
    {
        const std::vector<std::string> id_strings = Split(id_str);

        runId = Parse(id_strings.at(0), "runId");
        lumiBlock = Parse(id_strings.at(1), "lumiBlock");
        eventId = Parse(id_strings.at(2), "eventId");
        if(id_strings.size() > 3)
            sampleId = Parse(id_strings.at(3), "sampleId");
    }

    template<typename Event>
    explicit EventIdentifier(const Event& event) :
        runId(event.run), lumiBlock(event.lumi), eventId(event.evt) {}

    bool operator == (const EventIdentifier& other) const { return !(*this != other); }

    bool operator != (const EventIdentifier& other) const
    {
        return runId != other.runId || lumiBlock != other.lumiBlock || eventId != other.eventId
                || sampleId != other.sampleId;
    }

    bool operator < (const EventIdentifier& other) const
    {
        if(sampleId != other.sampleId) return sampleId < other.sampleId;
        if(runId != other.runId) return runId < other.runId;
        if(lumiBlock != other.lumiBlock) return lumiBlock < other.lumiBlock;
        return eventId < other.eventId;
    }

    std::string ToString() const
    {
        std::ostringstream ss;
        ss << runId << separator << lumiBlock << separator << eventId;
        if(sampleId != Undef_id)
            ss << separator << sampleId;
        return ss.str();
    }

    const std::string& GetLegendString() const { return sampleId == Undef_id ? LegendString(3) : LegendString(4); }

    static std::vector<std::string> Split(const std::string& id_str)
    {
        static const std::string separators(1, separator);
        std::vector<std::string> id_strings;
        boost::split(id_strings, id_str, boost::is_any_of(separators), boost::token_compress_on);
        if(id_strings.size() < 3 || id_strings.size() > 4)
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

inline std::ostream& operator<<(std::ostream& s, const EventIdentifier& event)
{
    s << event.ToString();
    return s;
}

inline std::istream& operator>>(std::istream& s, EventIdentifier& event)
{
    std::string str;
    s >> str;
    event = EventIdentifier(str);
    return s;
}

} // namespace analysis
