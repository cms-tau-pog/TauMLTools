/*! Definition of CMS event identifier.
This file is part of https://github.com/hh-italian-group/TauMLTools. */

#include "TauMLTools/Core/interface/EventIdentifier.h"

#include <sstream>
#include <map>
#include <boost/algorithm/string.hpp>
#include "TauMLTools/Core/interface/exception.h"

namespace analysis {

const EventIdentifier& EventIdentifier::Undef_event()
{
    static const EventIdentifier undef_event;
    return undef_event;
}

const std::vector<std::string>& EventIdentifier::Names()
{
    static const std::vector<std::string> names = { "run", "lumi", "evt", "sampleId" };
    return names;
}

const std::string& EventIdentifier::LegendString(size_t n)
{
    static const auto init_legends = []() {
        std::map<size_t, std::string> map;
        std::ostringstream ss;
        for(size_t k = 0; k < Names().size(); ++k) {
            ss << Names().at(k);
            map[k] = ss.str();
            ss << separator;
        }
        return map;
    };
    static const std::map<size_t, std::string> legends = init_legends();
    return legends.at(n);
}

EventIdentifier::EventIdentifier(IdType _runId, IdType _lumiBlock, IdType _eventId, IdType _sampleId) :
    runId(_runId), lumiBlock(_lumiBlock), eventId(_eventId), sampleId(_sampleId) {}

EventIdentifier::EventIdentifier(const std::string& id_str)
{
    const std::vector<std::string> id_strings = Split(id_str);

    runId = Parse(id_strings.at(0), "runId");
    lumiBlock = Parse(id_strings.at(1), "lumiBlock");
    eventId = Parse(id_strings.at(2), "eventId");
    if(id_strings.size() > 3)
        sampleId = Parse(id_strings.at(3), "sampleId");
}

bool EventIdentifier::operator == (const EventIdentifier& other) const { return !(*this != other); }

bool EventIdentifier::operator != (const EventIdentifier& other) const
{
    return runId != other.runId || lumiBlock != other.lumiBlock || eventId != other.eventId
            || sampleId != other.sampleId;
}

bool EventIdentifier::operator < (const EventIdentifier& other) const
{
    if(sampleId != other.sampleId) return sampleId < other.sampleId;
    if(runId != other.runId) return runId < other.runId;
    if(lumiBlock != other.lumiBlock) return lumiBlock < other.lumiBlock;
    return eventId < other.eventId;
}

std::string EventIdentifier::ToString() const
{
    std::ostringstream ss;
    ss << runId << separator << lumiBlock << separator << eventId;
    if(sampleId != Undef_id)
        ss << separator << sampleId;
    return ss.str();
}

const std::string& EventIdentifier::GetLegendString() const
{
    return sampleId == Undef_id ? LegendString(3) : LegendString(4);
}

std::vector<std::string> EventIdentifier::Split(const std::string& id_str)
{
    static const std::string separators(1, separator);
    std::vector<std::string> id_strings;
    boost::split(id_strings, id_str, boost::is_any_of(separators), boost::token_compress_on);
    if(id_strings.size() < 3 || id_strings.size() > 4)
        throw exception("Invalid event identifier '%1%'") % id_str;
    return id_strings;
}

EventIdentifier::IdType EventIdentifier::Parse(const std::string& str, const std::string& name)
{
    std::istringstream ss(str);
    IdType id;
    ss >> id;
    if(ss.fail())
        throw exception("Invalid %1% = '%2%'.") % name % str;
    return id;
}

std::ostream& operator<<(std::ostream& s, const EventIdentifier& event)
{
    s << event.ToString();
    return s;
}

std::istream& operator>>(std::istream& s, EventIdentifier& event)
{
    std::string str;
    s >> str;
    event = EventIdentifier(str);
    return s;
}

} // namespace analysis
