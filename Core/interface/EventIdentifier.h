/*! Definition of CMS event identifier.
This file is part of https://github.com/hh-italian-group/TauMLTools. */

#pragma once

#include <map>
#include "exception.h"

namespace analysis {

struct EventIdentifier {
    using IdType = unsigned long long;
    static constexpr IdType Undef_id = std::numeric_limits<IdType>::max();
    static constexpr char separator = ':';

    IdType runId{Undef_id}, lumiBlock{Undef_id}, eventId{Undef_id}, sampleId{Undef_id};

    static const EventIdentifier& Undef_event();
    static const std::vector<std::string>& Names();
    static const std::string& LegendString(size_t n);

    EventIdentifier() {}
    EventIdentifier(IdType _runId, IdType _lumiBlock, IdType _eventId, IdType _sampleId = Undef_id);

    explicit EventIdentifier(const std::string& id_str);

    template<typename Event>
    explicit EventIdentifier(const Event& event) :
        runId(event.run), lumiBlock(event.lumi), eventId(event.evt) {}

    bool operator == (const EventIdentifier& other) const;
    bool operator != (const EventIdentifier& other) const;
    bool operator < (const EventIdentifier& other) const;

    std::string ToString() const;
    const std::string& GetLegendString() const;
    static std::vector<std::string> Split(const std::string& id_str);

private:
    static IdType Parse(const std::string& str, const std::string& name);
};

std::ostream& operator<<(std::ostream& s, const EventIdentifier& event);
std::istream& operator>>(std::istream& s, EventIdentifier& event);

} // namespace analysis
