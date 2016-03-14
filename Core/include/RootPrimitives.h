/*! Definition of the primitives that extend CERN ROOT functionality.
This file is part of https://github.com/hh-italian-group/AnalysisTools. */

#pragma once

#include<TF1.h>
#include "exception.h"

namespace root_ext {

template<typename T>
std::string ToString(const T& t)
{
    std::ostringstream ss;
    ss << t;
    return ss.str();
}

template<typename T>
bool TryParse(const std::string& str, T& t)
{
    try {
        std::stringstream ss(str);
        ss >> t;
        return !ss.fail();
    } catch(analysis::exception&) {}
    return false;
}

template<typename T>
T Parse(const std::string& str)
{
    T t;
    std::istringstream ss(str);
    ss >> t;
    if(ss.fail())
        throw analysis::exception("Parse of string '%1%' to %2% is failed.") % str % typeid(T).name();
    return t;
}

template<typename T>
struct Range {
    Range() : _min(0), _max(0) {}
    Range(const T& min, const T& max) : _min(min), _max(max)
    {
        if(!IsValid(min, max))
            throw analysis::exception("Invalid range [%1%, %2%].") % min % max;
    }

    const T& min() const { return _min; }
    const T& max() const { return _max; }
    bool Contains(const T& v) const { return v >= min() && v <= max(); }
    static bool IsValid(const T& min, const T& max) { return min <= max; }

private:
    T _min, _max;
};

template<typename T>
std::ostream& operator<<(std::ostream& s, const Range<T>& r)
{
    s << boost::format("%1% %2%") % r.min() % r.max();
    return s;
}

template<typename T>
std::istream& operator>>(std::istream& s, Range<T>& r)
{
    T min, max;
    s >> min >> max;
    if(s.fail())
        throw analysis::exception("Invalid range.");
    r = Range<T>(min, max);
    return s;
}

template<typename T>
struct RelativeRange {

    RelativeRange() : _down(0), _up(0) {}
    RelativeRange(const T& down, const T& up) : _down(down), _up(up)
    {
        if(!IsValid(down, up))
            throw analysis::exception("Invalid relative range [%1%, %2%].") % down % up;
    }

    const T& down() const { return _down; }
    const T& up() const { return _up; }
    Range<T> ToAbsoluteRange(const T& v) const { return Range<T>(v + down(), v + up()); }
    static bool IsValid(const T& down, const T& up) { return down <= 0 && up >= 0; }

private:
    T _down, _up;
};

template<typename T>
std::ostream& operator<<(std::ostream& s, const RelativeRange<T>& r)
{
    s << boost::format("%1% %2%") % r.down() % r.up();
    return s;
}

template<typename T>
std::istream& operator>>(std::istream& s, RelativeRange<T>& r)
{
    T down, up;
    s >> down >> up;
    if(s.fail())
        throw analysis::exception("Invalid relative range.");
    r = RelativeRange<T>(down, up);
    return s;
}

struct NumericalExpression {
    NumericalExpression() : _value(0) {}
    NumericalExpression(const std::string& expression)
        : _expression(expression)
    {
        const std::string formula = boost::str(boost::format("x*(%1%)") % expression);
        TF1 fn("", formula.c_str(), 0, 1);
//        if(!fn.IsValid())
//            throw analysis::exception("Invalid numerical expression '%1%'") % expression;
        _value = fn.Eval(1);
    }

    const std::string& expression() const { return _expression; }
    double value() const { return _value; }
    operator double() const { return _value; }

private:
    std::string _expression;
    double _value;
};

std::ostream& operator<<(std::ostream& s, const NumericalExpression& e)
{
    s << e.expression();
    return s;
}

std::istream& operator>>(std::istream& s, NumericalExpression& e)
{
    std::string line;
    std::getline(s, line);
    e = NumericalExpression(line);
    return s;
}

} // namespace root_ext
