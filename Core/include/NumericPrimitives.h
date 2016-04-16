/*! Definition of the primitives that extend CERN ROOT functionality.
This file is part of https://github.com/hh-italian-group/AnalysisTools. */

#pragma once

#include <boost/math/constants/constants.hpp>
#include <TF1.h>
#include "exception.h"

namespace analysis {

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
    } catch(exception&) {}
    return false;
}

template<typename T>
T Parse(const std::string& str)
{
    T t;
    std::istringstream ss(str);
    ss >> t;
    if(ss.fail())
        throw exception("Parse of string '%1%' to %2% is failed.") % str % typeid(T).name();
    return t;
}

template<typename T>
struct Range {
    Range() : _min(0), _max(0) {}
    Range(const T& min, const T& max) : _min(min), _max(max)
    {
        if(!IsValid(min, max))
            throw exception("Invalid range [%1%, %2%].") % min % max;
    }

    const T& min() const { return _min; }
    const T& max() const { return _max; }
    T size() const { return max() - min(); }
    bool Contains(const T& v) const { return v >= min() && v <= max(); }
    static bool IsValid(const T& min, const T& max) { return min <= max; }

    Range<T> Extend(const T& v) const
    {
        if(Contains(v))
            return *this;
        return Range<T>(std::min(min(), v), std::max(max(), v));
    }

    bool Includes(const Range<T>& other) const { return min() <= other.min() && max() >= other.max(); }
    bool Overlaps(const Range<T>& other) const { return min() <= other.max() && other.min() <= max();  }
    Range<T> Combine(const Range<T>& other) const
    {
        if(!Overlaps(other))
            throw exception("Unable to combine non overlapping ranges.");
        return Range<T>(std::min(min(), other.min()), std::max(max(), other.max()));
    }

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
        throw exception("Invalid range.");
    r = Range<T>(min, max);
    return s;
}

template<typename T>
struct RelativeRange {
    RelativeRange() : _down(0), _up(0) {}
    RelativeRange(const T& down, const T& up) : _down(down), _up(up)
    {
        if(!IsValid(down, up))
            throw exception("Invalid relative range [%1%, %2%].") % down % up;
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
        throw exception("Invalid relative range.");
    r = RelativeRange<T>(down, up);
    return s;
}

template<unsigned n_pi_per_period_num, unsigned n_pi_per_period_denom = 1>
struct Angle {
    enum class Interval { Symmetric, Positive };
    static constexpr double Pi() { return boost::math::constants::pi<double>(); }
    static constexpr double NumberOfPiPerPeriod() { return double(n_pi_per_period_num) / n_pi_per_period_denom; }
    static constexpr double FullPeriod() { return n_pi_per_period_num * Pi() / n_pi_per_period_denom; }
    static constexpr double HalfPeriod() { return FullPeriod() / 2; }

    Angle() : _value(0), _interval(Interval::Symmetric) {}
    Angle(double value, Interval interval = Interval::Symmetric)
        : _value(AdjustValue(value, interval)), _interval(interval) {}

    double value() const { return _value; }
    Interval interval() const { return _interval; }

    Angle operator+(const Angle& other) const { return Angle(value() + other.value(), interval()); }
    Angle operator-(const Angle& other) const { return Angle(value() - other.value(), interval()); }

    static const Range<double>& AngleValuesRange(Interval interval)
    {
        static const std::map<Interval, Range<double>> range_map = {
            { Interval::Symmetric, { -HalfPeriod(), HalfPeriod() } },
            { Interval::Positive, { 0, FullPeriod() } }
        };
        return range_map.at(interval);
    }

    static double AdjustValue(double value, Interval interval)
    {
        const Range<double>& range = AngleValuesRange(interval);
        value -= FullPeriod() * std::floor(value/FullPeriod());
        while(value < range.min() || value >= range.max())
            value += value < range.min() ? FullPeriod() : -FullPeriod();
        return value;
    }

private:
    double _value;
    Interval _interval;
};

template<unsigned n_pi_per_period_num, unsigned n_pi_per_period_denom>
std::ostream& operator<<(std::ostream& s, const Angle<n_pi_per_period_num, n_pi_per_period_denom>& a)
{
    s << a.value();
    return s;
}

template<unsigned n_pi_per_period_num, unsigned n_pi_per_period_denom>
std::istream& operator>>(std::istream& s, const Angle<n_pi_per_period_num, n_pi_per_period_denom>& a)
{
    double value;
    s >> value;
    a = Angle<n_pi_per_period_num, n_pi_per_period_denom>(value);
    return s;
}

template<unsigned n_pi_per_period_num, unsigned n_pi_per_period_denom>
struct Range<Angle<n_pi_per_period_num, n_pi_per_period_denom>> {
    using A = Angle<n_pi_per_period_num, n_pi_per_period_denom>;

    Range() : _min(0), _max(0) {}
    Range(const A& min, const A& max) : _min(min), _max(max.value(), min.interval()) {}

    const A& min() const { return _min; }
    const A& max() const { return _max; }
    A size() const { return A(_max.value() - _min.value(), A::Interval::Positive); }
    Range<double> ToValueRange() const
    {
        const double min_value = min().value();
        double max_value = max().value();
        if(max_value < min_value)
            max_value += A::FullPeriod();
        return Range<double>(min_value, max_value);
    }

    bool Contains(const A& a) const
    {
        const Range<double> min_a_value_range = Range<A>(min(), a).ToValueRange();
        return ToValueRange().Contains(min_a_value_range.max());
    }

    static bool IsValid(const A& min, const A& max) { return true; }

    Range<A> Extend(const A& a) const
    {
        if(Contains(a))
            return *this;
        const A a_fixed(a.value(), min().interval());
        const Range<A> extend_min(a_fixed, max()), extend_max(min(), a_fixed);
        return extend_max.size().value() < extend_min.size().value() ? extend_max : extend_min;
    }

    bool Includes(const Range<A>& other) const
    {
        return Contains(other.min()) && Contains(other.max());
    }

    bool Overlaps(const Range<A>& other) const
    {
        return Contains(other.min()) || Contains(other.max()) || other.Contains(min());
    }

    Range<A> Combine(const Range<A>& other) const
    {
        if(!Overlaps(other))
            throw exception("Unable to combine non overlapping ranges.");
        if(Includes(other))
            return *this;
        if(other.Includes(*this))
            return other;
        if(Contains(other.min()))
            return Range<A>(min(), other.max());
        return Range<A>(other.min(), max());
    }

private:
    A _min, _max;
};

struct NumericalExpression {
    NumericalExpression() : _value(0) {}
    NumericalExpression(const std::string& expression)
        : _expression(expression)
    {
        const std::string formula = boost::str(boost::format("x*(%1%)") % expression);
        TF1 fn("", formula.c_str(), 0, 1);
//        if(!fn.IsValid())
//            throw exception("Invalid numerical expression '%1%'") % expression;
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

} // namespace analysis
