/*! Definition of the primitives that extend CERN ROOT functionality.
This file is part of https://github.com/hh-italian-group/AnalysisTools. */

#pragma once

#include <type_traits>
#include <boost/math/constants/constants.hpp>
#include <TF1.h>
#include "TextIO.h"

namespace analysis {

namespace detail {
    template<typename T, bool fundamental = std::is_fundamental<T>::value>
    struct ConstRefType;

    template<typename T>
    struct ConstRefType<T, true> { using Type = T; };

    template<typename T>
    struct ConstRefType<T, false> { using Type = const T&; };
}

template<typename T>
struct Range {
    using ValueType = T;
    using ConstRefType = typename detail::ConstRefType<T>::Type;
    Range() : _min(0), _max(0) {}
    Range(ConstRefType min, ConstRefType max) : _min(min), _max(max)
    {
        if(!IsValid(min, max))
            throw exception("Invalid range [%1%, %2%].") % min % max;
    }
    virtual ~Range() {}

    ConstRefType min() const { return _min; }
    ConstRefType max() const { return _max; }
    T size() const { return max() - min(); }
    bool Contains(ConstRefType v) const { return v >= min() && v <= max(); }
    static bool IsValid(ConstRefType min, ConstRefType max) { return min <= max; }

    Range<T> Extend(ConstRefType v) const
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

    std::string ToString(char sep = ' ') const
    {
        std::ostringstream ss;
        ss << min() << sep << max();
        return ss.str();
    }

    static Range<T> Parse(const std::string& str, const std::string& separators=": \n")
    {
        const auto values = SplitValueList(str, true, separators, true);
        if(values.size() != 2)
            throw exception("Invalid range '%1%'.") % str;
        return Make(values);
    }

    static Range<T> Read(std::istream& stream, const std::string& separators=": \n")
    {
        const auto values = ReadValueList(stream, 2, true, separators, true);
        return Make(values);
    }

private:
    static Range<T> Make(const std::vector<std::string>& values)
    {
        const T min = ::analysis::Parse<T>(values.at(0));
        const T max = ::analysis::Parse<T>(values.at(1));
        return Range<T>(min, max);
    }

private:
    T _min, _max;
};

template<typename T>
std::ostream& operator<<(std::ostream& s, const Range<T>& r)
{
    s << r.ToString(':');
    return s;
}

template<typename T>
std::istream& operator>>(std::istream& s, Range<T>& r)
{
    r = Range<T>::Read(s);
    return s;
}

template<typename T>
struct RelativeRange {
    using ValueType = T;
    using ConstRefType = typename detail::ConstRefType<T>::Type;
    RelativeRange() : _down(0), _up(0) {}
    RelativeRange(ConstRefType down, ConstRefType up) : _down(down), _up(up)
    {
        if(!IsValid(down, up))
            throw exception("Invalid relative range [%1%, %2%].") % down % up;
    }

    ConstRefType down() const { return _down; }
    ConstRefType up() const { return _up; }
    Range<T> ToAbsoluteRange(ConstRefType v) const { return Range<T>(v + down(), v + up()); }
    static bool IsValid(ConstRefType down, ConstRefType up) { return down <= 0 && up >= 0; }

    std::string ToString(char sep = ' ') const
    {
        std::ostringstream ss;
        ss << down() << sep << up();
        return ss.str();
    }

    static RelativeRange<T> Parse(const std::string& str, const std::string& separators=": \n")
    {
        const auto values = SplitValueList(str, true, separators, true);
        if(values.size() != 2)
            throw exception("Invalid relative range '%1%'.") % str;
        return Make(values);
    }

    static RelativeRange<T> Read(std::istream& stream, const std::string& separators=": \n")
    {
        const auto values = ReadValueList(stream, 2, true, separators, true);
        return Make(values);
    }

private:
    static RelativeRange<T> Make(const std::vector<std::string>& values)
    {
        const T down = ::analysis::Parse<T>(values.at(0));
        const T up = ::analysis::Parse<T>(values.at(1));
        return RelativeRange<T>(down, up);
    }

private:
    T _down, _up;
};

template<typename T>
std::ostream& operator<<(std::ostream& s, const RelativeRange<T>& r)
{
    s << r.ToString(':');
    return s;
}

template<typename T>
std::istream& operator>>(std::istream& s, RelativeRange<T>& r)
{
    r = RelativeRange<T>::Read(s);
    return s;
}

template<typename T>
struct RangeWithStep : public Range<T> {
    using ValueType = typename Range<T>::ValueType;
    using ConstRefType = typename Range<T>::ConstRefType;

    struct iterator {
        iterator(const RangeWithStep<T>& _range, size_t _pos) : range(&_range), pos(_pos) {}
        iterator& operator++() { ++pos; return *this; }
        iterator operator++(int) { iterator iter(*this); operator++(); return iter; }
        bool operator==(const iterator& other) { return range == other.range && pos == other.pos;}
        bool operator!=(const iterator& other) { return !(*this == other); }
        T operator*() { return range->grid_point_value(pos); }
    private:
        const RangeWithStep<T> *range;
        size_t pos;
    };

    RangeWithStep() : _step(0) {}
    RangeWithStep(ConstRefType min, ConstRefType max, ConstRefType step) : Range<T>(min, max), _step(step) {}

    ConstRefType step() const { return _step; }
    T grid_point_value(size_t index) const { return this->min() + index * step(); }
    size_t n_grid_points() const
    {
        size_t n_points = (this->max() - this->min()) / step();
        if(this->Contains(grid_point_value(n_points)))
            ++n_points;
        return n_points;
    }

    iterator begin() const { return iterator(*this, 0); }
    iterator end() const { return iterator(*this, n_grid_points()); }

    std::string ToString(char sep = ' ') const
    {
        std::ostringstream ss;
        ss << this->min() << sep << this->max() << sep << step();
        return ss.str();
    }

    static RangeWithStep<T> Parse(const std::string& str, const std::string& separators=": \n")
    {
        const auto values = SplitValueList(str, true, separators, true);
        if(values.size() != 3)
            throw exception("Invalid range with step '%1%'.") % str;
        return Make(values);
    }

    static RangeWithStep<T> Read(std::istream& stream, const std::string& separators=": \n")
    {
        const auto values = ReadValueList(stream, 3, true, separators, true);
        return Make(values);
    }

private:
    static RangeWithStep<T> Make(const std::vector<std::string>& values)
    {
        const T min = ::analysis::Parse<T>(values.at(0));
        const T max = ::analysis::Parse<T>(values.at(1));
        const T step = ::analysis::Parse<T>(values.at(2));
        return RangeWithStep<T>(min, max, step);
    }

private:
    T _step;
};

template<typename T>
std::ostream& operator<<(std::ostream& s, const RangeWithStep<T>& r)
{
    s << r.ToString(':');
    return s;
}

template<typename T>
std::istream& operator>>(std::istream& s, RangeWithStep<T>& r)
{
    r = RangeWithStep<T>::Read(s);
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
    using ValueType = A;

    Range() : _min(0), _max(0) {}
    Range(const A& min, const A& max) : _min(min), _max(max.value(), min.interval()) {}
    virtual ~Range() {}

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
    std::string ToString(char sep = ' ') const
    {
        std::ostringstream ss;
        ss << min() << sep << max();
        return ss.str();
    }

    static Range<A> Parse(const std::string& str, const std::string& separators=": \n")
    {
        const auto values = SplitValueList(str, true, separators, true);
        if(values.size() != 2)
            throw exception("Invalid angle range '%1%'.") % str;
        return Make(values);
    }

    static Range<A> Read(std::istream& stream, const std::string& separators=": \n")
    {
        const auto values = ReadValueList(stream, 2, true, separators, true);
        return Make(values);
    }

private:
    static Range<A> Make(const std::vector<std::string>& values)
    {
        const A min = ::analysis::Parse<A>(values.at(0));
        const A max = ::analysis::Parse<A>(values.at(1));
        return Range<A>(min, max);
    }

private:
    A _min, _max;
};

template<typename Range>
struct RangeMultiD {
public:
    using ValueType = typename Range::ValueType;
    explicit RangeMultiD(size_t n_dim) : ranges(n_dim) {}
    explicit RangeMultiD(const std::vector<Range>& _ranges) : ranges(_ranges) {}

    size_t GetNumberOfDimensions() const { return ranges.size(); }
    const Range& GetRange(size_t dim_id) const { Check(dim_id); return ranges.at(dim_id - 1); }
    Range& GetRange(size_t dim_id) { Check(dim_id); return ranges.at(dim_id - 1); }

    bool Contains(const std::vector<ValueType>& point) const
    {
        if(point.size() != GetNumberOfDimensions())
            throw exception("Invalid number of dimensions.");
        for(size_t n = 0; n < ranges.size(); ++n)
            if(!ranges.at(n).Contains(point.at(n))) return false;
        return true;
    }

private:
    void Check(size_t dim_id) const
    {
        if(!dim_id || dim_id > GetNumberOfDimensions())
            throw exception("Wrong dimension id = %1%") % dim_id;
    }

private:
    std::vector<Range> ranges;
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

inline std::ostream& operator<<(std::ostream& s, const NumericalExpression& e)
{
    s << e.expression();
    return s;
}

inline std::istream& operator>>(std::istream& s, NumericalExpression& e)
{
    std::string line;
    std::getline(s, line);
    e = NumericalExpression(line);
    return s;
}

} // namespace analysis
