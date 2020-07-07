/*! Definition of the primitives that extend CERN ROOT functionality.
This file is part of https://github.com/hh-italian-group/TauMLTools. */

#pragma once

#include <type_traits>
#include <boost/math/constants/constants.hpp>
#include <TF1.h>
#include "TextIO.h"

namespace analysis {

enum class RangeBoundaries { Open, MinIncluded, MaxIncluded, BothIncluded };

namespace detail {
template<typename T, bool is_integral = std::is_integral<T>::value>
struct RangeSize {
    static T size(const T& min, const T& max, RangeBoundaries) { return max - min; }
};

template<typename T>
struct RangeSize<T, true> {
    static T size(T min, T max, RangeBoundaries boundaries)
    {
        T delta = max - min;
        if(boundaries == RangeBoundaries::BothIncluded)
            delta += T(1);
        else if(boundaries == RangeBoundaries::Open && delta != T(0))
            delta -= T(1);
        return delta;
    }
};

template<typename T, bool is_unsigned = std::is_unsigned<T>::value>
struct Abs {
    static T abs(T value) { return std::abs(value); }
};

template<typename T>
struct Abs<T, true> {
    static T abs(T value) { return value; }
};

template<typename T>
inline T FloatRound(T x, T /*ref*/)
{
    return x;
}

template<>
inline float FloatRound<float>(float x, float ref)
{
    static constexpr float precision = 1e-6f;
    if(ref == 0) return x;
    return std::round(x / ref / precision) * precision * ref;
}

template<>
inline double FloatRound<double>(double x, double ref)
{
    static constexpr double precision = 1e-8;
    if(ref == 0) return x;
    return std::round(x / ref / precision) * precision * ref;
}


template<typename T>
inline size_t GetNumberOfGridPoints(T min, T max, T step)
{
    return static_cast<size_t>((max - min) / step);
}

template<>
inline size_t GetNumberOfGridPoints<float>(float min, float max, float step)
{
    static constexpr float precision = 1e-6f;
    return static_cast<size_t>(std::round((max - min) / step / precision) * precision);
}

template<>
inline size_t GetNumberOfGridPoints<double>(double min, double max, double step)
{
    static constexpr double precision = 1e-8;
    return static_cast<size_t>(std::round((max - min) / step / precision) * precision);
}

} // namespace detail

template<typename T>
struct Range {
    using ValueType = T;
    using ConstRefType =
        typename std::conditional<std::is_fundamental<ValueType>::value, ValueType, const ValueType&>::type;

    static const std::pair<char, char> GetBoundariesSymbols(RangeBoundaries b)
    {
        static const std::map<RangeBoundaries, std::pair<char, char>> symbols = {
            { RangeBoundaries::Open, { '(', ')' } },
            { RangeBoundaries::MinIncluded, { '[', ')' } },
            { RangeBoundaries::MaxIncluded, { '(', ']' } },
            { RangeBoundaries::BothIncluded, { '[', ']' } },
        };
        return symbols.at(b);
    }

    static RangeBoundaries CreateBoundaries(bool include_min, bool include_max)
    {
        if(include_min && !include_max)
            return RangeBoundaries::MinIncluded;
        if(!include_min && include_max)
            return RangeBoundaries::MaxIncluded;
        if(include_min && include_max)
            return RangeBoundaries::BothIncluded;
        return RangeBoundaries::Open;
    }

    Range() : _min(0), _max(0) {}
    Range(ConstRefType min, ConstRefType max, RangeBoundaries boundaries = RangeBoundaries::BothIncluded) :
        _min(min), _max(max), _boundaries(boundaries)
    {
        if(!IsValid(min, max))
            throw exception("Invalid range [%1%, %2%].") % min % max;
    }
    Range(const Range<T>& other) : _min(other._min), _max(other._max), _boundaries(other._boundaries) {}
    Range(const Range<T>& other, RangeBoundaries boundaries) :
        _min(other._min), _max(other._max), _boundaries(boundaries) {}
    virtual ~Range() {}
    Range<T>& operator=(const Range<T>& other)
    {
        _min = other._min;
        _max = other._max;
        _boundaries = other._boundaries;
        return *this;
    }

    ConstRefType min() const { return _min; }
    ConstRefType max() const { return _max; }
    T size() const { return detail::RangeSize<T>::size(min(), max(), boundaries()); }

    RangeBoundaries boundaries() const { return _boundaries; }
    bool min_included() const
    {
        return boundaries() == RangeBoundaries::MinIncluded || boundaries() == RangeBoundaries::BothIncluded;
    }
    bool max_included() const
    {
        return boundaries() == RangeBoundaries::MaxIncluded || boundaries() == RangeBoundaries::BothIncluded;
    }

    bool Contains(ConstRefType v) const
    {
        if(min() == max())
            return (min_included() || max_included()) && v == min();
        const bool min_cond = (min_included() && v >= min()) || v > min();
        const bool max_cond = (max_included() && v <= max()) || v < max();
        return min_cond && max_cond;
    }
    static bool IsValid(ConstRefType min, ConstRefType max) { return min <= max; }

    Range<T> Extend(ConstRefType v, bool include = true) const
    {
        if(Contains(v))
            return *this;
        const auto new_min = std::min(min(), v);
        const auto new_max = std::max(max(), v);
        RangeBoundaries b;
        if(new_min == v) {
            if(include)
                b = max_included() ? RangeBoundaries::BothIncluded : RangeBoundaries::MinIncluded;
            else
                b = max_included() ? RangeBoundaries::MaxIncluded : RangeBoundaries::Open;
        } else {
            if(include)
                b = min_included() ? RangeBoundaries::BothIncluded : RangeBoundaries::MaxIncluded;
            else
                b = min_included() ? RangeBoundaries::MinIncluded : RangeBoundaries::Open;
        }
        return Range<T>(new_min, new_max, b);
    }

    bool operator ==(const Range<T>& other) const
    {
        return min() == other.min() && max() == other.max() && boundaries() == other.boundaries();
    }
    bool operator !=(const Range<T>& other) const { return !(*this == other); }

    bool Includes(const Range<T>& other) const
    {
        const bool min_cond = min() == other.min() ? min_included() || !other.min_included() : min() < other.min();
        const bool max_cond = max() == other.max() ? max_included() || !other.max_included() : max() > other.max();
        return min_cond && max_cond;
    }
    bool Overlaps(const Range<T>& other) const
    {
        if(min() == max())
            return (min_included() || max_included()) && other.Contains(min());
        const bool min_cond = min() == other.max() ? min_included() && other.max_included() : min() < other.max();
        const bool max_cond = max() == other.min() ? max_included() && other.min_included() : max() > other.min();
        return min_cond && max_cond;
    }
    Range<T> Combine(const Range<T>& other) const
    {
        if(!Overlaps(other))
            throw exception("Unable to combine non overlapping ranges.");
        const auto new_min = std::min(min(), other.min());
        const auto new_max = std::max(max(), other.max());
        const bool include_min = (new_min == min() && min_included())
                || (new_min == other.min() && other.min_included());
        const bool include_max = (new_max == max() && max_included())
                || (new_max == other.max() && other.max_included());
        const RangeBoundaries b = CreateBoundaries(include_min, include_max);
        return Range<T>(new_min, new_max, b);
    }

    std::string ToString(char sep = ':') const
    {
        std::ostringstream ss;
        const auto b_sym = GetBoundariesSymbols(boundaries());
        if(boundaries() != RangeBoundaries::BothIncluded)
            ss << b_sym.first;
        ss << min() << sep << max();
        if(boundaries() != RangeBoundaries::BothIncluded)
            ss << b_sym.second;
        return ss.str();
    }

    static Range<T> Parse(const std::string& str, const std::string& separators=": \t")
    {
        const auto values = SplitValueList(str, true, separators, true);
        if(values.size() != 2)
            throw exception("Invalid range '%1%'.") % str;
        return Make(values);
    }

    static Range<T> Read(std::istream& stream, const std::string& separators=": \t")
    {
        const auto values = ReadValueList(stream, 2, true, separators, true);
        return Make(values);
    }

private:
    static Range<T> Make(const std::vector<std::string>& values)
    {
        static const auto opened_b_symbols = GetBoundariesSymbols(RangeBoundaries::Open);
        static const auto closed_b_symbols = GetBoundariesSymbols(RangeBoundaries::BothIncluded);
        bool include_min = true, include_max = true;
        std::string min_str = values.at(0), max_str = values.at(1);
        if(min_str.size() && (min_str.front() == opened_b_symbols.first || min_str.front() == closed_b_symbols.first)) {
            include_min = min_str.front() == closed_b_symbols.first;
            min_str.erase(0, 1);
        }
        if(max_str.size() && (max_str.back() == opened_b_symbols.second || max_str.back() == closed_b_symbols.second)) {
            include_max = max_str.back() == closed_b_symbols.second;
            max_str.erase(max_str.size() - 1, 1);
        }
        const T min = ::analysis::Parse<T>(min_str);
        const T max = ::analysis::Parse<T>(max_str);
        const RangeBoundaries b = CreateBoundaries(include_min, include_max);
        return Range<T>(min, max, b);
    }

private:
    T _min, _max;
    RangeBoundaries _boundaries;
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
    using ConstRefType = typename Range<T>::ConstRefType;
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

    static RelativeRange<T> Parse(const std::string& str, const std::string& separators=": \t")
    {
        const auto values = SplitValueList(str, true, separators, true);
        if(values.size() != 2)
            throw exception("Invalid relative range '%1%'.") % str;
        return Make(values);
    }

    static RelativeRange<T> Read(std::istream& stream, const std::string& separators=": \t")
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


    enum class PrintMode { Step = 0, NGridPoints = 1, NBins = 2 };
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
    RangeWithStep(ConstRefType min, ConstRefType max, ConstRefType step) :
        Range<T>(min, max, RangeBoundaries::BothIncluded), _step(step)
    {
    }

    ConstRefType step() const { return _step; }
    T grid_point_value(size_t index) const
    {
        const T ref = std::max<T>(detail::Abs<T>::abs(this->min()), detail::Abs<T>::abs(this->max()));
        return detail::FloatRound<T>(this->min() + T(index) * step(), ref);
    }
    size_t n_grid_points() const
    {
        if(this->max() == this->min()) return 1;
        if(step() == 0)
            throw exception("Number of grid points is not defined for a non-point range with the step = 0.");
        size_t n_points = detail::GetNumberOfGridPoints(this->min(), this->max(), step());
        if(this->Contains(grid_point_value(n_points)))
            ++n_points;
        return n_points;
    }
    size_t n_bins() const { return n_grid_points() - 1; }

    size_t find_bin(T value) const
    {
        if(!this->Contains(value))
            throw exception("find_bin: value is out of range.");
        if(n_bins() == 0)
            throw exception("find_bin: number of bins is 0.");
        size_t bin_id = static_cast<size_t>((value - this->min()) / step());
        if(bin_id == n_bins())
            --bin_id;
        return bin_id;
    }

    iterator begin() const { return iterator(*this, 0); }
    iterator end() const { return iterator(*this, n_grid_points()); }

    std::string ToString(PrintMode mode = PrintMode::Step) const
    {
        std::ostringstream ss;
        ss << this->min() << Separators().at(0) << this->max() << Separators().at(static_cast<size_t>(mode));
        if(mode == PrintMode::Step)
           ss << step();
        else if(mode == PrintMode::NGridPoints)
            ss << n_grid_points();
        else if(mode == PrintMode::NBins)
            ss << n_bins();
        else
            throw exception("Unsupported RangeWithStep::PrintMode = %1%.") % static_cast<size_t>(mode);
        return ss.str();
    }

    static RangeWithStep<T> Parse(const std::string& str)
    {
        const size_t first_split_pos = str.find_first_of(Separators());
        if(first_split_pos != std::string::npos) {
            const size_t last_split_pos = str.find_first_of(Separators(), first_split_pos + 1);
            if(last_split_pos != std::string::npos) {
                const size_t end_split_pos = str.find_last_of(Separators());
                if(last_split_pos == end_split_pos) {
                    const size_t sep_pos = Separators().find(str.at(last_split_pos));
                    const PrintMode mode = static_cast<PrintMode>(sep_pos);
                    std::vector<std::string> values;
                    values.push_back(str.substr(0, first_split_pos));
                    values.push_back(str.substr(first_split_pos + 1, last_split_pos - first_split_pos - 1));
                    values.push_back(str.substr(last_split_pos + 1));
                    return Make(values, mode);
                }
            }
        }
        throw exception("Invalid range with step '%1%'.") % str;
    }

private:
    static RangeWithStep<T> Make(const std::vector<std::string>& values, PrintMode mode)
    {
        const T min = ::analysis::Parse<T>(values.at(0));
        const T max = ::analysis::Parse<T>(values.at(1));
        T step(0);
        if(mode == PrintMode::Step) {
            step = ::analysis::Parse<T>(values.at(2));
        } else if(mode == PrintMode::NGridPoints) {
            size_t n = ::analysis::Parse<size_t>(values.at(2));
            if(n == 0 || (n == 1 && max != min) || (n != 1 && max == min))
                throw exception("Invalid number of grid points.");
            if(max != min)
                step = (max - min) / T(n - 1);
        } else if(mode == PrintMode::NBins) {
            size_t n = ::analysis::Parse<size_t>(values.at(2));
            if((n == 0 && max != min) || (n != 0 && max == min))
                throw exception("Invalid number of bins.");
            if(max != min)
                step = (max - min) / T(n);
        } else {
            throw exception("Unsupported RangeWithStep::PrintMode = %1%.") % static_cast<size_t>(mode);
        }
        return RangeWithStep<T>(min, max, step);
    }

    static const std::string& Separators() { static const std::string sep = ":|/"; return sep; }

private:
    T _step;
};

template<typename T>
std::ostream& operator<<(std::ostream& s, const RangeWithStep<T>& r)
{
    s << r.ToString();
    return s;
}

template<typename T>
std::istream& operator>>(std::istream& s, RangeWithStep<T>& r)
{
    std::string str;
    s >> str;
    r = RangeWithStep<T>::Parse(str);
    return s;
}

template<unsigned n_pi_per_period_num, unsigned n_pi_per_period_denom = 1>
struct Angle {
    enum class Interval { Symmetric, Positive };
    static constexpr double Pi() { return boost::math::constants::pi<double>(); }
    static constexpr double NumberOfPiPerPeriod() { return double(n_pi_per_period_num) / n_pi_per_period_denom; }
    static constexpr double FullPeriod() { return n_pi_per_period_num * Pi() / n_pi_per_period_denom; }
    static constexpr double HalfPeriod() { return FullPeriod() / 2; }
    static constexpr double RadiansToDegreesFactor() { return 180. / Pi(); }

    Angle() : _value(0), _interval(Interval::Symmetric) {}
    Angle(double value, Interval interval = Interval::Symmetric)
        : _value(AdjustValue(value, interval)), _interval(interval) {}

    double value() const { return _value; }
    double value_degrees() const { return value() * RadiansToDegreesFactor(); }
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
std::istream& operator>>(std::istream& s, Angle<n_pi_per_period_num, n_pi_per_period_denom>& a)
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

    static bool IsValid(const A& /*min*/, const A& /*max*/) { return true; }

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

    static Range<A> Parse(const std::string& str, const std::string& separators=": \t")
    {
        const auto values = SplitValueList(str, true, separators, true);
        if(values.size() != 2)
            throw exception("Invalid angle range '%1%'.") % str;
        return Make(values);
    }

    static Range<A> Read(std::istream& stream, const std::string& separators=": \t")
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

template<typename Range>
struct MultiRange {
    using ValueType = typename Range::ValueType;
    using ConstRefType = typename Range::ConstRefType;
    using RangeVec = std::vector<Range>;

    static const std::string& Separator() { static const std::string sep = ", "; return sep; }

    MultiRange() {}
    explicit MultiRange(const RangeVec& _ranges) : ranges(_ranges) {}

    bool Contains(const ValueType& point) const
    {
        for(const auto& range : ranges) {
            if(range.Contains(point))
                return true;
        }
        return false;
    }

    bool Overlaps(const Range& other) const
    {
        for(const auto& range : ranges) {
            if(range.Overlaps(other))
                return true;
        }
        return false;
    }

    std::string ToString() const
    {
        std::ostringstream ss;
        for(const auto& range : ranges)
            ss << range << Separator();
        std::string str = ss.str();
        if(str.size())
            str.erase(str.size() - Separator().size());
        return str;
    }

    static MultiRange<Range> Parse(const std::string& str)
    {
        const auto range_strs = SplitValueList(str, true, Separator(), true);
        RangeVec ranges;
        for(const auto& range_str : range_strs)
            ranges.push_back(::analysis::Parse<Range>(range_str));
        return MultiRange<Range>(ranges);
    }

private:
    RangeVec ranges;
};

template<typename Range>
std::ostream& operator<<(std::ostream& s, const MultiRange<Range>& r)
{
    s << r.ToString();
    return s;
}

template<typename Range>
std::istream& operator>>(std::istream& s, MultiRange<Range>& r)
{
    std::string str;
    std::getline(s, str);
    r = MultiRange<Range>::Parse(str);
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

struct Grid_ND {
    using Position = std::vector<size_t>;

    struct iterator {
        iterator(const Position& _pos, const Position& _limits) : pos(_pos), limits(&_limits) {}

        bool operator==(const iterator& other) const {
            if(pos.size() != other.pos.size()) return false;
            for(size_t n = 0; n < pos.size(); ++n) {
                if(pos.at(n) != other.pos.at(n)) return false;
            }
            return true;
        }

        bool operator!=(const iterator& other) { return !(*this == other); }

        iterator& operator++()
        {
            ++pos.at(0);
            for(size_t n = 0; n < pos.size() - 1 && pos.at(n) >= limits->at(n); ++n) {
                ++pos.at(n+1);
                pos.at(n) = 0;
            }
            return *this;
        }

        iterator operator++(int)
        {
            iterator cp(*this);
            ++(*this);
            return cp;
        }

        const Position& operator*() const { return pos; }
        const Position* operator->() const { return &pos; }

    private:
        Position pos;
        const Position* limits;
    };

    explicit Grid_ND(const Position& _limits) : limits(_limits)
    {
        if(!limits.size())
            throw exception("Grid dimensions should be > 0");
        for(size_t limit : limits) {
            if(!limit)
                throw exception("Grid range limit should be > 0.");
        }
    }

    iterator begin() const
    {
        Position pos;
        pos.assign(limits.size(), 0);
        return iterator(pos, limits);
    }

    iterator end() const
    {
        Position pos;
        pos.assign(limits.size(), 0);
        pos.back() = limits.back();
        return iterator(pos, limits);
    }

private:
    Position limits;
};

} // namespace analysis
