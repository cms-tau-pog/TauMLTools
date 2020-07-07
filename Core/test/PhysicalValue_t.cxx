/*! Test PhysicalValue class.
This file is part of https://github.com/hh-italian-group/TauMLTools. */

#include <iostream>
#include "TauMLTools/Core/interface/PhysicalValue.h"

#define BOOST_TEST_MODULE PhysicalValue_t
#define BOOST_TEST_DYN_LINK
#include <boost/test/unit_test.hpp>

using PV = analysis::PhysicalValue;

BOOST_AUTO_TEST_CASE(basic_arithmetics)
{
    PV x(5, 3), y(7, 4), z = x + y;
    BOOST_TEST(z.GetValue() == 12.);
    BOOST_TEST(z.GetStatisticalError() == 5.);

    z = x - y;
    BOOST_TEST(z.GetValue() == -2.);
    BOOST_TEST(z.GetStatisticalError() == 5.);

    z = x * y;
    BOOST_TEST(z.GetValue() == 35.);
    BOOST_TEST(z.GetStatisticalError() == 29.);

    z = x / y;
    BOOST_TEST(z.GetValue() == 0.7142857142857143);
    BOOST_TEST(z.GetStatisticalError() == 0.5918367346938775);
}

BOOST_AUTO_TEST_CASE(pv_to_string)
{
    PV x(1.235, 0.12);
    BOOST_TEST(x.ToString<char>(false, false) == std::string("1.24"));
    BOOST_TEST(x.ToString<char>(true, false) == std::string("1.24 +/- 0.12"));
    const bool wstring_test_1 = x.ToString<wchar_t>(false, false) == std::wstring(L"1.24");
    BOOST_TEST(wstring_test_1);
    const bool wstring_test_2 = x.ToString<wchar_t>(true, false) == std::wstring(L"1.24 \u00B1 0.12");
    BOOST_TEST(wstring_test_2);
}

BOOST_AUTO_TEST_CASE(std_arithmetics)
{
    PV x(-7., 3.);
    x = std::abs(x);
    BOOST_TEST(x.GetValue() == 7);
    BOOST_TEST(x.GetStatisticalError() == 3);

    PV y = std::sqrt(x);
    BOOST_TEST(y.GetValue() == 2.6457513110645907);
    BOOST_TEST(std::abs(y.GetStatisticalError() - 0.56694670951384084) < 1e-14);

    y = std::exp(x);
    BOOST_TEST(y.GetValue() == 1096.6331584284585);
    BOOST_TEST(y.GetStatisticalError() == 3289.8994752853755);

    y = std::log(x);
    BOOST_TEST(y.GetValue() == 1.9459101490553132);
    BOOST_TEST(y.GetStatisticalError() == 0.42857142857142855);

    y = std::pow(x, 5);
    BOOST_TEST(y.GetValue() == 16807.);
    BOOST_TEST(y.GetStatisticalError() == 36015.);

    y = std::pow(x, -0.33);
    BOOST_TEST(y.GetValue() == 0.5261597794341889);
    BOOST_TEST(y.GetStatisticalError() == 0.074414025948549572643);
}
