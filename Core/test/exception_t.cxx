/*! Test exception class.
This file is part of https://github.com/hh-italian-group/AnalysisTools. */

#include <iostream>
#include "AnalysisTools/Core/include/exception.h"

#define BOOST_TEST_MODULE exception_t
#define BOOST_TEST_DYN_LINK
#include <boost/test/unit_test.hpp>

using exception = analysis::exception;

BOOST_AUTO_TEST_CASE(message_test)
{
    {
        exception e("test");
        BOOST_TEST(e.message() == "test");
        BOOST_TEST(e.what() == "test");
    }

    {
        exception e("hello %1.");
        BOOST_TEST(e.message() == "bad formatted error message = 'hello %1.'.");
    }

    {
        exception e("hello %1%.");
        BOOST_TEST(e.message() == "too few arguments are provided to the error message = 'hello %1%.'.");
    }

    {
        exception e("hello %1%.");
        e % 2;
        BOOST_TEST(e.message() == "hello 2.");
    }

    {
        exception e("hello %1%.");
        e % 2 % 3.14;
        BOOST_TEST(e.message() == "too many arguments are provided to the error message = 'hello %1%.'.");
    }

    {
        exception e("hello %1%, %2%.");
        e % 2 % 3.14;
        BOOST_TEST(e.message() == "hello 2, 3.14.");
    }
}
