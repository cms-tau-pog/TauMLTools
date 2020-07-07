/*! Test exception class.
This file is part of https://github.com/hh-italian-group/TauMLTools. */

#include <iostream>
#include "TauMLTools/Core/interface/exception.h"

#define BOOST_TEST_MODULE exception_t
#define BOOST_TEST_DYN_LINK
#include <boost/test/unit_test.hpp>

using exception = analysis::exception;
namespace {
struct Foo {
    bool throw_when_creating{false}, throw_when_printing{false};

    Foo(bool _throw_when_creating, bool _throw_when_printing) :
        throw_when_creating(_throw_when_creating), throw_when_printing(_throw_when_printing)
    {
        if(throw_when_creating)
            throw std::runtime_error("Unable to create Foo.");
    }
};
std::ostream& operator<<(std::ostream& os, const Foo& foo)
{
    if(foo.throw_when_printing)
        throw std::runtime_error("Unable to convert Foo to string.");
    os << "Foo";
    return os;
}
}

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

    try {
        throw exception("hello %1%.") % Foo(false, false);
    } catch(exception& e) {
        BOOST_TEST(e.message() == "hello Foo.");
    }

    try {
        throw exception("hello %1%.") % Foo(false, true);
    } catch(exception& e) {
        BOOST_TEST(e.message() == "An exception has been raised while creating an error message."
                                  " Error message = 'hello %1%.'."
                                  " Exception message = 'Unable to convert Foo to string.'.");
    }

    try {
        throw exception("hello %1%.") % Foo(true, false);
    } catch(std::runtime_error& e) {
        BOOST_TEST(e.what() == "Unable to create Foo.");
    }

}
