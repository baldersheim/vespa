// Copyright Vespa.ai. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root.

#include <vespa/vespalib/stllike/string.hpp>
#include <istream>
#include <ostream>

namespace vespalib {

template<uint32_t SS>
std::ostream & operator << (std::ostream & os, const small_string<SS> & v)
{
     return os << v.buffer();
}

template<uint32_t SS>
std::istream & operator >> (std::istream & is, small_string<SS> & v)
{
    std::string s;
    is >> s;
    v = s;
    return is;
}

template std::ostream & operator << (std::ostream & os, const string & v);
template std::istream & operator >> (std::istream & is, string & v);

template class small_string<48>;

template string operator + (const string & a, const string & b) noexcept;
template string operator + (const string & a, stringref b) noexcept;
template string operator + (stringref a, const string & b) noexcept;
template string operator + (const string & a, const char * b) noexcept;
template string operator + (const  char * a, const string & b) noexcept;

const string &empty_string() noexcept {
    static string empty;
    return empty;
}

}
namespace std {
vespalib::string
operator + (std::string_view a, const char * b) noexcept
{
    vespalib::string t(a);
    t += b;
    return t;
}

vespalib::string
operator + (const char * a, std::string_view b) noexcept
{
    vespalib::string t(a);
    t += b;
    return t;
}

vespalib::string
operator + (std::string_view a, std::string_view b) noexcept {
    vespalib::string t(a);
    t += b;
    return t;
}
}