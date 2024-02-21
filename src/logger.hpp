#ifndef JPEGGPU_LOGGER_HPP_
#define JPEGGPU_LOGGER_HPP_

#include <cstdarg>
#include <cstdio>

namespace jpeggpu {

struct logger {

    // TODO improve semantics for pointer types,
    //   how to make it accessible from everywhere
    void operator()(const char* t_format, ...)
    {
        if (!do_logging) {
            return;
        }
        va_list argptr;
        va_start(argptr, t_format);
        vprintf(t_format, argptr);
        va_end(argptr);
    }

    bool do_logging;
};

} // namespace jpeggpu

#endif // JPEGGPU_LOGGER_HPP_
