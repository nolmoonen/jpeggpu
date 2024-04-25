#ifndef JPEGGPU_LOGGER_HPP_
#define JPEGGPU_LOGGER_HPP_

#include <cstdarg>
#include <cstdio>

namespace jpeggpu {

struct logger {

    void log(const char* t_format, ...)
    {
        if (!do_logging) {
            return;
        }
        va_list argptr;
        va_start(argptr, t_format);
        vprintf(t_format, argptr);
        va_end(argptr);
    }

    void set_logging(bool do_logging) { this->do_logging = do_logging; }

    bool do_logging;
};

} // namespace jpeggpu

#endif // JPEGGPU_LOGGER_HPP_
