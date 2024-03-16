#include "logger.hpp"

using namespace jpeggpu;

namespace {
bool do_logging;
}

void jpeggpu::log(const char* t_format, ...)
{
    if (!do_logging) {
        return;
    }
    va_list argptr;
    va_start(argptr, t_format);
    vprintf(t_format, argptr);
    va_end(argptr);
}

void jpeggpu::set_logging(bool do_logging) { ::do_logging = do_logging; }
