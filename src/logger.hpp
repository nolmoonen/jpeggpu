#ifndef JPEGGPU_LOGGER_HPP_
#define JPEGGPU_LOGGER_HPP_

#include <cstdarg>
#include <cstdio>

namespace jpeggpu {

void log(const char* t_format, ...);

void set_logging(bool do_logging);

} // namespace jpeggpu

#endif // JPEGGPU_LOGGER_HPP_
