// Copyright (c) 2024 Nol Moonen
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program.  If not, see <https://www.gnu.org/licenses/>.

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
