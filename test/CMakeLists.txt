# Copyright (c) 2024 Nol Moonen
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https:#www.gnu.org/licenses/>.

# TODO add option to disable test
add_executable(jpeggpu_test test.cpp)
target_link_libraries(jpeggpu_test PRIVATE jpeggpu CUDA::nvjpeg)
target_include_directories(jpeggpu_test PRIVATE "${PROJECT_SOURCE_DIR}/util")
configure_file(test.sh "${CMAKE_BINARY_DIR}/test.sh" COPYONLY)
