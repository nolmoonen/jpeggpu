// Copyright (c) 2026 Nol Moonen
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

#ifndef JPEGGPU_UTIL_CUH_
#define JPEGGPU_UTIL_CUH_

#include <cub/thread/thread_operators.cuh>
#include <cub/version.cuh>
#include <cuda/functional>

namespace jpeggpu {

#if CUB_VERSION < 300000
using dev_eq = cub::Equality;
#else
using dev_eq = cuda::std::equal_to<>;
#endif

} // namespace jpeggpu

#endif // JPEGGPU_UTIL_CUH_
