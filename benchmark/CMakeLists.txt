# TODO add option to disable benchmarks

set(THREADS_PREFER_PTHREAD_FLAG ON)
find_package(Threads REQUIRED)

add_executable(benchmark_all benchmark.cpp)
target_link_libraries(benchmark_all PRIVATE jpeggpu CUDA::nvjpeg Threads::Threads)
