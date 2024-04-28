# TODO add option to disable benchmarks

set(THREADS_PREFER_PTHREAD_FLAG ON)
find_package(Threads REQUIRED)

add_executable(jpeggpu_benchmark benchmark.cpp)
target_link_libraries(jpeggpu_benchmark PRIVATE jpeggpu CUDA::nvjpeg Threads::Threads)
