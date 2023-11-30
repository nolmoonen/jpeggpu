cmake_minimum_required(VERSION 3.16)
project(jpeggpu LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/src")
add_library(jpeggpu STATIC
    "${SOURCE_DIR}/jpeggpu.cpp"
    "${SOURCE_DIR}/idct.cpp"
    "${SOURCE_DIR}/reader.cpp"
    "${SOURCE_DIR}/decoder.cpp")
target_include_directories(jpeggpu PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include")

add_executable(example example.c)
target_link_libraries(example PRIVATE jpeggpu)

add_executable(example_cpp example.cpp)
target_link_libraries(example_cpp PRIVATE jpeggpu)