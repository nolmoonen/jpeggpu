cmake_minimum_required(VERSION 3.16)
project(jpeggpu LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_library(jpeggpu STATIC "${CMAKE_CURRENT_SOURCE_DIR}/src/jpeggpu.cpp")
target_include_directories(jpeggpu PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include")

add_executable(jpeggputool main.cpp)
target_link_libraries(jpeggputool PRIVATE jpeggpu)