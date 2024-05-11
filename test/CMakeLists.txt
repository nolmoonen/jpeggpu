# TODO add option to disable test
add_executable(jpeggpu_test test.cpp)
target_link_libraries(jpeggpu_test PRIVATE jpeggpu CUDA::nvjpeg)
configure_file(test.sh "${CMAKE_BINARY_DIR}/test.sh" COPYONLY)
