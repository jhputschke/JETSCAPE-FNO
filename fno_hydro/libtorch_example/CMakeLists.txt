#On Mac before: xattr -r -d com.apple.quarantine /Users/du8478/temp/libtorch/lib/*.dylib
#cmake -DCMAKE_PREFIX_PATH=/absolute/path/to/libtorch ..

#Alternative:
# Set the path to your libtorch installation
set(LIBTORCH_PATH "../libtorch") # Replace with the actual path
find_package(Torch REQUIRED PATHS "${LIBTORCH_PATH}")

cmake_minimum_required(VERSION 3.15 FATAL_ERROR)
project(libtorch_exampel)

find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

add_executable(test test.cc)
target_link_libraries(test "${TORCH_LIBRARIES}")
set_property(TARGET test PROPERTY CXX_STANDARD 17)

add_executable(test_load test_load.cc)
target_link_libraries(test_load "${TORCH_LIBRARIES}")
set_property(TARGET test_load PROPERTY CXX_STANDARD 17)
