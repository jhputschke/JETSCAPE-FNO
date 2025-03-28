cmake_minimum_required(VERSION 3.15 FATAL_ERROR)
project (libFnoModule)

#for MacOSX
cmake_policy(SET CMP0042 NEW)
#cmake_policy(SET CMP0079 NEW)

set(LIBTORCH_PATH "../libtorch") # Replace with the actual path
find_package(Torch REQUIRED PATHS "${LIBTORCH_PATH}")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

#add_executable(test test.cc)
#target_link_libraries(test "${TORCH_LIBRARIES}")
#set_property(TARGET test PROPERTY CXX_STANDARD 17)

set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "./lib")

set (SOURCES FnoHydro.cc)

#message(${TORCH_LIBRARIES})
#message(${TORCH_INCLUDE_DIRS})
#message(${CMAKE_CXX_FLAGS})

add_library(FnoModule SHARED ${SOURCES})
target_link_libraries(FnoModule "${TORCH_LIBRARIES}")
target_link_libraries(FnoModule JetScape)
target_link_libraries(FnoModule ${ROOT_LIBRARIES})
