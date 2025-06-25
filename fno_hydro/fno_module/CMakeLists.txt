#Original Mac ...
# cmake_minimum_required(VERSION 3.15 FATAL_ERROR)
# project (libFnoModule)

# #for MacOSX
# cmake_policy(SET CMP0042 NEW)
# #cmake_policy(SET CMP0079 NEW)

# set(LIBTORCH_PATH "../libtorch") # Replace with the actual path
# find_package(Torch REQUIRED PATHS "${LIBTORCH_PATH}")
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

# #add_executable(test test.cc)
# #target_link_libraries(test "${TORCH_LIBRARIES}")
# #set_property(TARGET test PROPERTY CXX_STANDARD 17)

# set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "./lib")

# set (SOURCES FnoHydro.cc)

# #message(${TORCH_LIBRARIES})
# #message(${TORCH_INCLUDE_DIRS})
# #message(${CMAKE_CXX_FLAGS})

# add_library(FnoModule SHARED ${SOURCES})
# target_link_libraries(FnoModule "${TORCH_LIBRARIES}")
# target_link_libraries(FnoModule JetScape)
# target_link_libraries(FnoModule ${ROOT_LIBRARIES})

###########################

cmake_minimum_required(VERSION 3.15 FATAL_ERROR)
project (libFnoModule)

#include(${ROOT_USE_FILE})
#for MacOSX
#cmake_policy(SET CMP0042 NEW)
#cmake_policy(SET CMP0079 NEW)

#set(LIBTORCH_PATH "../libtorch") # Replace with the actual path
#find_package(Torch REQUIRED PATHS "${LIBTORCH_PATH}")
#add_executable(test test.cc)
#target_link_libraries(test "${TORCH_LIBRARIES}")
#set_property(TARGET test PROPERTY CXX_STANDARD 17)

set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "./lib")
set(CMAKE_INSTALL_LIBDIR "./")

set (SOURCES FnoHydro.cc FnoRooIn.cc PGunFno.cc)

#message(${TORCH_LIBRARIES})
#message(${TORCH_INCLUDE_DIRS})
message(${CMAKE_CXX_FLAGS})

ROOT_GENERATE_DICTIONARY(FnoModuleDict LINKDEF FnoModuleLinkDef.h)
#add_library(FnoModuleRootDict SHARED FnoModuleRootDictDict.cxx)
#target_link_libraries(FnoModuleRootDict ${ROOT_LIBRARIES})
#target_include_directories(FnoModuleRootDict PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

add_library(FnoModule SHARED ${SOURCES} FnoModuleDict.cxx)
target_link_libraries(FnoModule "${TORCH_LIBRARIES}")
target_link_libraries(FnoModule JetScape)
target_link_libraries(FnoModule ${ROOT_LIBRARIES})
