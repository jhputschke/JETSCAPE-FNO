#Original Mac ...
# make_minimum_required(VERSION 3.10 FATAL_ERROR)
# project (fnoHydroExample)

# #for MacOSX
# cmake_policy(SET CMP0042 NEW)
# cmake_policy(SET CMP0022 NEW)

# add_subdirectory(../fno_module ./lib) #Maybe change ...

# add_executable(fnoHydroTest ./fnoHydroTest.cc)
# target_link_libraries(fnoHydroTest FnoModule ${ROOT_LIBRARIES})

##################

cmake_minimum_required(VERSION 3.10 FATAL_ERROR)
project (fnoHydroExample)

#for MacOSX
#cmake_policy(SET CMP0042 NEW)
#cmake_policy(SET CMP0022 NEW)

set(CMAKE_PREFIX_PATH "../libtorch/share/cmake/Torch")
find_package(Torch REQUIRED) # PATHS "${LIBTORCH_PATH}")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")
include_directories(${TORCH_INCLUDE_DIRS})

add_subdirectory(../fno_module ./lib) # ./lib) #Maybe change ...

#set(LIBTORCH_PATH "../libtorch") # Replace with the actual path
#find_package(Torch REQUIRED PATHS "${LIBTORCH_PATH}")
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

add_executable(fnoHydroTest ./fnoHydroTest.cc)
target_link_libraries(fnoHydroTest FnoModule ${ROOT_LIBRARIES})
#set_property(TARGET fnoHydroTest PROPERTY CXX_STANDARD 17)

add_executable(fnoRooInTest ./fnoRooInTest.cc)
target_link_libraries(fnoRooInTest FnoModule ${ROOT_LIBRARIES})

add_executable(simpleReaderAna ./simpleReaderAna.cc)
target_link_libraries(simpleReaderAna JetScape ${ROOT_LIBRARIES})

add_executable(bulkAna ./bulkAna.cc)
target_link_libraries(bulkAna JetScape ${ROOT_LIBRARIES})
