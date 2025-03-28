cmake_minimum_required(VERSION 3.10 FATAL_ERROR)
project (fnoHydroExample)

#for MacOSX
cmake_policy(SET CMP0042 NEW)
cmake_policy(SET CMP0022 NEW)

add_subdirectory(../fno_module ./lib) #Maybe change ...

add_executable(fnoHydroTest ./fnoHydroTest.cc)
target_link_libraries(fnoHydroTest FnoModule ${ROOT_LIBRARIES})
