cmake_minimum_required(VERSION 3.15)

set(PYJETSCAPE_SOURCES
    pyjetscape_core.cc
    bind_framework.cc
    bind_evolution.cc
    bind_initial_state.cc
    bind_fluid_dynamics.cc
    bind_music.cc
)

pybind11_add_module(pyjetscape_core ${PYJETSCAPE_SOURCES})

target_include_directories(pyjetscape_core PRIVATE
    ${CMAKE_SOURCE_DIR}/src/framework
    ${CMAKE_SOURCE_DIR}/src/initialstate
    ${CMAKE_SOURCE_DIR}/src/preequilibrium
    ${CMAKE_SOURCE_DIR}/src/hydro
    ${CMAKE_SOURCE_DIR}/src/liquefier
    ${CMAKE_SOURCE_DIR}/src/jet
    ${CMAKE_SOURCE_DIR}/src/hadronization
    ${CMAKE_SOURCE_DIR}/src/afterburner
    ${CMAKE_SOURCE_DIR}/external_packages
    ${CMAKE_SOURCE_DIR}/external_packages/gtl/include
    ${CMAKE_SOURCE_DIR}/external_packages/trento/src
)

target_link_libraries(pyjetscape_core PRIVATE JetScape)

# Output the .so directly into python/jetscape/ for easy import
set_target_properties(pyjetscape_core PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/python/jetscape"
)

message(STATUS "pyjetscape_core will be built and installed to ${CMAKE_SOURCE_DIR}/python/jetscape/")
