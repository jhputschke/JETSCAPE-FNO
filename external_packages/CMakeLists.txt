cmake_minimum_required(VERSION 3.5 FATAL_ERROR)
project (libJetScapeThird)

# ── pybind11 via FetchContent (enabled by -DUSE_PYTHON=ON) ──────────────────
if (USE_PYTHON)
  include(FetchContent)
  FetchContent_Declare(
    pybind11
    GIT_REPOSITORY https://github.com/pybind/pybind11.git
    GIT_TAG        v2.11.1
    GIT_SHALLOW    TRUE
  )
  FetchContent_MakeAvailable(pybind11)
  message(STATUS "pybind11 fetched and available for pyjetscape_core build.")
endif (USE_PYTHON)

#for MacOSX
cmake_policy(SET CMP0042 NEW)

set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "../lib/")

set (SOURCES
    fjcore.cc
    tinyxml2.cc
    cornelius.cpp
)

add_subdirectory( hydro_from_external_file )

if (${ZLIB_FOUND})
    set (SOURCES ${SOURCES}  gzstream.cc )
endif()

add_library(JetScapeThird SHARED ${SOURCES})
if (${ZLIB_FOUND})
    target_link_libraries(JetScapeThird ${ZLIB_LIBRARIES} )
endif()


