cmake_minimum_required(VERSION 3.10 FATAL_ERROR)
project (libJetScapeRootBulk)

#for MacOSX
cmake_policy(SET CMP0042 NEW)
cmake_policy(SET CMP0022 NEW)

set (SOURCES )
set (HEADERS )

#---Define useful ROOT functions and macros (e.g. ROOT_GENERATE_DICTIONARY)
include(${ROOT_USE_FILE})

set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "../../")
set(CMAKE_INSTALL_LIBDIR "./")

message("ROOT_GENERATE_DICTIONARY .. ")
include_directories(./)
add_definitions(${ROOT_CXX_FLAGS})

#ROOT_GENERATE_DICTIONARY(G__JetScapeRootLibBulkDict ${HEADERS} LINKDEF JetScapeRootBulkLinkDef.h)
#set (SOURCES ${SOURCES} G__JetScapeRootBulkLibDict.cxx)
#add_executable(bulkTest ./bulkTest.cc G__JetScapeRootBulkLibDict.cxx)
#target_link_libraries(bulkTest JetScape ${ROOT_LIBRARIES})

#add_library(JetScapeRoot SHARED ${SOURCES} G__JetScapeRootLibDict.cxx)
#add_dependencies(JetScapeRoot G__JetScapeRootLibDict)

add_executable(bulkTest ./bulkTest.cc)
target_link_libraries(bulkTest JetScape ${ROOT_LIBRARIES})
#Attach dictionaries to the executable. First, tell it where to look for headers required by the dictionaries:
target_include_directories(bulkTest PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
# Then generate dictionaries and add them as a dependency of the executable (via the MODULE parameter):
ROOT_GENERATE_DICTIONARY(bulkTest_dict MODULE bulkTest LINKDEF JetScapeRootBulkLinkDef.h)

add_executable(bulkRootWriter ./bulkRootWriter.cc)
target_link_libraries(bulkRootWriter JetScape ${ROOT_LIBRARIES})
#Attach dictionaries to the executable. First, tell it where to look for headers required by the dictionaries:
target_include_directories(bulkRootWriter PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
# Then generate dictionaries and add them as a dependency of the executable (via the MODULE parameter):
ROOT_GENERATE_DICTIONARY(bulkRootWriter_dict MODULE bulkRootWriter LINKDEF JetScapeRootBulkLinkDef.h)

#Have to ln -s src/lib/libbulkTest_rdict.pcm . otherwise error, change location or create a real library ....
