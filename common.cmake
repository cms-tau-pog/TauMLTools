execute_process(COMMAND "$ENV{ROOTSYS}/bin/root-config" --incdir OUTPUT_VARIABLE ROOT_INCLUDE_PATH
                OUTPUT_STRIP_TRAILING_WHITESPACE)

unset(scram_path CACHE)
find_file(scram_path "scram")
if(${scram_path} STREQUAL "scram_path-NOTFOUND")
    find_package(Boost)
    if(NOT ${Boost_FOUND})
        message(FATAL_ERROR "Boost package not found.")
    endif(NOT ${Boost_FOUND})
    set(BOOST_INCLUDE_PATH ${Boost_INCLUDE_DIRS})
    set(BOOST_LIB_PATH ${Boost_LIBRARY_DIRS})
    set(CMSSW_RELEASE_BASE_SRC "$ENV{CMSSW_RELEASE_BASE}/src")
else(${scram_path} STREQUAL "scram_path-NOTFOUND")
    execute_process(COMMAND scram tool info boost
                    COMMAND grep -e "^INCLUDE"
                    COMMAND sed "s/^INCLUDE=//"
                    OUTPUT_VARIABLE BOOST_INCLUDE_PATH
                    OUTPUT_STRIP_TRAILING_WHITESPACE)
    execute_process(COMMAND scram tool info boost
                    COMMAND grep -e "^LIBDIR"
                    COMMAND sed "s/^LIBDIR=//"
                    OUTPUT_VARIABLE BOOST_LIB_PATH
                    OUTPUT_STRIP_TRAILING_WHITESPACE)
    set(CMSSW_RELEASE_BASE_SRC "$ENV{CMSSW_RELEASE_BASE_SRC}")
endif(${scram_path} STREQUAL "scram_path-NOTFOUND")

include_directories(SYSTEM "${ROOT_INCLUDE_PATH}" "${BOOST_INCLUDE_PATH}")
include_directories(SYSTEM "${CMSSW_RELEASE_BASE_SRC}")

execute_process(COMMAND sh -c "cd ${CMAKE_CURRENT_SOURCE_DIR}/.. ; pwd" OUTPUT_VARIABLE CMSSW_BASE_SRC
                OUTPUT_STRIP_TRAILING_WHITESPACE)
include_directories("${CMSSW_BASE_SRC}")

execute_process(COMMAND "$ENV{ROOTSYS}/bin/root-config" "--libs" OUTPUT_VARIABLE root_base_libs
                OUTPUT_STRIP_TRAILING_WHITESPACE)

#set(root_all_libs "${root_base_libs} -lMathMore -lGenVector -lTMVA -lASImage")
#set(all_libs "${root_all_libs}")

execute_process(COMMAND "$ENV{ROOTSYS}/bin/root-config" --libdir OUTPUT_VARIABLE ROOT_LIB_PATH
                OUTPUT_STRIP_TRAILING_WHITESPACE)

find_library(lib_root_core Core PATHS ${ROOT_LIB_PATH})
find_library(lib_root_hist Hist PATHS ${ROOT_LIB_PATH})
find_library(lib_root_io RIO PATHS ${ROOT_LIB_PATH})
find_library(lib_root_tree Tree PATHS ${ROOT_LIB_PATH})
find_library(lib_root_phys Physics PATHS ${ROOT_LIB_PATH})
find_library(lib_root_graf Graf PATHS ${ROOT_LIB_PATH})
find_library(lib_root_gpad Gpad PATHS ${ROOT_LIB_PATH})
find_library(lib_root_matrix Matrix PATHS ${ROOT_LIB_PATH})
set(ROOT_LIBS ${lib_root_core} ${lib_root_hist} ${lib_root_io} ${lib_root_tree} ${lib_root_phys} ${lib_root_graf} ${lib_root_gpad} ${lib_root_matrix})
find_library(lib_boost_po NAMES boost_program_options boost_program_options-mt PATHS ${BOOST_LIB_PATH} NO_DEFAULT_PATH)
find_library(lib_boost_fs NAMES boost_filesystem boost_filesystem-mt PATHS ${BOOST_LIB_PATH} NO_DEFAULT_PATH)
find_library(lib_boost_regex NAMES boost_regex boost_regex-mt PATHS ${BOOST_LIB_PATH} NO_DEFAULT_PATH)
find_library(lib_boost_sys NAMES boost_system boost_system-mt PATHS ${BOOST_LIB_PATH} NO_DEFAULT_PATH)
set(BOOST_LIBS ${lib_boost_po} ${lib_boost_fs} ${lib_boost_regex} ${lib_boost_sys})
set(ALL_LIBS ${ROOT_LIBS} ${BOOST_LIBS})

SET(CMAKE_BUILD_WITH_INSTALL_RPATH TRUE)
SET(CMAKE_INSTALL_RPATH "$ENV{ROOTSYS}/lib;${BOOST_LIB_PATH}")

file(GLOB_RECURSE HEADER_LIST "*.h" "*.hh")
add_custom_target(headers SOURCES ${HEADER_LIST})

file(GLOB_RECURSE SOURCE_LIST "*.cxx" "*.C" "*.cpp" "*.cc")
add_custom_target(sources SOURCES ${SOURCE_LIST})

file(GLOB_RECURSE EXE_SOURCE_LIST "*.cxx")

file(GLOB_RECURSE SCRIPT_LIST "*.sh" "*.py")
add_custom_target(scripts SOURCES ${SCRIPT_LIST})

file(GLOB_RECURSE CONFIG_LIST "*.cfg" "*.xml")
add_custom_target(configs SOURCES ${CONFIG_LIST})

set(CMAKE_CXX_COMPILER g++)
set(CMAKE_CXX_FLAGS "-std=c++14 -Wall -O3")
