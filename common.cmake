set(CMAKE_CXX_COMPILER g++)

execute_process(COMMAND "$ENV{ROOTSYS}/bin/root-config" --incdir OUTPUT_VARIABLE ROOT_INCLUDE_PATH
                OUTPUT_STRIP_TRAILING_WHITESPACE)

unset(scram_path CACHE)
find_file(scram_path "scram")
if(${scram_path} STREQUAL "scram_path-NOTFOUND")
    unset(boost_path CACHE)
    find_file(boost_path "boost")
    if(${boost_path} STREQUAL "boost_path-NOTFOUND")
        message(FATAL_ERROR "Boost include path not found.")
    endif(${boost_path} STREQUAL "boost_path-NOTFOUND")
    execute_process(COMMAND sh -c "cd ${boost_path}/.. ; pwd" OUTPUT_VARIABLE BOOST_INCLUDE_PATH
                    OUTPUT_STRIP_TRAILING_WHITESPACE)
    set(CMSSW_RELEASE_BASE_SRC "$ENV{CMSSW_RELEASE_BASE}/src")
else(${scram_path} STREQUAL "scram_path-NOTFOUND")
    execute_process(COMMAND scram tool info boost
                    COMMAND grep -e "^INCLUDE"
                    COMMAND sed "s/^INCLUDE=//"
                    OUTPUT_VARIABLE BOOST_INCLUDE_PATH
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
set(root_all_libs "${root_base_libs} -lMathMore -lGenVector -lTMVA -lASImage")
set(all_libs "${root_all_libs}")

SET(CMAKE_BUILD_WITH_INSTALL_RPATH TRUE)
SET(CMAKE_INSTALL_RPATH "$ENV{ROOTSYS}/lib")

file(GLOB_RECURSE HEADER_LIST "*.h" "*.hh")
add_custom_target(headers SOURCES ${HEADER_LIST})

file(GLOB_RECURSE SOURCE_LIST "*.cxx" "*.C" "*.cpp" "*.cc")
add_custom_target(sources SOURCES ${SOURCE_LIST})

file(GLOB_RECURSE EXE_SOURCE_LIST "*.cxx")

file(GLOB_RECURSE SCRIPT_LIST "*.sh" "*.py")
add_custom_target(scripts SOURCES ${SCRIPT_LIST})

file(GLOB_RECURSE CONFIG_LIST "*.cfg" "*.xml")
add_custom_target(configs SOURCES ${CONFIG_LIST})

set(CMAKE_CXX_FLAGS "-std=c++14 -Wall")

