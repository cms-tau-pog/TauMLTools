set(CMAKE_CXX_COMPILER g++)

include_directories(SYSTEM "$ENV{ROOT_INCLUDE_PATH}" "$ENV{BOOST_INCLUDE_PATH}")
include_directories(SYSTEM "$ENV{CMSSW_RELEASE_BASE_SRC}")

set(CMSSW_BASE_SRC "${CMAKE_CURRENT_SOURCE_DIR}/..")
include_directories("${CMSSW_BASE_SRC}")

execute_process(COMMAND "$ENV{ROOTSYS}/bin/root-config" "--libs" OUTPUT_VARIABLE root_base_libs
                OUTPUT_STRIP_TRAILING_WHITESPACE)
set(root_all_libs "${root_base_libs} -lMathMore -lGenVector -lTMVA -lASImage")
set(all_libs "${root_all_libs}")

file(GLOB_RECURSE HEADER_LIST "*.h" "*.hh")
add_custom_target(headers SOURCES ${HEADER_LIST})

file(GLOB_RECURSE SOURCE_LIST "*.cxx" "*.C" "*.cpp" "*.cc")
add_custom_target(sources SOURCES ${SOURCE_LIST})

file(GLOB_RECURSE EXE_SOURCE_LIST "*.cxx")

file(GLOB_RECURSE SCRIPT_LIST "*.sh" "*.py")
add_custom_target(scripts SOURCES ${SCRIPT_LIST})

file(GLOB_RECURSE CONFIG_LIST "*.cfg" "*.xml")
add_custom_target(configs SOURCES ${CONFIG_LIST})

set(CMAKE_CXX_FLAGS "-std=c++11 -Wall")
