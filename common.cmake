include_directories(SYSTEM $ENV{GCC_INCLUDE_PATH} $ENV{ROOT_INCLUDE_PATH} $ENV{BOOST_INCLUDE_PATH})
include_directories(SYSTEM $ENV{CMSSW_RELEASE_BASE_SRC} $ENV{HEPMC_INCLUDE_PATH})
include_directories($ENV{CMSSW_BASE_SRC})

file(GLOB_RECURSE HEADER_LIST "*.h" "*.hh")
add_custom_target(headers SOURCES ${HEADER_LIST})

file(GLOB_RECURSE SOURCE_LIST "*.cxx" "*.C" "*.cpp" "*.cc")
add_custom_target(sources SOURCES ${SOURCE_LIST})

file(GLOB_RECURSE SCRIPT_LIST "*.sh" "*.py")
add_custom_target(scripts SOURCES ${SCRIPT_LIST})

file(GLOB_RECURSE CONFIG_LIST "*.cfg" "*.xml")
add_custom_target(configs SOURCES ${CONFIG_LIST})

set(CMAKE_CXX_FLAGS "-std=c++11 -Wall -O3")
