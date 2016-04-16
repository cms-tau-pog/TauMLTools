find_package(ROOT REQUIRED COMPONENTS Core Hist RIO Tree Physics Graf Gpad Matrix)
find_package(BoostEx REQUIRED COMPONENTS program_options filesystem regex system)
include_directories(SYSTEM ${Boost_INCLUDE_DIRS} ${ROOT_INCLUDE_DIR})
set(ALL_LIBS ${Boost_LIBRARIES} ${ROOT_LIBRARIES})

SET(CMAKE_BUILD_WITH_INSTALL_RPATH TRUE)
SET(CMAKE_INSTALL_RPATH "${Boost_LIBRARY_DIRS};${ROOT_LIBRARY_DIR}")

set(CMAKE_CXX_COMPILER g++)
set(CMAKE_CXX_FLAGS "-std=c++14 -Wall -O3")

execute_process(COMMAND root-config --incdir OUTPUT_VARIABLE ROOT_INCLUDE_PATH OUTPUT_STRIP_TRAILING_WHITESPACE)

find_file(scram_path "scram")
if(scram_path)
    set(CMSSW_RELEASE_BASE_SRC "$ENV{CMSSW_RELEASE_BASE}/src")
else()
    set(CMSSW_RELEASE_BASE_SRC "$ENV{CMSSW_RELEASE_BASE_SRC}")
endif()

include_directories(SYSTEM "${CMSSW_RELEASE_BASE_SRC}")

execute_process(COMMAND sh -c "cd ${CMAKE_CURRENT_SOURCE_DIR}/.. ; pwd" OUTPUT_VARIABLE CMSSW_BASE_SRC
                OUTPUT_STRIP_TRAILING_WHITESPACE)
include_directories("${CMSSW_BASE_SRC}")

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
