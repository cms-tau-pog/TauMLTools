project(TauML)
cmake_minimum_required(VERSION 3.6)

get_filename_component(AnalysisTools_DIR "${PROJECT_SOURCE_DIR}/../AnalysisTools" ABSOLUTE)
include("${AnalysisTools_DIR}/cmake/include/common.cmake")

file(GLOB_RECURSE ANA_TOOLS_SRC "${AnalysisTools_DIR}/*.cpp")
add_library(AnalysisTools STATIC ${ANA_TOOLS_SRC} "${RootDict}")
add_dependencies(AnalysisTools GenerateRootDict)

#add_library(TauML STATIC ${SOURCE_LIST})

foreach(exe_name ${EXE_LIST})
    target_link_libraries("${exe_name}" AnalysisTools)
endforeach()
