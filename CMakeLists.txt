project(TauML)
cmake_minimum_required(VERSION 3.6)

get_filename_component(AnalysisTools_DIR "${PROJECT_SOURCE_DIR}/../AnalysisTools" ABSOLUTE)
include("${AnalysisTools_DIR}/cmake/include/common.cmake")

add_library("TauML" OBJECT ${HEADER_LIST} ${SOURCE_LIST} ${EXE_SOURCE_LIST} ${SCRIPT_LIST} ${CONFIG_LIST})
set_target_properties("TauML" PROPERTIES EXCLUDE_FROM_ALL 1)
