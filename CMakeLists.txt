project(TauML)
cmake_minimum_required(VERSION 3.6)

get_filename_component(AnalysisTools_DIR "${PROJECT_SOURCE_DIR}/../AnalysisTools" ABSOLUTE)
include("${AnalysisTools_DIR}/cmake/include/common.cmake")

execute_process(COMMAND scram tool info Eigen OUTPUT_VARIABLE SCRAM_E_INFO)
string(REGEX MATCH "\nINCLUDE=([^\n]*)" E_STR ${SCRAM_E_INFO})
set(Eigen_INCLUDE_DIRS "${CMAKE_MATCH_1}")

execute_process(COMMAND scram tool info protobuf OUTPUT_VARIABLE SCRAM_PB_INFO)
string(REGEX MATCH "\nINCLUDE=([^\n]*)" PB_STR ${SCRAM_PB_INFO})
set(Protobuf_INCLUDE_DIRS "${CMAKE_MATCH_1}")
string(REGEX MATCH "\nLIBDIR=([^\n]*)" PB_STR ${SCRAM_PB_INFO})
set(Protobuf_LIBRARY_DIRS "${CMAKE_MATCH_1}")
set(Protobuf_LIBRARIES "${CMAKE_MATCH_1}/libprotobuf.so")

execute_process(COMMAND scram tool info tensorflow-c OUTPUT_VARIABLE SCRAM_TFC_INFO)
string(REGEX MATCH "\nINCLUDE=([^\n]*)" TFC_STR ${SCRAM_TFC_INFO})
set(APPEND Tensorflow-c_INCLUDE_DIRS "${CMAKE_MATCH_1}")
string(REGEX MATCH "\nLIBDIR=([^\n]*)" TFC_STR ${SCRAM_TFC_INFO})
set(Tensorflow-c_LIBRARY_DIRS "${CMAKE_MATCH_1}")
set(Tensorflow-c_LIBRARIES "${CMAKE_MATCH_1}/libtensorflow.so")

execute_process(COMMAND scram tool info tensorflow-cc OUTPUT_VARIABLE SCRAM_TFCC_INFO)
string(REGEX MATCH "\nINCLUDE=([^\n]*)" TFCC_STR ${SCRAM_TFCC_INFO})
set(Tensorflow-cc_INCLUDE_DIRS "${CMAKE_MATCH_1}")
string(REGEX MATCH "\nLIBDIR=([^\n]*)" TFCC_STR ${SCRAM_TFCC_INFO})
set(Tensorflow-cc_LIBRARY_DIRS "${CMAKE_MATCH_1}")
set(Tensorflow-cc_LIBRARIES "${Tensorflow-cc_LIBRARY_DIRS}/libtensorflow_cc.so")

include_directories(SYSTEM ${Tensorflow-cc_INCLUDE_DIRS} ${Tensorflow-c_INCLUDE_DIRS} ${Eigen_INCLUDE_DIRS} ${Protobuf_INCLUDE_DIRS})

target_link_libraries("ApplyTraining" ${Tensorflow-cc_LIBRARIES} ${Tensorflow-c_LIBRARIES} ${Protobuf_LIBRARIES} pthread $ENV{CMSSW_RELEASE_BASE}/lib/slc6_amd64_gcc630/libPhysicsToolsTensorFlow.so)

add_library("TauML" OBJECT ${HEADER_LIST} ${SOURCE_LIST} ${EXE_SOURCE_LIST} ${SCRIPT_LIST} ${CONFIG_LIST})
set_target_properties("TauML" PROPERTIES EXCLUDE_FROM_ALL 1)
