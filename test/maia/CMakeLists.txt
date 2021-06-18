cmake_minimum_required (VERSION 3.15)

project(MAIA_TEST)

set(CMAKE_CXX_STANDARD 17)

include(apollo.cmake)

function(generate_baseline_output TEST)
  add_custom_command(OUTPUT ${TEST_DIR}/outputs
                     COMMAND ${Python3_EXECUTABLE} ${MAIA_SOURCE_DIR}/execute_model.py --input ${TEST_DIR}/${TEST}.f32.onnx
                     MAIN_DEPENDENCY ${MAIA_SOURCE_DIR}/execute_model.py
                     DEPEND ${TEST_DIR}/${TEST}.f32.onnx
                     VERBATIM)
  add_custom_target(${TEST}-ort DEPENDS ${TEST_DIR}/outputs)
  add_dependencies(MAIA_TEST ${TEST}-ort)
endfunction()

if (CMAKE_GENERATOR MATCHES "Visual Studio")
  message(WARNING "MAIA NPU testing does not support Visual Studio; please use ninja.")
  return()
endif()

if (NOT APOLLO_SDK_PATH)
  set(APOLLO_SDK_PATH "APOLLO_SDK_PATH-NOTFOUND" CACHE PATH "Path to Apollo SDK." FORCE)
endif()

message(STATUS "Using Apollo SDK: ${APOLLO_SDK_PATH}")
message(STATUS "Using LLVM: ${LLVM_TOOLS_BINARY_DIR}")
message(STATUS "Using ONNX MLIR: ${ONNX_MLIR_RUNTIME_PATH}")

file(TO_CMAKE_PATH "${APOLLO_SDK_PATH}" ApolloSDK_PATH)

set(ApolloSDK_Lib_PATH ${ApolloSDK_PATH}/lib)
set(ApolloSDK_Package_PATH ${ApolloSDK_PATH}/packages)

set(CMAKE_MODULE_PATH "${ApolloSDK_PATH}/cmake")

set(MAIA_BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR})
set(MAIA_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR})
set(HOST_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/include)

# build our test lists
#  `make_tests --list` produces a tab-delimited output which is parsed by this code
execute_process(COMMAND ${Python3_EXECUTABLE} ${MAIA_SOURCE_DIR}/make_tests.py --list
                OUTPUT_VARIABLE TEST_CONFIG)
string(REPLACE "\n" ";" TEST_CONFIG ${TEST_CONFIG})
foreach(ENTRY ${TEST_CONFIG})
  string(REGEX REPLACE "^(.*)\t(.*)$" "\\1" TEST_NAME ${ENTRY})
  list(APPEND TEST_LIST ${TEST_NAME})
  string(REGEX MATCH "^.*\t.*EXECUTE" TEST_EXECUTE ${ENTRY})
  if(NOT ${TEST_EXECUTE} EQUAL "")
    list(APPEND EXECUTE_LIST ${TEST_NAME})
  endif()
  string(REGEX MATCH "^.*\t.*DIFF" TEST_DIFF ${ENTRY})
  if(NOT ${TEST_DIFF} EQUAL "")
    list(APPEND DIFF_LIST ${TEST_NAME})
  endif()
endforeach()

# compute a list of all the python scripts in our domain.
#   - many of these scripts are used as modules by the python scripts below
#   - if these scripts change we want to make sure to rebuild
file(GLOB_RECURSE PY_SCRIPT_LIST ${MAIA_SOURCE_DIR}/*.py)

# this is the "parent" target for all of the tests we will be building
add_custom_target(MAIA_TEST)

# walk the list of tests computed above and add appropriate builds/tests for each
set(TEST_INDEX 0)
foreach(TEST ${TEST_LIST})
  list(FIND EXECUTE_LIST ${TEST} TEST_EXECUTE)
  list(FIND DIFF_LIST ${TEST} TEST_DIFF)
  set(TEST_DIR ${MAIA_BINARY_DIR}/${TEST})

  # generate the ONNX graphs for this test
  add_custom_command(OUTPUT ${TEST_DIR}/${TEST}.onnx ${TEST_DIR}/${TEST}.f32.onnx ${TEST_DIR}/host.cpp
                     COMMAND ${Python3_EXECUTABLE} ${MAIA_SOURCE_DIR}/make_tests.py --outdir=${MAIA_BINARY_DIR} --test ${TEST} --test-index ${TEST_INDEX}
                     MAIN_DEPENDENCY ${MAIA_SOURCE_DIR}/make_tests.py
                     DEPENDS ${PY_SCRIPT_LIST}
                     VERBATIM)

  # invoke the MAIA compiler
  apollo_compile_graph(${TEST})

  if(${TEST_EXECUTE} GREATER_EQUAL 0)
    # compile output code and execute as a test
    apollo_compile_host(${TEST})
    apollo_compile_npu(${TEST})
    generate_baseline_output(${TEST})
    add_test(NAME maia-execute-${TEST}
             COMMAND ${TEST}.host ${TEST}_NPU.zip
             WORKING_DIRECTORY ${TEST_DIR})
  elseif(${TEST_DIFF} GREATER_EQUAL 0)
# This is skeleton/placeholder code for what the diff tests should look like.
# Remaining work to enable these:
#   - Determine if we really want to use FileCheck
#   - Determine if this is the right set of diffs
#   - Create baseline "golden" files
#   add_test(NAME maia-diff-${TEST}-asm COMMAND FileCheck --input-file=./${TEST}.s -check-prefix=CHECK-ASM ${MAIA_SOURCE_DIR}/golden/${TEST}.golden)
#   add_test(NAME maia-diff-${TEST}-tcp-h COMMAND FileCheck --input-file=./${TEST}.tcp.h -check-prefix=CHECK-TCP-H ${MAIA_SOURCE_DIR}/golden/${TEST}.golden)
#   add_test(NAME maia-diff-${TEST}-tvp-h COMMAND FileCheck --input-file=./${TEST}.tvp.h -check-prefix=CHECK-TVP-H ${MAIA_SOURCE_DIR}/golden/${TEST}.golden)
#   add_test(NAME maia-diff-${TEST}-tcp-cpp COMMAND FileCheck --input-file=./${TEST}.tcp.cpp -check-prefix=CHECK-TCP-CPP ${MAIA_SOURCE_DIR}/golden/${TEST}.golden)
#   add_test(NAME maia-diff-${TEST}-tvp-cpp COMMAND FileCheck --input-file=./${TEST}.tvp.cpp -check-prefix=CHECK-TVP-CPP ${MAIA_SOURCE_DIR}/golden/${TEST}.golden)
  endif()

  math(EXPR TEST_INDEX "${TEST_INDEX} + 1")
endforeach()
