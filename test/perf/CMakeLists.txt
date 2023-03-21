# SPDX-License-Identifier: Apache-2.0

add_onnx_mlir_library(PerfLib
  PerfHelper.cpp
  EXCLUDE_FROM_OM_LIBS

  LINK_LIBS PUBLIC
  benchmark 
  LLVMSupport
  )

add_custom_target(perf)
set_target_properties(perf PROPERTIES FOLDER "Perf")

add_custom_target(check-onnx-perf
COMMENT "Running the ONNX-MLIR performance regression tests, generating .perf.txt log files. Tune onnx-mlir compiler options using PERF_ARGS env var."
COMMAND "${CMAKE_CTEST_COMMAND}" -L perf --output-on-failure -C $<CONFIG> --force-new-ctest-process
  USES_TERMINAL
  DEPENDS perf
  )
set_target_properties(check-onnx-perf PROPERTIES FOLDER "Perf")
# Exclude the target from the default VS build
set_target_properties(check-onnx-perf PROPERTIES EXCLUDE_FROM_DEFAULT_BUILD ON)

# add_perf_unittest(test_name sources... options...
#   This function (generally) has the same semantic as add_onnx_mlir_executable.
#   A test with test_name is added as a ctest to the perf testsuite and
#   all the rest of the arguments are passed directly to add_onnx_mlir_executable.
#   The function usage is meant to look like a call to add_onnx_mlir_executable
#   for readability.
#   )
function(add_perf_unittest test_name)
  add_onnx_mlir_executable(${test_name} NO_INSTALL ${ARGN})

  add_dependencies(perf ${test_name})
  get_target_property(perf_suite_folder perf FOLDER)
  if (perf_suite_folder)
    set_property(TARGET ${test_name} PROPERTY FOLDER "${perf_suite_folder}")
  endif ()

  # Command to run a test.
  set(PERF_FILE ${CMAKE_CURRENT_BINARY_DIR}/${test_name}.perf.txt)
  add_test(NAME ${test_name} COMMAND ${test_name} --benchmark_out=${PERF_FILE} WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})
  set_tests_properties(${test_name} PROPERTIES LABELS perf)
endfunction()

set(TEST_LINK_LIBS ModelLib PerfLib)

add_perf_unittest(PerfGemm
  PerfGemm.cpp
  LINK_LIBS PRIVATE ${TEST_LINK_LIBS}
  )

add_perf_unittest(PerfConv
  PerfConv.cpp
  LINK_LIBS PRIVATE ${TEST_LINK_LIBS}
  )

add_perf_unittest(PerfRNN
  PerfRNN.cpp
  LINK_LIBS PRIVATE ${TEST_LINK_LIBS}
  )

add_perf_unittest(PerfElementwise
  PerfElementwise.cpp
  LINK_LIBS PRIVATE ${TEST_LINK_LIBS}
  )

