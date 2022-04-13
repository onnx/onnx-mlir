# SPDX-License-Identifier: Apache-2.0

set(ONNX_NUMERICALTEST_SRC_DIR ${ONNX_MLIR_SRC_ROOT}/test/numerical)
set(ONNX_NUMERICALTEST_BIN_DIR ${ONNX_MLIR_BIN_ROOT}/test/numerical)

add_custom_target(numerical-nnpa)
set_target_properties(numerical-nnpa PROPERTIES FOLDER "Tests")

set(ZDNN_LIBRARY_DIR $ENV{ZDNN_LIBRARY_DIR})
set(ZDNN_LIB "${ZDNN_LIBRARY_DIR}/libzdnn.so")

add_custom_target(check-onnx-numerical-nnpa
  COMMENT "Running the ONNX-MLIR numerical regression tests for NNPA"
  COMMAND "${CMAKE_CTEST_COMMAND}" -L numerical-nnpa --output-on-failure -C $<CONFIG> --force-new-ctest-process
  USES_TERMINAL
  DEPENDS numerical-nnpa
  )
set_target_properties(check-onnx-numerical-nnpa PROPERTIES FOLDER "Tests")
# Exclude the target from the default VS build
set_target_properties(check-onnx-numerical-nnpa PROPERTIES EXCLUDE_FROM_DEFAULT_BUILD ON)

# add_numerical_unittest(test_name sources... options...
#   This function (generally) has the same semantic as add_onnx_mlir_executable.
#   A test with test_name is added as a ctest to the numerical testsuite and
#   all the rest of the arguments are passed directly to add_onnx_mlir_executable.
#   The function usage is meant to look like a call to add_onnx_mlir_executable
#   for readability.
#   )
function(add_numerical_unittest test_name)
  add_onnx_mlir_executable(${test_name} NO_INSTALL ${ARGN})

  add_dependencies(numerical-nnpa ${test_name})
  get_target_property(test_suite_folder numerical-nnpa FOLDER)
  if (test_suite_folder)
    set_property(TARGET ${test_name} PROPERTY FOLDER "${test_suite_folder}")
  endif ()

  # Force -O3 when invoking the numerical tests, so that the models in the test
  # are compiled with -O3. TODO: make that a runtime env variable.
  add_test(NAME ${test_name} COMMAND ${test_name} -O3 --maccel=NNPA WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})
  set_tests_properties(${test_name} PROPERTIES LABELS numerical-nnpa)
  set_tests_properties(${test_name} PROPERTIES ENVIRONMENT "TEST_ATOL=0.01;TEST_RTOL=0.05;LD_PRELOAD=${ZDNN_LIB}")
endfunction()

# All libraries and executables coming from llvm or ONNX-MLIR have had their
# compile flags updated via llvm_update_compile_flags, so we need to do that to
# rapidcheck as well, so that we can successfully link against it. Otherwise some
# of the flags for exceptions (among others) are not set correctly.
llvm_update_compile_flags(rapidcheck)

# The CompilerUtils ExecutionSession are also included in ModelLib,
# but it did not compile when I removed these two. TODO, figure out why.
set(TEST_LINK_LIBS rapidcheck ModelLib)
# set(TEST_LINK_LIBS rapidcheck ModelLib ${ZDNN_LIBRARY_DIR}/libzdnn.so)

add_numerical_unittest(TestConv-NNPA
  ${ONNX_NUMERICALTEST_SRC_DIR}/TestConv.cpp
  LINK_LIBS PRIVATE ${TEST_LINK_LIBS}
  )

add_numerical_unittest(TestMatMul2D-NNPA
  ${ONNX_NUMERICALTEST_SRC_DIR}/TestMatMul2D.cpp
  LINK_LIBS PRIVATE ${TEST_LINK_LIBS}
  )

add_numerical_unittest(TestGemm-NNPA
  ${ONNX_NUMERICALTEST_SRC_DIR}/TestGemm.cpp
  LINK_LIBS PRIVATE ${TEST_LINK_LIBS}
  )

add_numerical_unittest(TestLSTM-NNPA
  ${ONNX_NUMERICALTEST_SRC_DIR}/TestLSTM.cpp
  LINK_LIBS PRIVATE ${TEST_LINK_LIBS}
  )

add_numerical_unittest(TestRNN-NNPA
  ${ONNX_NUMERICALTEST_SRC_DIR}/TestRNN.cpp
  LINK_LIBS PRIVATE ${TEST_LINK_LIBS}
  )

add_numerical_unittest(TestGRU-NNPA
  ${ONNX_NUMERICALTEST_SRC_DIR}/TestGRU.cpp
  LINK_LIBS PRIVATE ${TEST_LINK_LIBS}
  )
