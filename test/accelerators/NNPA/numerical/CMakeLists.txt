# SPDX-License-Identifier: Apache-2.0

set(ONNX_NUMERICALTEST_SRC_DIR ${ONNX_MLIR_SRC_ROOT}/test/numerical)

add_custom_target(numerical-nnpa)
set_target_properties(numerical-nnpa PROPERTIES FOLDER "Tests")

add_custom_target(check-onnx-numerical-nnpa
  COMMENT "Running the ONNX-MLIR numerical regression tests for NNPA"
  # To display logs for the tests, add "--verbose" in following command
  COMMAND "${CMAKE_CTEST_COMMAND}" -L numerical-nnpa --output-on-failure -C $<CONFIG> --force-new-ctest-process
  USES_TERMINAL
  DEPENDS numerical-nnpa
  )
set_target_properties(check-onnx-numerical-nnpa PROPERTIES FOLDER "Tests")

# add_numerical_unittest(test_name check_instruction sources... options...
#   This function (generally) has the same semantic as add_onnx_mlir_executable.
#   A test with test_name is added as a ctest to the numerical testsuite and
#   all the rest of the arguments are passed directly to add_onnx_mlir_executable.
#   The function usage is meant to look like a call to add_onnx_mlir_executable
#   for readability.
#   The check_instruction is an instruction name which should be included in
#   generated library.
#   )
function(add_numerical_unittest test_name)
  add_onnx_mlir_executable(${test_name} NO_INSTALL ${ARGN})

  add_dependencies(numerical-nnpa ${test_name})
  get_target_property(test_suite_folder numerical-nnpa FOLDER)
  if (test_suite_folder)
    set_property(TARGET ${test_name} PROPERTY FOLDER "${test_suite_folder}")
  endif ()

  # Optimization level set by ONNX_MLIR_TEST_OPTLEVEL, defaults to 3
  add_test(NAME ${test_name}
    COMMAND ${test_name} -O${ONNX_MLIR_TEST_OPTLEVEL} --maccel=NNPA
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    )
  set_tests_properties(${test_name} PROPERTIES LABELS numerical-nnpa)

  # Set environment variables
  set(TEST_ENV_VAR "TEST_ATOL=0.01;TEST_RTOL=0.05")
  if (${test_name}_instruction)
    set(TEST_ENV_VAR "${TEST_ENV_VAR};${test_name}_INSTRUCTION=${${test_name}_instruction}")
  endif ()
  if (${test_name}_data_range)
     set(TEST_ENV_VAR "${TEST_ENV_VAR};${test_name}_DATARANGE=${${test_name}_data_range}")
  endif ()
  set_tests_properties(${test_name} PROPERTIES ENVIRONMENT "${TEST_ENV_VAR}")
endfunction()

# All libraries and executables coming from llvm or ONNX-MLIR have had their
# compile flags updated via llvm_update_compile_flags, so we need to do that to
# rapidcheck as well, so that we can successfully link against it. Otherwise some
# of the flags for exceptions (among others) are not set correctly.
llvm_update_compile_flags(rapidcheck)

# The CompilerUtils ExecutionSession are also included in ModelLib,
# but it did not compile when I removed these two. TODO, figure out why.
set(TEST_LINK_LIBS rapidcheck ModelLib)

# Conv2D
set(TestConvNNPA_instruction zdnn_conv2d)
set(TestConvNNPA_data_range 0.001)
add_numerical_unittest(TestConvNNPA
   ${ONNX_NUMERICALTEST_SRC_DIR}/TestConv.cpp
   DEFINE PRIVATE TEST_CONV_STATIC TEST_CONV_D1 TEST_CONV_VALID_UPPER
   LINK_LIBS PRIVATE ${TEST_LINK_LIBS}
   )

# MatMul2D
set(TestMatMul2DNNPA_instruction zdnn_matmul_op)
add_numerical_unittest(TestMatMul2DNNPA
  ${ONNX_NUMERICALTEST_SRC_DIR}/TestMatMul2D.cpp
  LINK_LIBS PRIVATE ${TEST_LINK_LIBS}
  )

# Gemm
set(TestGemmNNPA_instruction zdnn_matmul_op)
add_numerical_unittest(TestGemmNNPA
  ${ONNX_NUMERICALTEST_SRC_DIR}/TestGemm.cpp
  DEFINE PRIVATE TEST_GEMM_ALPHA_BETA_1
  LINK_LIBS PRIVATE ${TEST_LINK_LIBS}
  )


# LSTM
set(TestLSTMNNPA_instruction zdnn_lstm)
add_numerical_unittest(TestLSTMNNPA
  ${ONNX_NUMERICALTEST_SRC_DIR}/TestLSTM.cpp
  DEFINE PRIVATE TEST_LSTM_NONEP_ONLY
  LINK_LIBS PRIVATE ${TEST_LINK_LIBS}
  )

# GRU
set(TestGRUNNPA_instruction zdnn_gru)
add_numerical_unittest(TestGRUNNPA
  ${ONNX_NUMERICALTEST_SRC_DIR}/TestGRU.cpp
  DEFINE PRIVATE TEST_GRU_L1
  LINK_LIBS PRIVATE ${TEST_LINK_LIBS}
  )
