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

add_dependencies(check-onnx-backend-numerical-nnpa check-onnx-numerical-nnpa)

# add_numerical_test(test_name sources... options...
#   This function (generally) has the same semantic as add_onnx_mlir_executable.
#   A test with test_name is added as a ctest to the numerical testsuite and
#   all the rest of the arguments are passed directly to add_onnx_mlir_executable.
#   The function usage is meant to look like a call to add_onnx_mlir_executable
#   for readability.
#   )
function(add_numerical_test test_name)
  add_onnx_mlir_executable(${test_name}
    ${ARGN}

    NO_INSTALL

    INCLUDE_DIRS PRIVATE
    ${ONNX_NUMERICALTEST_SRC_DIR}
    )

  add_dependencies(numerical-nnpa ${test_name})
  get_target_property(test_suite_folder numerical-nnpa FOLDER)
  if (test_suite_folder)
    set_property(TARGET ${test_name} PROPERTY FOLDER "${test_suite_folder}")
  endif ()

  # Optimization level set by ONNX_MLIR_TEST_OPTLEVEL, defaults to 3
  add_test(NAME ${test_name}
    COMMAND ${test_name} -O${ONNX_MLIR_TEST_OPTLEVEL} --march=z16 --maccel=NNPA
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    )
  set_tests_properties(${test_name} PROPERTIES LABELS numerical-nnpa)

  # Set environment variables
  set(TEST_ENV_VAR "TEST_ATOL=0.01;TEST_RTOL=0.05")
  if (${test_name}_instruction)
    set(TEST_ENV_VAR "${TEST_ENV_VAR};TEST_INSTRUCTION=${${test_name}_instruction}")
  endif ()
  if (${test_name}_data_range)
     set(TEST_ENV_VAR "${TEST_ENV_VAR};TEST_DATARANGE=${${test_name}_data_range}")
  endif ()
  if (${test_name}_config)
     set(TEST_ENV_VAR "${TEST_ENV_VAR};TEST_CONFIG=${${test_name}_config}")
  endif ()
  set_tests_properties(${test_name} PROPERTIES ENVIRONMENT "${TEST_ENV_VAR}")
endfunction()

# The CompilerUtils ExecutionSession are also included in ModelLib,
# but it did not compile when I removed these two. TODO, figure out why.
set(TEST_LINK_LIBS rapidcheck ModelLib)

# Conv2D
set(TestConvNNPA_instruction zdnn_conv2d)
set(TestConvNNPA_data_range -0.1,0.1)
# Automatically set following config when using --maccel=NNPA
# set(TestConvNNPA_config "-dim=static -dilation=1 -padding=valid_upper")
add_numerical_test(TestConvNNPA
   ${ONNX_NUMERICALTEST_SRC_DIR}/TestConv.cpp
   LINK_LIBS PRIVATE ${TEST_LINK_LIBS}
   )

# MatMul2D
set(TestMatMul2DNNPA_instruction zdnn_matmul_op)
add_numerical_test(TestMatMul2DNNPA
  ${ONNX_NUMERICALTEST_SRC_DIR}/TestMatMul2D.cpp
  LINK_LIBS PRIVATE ${TEST_LINK_LIBS}
  )

# Gemm
set(TestGemmNNPA_instruction zdnn_matmul_op)
# Automatically set following config when using --maccel=NNPA
# set(TestGemmNNPA_config "-alpha=1 -beta=1")
add_numerical_test(TestGemmNNPA
  ${ONNX_NUMERICALTEST_SRC_DIR}/TestGemm.cpp
  LINK_LIBS PRIVATE ${TEST_LINK_LIBS}
  )

# LSTM
set(TestLSTMNNPA_instruction zdnn_lstm)
# Automatically set following config when using --maccel=NNPA
set(TestLSTMNNPA_config "-peephole=0")
add_numerical_test(TestLSTMNNPA
  ${ONNX_NUMERICALTEST_SRC_DIR}/TestLSTM.cpp
  LINK_LIBS PRIVATE ${TEST_LINK_LIBS}
  )

# GRU
set(TestGRUNNPA_instruction zdnn_gru)
# Automatically set following config when using --maccel=NNPA
set(TestGRUNNPA_config "-linearBeforeReset=1")
add_numerical_test(TestGRUNNPA
  ${ONNX_NUMERICALTEST_SRC_DIR}/TestGRU.cpp
  LINK_LIBS PRIVATE ${TEST_LINK_LIBS}
  )


# LeakyRelu
set(TestLeakyReluNNPA_instruction zdnn_mul)
# Automatically set following config when using --maccel=NNPA
add_numerical_test(TestLeakyReluNNPA
  ${ONNX_NUMERICALTEST_SRC_DIR}/TestLeakyRelu.cpp
  LINK_LIBS PRIVATE ${TEST_LINK_LIBS}
  )
