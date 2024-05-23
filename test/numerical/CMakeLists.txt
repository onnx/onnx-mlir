# SPDX-License-Identifier: Apache-2.0

add_custom_target(numerical)
set_target_properties(numerical PROPERTIES FOLDER "Tests")

add_custom_target(check-onnx-numerical
  COMMENT "Running the ONNX-MLIR numerical regression tests"
  COMMAND "${CMAKE_CTEST_COMMAND}" -L numerical --output-on-failure -C $<CONFIG> --force-new-ctest-process
  USES_TERMINAL
  DEPENDS numerical
  )
set_target_properties(check-onnx-numerical PROPERTIES FOLDER "Tests")
# Exclude the target from the default VS build
set_target_properties(check-onnx-numerical PROPERTIES EXCLUDE_FROM_DEFAULT_BUILD ON)

add_dependencies(check-onnx-backend-numerical check-onnx-numerical)

# add_numerical_test(test_name sources... options...
#   This function (generally) has the same semantic as add_onnx_mlir_executable.
#   A test with test_name is added as a ctest to the numerical testsuite and
#   all the rest of the arguments are passed directly to add_onnx_mlir_executable.
#   The function usage is meant to look like a call to add_onnx_mlir_executable
#   for readability.
#   )
function(add_numerical_test test_name)
  add_onnx_mlir_executable(${test_name} NO_INSTALL ${ARGN})

  add_dependencies(numerical ${test_name})
  get_target_property(test_suite_folder numerical FOLDER)
  if (test_suite_folder)
    set_property(TARGET ${test_name} PROPERTY FOLDER "${test_suite_folder}")
  endif ()

  # Optimization level set by ONNX_MLIR_TEST_OPTLEVEL, defaults to 3
  add_test(NAME ${test_name}
    COMMAND ${test_name} -O${ONNX_MLIR_TEST_OPTLEVEL}
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    )
  set_tests_properties(${test_name} PROPERTIES LABELS numerical)
endfunction()

# The CompilerUtils ExecutionSession are also included in ModelLib,
# but it did not compile when I removed these two. TODO, figure out why.
set(TEST_LINK_LIBS rapidcheck ModelLib)

add_numerical_test(TestConv
  TestConv.cpp
  LINK_LIBS PRIVATE ${TEST_LINK_LIBS}
  )

add_numerical_test(TestMatMul2D
  TestMatMul2D.cpp
  LINK_LIBS PRIVATE ${TEST_LINK_LIBS}
  )

  add_numerical_test(TestMatMulBroadcast
  TestMatMulBroadcast.cpp
  LINK_LIBS PRIVATE ${TEST_LINK_LIBS}
  )

add_numerical_test(TestGemm
  TestGemm.cpp
  LINK_LIBS PRIVATE ${TEST_LINK_LIBS}
  )

add_numerical_test(TestLSTM
  TestLSTM.cpp
  LINK_LIBS PRIVATE ${TEST_LINK_LIBS}
  )

add_numerical_test(TestRNN
  TestRNN.cpp
  LINK_LIBS PRIVATE ${TEST_LINK_LIBS}
  )

add_numerical_test(TestGRU
  TestGRU.cpp
  LINK_LIBS PRIVATE ${TEST_LINK_LIBS}
  )

add_numerical_test(TestLoop
  TestLoop.cpp
  LINK_LIBS PRIVATE ${TEST_LINK_LIBS}
  )

add_numerical_test(TestScan
  TestScan.cpp
  LINK_LIBS PRIVATE ${TEST_LINK_LIBS}
  )

  add_numerical_test(TestElementwise
  TestElementwise.cpp
  LINK_LIBS PRIVATE ${TEST_LINK_LIBS}
  )

add_numerical_test(TestSoftplus
  TestSoftplus.cpp
  LINK_LIBS PRIVATE ${TEST_LINK_LIBS}
  )
