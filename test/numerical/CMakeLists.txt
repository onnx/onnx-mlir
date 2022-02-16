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

# add_numerical_unittest(test_name sources... options...
#   This function (generally) has the same semantic as add_onnx_mlir_executable.
#   A test with test_name is added as a ctest to the numerical testsuite and
#   all the rest of the arguments are passed directly to add_onnx_mlir_executable.
#   The function usage is meant to look like a call to add_onnx_mlir_executable
#   for readability.
#   )
function(add_numerical_unittest test_name)
  add_onnx_mlir_executable(${test_name} NO_INSTALL ${ARGN})

  add_dependencies(numerical ${test_name})
  get_target_property(test_suite_folder numerical FOLDER)
  if (test_suite_folder)
    set_property(TARGET ${test_name} PROPERTY FOLDER "${test_suite_folder}")
  endif ()

  # Force -O3 when invoking the numerical tests, so that the models in the test
  # are compiled with -O3. TODO: make that a runtime env variable.
  add_test(NAME ${test_name} COMMAND ${test_name} -O3 WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})
  set_tests_properties(${test_name} PROPERTIES LABELS numerical)

  if (WIN32)
    # On Windows, we need a .def file to specify the export functions from a DLL.
    # The name of the .def file must match the name of the DLL being created, so
    # we follow the same naming convention as the tests (e.g. TestConv_main_graph)
    # (see SHARED_LIB_BASE in each of the test files).
    configure_file(
      ${CMAKE_CURRENT_SOURCE_DIR}/numerical.def
      ${CMAKE_CURRENT_BINARY_DIR}/${test_name}_main_graph.def
      COPYONLY
      )
  endif()
endfunction()

# All libraries and executables coming from llvm or ONNX-MLIR have had their
# compile flags updated via llvm_update_compile_flags, so we need to do that to
# rapidcheck as well, so that we can successfully link against it. Otherwise some
# of the flags for exceptions (among others) are not set correctly.
llvm_update_compile_flags(rapidcheck)

# The CompilerUtils ExecutionSession are also included in ModelLib,
# but it did not compile when I removed these two. TODO, figure out why.
set(TEST_LINK_LIBS rapidcheck ModelLib)

add_numerical_unittest(TestConv
  TestConv.cpp
  LINK_LIBS PRIVATE ${TEST_LINK_LIBS}
  )

add_numerical_unittest(TestMatMul2D
  TestMatMul2D.cpp
  LINK_LIBS PRIVATE ${TEST_LINK_LIBS}
  )

add_numerical_unittest(TestGemm
  TestGemm.cpp
  LINK_LIBS PRIVATE ${TEST_LINK_LIBS}
  )

add_numerical_unittest(TestLSTM
  TestLSTM.cpp
  LINK_LIBS PRIVATE ${TEST_LINK_LIBS}
  )

add_numerical_unittest(TestRNN
  TestRNN.cpp
  LINK_LIBS PRIVATE ${TEST_LINK_LIBS}
  )

add_numerical_unittest(TestGRU
  TestGRU.cpp
  LINK_LIBS PRIVATE ${TEST_LINK_LIBS}
  )

add_numerical_unittest(TestLoop
  TestLoop.cpp
  LINK_LIBS PRIVATE ${TEST_LINK_LIBS}
  )
