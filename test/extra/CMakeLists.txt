# SPDX-License-Identifier: Apache-2.0

add_custom_target(extra)
set_target_properties(extra PROPERTIES FOLDER "Tests")

add_custom_target(check-onnx-extra
  COMMENT "Running the ONNX-MLIR extra unit tests"
  COMMAND "${CMAKE_CTEST_COMMAND}" -L extra --output-on-failure -C $<CONFIG> --force-new-ctest-process
  USES_TERMINAL
  DEPENDS extra
  )
set_target_properties(check-onnx-extra PROPERTIES FOLDER "Tests")
# Exclude the target from the default VS build
set_target_properties(check-onnx-extra PROPERTIES EXCLUDE_FROM_DEFAULT_BUILD ON)

# add_unittest(test_name sources... options...
#   This function (generally) has the same semantic as add_onnx_mlir_executable.
#   A test with test_name is added as a ctest to the extra unit tests suite and
#   all the rest of the arguments are passed directly to add_onnx_mlir_executable.
#   The function usage is meant to look like a call to add_onnx_mlir_executable
#   for readability.
#   )
function(add_extra_unittest test_name)
  add_onnx_mlir_executable(${test_name} NO_INSTALL ${ARGN})

  add_dependencies(extra ${test_name})
  get_target_property(test_suite_folder extra FOLDER)
  if (test_suite_folder)
    set_property(TARGET ${test_name} PROPERTY FOLDER "${test_suite_folder}")
  endif ()

  add_test(NAME ${test_name} COMMAND ${test_name} WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})
  set_tests_properties(${test_name} PROPERTIES LABELS extra)

  if (WIN32)
    # On Windows, we need a .def file to specify the export functions from a DLL.
    # The name of the .def file must match the name of the DLL being created, so
    # we follow the same naming convention as the tests (e.g. TestCategoryMapper_main_graph)
    # (see SHARED_LIB_BASE in each of the test files).
    configure_file(
      ${CMAKE_CURRENT_SOURCE_DIR}/extra.def
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

set(TEST_LINK_LIBS rapidcheck CompilerUtils ExecutionSession)

add_extra_unittest(TestCategoryMapper
  TestHelper.cpp
  TestCategoryMapper.cpp

  LINK_LIBS PRIVATE ${TEST_LINK_LIBS}
  )
