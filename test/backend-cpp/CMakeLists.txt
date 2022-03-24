# SPDX-License-Identifier: Apache-2.0

add_custom_target(backend-cpp)
set_target_properties(backend-cpp PROPERTIES FOLDER "Tests")

add_custom_target(check-onnx-backend-cpp
  COMMENT "Running the ONNX-MLIR backend-cpp tests"
  COMMAND "${CMAKE_CTEST_COMMAND}" -L backend-cpp --output-on-failure -C $<CONFIG> --force-new-ctest-process
  USES_TERMINAL
  DEPENDS backend-cpp
  )
set_target_properties(check-onnx-backend-cpp PROPERTIES FOLDER "Tests")
# Exclude the target from the default VS build
set_target_properties(check-onnx-backend-cpp PROPERTIES EXCLUDE_FROM_DEFAULT_BUILD ON)

# add_unittest(test_name sources... options...
#   This function (generally) has the same semantic as add_onnx_mlir_executable.
#   A test with test_name is added as a ctest to the backend-cpp tests suite and
#   all the rest of the arguments are passed directly to add_onnx_mlir_executable.
#   The function usage is meant to look like a call to add_onnx_mlir_executable
#   for readability.
#   )
function(add_backend_cpp_unittest test_name)
  add_onnx_mlir_executable(${test_name} NO_INSTALL ${ARGN})

  add_dependencies(backend-cpp ${test_name})
  get_target_property(test_suite_folder backend-cpp FOLDER)
  if (test_suite_folder)
    set_property(TARGET ${test_name} PROPERTY FOLDER "${test_suite_folder}")
  endif ()

  add_test(NAME ${test_name} COMMAND ${test_name} WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})
  set_tests_properties(${test_name} PROPERTIES LABELS backend-cpp)

  if (WIN32)
    # On Windows, we need a .def file to specify the export functions from a DLL.
    # The name of the .def file must match the name of the DLL being created, so
    # we follow the same naming convention as the tests (e.g. TestCategoryMapper_main_graph)
    # (see SHARED_LIB_BASE in each of the test files).
    configure_file(
      ${CMAKE_CURRENT_SOURCE_DIR}/backend-cpp.def
      ${CMAKE_CURRENT_BINARY_DIR}/${test_name}_main_graph.def
      COPYONLY
      )
  endif()
endfunction()

set(TEST_LINK_LIBS CompilerUtils ExecutionSession)

add_backend_cpp_unittest(TestCategoryMapper
  ModelBuilder.cpp
  TestCategoryMapper.cpp

  LINK_LIBS PRIVATE ${TEST_LINK_LIBS}
  )
