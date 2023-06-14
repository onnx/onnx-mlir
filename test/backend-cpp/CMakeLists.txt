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

add_dependencies(check-onnx-backend-numerical check-onnx-backend-cpp)

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
endfunction()

add_backend_cpp_unittest(TestCategoryMapper
  TestCategoryMapper.cpp

  LINK_LIBS PRIVATE
  ModelLib
  )
