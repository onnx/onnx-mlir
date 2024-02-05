# SPDX-License-Identifier: Apache-2.0

add_custom_target(unittest)
set_target_properties(unittest PROPERTIES FOLDER "Tests")

add_custom_target(check-unittest
  COMMENT "Running the ONNX-MLIR unit tests"
  COMMAND "${CMAKE_CTEST_COMMAND}" -L unittest --output-on-failure -C $<CONFIG> --force-new-ctest-process
  USES_TERMINAL
  DEPENDS unittest
  )
set_target_properties(check-unittest PROPERTIES FOLDER "Tests")
# Exclude the target from the default VS build
set_target_properties(check-unittest PROPERTIES EXCLUDE_FROM_DEFAULT_BUILD ON)

# add_unittest(test_name sources... options...
#   This function (generally) has the same semantic as add_onnx_mlir_executable.
#   A test with test_name is added as a ctest to the unittest testsuite and
#   all the rest of the arguments are passed directly to add_onnx_mlir_executable.
#   The function usage is meant to look like a call to add_onnx_mlir_executable
#   for readability.
#   )
function(add_unittest test_name)
  add_onnx_mlir_executable(${test_name} NO_INSTALL ${ARGN})

  add_dependencies(unittest ${test_name})
  get_target_property(test_suite_folder unittest FOLDER)
  if (test_suite_folder)
    set_property(TARGET ${test_name} PROPERTY FOLDER "${test_suite_folder}")
  endif ()

  # Optimization level set by ONNX_MLIR_TEST_OPTLEVEL, defaults to 3
  add_test(NAME ${test_name} COMMAND ${test_name})
  set_tests_properties(${test_name} PROPERTIES LABELS unittest)
endfunction()

add_unittest(TestInstrumentation
  TestInstrumentation.cpp

  INCLUDE_DIRS PUBLIC
  ${ONNX_MLIR_SRC_ROOT}/include

  LINK_LIBS PRIVATE
  cruntime
  )

add_subdirectory(BType)
add_subdirectory(CustomFn)
add_subdirectory(DisposableElementsAttr)
add_subdirectory(Einsum)
add_subdirectory(Runtime)
add_subdirectory(SmallFP)
add_subdirectory(Strides)
