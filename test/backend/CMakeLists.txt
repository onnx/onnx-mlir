# SPDX-License-Identifier: Apache-2.0

file(GENERATE
  OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/$<CONFIG>/test.py
  INPUT ${CMAKE_CURRENT_SOURCE_DIR}/test.py
  )

configure_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/test_config.py.in
  ${CMAKE_CURRENT_BINARY_DIR}/test_config.py.cfg
  @ONLY
  )

file(GENERATE
  OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/$<CONFIG>/test_config.py
  INPUT ${CMAKE_CURRENT_BINARY_DIR}/test_config.py.cfg
  )

# Detect pytest-xdist for parallel backend tests
execute_process(
  COMMAND ${Python3_EXECUTABLE} -m pip show pytest-xdist
  RESULT_VARIABLE PYTEST_XDIST_FOUND
  OUTPUT_QUIET
  ERROR_QUIET
)
if (${PYTEST_XDIST_FOUND} EQUAL 0)
  message(STATUS "Parallel backend tests  : ON")
  set(BACKEND_TEST_COMMAND "pytest")
  set(BACKEND_TEST_ARGS "--forked" "-n" "$$(nproc)")
else()
  message(STATUS "Parallel backend tests  : OFF (install pytest-xdist to enable)")
  set(BACKEND_TEST_COMMAND ${Python3_EXECUTABLE})
  set(BACKEND_TEST_ARGS "")
endif()

set(LD_PRELOADS $<TARGET_FILE:cruntime>)

get_filename_component(CXX_BIN_DIR ${CMAKE_CXX_COMPILER} PATH)
set(LIBSTDCXX_LIB "${CXX_BIN_DIR}/../lib64/libstdc++.so")
if(EXISTS ${LIBSTDCXX_LIB})
  set(LD_PRELOADS ${LIBSTDCXX_LIB}:${LD_PRELOADS})
endif()

add_custom_target(check-onnx-backend
  COMMAND
    ONNX_HOME=${CMAKE_CURRENT_BINARY_DIR} ${BACKEND_TEST_COMMAND}
    ${BACKEND_TEST_ARGS} ${CMAKE_CURRENT_BINARY_DIR}/${CMAKE_CFG_INTDIR}/test.py
  DEPENDS
    ${CMAKE_CURRENT_BINARY_DIR}/${CMAKE_CFG_INTDIR}/test.py
    ${CMAKE_CURRENT_BINARY_DIR}/${CMAKE_CFG_INTDIR}/test_config.py
   )

add_custom_target(check-onnx-backend-dynamic
  COMMAND
    TEST_DYNAMIC=true ${BACKEND_TEST_COMMAND}
    ${BACKEND_TEST_ARGS} ${CMAKE_CURRENT_BINARY_DIR}/${CMAKE_CFG_INTDIR}/test.py
  DEPENDS
    ${CMAKE_CURRENT_BINARY_DIR}/${CMAKE_CFG_INTDIR}/test.py
    ${CMAKE_CURRENT_BINARY_DIR}/${CMAKE_CFG_INTDIR}/test_config.py
  )

add_custom_target(clean-onnx-backend
  COMMAND
    ${CMAKE_COMMAND} -E remove
    ${CMAKE_CURRENT_BINARY_DIR}/*.onnx
    ${CMAKE_CURRENT_BINARY_DIR}/*.so
  )

add_dependencies(check-onnx-backend onnx-mlir)
add_dependencies(check-onnx-backend PyRuntime)
add_dependencies(check-onnx-backend-dynamic onnx-mlir)
add_dependencies(check-onnx-backend-dynamic PyRuntime)
