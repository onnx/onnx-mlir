# SPDX-License-Identifier: Apache-2.0

file(GENERATE
  OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/conftest.py
  INPUT ${CMAKE_CURRENT_SOURCE_DIR}/conftest.py
  )

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
  
configure_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/test_config_compilerlib.py.in
  ${CMAKE_CURRENT_BINARY_DIR}/test_config_compilerlib.py.cfg
  @ONLY
  )

file(GENERATE
  OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/$<CONFIG>/test_config_compilerlib.py
  INPUT ${CMAKE_CURRENT_BINARY_DIR}/test_config_compilerlib.py.cfg
  )

# CMAKE_CFG_INTDIR is . for single-config generators such as make, and
# it has a value (e.g. $(Configuration)) otherwise, so we can use it to
# determine whether we are dealing with a multi-config generator.
if (NOT "${CMAKE_CFG_INTDIR}" STREQUAL ".")
  set(FILE_GENERATE_DIR ${CMAKE_CURRENT_BINARY_DIR}/${CMAKE_CFG_INTDIR})
else()
  set(FILE_GENERATE_DIR ${CMAKE_CURRENT_BINARY_DIR}/${CMAKE_BUILD_TYPE})
endif()

# Detect pytest-xdist for parallel backend tests
execute_process(
  COMMAND ${Python3_EXECUTABLE} -m pip show pytest-xdist
  RESULT_VARIABLE PYTEST_XDIST_FOUND
  OUTPUT_QUIET
  ERROR_QUIET
)
if (${PYTEST_XDIST_FOUND} EQUAL 0)
  message(STATUS "Parallel backend tests      : ON")
  set(BACKEND_TEST_COMMAND "${Python3_EXECUTABLE}" "-m" "pytest")
  set(BACKEND_TEST_ARGS "--forked" "-n" "$$\{NPROC:-auto\}" "-q" "--silent")
else()
  message(STATUS "Parallel backend tests      : OFF (install pytest-xdist to enable)")
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
    ${BACKEND_TEST_ARGS} ${FILE_GENERATE_DIR}/test.py
  DEPENDS
    ${FILE_GENERATE_DIR}/test.py
    ${FILE_GENERATE_DIR}/test_config.py
   )

add_custom_target(check-onnx-backend-dynamic
  COMMAND
    TEST_DYNAMIC=true ${BACKEND_TEST_COMMAND}
    ${BACKEND_TEST_ARGS} ${FILE_GENERATE_DIR}/test.py
  DEPENDS
    ${FILE_GENERATE_DIR}/test.py
    ${FILE_GENERATE_DIR}/test_config.py
  )

add_custom_target(check-onnx-backend-constant
  COMMAND
  TEST_CONSTANT=true ${BACKEND_TEST_COMMAND}
    ${BACKEND_TEST_ARGS} ${FILE_GENERATE_DIR}/test.py
  DEPENDS
    ${FILE_GENERATE_DIR}/test.py
    ${FILE_GENERATE_DIR}/test_config.py
  )

add_custom_target(clean-onnx-backend
  COMMAND
    ${CMAKE_COMMAND} -E remove
    ${CMAKE_CURRENT_BINARY_DIR}/*.onnx
    ${CMAKE_CURRENT_BINARY_DIR}/*.so
  )
  
add_custom_target(check-onnx-backend-compilerlib
  COMMAND
    TEST_COMPILERLIB=true ONNX_HOME=${CMAKE_CURRENT_BINARY_DIR} ${BACKEND_TEST_COMMAND}
    ${BACKEND_TEST_ARGS} ${FILE_GENERATE_DIR}/test.py
  DEPENDS
    ${FILE_GENERATE_DIR}/test.py
    ${FILE_GENERATE_DIR}/test_config_compilerlib.py
   )

add_dependencies(check-onnx-backend onnx-mlir)
add_dependencies(check-onnx-backend PyRuntime)
add_dependencies(check-onnx-backend-dynamic onnx-mlir)
add_dependencies(check-onnx-backend-dynamic PyRuntime)
add_dependencies(check-onnx-backend-constant onnx-mlir)
add_dependencies(check-onnx-backend-constant PyRuntime)

add_dependencies(check-onnx-backend-compilerlib CompilerLibTest)
add_dependencies(check-onnx-backend-compilerlib PyRuntime)
