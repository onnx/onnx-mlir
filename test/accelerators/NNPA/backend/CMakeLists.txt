# SPDX-License-Identifier: Apache-2.0

set(ONNX_BACKENDTEST_SRC_DIR ${ONNX_MLIR_SRC_ROOT}/test/backend)
set(ONNX_BACKENDTEST_BIN_DIR ${ONNX_MLIR_BIN_ROOT}/test/backend)

file(GENERATE
  OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/conftest.py
  INPUT ${ONNX_BACKENDTEST_SRC_DIR}/conftest.py
  )

file(GENERATE
  OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/$<CONFIG>/test_nnpa.py
  INPUT ${CMAKE_CURRENT_SOURCE_DIR}/test_nnpa.py
  )

file(GENERATE
  OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/$<CONFIG>/inference_backend_nnpa.py
  INPUT ${CMAKE_CURRENT_SOURCE_DIR}/inference_backend_nnpa.py
  )

file(GENERATE
  OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/$<CONFIG>/signature_backend.py
  INPUT ${ONNX_BACKENDTEST_SRC_DIR}/signature_backend.py
  )

file(GENERATE
  OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/$<CONFIG>/variables.py
  INPUT ${ONNX_BACKENDTEST_SRC_DIR}/variables.py
  )

file(GENERATE
  OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/$<CONFIG>/common_nnpa.py
  INPUT ${CMAKE_CURRENT_SOURCE_DIR}/common_nnpa.py
  )

configure_file(
 ${ONNX_BACKENDTEST_SRC_DIR}/test_config.py.in
 ${ONNX_BACKENDTEST_BIN_DIR}/test_config.py.cfg
 @ONLY
 )

file(GENERATE
 OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/$<CONFIG>/test_config.py
 INPUT ${ONNX_BACKENDTEST_BIN_DIR}/test_config.py.cfg
 )
  
configure_file(
  ${ONNX_BACKENDTEST_SRC_DIR}/test_config_compilerlib.py.in
  ${ONNX_BACKENDTEST_BIN_DIR}/test_config_compilerlib.py.cfg
  @ONLY
  )

file(GENERATE
 OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/$<CONFIG>/test_config_compilerlib.py
 INPUT ${ONNX_BACKENDTEST_BIN_DIR}/test_config_compilerlib.py.cfg
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
  message(STATUS "  Parallel backend tests    : ON")
  set(BACKEND_TEST_COMMAND "${Python3_EXECUTABLE}" "-m" "pytest")
  set(BACKEND_TEST_ARGS "--forked" "-n" "$$\{NPROC:-auto\}" "-q" "--silent")
else()
  message(STATUS "  Parallel backend tests    : OFF (install pytest-xdist to enable)")
  set(BACKEND_TEST_COMMAND ${Python3_EXECUTABLE})
  set(BACKEND_TEST_ARGS "")
endif()

set(LD_PRELOADS $<TARGET_FILE:cruntime>)

get_filename_component(CXX_BIN_DIR ${CMAKE_CXX_COMPILER} PATH)
set(LIBSTDCXX_LIB "${CXX_BIN_DIR}/../lib64/libstdc++.so")
if (EXISTS ${LIBSTDCXX_LIB})
  set(LD_PRELOADS ${LIBSTDCXX_LIB}:${LD_PRELOADS})
endif()

set(ZDNN_LIBRARY_DIR $ENV{ZDNN_LIBRARY_DIR})
set(ZDNN_LIB "${ZDNN_LIBRARY_DIR}/libzdnn.so")

# ${ONNX_HOME} is the directory where onnx downloads real model files.
# Model files are saved under ${ONNX_HOME}/models/model_name/model.onnx.
# C/C++ and JNI tests run in parallel so they must use a different
# ONNX_HOME to avoid conflicts.
add_custom_target(check-onnx-backend-nnpa
  COMMAND
    ONNX_HOME=${CMAKE_CURRENT_BINARY_DIR}/cpp
    LD_PRELOAD=${ZDNN_LIB} ${BACKEND_TEST_COMMAND} ${BACKEND_TEST_ARGS} ${FILE_GENERATE_DIR}/test_nnpa.py
  DEPENDS
    ${FILE_GENERATE_DIR}/test_nnpa.py
    ${FILE_GENERATE_DIR}/test_config.py
   )

add_custom_target(check-onnx-backend-nnpa-dynamic
  COMMAND
    TEST_DYNAMIC=true
    LD_PRELOAD=${ZDNN_LIB} ${BACKEND_TEST_COMMAND} ${BACKEND_TEST_ARGS} ${FILE_GENERATE_DIR}/test_nnpa.py
  DEPENDS
    ${FILE_GENERATE_DIR}/test_nnpa.py
    ${FILE_GENERATE_DIR}/test_config.py
  )

add_custom_target(check-onnx-backend-nnpa-constant
  COMMAND
    TEST_CONSTANT=true
    ${BACKEND_TEST_COMMAND} ${BACKEND_TEST_ARGS} ${FILE_GENERATE_DIR}/test_nnpa.py
  DEPENDS
    ${FILE_GENERATE_DIR}/test_nnpa.py
    ${FILE_GENERATE_DIR}/test_config.py
  )

add_custom_target(check-onnx-backend-nnpa-signature
  COMMAND
    TEST_SIGNATURE=true
    ${BACKEND_TEST_COMMAND} ${BACKEND_TEST_ARGS} ${FILE_GENERATE_DIR}/test_nnpa.py
  DEPENDS
    ${FILE_GENERATE_DIR}/test_nnpa.py
    ${FILE_GENERATE_DIR}/test_config.py
  )

add_custom_target(check-onnx-backend-nnpa-compilerlib
  COMMAND
    TEST_COMPILERLIB=true ONNX_HOME=${CMAKE_CURRENT_BINARY_DIR}
    ${BACKEND_TEST_COMMAND} ${BACKEND_TEST_ARGS} ${FILE_GENERATE_DIR}/test_nnpa.py
  DEPENDS
    ${FILE_GENERATE_DIR}/test_nnpa.py
    ${FILE_GENERATE_DIR}/test_config_compilerlib.py
   )

add_custom_target(clean-onnx-backend-nnpa
  COMMAND
    ${CMAKE_COMMAND} -E remove
    ${CMAKE_CURRENT_BINARY_DIR}/*.onnx
    ${CMAKE_CURRENT_BINARY_DIR}/*.so
  )

add_dependencies(check-onnx-backend-nnpa onnx-mlir)
add_dependencies(check-onnx-backend-nnpa PyRuntime)
add_dependencies(check-onnx-backend-nnpa-dynamic onnx-mlir)
add_dependencies(check-onnx-backend-nnpa-dynamic PyRuntime)
add_dependencies(check-onnx-backend-nnpa-constant onnx-mlir)
add_dependencies(check-onnx-backend-nnpa-constant PyRuntime)
add_dependencies(check-onnx-backend-nnpa-signature onnx-mlir)
add_dependencies(check-onnx-backend-nnpa-signature PyRuntime)
add_dependencies(check-onnx-backend-nnpa-compilerlib CompilerLibTest)
add_dependencies(check-onnx-backend-nnpa-compilerlib PyRuntime)

if (ONNX_MLIR_ENABLE_JNI)
  message(STATUS "  JNI backend tests         : ON")
  message(STATUS "  JSONITER_JAR              : ${JSONITER_JAR}")
  add_custom_target(check-onnx-backend-nnpa-jni
    COMMAND
      ONNX_HOME=${CMAKE_CURRENT_BINARY_DIR}/jni
       TEST_EMIT=jni JSONITER_JAR=${JSONITER_JAR}
      ${BACKEND_TEST_COMMAND} ${BACKEND_TEST_ARGS} ${FILE_GENERATE_DIR}/test_nnpa.py
    DEPENDS
      ${FILE_GENERATE_DIR}/test_nnpa.py
      ${FILE_GENERATE_DIR}/test_config.py
    )

  add_custom_target(check-onnx-backend-nnpa-dynamic-jni
    COMMAND
      TEST_DYNAMIC=true TEST_EMIT=jni JSONITER_JAR=${JSONITER_JAR}
      ${BACKEND_TEST_COMMAND} ${BACKEND_TEST_ARGS} ${FILE_GENERATE_DIR}/test_nnpa.py
    DEPENDS
      ${FILE_GENERATE_DIR}/test_nnpa.py
      ${FILE_GENERATE_DIR}/test_config.py
    )

  add_custom_target(check-onnx-backend-nnpa-constant-jni
    COMMAND
      TEST_CONSTANT=true TEST_EMIT=jni JSONITER_JAR=${JSONITER_JAR}
      ${BACKEND_TEST_COMMAND} ${BACKEND_TEST_ARGS} ${FILE_GENERATE_DIR}/test_nnpa.py
    DEPENDS
      ${FILE_GENERATE_DIR}/test_nnpa.py
      ${FILE_GENERATE_DIR}/test_config.py
    )

  add_dependencies(check-onnx-backend-nnpa-jni onnx-mlir)
  add_dependencies(check-onnx-backend-nnpa-jni PyRuntime)
  add_dependencies(check-onnx-backend-nnpa-jni javaruntime)
  add_dependencies(check-onnx-backend-nnpa-jni jniruntime)
  add_dependencies(check-onnx-backend-nnpa-dynamic-jni onnx-mlir)
  add_dependencies(check-onnx-backend-nnpa-dynamic-jni PyRuntime)
  add_dependencies(check-onnx-backend-nnpa-dynamic-jni javaruntime)
  add_dependencies(check-onnx-backend-nnpa-dynamic-jni jniruntime)
  add_dependencies(check-onnx-backend-nnpa-constant-jni onnx-mlir)
  add_dependencies(check-onnx-backend-nnpa-constant-jni PyRuntime)
  add_dependencies(check-onnx-backend-nnpa-constant-jni javaruntime)
  add_dependencies(check-onnx-backend-nnpa-constant-jni jniruntime)
else()
  message(STATUS "  JNI backend-nnpa tests         : OFF")
endif()
