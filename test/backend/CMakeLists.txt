# SPDX-License-Identifier: Apache-2.0

file(GENERATE
  OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/conftest.py
  INPUT ${CMAKE_CURRENT_SOURCE_DIR}/conftest.py
  )

file(GENERATE
  OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/$<CONFIG>/test.py
  INPUT ${CMAKE_CURRENT_SOURCE_DIR}/test.py
  )

file(GENERATE
  OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/$<CONFIG>/inference_backend.py
  INPUT ${CMAKE_CURRENT_SOURCE_DIR}/inference_backend.py
  )

file(GENERATE
  OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/$<CONFIG>/signature_backend.py
  INPUT ${CMAKE_CURRENT_SOURCE_DIR}/signature_backend.py
  )

file(GENERATE
  OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/$<CONFIG>/input_verification_backend.py
  INPUT ${CMAKE_CURRENT_SOURCE_DIR}/input_verification_backend.py
  )

file(GENERATE
  OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/$<CONFIG>/variables.py
  INPUT ${CMAKE_CURRENT_SOURCE_DIR}/variables.py
  )

file(GENERATE
  OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/$<CONFIG>/common.py
  INPUT ${CMAKE_CURRENT_SOURCE_DIR}/common.py
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
  message(STATUS "Parallel backend tests   : ON")
  set(BACKEND_TEST_COMMAND "${Python3_EXECUTABLE}" "-m" "pytest")
  set(BACKEND_TEST_ARGS "--forked" "-n" "$$\{NPROC:-auto\}" "-q" "--silent")
else()
  message(STATUS "Parallel backend tests   : OFF (install pytest-xdist to enable)")
  set(BACKEND_TEST_COMMAND ${Python3_EXECUTABLE})
  set(BACKEND_TEST_ARGS "")
endif()

include(setup_model_download.cmake)

# ${ONNX_HOME} is the directory where onnx downloads real model files.
# Model files are saved under ${ONNX_HOME}/models/model_name/model.onnx.
# C/C++ and JNI tests run in parallel so they must use a different
# ONNX_HOME to avoid conflicts.
#
# We now have our own model download targets and different backend tests
# share the pre-downloaded model files. So we no longer need a different
# ONNX_HOME for C/C++ and JNI tests.
#
# We can either set ONNX_HOME=${FILE_GENERATE_DIR} or we can set
# ONNX_MODELS=${FILE_GENERATE_DIR}/models (more flexible since we can
# change the "models" subdirectory to anything we like)
add_custom_target(check-onnx-backend
  COMMAND
    ONNX_MODELS=${FILE_GENERATE_DIR}/models
    ${BACKEND_TEST_COMMAND} ${BACKEND_TEST_ARGS} ${FILE_GENERATE_DIR}/test.py
  DEPENDS
    ${FILE_GENERATE_DIR}/test.py
    ${FILE_GENERATE_DIR}/test_config.py
  )

add_custom_target(check-onnx-backend-dynamic
  COMMAND
    ONNX_MODELS=${FILE_GENERATE_DIR}/models
    TEST_DYNAMIC=true
    ${BACKEND_TEST_COMMAND} ${BACKEND_TEST_ARGS} ${FILE_GENERATE_DIR}/test.py
  DEPENDS
    ${FILE_GENERATE_DIR}/test.py
    ${FILE_GENERATE_DIR}/test_config.py
  )

add_custom_target(check-onnx-backend-constant
  COMMAND
    ONNX_MODELS=${FILE_GENERATE_DIR}/models
    TEST_CONSTANT=true
    ${BACKEND_TEST_COMMAND} ${BACKEND_TEST_ARGS} ${FILE_GENERATE_DIR}/test.py
  DEPENDS
    ${FILE_GENERATE_DIR}/test.py
    ${FILE_GENERATE_DIR}/test_config.py
  )

add_custom_target(check-onnx-backend-signature
  COMMAND
    TEST_SIGNATURE=true
    ${BACKEND_TEST_COMMAND} ${BACKEND_TEST_ARGS} ${FILE_GENERATE_DIR}/test.py
  DEPENDS
    ${FILE_GENERATE_DIR}/test.py
    ${FILE_GENERATE_DIR}/test_config.py
  )

# Generate all the backend test cases for CPU provided by the ONNX package.
# The result is recorded in file all_test_names.txt
add_custom_target(check-onnx-backend-case
  COMMAND
    TEST_CASE_CHECK=true
    ${BACKEND_TEST_COMMAND} ${BACKEND_TEST_ARGS} ${FILE_GENERATE_DIR}/test.py
  DEPENDS
    ${FILE_GENERATE_DIR}/test.py
    ${FILE_GENERATE_DIR}/test_config.py
  )

# Do not use pytest-xdist for this check because pytest-xdist swallows stdout
# and this test cannot check messages from stdout.
add_custom_target(check-onnx-backend-input-verification
  COMMAND
    TEST_INPUT_VERIFICATION=true
    ${Python3_EXECUTABLE} ${FILE_GENERATE_DIR}/test.py --input_verification
  DEPENDS
    ${FILE_GENERATE_DIR}/test.py
    ${FILE_GENERATE_DIR}/test_config.py
  )

add_custom_target(check-onnx-backend-compilerlib
  COMMAND
    TEST_COMPILERLIB=true ONNX_HOME=${CMAKE_CURRENT_BINARY_DIR}
    ${BACKEND_TEST_COMMAND} ${BACKEND_TEST_ARGS} ${FILE_GENERATE_DIR}/test.py
  DEPENDS
    ${FILE_GENERATE_DIR}/test.py
    ${FILE_GENERATE_DIR}/test_config_compilerlib.py
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
add_dependencies(check-onnx-backend-constant onnx-mlir)
add_dependencies(check-onnx-backend-constant PyRuntime)
add_dependencies(check-onnx-backend-signature onnx-mlir)
add_dependencies(check-onnx-backend-signature PyRuntime)
add_dependencies(check-onnx-backend-input-verification onnx-mlir)
add_dependencies(check-onnx-backend-input-verification PyRuntime)
add_dependencies(check-onnx-backend-compilerlib CompilerLibTest)
add_dependencies(check-onnx-backend-compilerlib PyRuntime)

setup_model_download(check-onnx-backend "")
setup_model_download(check-onnx-backend-dynamic --dynamic)
setup_model_download(check-onnx-backend-constant --constant)

add_dependencies(check-onnx-backend-numerical check-onnx-backend)
add_dependencies(check-onnx-backend-numerical check-onnx-backend-dynamic)
add_dependencies(check-onnx-backend-numerical check-onnx-backend-constant)
add_dependencies(check-onnx-backend-numerical check-onnx-backend-signature)
add_dependencies(check-onnx-backend-numerical check-onnx-backend-input-verification)

if (ONNX_MLIR_ENABLE_JNI)
  message(STATUS "JNI backend tests        : ON")
  message(STATUS "JSONITER_JAR             : ${JSONITER_JAR}")
  add_custom_target(check-onnx-backend-jni
    COMMAND
      ONNX_MODELS=${FILE_GENERATE_DIR}/models
      TEST_EMIT=jni JSONITER_JAR=${JSONITER_JAR}
      ${BACKEND_TEST_COMMAND} ${BACKEND_TEST_ARGS} ${FILE_GENERATE_DIR}/test.py
    DEPENDS
      ${FILE_GENERATE_DIR}/test.py
      ${FILE_GENERATE_DIR}/test_config.py
    )

  add_custom_target(check-onnx-backend-dynamic-jni
    COMMAND
      ONNX_MODELS=${FILE_GENERATE_DIR}/models
      TEST_DYNAMIC=true TEST_EMIT=jni JSONITER_JAR=${JSONITER_JAR}
      ${BACKEND_TEST_COMMAND} ${BACKEND_TEST_ARGS} ${FILE_GENERATE_DIR}/test.py
    DEPENDS
      ${FILE_GENERATE_DIR}/test.py
      ${FILE_GENERATE_DIR}/test_config.py
    )

  add_custom_target(check-onnx-backend-constant-jni
    COMMAND
      ONNX_MODELS=${FILE_GENERATE_DIR}/models
      TEST_CONSTANT=true TEST_EMIT=jni JSONITER_JAR=${JSONITER_JAR}
      ${BACKEND_TEST_COMMAND} ${BACKEND_TEST_ARGS} ${FILE_GENERATE_DIR}/test.py
    DEPENDS
      ${FILE_GENERATE_DIR}/test.py
      ${FILE_GENERATE_DIR}/test_config.py
    )

  add_dependencies(check-onnx-backend-jni onnx-mlir)
  add_dependencies(check-onnx-backend-jni PyRuntime)
  add_dependencies(check-onnx-backend-jni javaruntime)
  add_dependencies(check-onnx-backend-jni jniruntime)
  add_dependencies(check-onnx-backend-dynamic-jni onnx-mlir)
  add_dependencies(check-onnx-backend-dynamic-jni PyRuntime)
  add_dependencies(check-onnx-backend-dynamic-jni javaruntime)
  add_dependencies(check-onnx-backend-dynamic-jni jniruntime)
  add_dependencies(check-onnx-backend-constant-jni onnx-mlir)
  add_dependencies(check-onnx-backend-constant-jni PyRuntime)
  add_dependencies(check-onnx-backend-constant-jni javaruntime)
  add_dependencies(check-onnx-backend-constant-jni jniruntime)

  setup_model_download(check-onnx-backend-jni "")
  setup_model_download(check-onnx-backend-dynamic-jni --dynamic)
  setup_model_download(check-onnx-backend-constant-jni --constant)

  add_dependencies(check-onnx-backend-numerical check-onnx-backend-jni)
  add_dependencies(check-onnx-backend-numerical check-onnx-backend-dynamic-jni)
  add_dependencies(check-onnx-backend-numerical check-onnx-backend-constant-jni)

else()
  message(STATUS "JNI backend tests         : OFF")
endif()
