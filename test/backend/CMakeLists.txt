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
  OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/$<CONFIG>/signature_backend.py
  INPUT ${CMAKE_CURRENT_SOURCE_DIR}/signature_backend.py
  )

file(GENERATE
  OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/$<CONFIG>/input_verification_backend.py
  INPUT ${CMAKE_CURRENT_SOURCE_DIR}/input_verification_backend.py
  )

file(GENERATE
  OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/$<CONFIG>/inference_backend.py
  INPUT ${CMAKE_CURRENT_SOURCE_DIR}/inference_backend.py
  )

file(GENERATE
  OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/$<CONFIG>/onnxmlir_node_tests.py
  INPUT ${CMAKE_CURRENT_SOURCE_DIR}/onnxmlir_node_tests.py
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
  set(BACKEND_TEST_ARGS "-n" "$$\{NPROC:-auto\}" "-q" "--silent")
else()
  message(STATUS "Parallel backend tests   : OFF (install pytest-xdist to enable)")
  set(BACKEND_TEST_COMMAND ${Python3_EXECUTABLE})
  set(BACKEND_TEST_ARGS "")
endif()

# ONNX repo now includes the "light" models for the backend test so
# models are no longer downloaded separately. The light models are
# installed along with the ONNX code. ONNX test runner will load
# the light models from the installation and generate input/output
# files in directories controlled by ONNX_HOME and ONNX_MODELS.
#
# Our test.py writes out the light models out to files, just like
# the node models, so they can be compiled by onnx-mlir.
#
# Since multiple check-onnx-??? targets are running concurrently
# and they may be compiling the same models, each target needs its
# own directory to hold the models and input/output files to avoid
# stepping on each other's toes.
add_custom_target(check-onnx-backend
  COMMAND
    ONNX_HOME=${FILE_GENERATE_DIR}/check-onnx-backend
    ${BACKEND_TEST_COMMAND} ${BACKEND_TEST_ARGS} ${FILE_GENERATE_DIR}/test.py
  DEPENDS
    ${FILE_GENERATE_DIR}/test.py
    ${FILE_GENERATE_DIR}/test_config.py
  )

add_custom_target(check-onnx-backend-dynamic
  COMMAND
    ONNX_HOME=${FILE_GENERATE_DIR}/check-onnx-backend-dynamic
    TEST_DYNAMIC=true
    ${BACKEND_TEST_COMMAND} ${BACKEND_TEST_ARGS} ${FILE_GENERATE_DIR}/test.py
  DEPENDS
    ${FILE_GENERATE_DIR}/test.py
    ${FILE_GENERATE_DIR}/test_config.py
  )

add_custom_target(check-onnx-backend-constant
  COMMAND
    ONNX_HOME=${FILE_GENERATE_DIR}/check-onnx-backend-constant
    TEST_CONSTANT=true
    ${BACKEND_TEST_COMMAND} ${BACKEND_TEST_ARGS} ${FILE_GENERATE_DIR}/test.py
  DEPENDS
    ${FILE_GENERATE_DIR}/test.py
    ${FILE_GENERATE_DIR}/test_config.py
  )

add_custom_target(check-onnx-backend-node
  COMMAND
    ONNX_HOME=${FILE_GENERATE_DIR}/check-onnx-backend-node
    TEST_TYPE=node
    ${BACKEND_TEST_COMMAND} ${BACKEND_TEST_ARGS} ${FILE_GENERATE_DIR}/test.py
  DEPENDS
    ${FILE_GENERATE_DIR}/test.py
    ${FILE_GENERATE_DIR}/test_config.py
  )

add_custom_target(check-onnx-backend-model
  COMMAND
    ONNX_HOME=${FILE_GENERATE_DIR}/check-onnx-backend-model
    TEST_TYPE=model
    ${BACKEND_TEST_COMMAND} ${BACKEND_TEST_ARGS} ${FILE_GENERATE_DIR}/test.py
  DEPENDS
    ${FILE_GENERATE_DIR}/test.py
    ${FILE_GENERATE_DIR}/test_config.py
  )

IF (NOT WIN32)
add_custom_target(check-onnx-backend-constants-to-file
  COMMAND
    ONNX_HOME=${FILE_GENERATE_DIR}/check-onnx-backend-constants-to-file
    TEST_CONSTANTS_TO_FILE=true
    ${BACKEND_TEST_COMMAND} ${BACKEND_TEST_ARGS} ${FILE_GENERATE_DIR}/test.py
  DEPENDS
    ${FILE_GENERATE_DIR}/test.py
    ${FILE_GENERATE_DIR}/test_config.py
  )
ENDIF()

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
    ${Python3_EXECUTABLE} ${FILE_GENERATE_DIR}/test.py
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
    ${CMAKE_CURRENT_BINARY_DIR}/*.constants.bin
  )

add_dependencies(check-onnx-backend onnx-mlir)
add_dependencies(check-onnx-backend PyRuntimeC)
add_dependencies(check-onnx-backend-dynamic onnx-mlir)
add_dependencies(check-onnx-backend-dynamic PyRuntimeC)
add_dependencies(check-onnx-backend-constant onnx-mlir)
add_dependencies(check-onnx-backend-constant PyRuntimeC)
IF (NOT WIN32)
add_dependencies(check-onnx-backend-constants-to-file onnx-mlir)
add_dependencies(check-onnx-backend-constants-to-file PyRuntimeC)
ENDIF()
add_dependencies(check-onnx-backend-node onnx-mlir)
add_dependencies(check-onnx-backend-node PyRuntimeC)
add_dependencies(check-onnx-backend-model onnx-mlir)
add_dependencies(check-onnx-backend-model PyRuntimeC)
add_dependencies(check-onnx-backend-signature onnx-mlir)
add_dependencies(check-onnx-backend-signature PyRuntimeC)
add_dependencies(check-onnx-backend-case PyRuntimeC)
add_dependencies(check-onnx-backend-input-verification onnx-mlir)
add_dependencies(check-onnx-backend-input-verification PyRuntimeC)
add_dependencies(check-onnx-backend-compilerlib CompilerLibTest)
add_dependencies(check-onnx-backend-compilerlib PyRuntimeC)

add_dependencies(check-onnx-backend-numerical check-onnx-backend)
add_dependencies(check-onnx-backend-numerical check-onnx-backend-dynamic)
add_dependencies(check-onnx-backend-numerical check-onnx-backend-constant)
IF (NOT WIN32)
add_dependencies(check-onnx-backend-numerical check-onnx-backend-constants-to-file)
ENDIF()
add_dependencies(check-onnx-backend-numerical check-onnx-backend-signature)
add_dependencies(check-onnx-backend-numerical check-onnx-backend-input-verification)

if (ONNX_MLIR_ENABLE_JNI)
  message(STATUS "JNI backend tests        : ON")
  message(STATUS "JSONITER_JAR             : ${JSONITER_JAR}")
  add_custom_target(check-onnx-backend-jni
    COMMAND
      ONNX_HOME=${FILE_GENERATE_DIR}/check-onnx-backend-jni
      TEST_EMIT=jni JSONITER_JAR=${JSONITER_JAR}
      ${BACKEND_TEST_COMMAND} ${BACKEND_TEST_ARGS} ${FILE_GENERATE_DIR}/test.py
    DEPENDS
      ${FILE_GENERATE_DIR}/test.py
      ${FILE_GENERATE_DIR}/test_config.py
    )

  add_custom_target(check-onnx-backend-dynamic-jni
    COMMAND
      ONNX_HOME=${FILE_GENERATE_DIR}/check-onnx-backend-dynamic-jni
      TEST_DYNAMIC=true TEST_EMIT=jni JSONITER_JAR=${JSONITER_JAR}
      ${BACKEND_TEST_COMMAND} ${BACKEND_TEST_ARGS} ${FILE_GENERATE_DIR}/test.py
    DEPENDS
      ${FILE_GENERATE_DIR}/test.py
      ${FILE_GENERATE_DIR}/test_config.py
    )

  add_custom_target(check-onnx-backend-constant-jni
    COMMAND
      ONNX_HOME=${FILE_GENERATE_DIR}/check-onnx-backend-constant-jni
      TEST_CONSTANT=true TEST_EMIT=jni JSONITER_JAR=${JSONITER_JAR}
      ${BACKEND_TEST_COMMAND} ${BACKEND_TEST_ARGS} ${FILE_GENERATE_DIR}/test.py
    DEPENDS
      ${FILE_GENERATE_DIR}/test.py
      ${FILE_GENERATE_DIR}/test_config.py
    )

  add_custom_target(check-onnx-backend-node-jni
    COMMAND
      ONNX_HOME=${FILE_GENERATE_DIR}/check-onnx-backend-node-jni
      TEST_TYPE=node TEST_EMIT=jni JSONITER_JAR=${JSONITER_JAR}
      ${BACKEND_TEST_COMMAND} ${BACKEND_TEST_ARGS} ${FILE_GENERATE_DIR}/test.py
    DEPENDS
      ${FILE_GENERATE_DIR}/test.py
      ${FILE_GENERATE_DIR}/test_config.py
    )

  add_custom_target(check-onnx-backend-model-jni
    COMMAND
      ONNX_HOME=${FILE_GENERATE_DIR}/check-onnx-backend-model-jni
      TEST_TYPE=model TEST_EMIT=jni JSONITER_JAR=${JSONITER_JAR}
      ${BACKEND_TEST_COMMAND} ${BACKEND_TEST_ARGS} ${FILE_GENERATE_DIR}/test.py
    DEPENDS
      ${FILE_GENERATE_DIR}/test.py
      ${FILE_GENERATE_DIR}/test_config.py
    )

  add_dependencies(check-onnx-backend-jni onnx-mlir)
  add_dependencies(check-onnx-backend-jni PyRuntimeC)
  add_dependencies(check-onnx-backend-jni javaruntime)
  add_dependencies(check-onnx-backend-jni jniruntime)
  add_dependencies(check-onnx-backend-dynamic-jni onnx-mlir)
  add_dependencies(check-onnx-backend-dynamic-jni PyRuntimeC)
  add_dependencies(check-onnx-backend-dynamic-jni javaruntime)
  add_dependencies(check-onnx-backend-dynamic-jni jniruntime)
  add_dependencies(check-onnx-backend-constant-jni onnx-mlir)
  add_dependencies(check-onnx-backend-constant-jni PyRuntimeC)
  add_dependencies(check-onnx-backend-constant-jni javaruntime)
  add_dependencies(check-onnx-backend-constant-jni jniruntime)
  add_dependencies(check-onnx-backend-node-jni onnx-mlir)
  add_dependencies(check-onnx-backend-node-jni PyRuntimeC)
  add_dependencies(check-onnx-backend-node-jni javaruntime)
  add_dependencies(check-onnx-backend-node-jni jniruntime)
  add_dependencies(check-onnx-backend-model-jni onnx-mlir)
  add_dependencies(check-onnx-backend-model-jni PyRuntimeC)
  add_dependencies(check-onnx-backend-model-jni javaruntime)
  add_dependencies(check-onnx-backend-model-jni jniruntime)

  add_dependencies(check-onnx-backend-numerical check-onnx-backend-jni)
  add_dependencies(check-onnx-backend-numerical check-onnx-backend-dynamic-jni)
  add_dependencies(check-onnx-backend-numerical check-onnx-backend-constant-jni)

else()
  message(STATUS "JNI backend tests         : OFF")
endif()
