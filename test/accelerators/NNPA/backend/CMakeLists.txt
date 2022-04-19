# SPDX-License-Identifier: Apache-2.0

set(ONNX_BACKENDTEST_SRC_DIR ${ONNX_MLIR_SRC_ROOT}/test/backend)
set(ONNX_BACKENDTEST_BIN_DIR ${ONNX_MLIR_BIN_ROOT}/test/backend)

file(GENERATE
  OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/conftest.py
  INPUT ${ONNX_BACKENDTEST_SRC_DIR}/conftest.py
  )

file(GENERATE
  OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/$<CONFIG>/test.py
  INPUT ${ONNX_BACKENDTEST_SRC_DIR}/test.py
  )

file(GENERATE
  OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/$<CONFIG>/inference_backend.py
  INPUT ${ONNX_BACKENDTEST_SRC_DIR}/inference_backend.py
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
  OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/$<CONFIG>/common.py
  INPUT ${ONNX_BACKENDTEST_SRC_DIR}/common.py
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

# Followings are test cases of test_to_enable_dict in test/backend/inference_backend.py.
# Only test cases including operations supported by zDNN are listed.
set(TEST_ABS "test_abs_cpu")
set(TEST_ADD "test_add_cpu test_add_bcast_cpu")
set(TEST_AVGPOOL "test_averagepool_1d_default_cpu test_averagepool_2d_ceil_cpu test_averagepool_2d_default_cpu test_averagepool_2d_pads_count_include_pad_cpu test_averagepool_2d_pads_cpu test_averagepool_2d_precomputed_pads_count_include_pad_cpu test_averagepool_2d_precomputed_pads_cpu test_averagepool_2d_precomputed_same_upper_cpu test_averagepool_2d_precomputed_strides_cpu test_averagepool_2d_same_lower_cpu test_averagepool_2d_same_upper_cpu test_averagepool_2d_strides_cpu test_averagepool_3d_default_cpu")
set(TEST_BNORM "test_batchnorm_epsilon_cpu test_batchnorm_example_cpu")
set(TEST_CONV "test_basic_conv_with_padding_cpu test_basic_conv_without_padding_cpu test_conv_with_autopad_same_cpu test_conv_with_strides_no_padding_cpu test_conv_with_strides_padding_cpu test_conv_with_strides_and_asymmetric_padding_cpu")
set(TEST_DIV "test_div_cpu test_div_bcast_cpu test_div_example_cpu")
set(TEST_EXP "test_exp_cpu test_exp_example_cpu")
set(TEST_GEMM "test_gemm_all_attributes_cpu test_gemm_alpha_cpu test_gemm_beta_cpu test_gemm_default_matrix_bias_cpu test_gemm_default_no_bias_cpu test_gemm_default_scalar_bias_cpu test_gemm_default_single_elem_vector_bias_cpu test_gemm_default_vector_bias_cpu test_gemm_default_zero_bias_cpu test_gemm_transposeA_cpu test_gemm_transposeB_cpu test_globalaveragepool_cpu")
set(TEST_GAVGPOOL "test_globalaveragepool_precomputed_cpu test_globalmaxpool_cpu test_globalmaxpool_precomputed_cpu")
set(TEST_GRU "test_gru_defaults_cpu test_gru_seq_length_cpu test_gru_with_initial_bias_cpu")
set(TEST_LOG "test_log_example_cpu test_log_cpu")
set(TEST_LOGSOFTMAX "test_logsoftmax_axis_2_cpu test_logsoftmax_example_1_cpu test_logsoftmax_negative_axis_cpu test_logsoftmax_large_number_cpu")
set(TEST_LSTM "test_lstm_defaults_cpu test_lstm_with_initial_bias_cpu test_lstm_with_peepholes_cpu")
set(TEST_MATMUL "test_matmul_2d_cpu test_matmul_3d_cpu test_matmul_4d_cpu")
set(TEST_MAX "test_max_example_cpu test_max_one_input_cpu test_max_two_inputs_cpu test_max_float32_cpu test_max_float64_cpu test_max_int8_cpu test_max_int16_cpu test_max_int32_cpu test_max_int64_cpu")
set(TEST_MAXPOOL "test_maxpool_1d_default_cpu test_maxpool_2d_ceil_cpu test_maxpool_2d_default_cpu test_maxpool_2d_dilations_cpu test_maxpool_2d_pads_cpu test_maxpool_2d_precomputed_pads_cpu test_maxpool_2d_precomputed_same_upper_cpu test_maxpool_2d_precomputed_strides_cpu test_maxpool_2d_same_lower_cpu test_maxpool_2d_same_upper_cpu test_maxpool_2d_strides_cpu test_maxpool_3d_default_cpu")
set(TEST_MIN "test_min_example_cpu test_min_one_input_cpu test_min_two_inputs_cpu test_min_float32_cpu test_min_float64_cpu test_min_int8_cpu test_min_int16_cpu test_min_int32_cpu test_min_int64_cpu")
set(TEST_MUL "test_mul_cpu test_mul_bcast_cpu test_mul_example_cpu")
set(TEST_REDUCEMEAN "test_reduce_mean_default_axes_keepdims_example_cpu test_reduce_mean_default_axes_keepdims_random_cpu test_reduce_mean_do_not_keepdims_example_cpu test_reduce_mean_do_not_keepdims_random_cpu test_reduce_mean_keepdims_example_cpu test_reduce_mean_keepdims_random_cpu test_reduce_mean_negative_axes_keepdims_example_cpu test_reduce_mean_negative_axes_keepdims_random_cpu")
set(TEST_RELU "test_relu_cpu")
set(TEST_SOFTMAX "test_softmax_axis_0_cpu test_softmax_axis_1_cpu test_softmax_axis_2_cpu test_softmax_default_axis_cpu test_softmax_example_cpu test_softmax_large_number_cpu")
set(TEST_SUB "test_sub_cpu test_sub_bcast_cpu test_sub_example_cpu test_sum_example_cpu")
set(TEST_SUM "test_sum_one_input_cpu test_sum_two_inputs_cpu")
set(TEST_MODELS "test_densenet121_cpu test_inception_v1_cpu test_resnet50_cpu test_shufflenet_cpu test_squeezenet_cpu test_vgg19_cpu")

set(NNPA_TESTS "${TEST_ABS} ${TEST_ADD} ${TEST_AVGPOOL} ${TEST_BNORM} ${TEST_CONV} ${TEST_DIV} ${TEST_EXP} ${TEST_GEMM} ${TEST_GAVGPOOL} ${TEST_GRU} ${TEST_LOG} ${TEST_LOGSOFTMAX} ${TEST_LSTM} ${TEST_MATMUL} ${TEST_MAX} ${TEST_MAXPOOL} ${TEST_MIN} ${TEST_MUL} ${TEST_REDUCEMEAN} ${TEST_RELU} ${TEST_SOFTMAX} ${TEST_SUB} ${TEST_SUM} ${TEST_MODELS}")

set(NNPA_TESTS_ENVS TEST_MACCEL=NNPA TEST_CASE_BY_USER=${NNPA_TESTS})

# ${ONNX_HOME} is the directory where onnx downloads real model files.
# Model files are saved under ${ONNX_HOME}/models/model_name/model.onnx.
# C/C++ and JNI tests run in parallel so they must use a different
# ONNX_HOME to avoid conflicts.
add_custom_target(check-onnx-backend-nnpa
  COMMAND
    ONNX_HOME=${CMAKE_CURRENT_BINARY_DIR}/cpp
    ${NNPA_TESTS_ENVS} ${BACKEND_TEST_COMMAND} ${BACKEND_TEST_ARGS} ${FILE_GENERATE_DIR}/test.py
  DEPENDS
    ${FILE_GENERATE_DIR}/test.py
    ${FILE_GENERATE_DIR}/test_config.py
   )

add_custom_target(check-onnx-backend-nnpa-dynamic
  COMMAND
    TEST_DYNAMIC=true
    ${NNPA_TESTS_ENVS} ${BACKEND_TEST_COMMAND} ${BACKEND_TEST_ARGS} ${FILE_GENERATE_DIR}/test.py
  DEPENDS
    ${FILE_GENERATE_DIR}/test.py
    ${FILE_GENERATE_DIR}/test_config.py
  )

add_custom_target(check-onnx-backend-nnpa-constant
  COMMAND
    TEST_CONSTANT=true
    ${NNPA_TESTS_ENVS} ${BACKEND_TEST_COMMAND} ${BACKEND_TEST_ARGS} ${FILE_GENERATE_DIR}/test.py
  DEPENDS
    ${FILE_GENERATE_DIR}/test.py
    ${FILE_GENERATE_DIR}/test_config.py
  )

add_custom_target(check-onnx-backend-nnpa-signature
  COMMAND
    TEST_SIGNATURE=true
    ${NNPA_TESTS_ENVS} ${BACKEND_TEST_COMMAND} ${BACKEND_TEST_ARGS} ${FILE_GENERATE_DIR}/test.py
  DEPENDS
    ${FILE_GENERATE_DIR}/test.py
    ${FILE_GENERATE_DIR}/test_config.py
  )

add_custom_target(check-onnx-backend-nnpa-compilerlib
  COMMAND
    TEST_COMPILERLIB=true ONNX_HOME=${CMAKE_CURRENT_BINARY_DIR}
    ${NNPA_TESTS_ENVS} ${BACKEND_TEST_COMMAND} ${BACKEND_TEST_ARGS} ${FILE_GENERATE_DIR}/test.py
  DEPENDS
    ${FILE_GENERATE_DIR}/test.py
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
      ${NNPA_TESTS_ENVS} ${BACKEND_TEST_COMMAND} ${BACKEND_TEST_ARGS} ${FILE_GENERATE_DIR}/test.py
    DEPENDS
      ${FILE_GENERATE_DIR}/test.py
      ${FILE_GENERATE_DIR}/test_config.py
    )

  add_custom_target(check-onnx-backend-nnpa-dynamic-jni
    COMMAND
      TEST_DYNAMIC=true TEST_EMIT=jni JSONITER_JAR=${JSONITER_JAR}
      ${NNPA_TESTS_ENVS} ${BACKEND_TEST_COMMAND} ${BACKEND_TEST_ARGS} ${FILE_GENERATE_DIR}/test.py
    DEPENDS
      ${FILE_GENERATE_DIR}/test.py
      ${FILE_GENERATE_DIR}/test_config.py
    )

  add_custom_target(check-onnx-backend-nnpa-constant-jni
    COMMAND
      TEST_CONSTANT=true TEST_EMIT=jni JSONITER_JAR=${JSONITER_JAR}
      ${NNPA_TESTS_ENVS} ${BACKEND_TEST_COMMAND} ${BACKEND_TEST_ARGS} ${FILE_GENERATE_DIR}/test.py
    DEPENDS
      ${FILE_GENERATE_DIR}/test.py
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
