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
  OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/$<CONFIG>/input_verification_backend.py
  INPUT ${ONNX_BACKENDTEST_SRC_DIR}/input_verification_backend.py
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
  message(STATUS "Parallel backend tests   : ON")
  set(BACKEND_TEST_COMMAND "${Python3_EXECUTABLE}" "-m" "pytest")
  set(BACKEND_TEST_ARGS "--forked" "-n" "$$\{NPROC:-auto\}" "-q" "--silent")
else()
  message(STATUS "Parallel backend tests   : OFF (install pytest-xdist to enable)")
  set(BACKEND_TEST_COMMAND ${Python3_EXECUTABLE})
  set(BACKEND_TEST_ARGS "")
endif()

# Followings are test cases of test_to_enable_dict in test/backend/inference_backend.py.
# Only test cases for operations supported by zDNN are listed. Non-supported cases are
# commented out. This list is set in the environment variable `TEST_CASE_BY_USER`. So,
# instruction name is added after test case name in each test case.
set(NNPA_TEST_LIST
    # Add
    test_add_cpu,zdnn_add
    # test_add_bcast_cpu # bcast not supported
    # AveragePool
    # test_averagepool_1d_default_cpu
    # test_averagepool_2d_ceil_cpu
    test_averagepool_2d_default_cpu,zdnn_avgpool2d
    # test_averagepool_2d_pads_count_include_pad_cpu
    # test_averagepool_2d_pads_cpu
    # test_averagepool_2d_precomputed_pads_count_include_pad_cpu
    test_averagepool_2d_precomputed_pads_cpu,zdnn_avgpool2d
    test_averagepool_2d_precomputed_same_upper_cpu,zdnn_avgpool2d
    test_averagepool_2d_precomputed_strides_cpu,zdnn_avgpool2d
    # test_averagepool_2d_same_lower_cpu
    test_averagepool_2d_same_upper_cpu,zdnn_avgpool2d
    test_averagepool_2d_strides_cpu,zdnn_avgpool2d
    # test_averagepool_3d_default_cpu
    # BatchNormalization (test mode)
    test_batchnorm_epsilon_cpu,zdnn_mul
    test_batchnorm_example_cpu,zdnn_mul
    # Conv
    test_basic_conv_with_padding_cpu,zdnn_conv2d
    test_basic_conv_without_padding_cpu,zdnn_conv2d
    # test_conv_with_autopad_same_cpu
    test_conv_with_strides_no_padding_cpu,zdnn_conv2d
    test_conv_with_strides_padding_cpu,zdnn_conv2d
    # test_conv_with_strides_and_asymmetric_padding_cpu
    # Div
    test_div_cpu,zdnn_div
    # test_div_bcast_cpu
    test_div_example_cpu,zdnn_div
    # Exp
    test_exp_cpu,zdnn_exp
    test_exp_example_cpu,zdnn_exp
    # Gemm
    # test_gemm_all_attributes_cpu
    # test_gemm_alpha_cpu
    # test_gemm_beta_cpu
    test_gemm_default_matrix_bias_cpu,zdnn_matmul_op
    test_gemm_default_no_bias_cpu,zdnn_matmul_op
    # test_gemm_default_scalar_bias_cpu
    # test_gemm_default_single_elem_vector_bias_cpu
    test_gemm_default_vector_bias_cpu,zdnn_matmul_op
    test_gemm_default_zero_bias_cpu,zdnn_matmul_op
    test_gemm_transposeA_cpu,zdnn_matmul_op
    test_gemm_transposeB_cpu,zdnn_matmul_op
    # Global Average Pool
    test_globalaveragepool_cpu,zdnn_meanreduce2d
    test_globalaveragepool_precomputed_cpu,zdnn_meanreduce2d
    # Global Max Pool
    # test_globalmaxpool_cpu
    # test_globalmaxpool_precomputed_cpu
    # GRU
    # test_gru_defaults_cpu
    # test_gru_seq_length_cpu
    # test_gru_with_initial_bias_cpu
    # Log
    test_log_example_cpu,zdnn_log
    test_log_cpu,zdnn_log
    # LogSoftmax
    # test_logsoftmax_axis_0_cpu
    # test_logsoftmax_axis_1_cpu
    test_logsoftmax_axis_2_cpu,zdnn_log
    test_logsoftmax_example_1_cpu,zdnn_softmax
    # test_logsoftmax_default_axis_cpu
    test_logsoftmax_negative_axis_cpu,zdnn_log
    # test_logsoftmax_large_number_cpu #  accuracy error in test_logsoftmax_large_number_cpu
    # LSTM
    test_lstm_defaults_cpu,zdnn_lstm
    test_lstm_with_initial_bias_cpu,zdnn_lstm
    # test_lstm_with_peepholes_cpu
    # Matmul
    test_matmul_2d_cpu,zdnn_matmul_op
    test_matmul_3d_cpu,zdnn_matmul_op
    # test_matmul_4d_cpu,zdnn_matmul_op # 4d not supported
    # Max
    # test_max_example_cpu
    #test_max_one_input_cpu
    test_max_two_inputs_cpu,zdnn_max
    # test_max_float16_cpu
    test_max_float32_cpu,zdnn_max
    # test_max_float64_cpu
    # test_max_int8_cpu
    # test_max_int16_cpu
    # test_max_int32_cpu
    # test_max_int64_cpu
    # test_max_uint8_cpu
    # test_max_uint16_cpu
    # test_max_uint32_cpu
    # test_max_uint64_cpu
    # MaxPoolSigleOut
    # test_maxpool_1d_default_cpu
    # test_maxpool_2d_ceil_cpu
    test_maxpool_2d_default_cpu,zdnn_avgpool2d
    # test_maxpool_2d_dilations_cpu
    # test_maxpool_2d_pads_cpu
    # test_maxpool_2d_precomputed_pads_cpu
    test_maxpool_2d_precomputed_same_upper_cpu,zdnn_avgpool2d
    test_maxpool_2d_precomputed_strides_cpu,zdnn_avgpool2d
    # test_maxpool_2d_same_lower_cpu
    test_maxpool_2d_same_upper_cpu,zdnn_avgpool2d
    test_maxpool_2d_strides_cpu,zdnn_avgpool2d
    # test_maxpool_3d_default_cpu
    # Min
    # test_min_example_cpu
    # test_min_one_input_cpu
    test_min_two_inputs_cpu,zdnn_min
    # test_min_float16_cpu
    test_min_float32_cpu,zdnn_min
    # test_min_float64_cpu
    # test_min_int8_cpu
    # test_min_int16_cpu
    # test_min_int32_cpu
    # test_min_int64_cpu
    # test_min_uint8_cpu
    # test_min_uint16_cpu
    # test_min_uint32_cpu
    # test_min_uint64_cpu
    # Mul
    test_mul_cpu,zdnn_mul
    # test_mul_bcast_cpu
    test_mul_example_cpu,zdnn_mul
    # ReduceMean
    # test_reduce_mean_default_axes_keepdims_example_cpu
    # test_reduce_mean_default_axes_keepdims_random_cpu
    # test_reduce_mean_do_not_keepdims_example_cpu
    # test_reduce_mean_do_not_keepdims_random_cpu
    # test_reduce_mean_keepdims_example_cpu
    # test_reduce_mean_keepdims_random_cpu
    # test_reduce_mean_negative_axes_keepdims_example_cpu
    # test_reduce_mean_negative_axes_keepdims_random_cpu
    # Relu
    test_relu_cpu,zdnn_relu
    # Softmax
    # test_softmax_axis_0_cpu
    # test_softmax_axis_1_cpu
    # test_softmax_axis_2_cpu
    # test_softmax_default_axis_cpu
    test_softmax_example_cpu,zdnn_softmax
    # test_softmax_large_number_cpu #  accuracy error
    # Sub
    test_sub_cpu,zdnn_sub
    # test_sub_bcast_cpu
    test_sub_example_cpu,zdnn_sub
    # Sum
    test_sum_example_cpu,zdnn_add
    # test_sum_one_input_cpu
    test_sum_two_inputs_cpu,zdnn_add
    # Model
    # test_densenet121_cpu  #  accurary error
    test_inception_v1_cpu,zdnn_conv2d
    test_resnet50_cpu,zdnn_conv2d
    test_shufflenet_cpu,zdnn_matmul_op
    test_squeezenet_cpu,zdnn_conv
    test_vgg19_cpu,zdnn_conv
)
set(ENV_TEST_CASE_BY_USER "")
foreach(test_name IN LISTS NNPA_TEST_LIST)
  set(ENV_TEST_CASE_BY_USER "${ENV_TEST_CASE_BY_USER} ${test_name}")
endforeach()

set(NNPA_TESTS_ENVS TEST_MACCEL=NNPA TEST_CASE_BY_USER=${ENV_TEST_CASE_BY_USER} TEST_ATOL=0.01 TEST_RTOL=0.05)

# ${ONNX_HOME} is the directory where onnx downloads real model files.
# Model files are saved under ${ONNX_HOME}/models/model_name/model.onnx.
# C/C++ and JNI tests run in parallel so they must use a different
# ONNX_HOME to avoid conflicts.
add_custom_target(check-onnx-backend-nnpa
  COMMAND
    TEST_INSTRUCTION_CHECK=true
    ONNX_HOME=${CMAKE_CURRENT_BINARY_DIR}/cpp
    ${NNPA_TESTS_ENVS} ${BACKEND_TEST_COMMAND} ${BACKEND_TEST_ARGS} ${FILE_GENERATE_DIR}/test.py
  DEPENDS
    ${FILE_GENERATE_DIR}/test.py
    ${FILE_GENERATE_DIR}/test_config.py
   )

add_custom_target(check-onnx-backend-nnpa-dynamic
  COMMAND
    TEST_INSTRUCTION_CHECK=true
    TEST_DYNAMIC=true
    ${NNPA_TESTS_ENVS} ${BACKEND_TEST_COMMAND} ${BACKEND_TEST_ARGS} ${FILE_GENERATE_DIR}/test.py
  DEPENDS
    ${FILE_GENERATE_DIR}/test.py
    ${FILE_GENERATE_DIR}/test_config.py
  )

add_custom_target(check-onnx-backend-nnpa-constant
  COMMAND
    # In some test cases such as `test_add_cpu`, operations are removed by optimization and
    # instruction check fails. So, currently instruction check is disabled in constant test.
    # TEST_INSTRUCTION_CHECK=true
    TEST_CONSTANT=true
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
add_dependencies(check-onnx-backend-nnpa-compilerlib CompilerLibTest)
add_dependencies(check-onnx-backend-nnpa-compilerlib PyRuntime)

if (ONNX_MLIR_ENABLE_JNI)
  message(STATUS "JNI backend tests        : ON")
  message(STATUS "JSONITER_JAR             : ${JSONITER_JAR}")
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
