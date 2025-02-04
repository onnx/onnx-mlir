# SPDX-License-Identifier: Apache-2.0

set(ONNX_BACKENDTEST_SRC_DIR ${ONNX_MLIR_SRC_ROOT}/test/backend)

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
  OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/$<CONFIG>/onnxmlir_node_tests.py
  INPUT ${ONNX_BACKENDTEST_SRC_DIR}/onnxmlir_node_tests.py
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
 ${CMAKE_CURRENT_BINARY_DIR}/test_config.py.cfg
 @ONLY
 )

file(GENERATE
 OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/$<CONFIG>/test_config.py
 INPUT ${CMAKE_CURRENT_BINARY_DIR}/test_config.py.cfg
 )

configure_file(
  ${ONNX_BACKENDTEST_SRC_DIR}/test_config_compilerlib.py.in
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

# Followings are test cases of test_to_enable_dict in test/backend/inference_backend.py.
# Only test cases for operations supported by zDNN are listed. Non-supported cases are
# commented out. This list is set in the environment variable `TEST_CASE_BY_USER`. So,
# instruction name and dimParams are added after test case name if necessary.
# For example, the following line
#    test_add_cpu,zdnn_add,"0:0=a,1=b,2=c|1:0=a,1=b,2=c"
# means that (1) the test name is "test_add_cpu", (2) instruction name is
# "zdnn_add" to check the function is called, (3) when dimParams is
# "NO_DYNAMIC_SHAPE_TEST", backend test is skipped, otherwise the string is
# passed as --dimParams option. "0:0=a,1=b,2=c|1:0=a,1=b,2=c" means that the
# first, second and third dimensions of the first and second input arguments # are the same respectively.
set(NNPA_TEST_LIST_z16

    # To rebuild after changes: make onnx_mlir_supported_ops
    # ==ARCH== NNPA
    # ==ADDITIONAL_PARAGRAPH== NNPA has hardware limitations in dimension index size and tensor size, which are described in [NNPALimit.hpp](../src/Accelerators/NNPA/Support/NNPALimit.hpp). They are large enough for normal use cases, but if your model exceeds the limitations, CPU is used instead of NNPA. NNPA currently only support DLFLOAT16 as its data type. Common data formats like FP32, FP16, BFLOAT need to undergo data conversions to the NNPA internal format DLFLOAT16. Hence ONNX ops which updated their tensors to BFLOAT16 will not be natively supported on NNPA.  Onnx-mlir with NNPA utilizes hardware when possible. To accomplish this, the compiler converts ONNX ops to [ZHigh](Dialects/zhigh.md) ops, [ZLow](Dialects/zlow.md) ops, and are processed by the [IBM Z Deep Neural Network Library (zDNN)](https://github.com/IBM/zDNN).

    # ==OP== Add
    # ==LEVEL== z16,arch15
    # ==MIN== 6
    # ==LIM== - Shape of input tensors must be the same since broadcasting is not supported.<br>- Input tensors must have static dimensions.
    # Scalar tensor not supported.
    test_add_cpu,zdnn_add_ext,"0:0=a,1=b,2=c|1:0=a,1=b,2=c"
    # test_add_bcast_cpu # bcast not supported

    # ==OP== AveragePool
    # ==LEVEL== z16,arch15
    # ==MIN== 1
    # ==LIM== - `auto_pad` must be `NOTSET`, `VALID`, and `SAME_UPPER`. If `NOTSET` is used, `pads` must be set so that the padding valid type or same upper.<br>- `ceil_mode` must be default value(0) <br>- Input and output tensors must be 4D tensors (N x C x H x W).<br>- `kernel_shape` must be static.<br>- `count_include_pad` must be default value(0).<br>- `ceil_mode` must be default value(0).
    # test_averagepool_1d_default_cpu
    # test_averagepool_2d_ceil_cpu
    test_averagepool_2d_default_cpu,zdnn_avgpool2d
    # test_averagepool_2d_pads_count_include_pad_cpu
    # test_averagepool_2d_pads_cpu
    # test_averagepool_2d_precomputed_pads_count_include_pad_cpu
    test_averagepool_2d_precomputed_pads_cpu,zdnn_avgpool2d,NO_DYNAMIC_SHAPE_TEST
    test_averagepool_2d_precomputed_same_upper_cpu,zdnn_avgpool2d
    test_averagepool_2d_precomputed_strides_cpu,zdnn_avgpool2d
    # test_averagepool_2d_same_lower_cpu
    test_averagepool_2d_same_upper_cpu,zdnn_avgpool2d
    test_averagepool_2d_strides_cpu,zdnn_avgpool2d
    # test_averagepool_3d_default_cpu

    # ==OP== BatchNormalization
    # ==LEVEL== z16,arch15
    # ==MIN== 6
    # ==LIM== Input and output tensor must be 4D(N x C x H x W).
    test_batchnorm_epsilon_cpu,zdnn_mul_ext,"0:0=a,1=b,2=c,3=d|1:0=b|2:0=b|3:0=b|4:0=b"
    test_batchnorm_example_cpu,zdnn_mul_ext,"0:0=a,1=b,2=c,3=d|1:0=b|2:0=b|3:0=b|4:0=b"
    
    # ==OP== Conv
    # ==LEVEL== z16,arch15
    # ==MIN== 1
    # ==LIM== - `auto_pad` must be `NOTSET`, `VALID`, and `SAME_UPPER`. If `NOTSET` is used, `pads` must be set so that the padding valid type or same upper.<br>- Dimension in Height and weight must be static.<br>- `group` must be default value(1).<br>- `dilations` must be default value(1).<br>- Input and output tensors must have 4D (N x C x H x W).<br>- `kernel_shape` must be static.
    test_basic_conv_with_padding_cpu,zdnn_conv2d,NO_DYNAMIC_SHAPE_TEST
    test_basic_conv_without_padding_cpu,zdnn_conv2d,NO_DYNAMIC_SHAPE_TEST
    # test_conv_with_autopad_same_cpu
    test_conv_with_strides_no_padding_cpu,zdnn_conv2d,NO_DYNAMIC_SHAPE_TEST
    test_conv_with_strides_padding_cpu,zdnn_conv2d,NO_DYNAMIC_SHAPE_TEST
    # test_conv_with_strides_and_asymmetric_padding_cpu

    # ==OP== ConvTranspose
    # ==LEVEL== z16,arch15
    # ==MIN== 1
    # ==LIM== - 1D and 3D not supported because Conv1D and Conv3D not supported in zDNN. non-default `dilations` not supported because dilated convolution not supported in zDNN.
    # Spatial dims must be static.
    # test_convtranspose_1d_cpu,zdnn_conv1d
    # test_convtranspose_3d_cpu
    test_convtranspose_autopad_same_cpu,zdnn_conv2d
    test_convtranspose_cpu,zdnn_conv2d
    # test_convtranspose_dilations_cpu,zdnn_conv2d
    test_convtranspose_kernel_shape_cpu,zdnn_conv2d
    test_convtranspose_output_shape_cpu,zdnn_conv2d
    test_convtranspose_pad_cpu,zdnn_conv2d
    test_convtranspose_pads_cpu,zdnn_conv2d

    # ==OP== Div
    # ==LEVEL== z16,arch15
    # ==MIN== 6
    # ==LIM== - Shape of input tensors must be the same since broadcasting is not supported.<br>- Input tensors must have static dimensions.
    test_div_cpu,zdnn_div_ext,"0:0=a,1=b,2=c|1:0=a,1=b,2=c"
    # test_div_bcast_cpu
    test_div_example_cpu,zdnn_div_ext,"0:0=a|1:0=a"

    # ==OP== Exp
    # ==LEVEL== z16,arch15
    # ==MIN== 6
    # ==LIM== Input tensor must have 4 dimensions.
    test_exp_cpu,zdnn_exp_ext
    test_exp_example_cpu,zdnn_exp_ext

    # ==OP== Gemm
    # ==LEVEL== z16,arch15
    # ==MIN== 6
    # ==LIM== - `alpha` and `beta` must be default value(1).<br>- Rank of `C` must be 1 or 2. If the rank is 1, the dimension of `C` must be the same with the seconde dimension of `B`.<br>

    # Commented out for the moment.
    # -`gemm_transposeA` and `gemm_transposeB` will require an "--march" or an NNPA level of at least arch15, and the "transA" or "transB" attribute must be non-zero.

    # test_gemm_all_attributes_cpu
    # test_gemm_alpha_cpu
    # test_gemm_beta_cpu
    test_gemm_default_matrix_bias_cpu,zdnn_matmul_op_ext
    test_gemm_default_no_bias_cpu,zdnn_matmul_op_ext
    # test_gemm_default_scalar_bias_cpu
    # test_gemm_default_single_elem_vector_bias_cpu
    test_gemm_default_vector_bias_cpu,zdnn_matmul_op_ext
    test_gemm_default_zero_bias_cpu,zdnn_matmul_op_ext
    test_gemm_transposeA_cpu,zdnn_matmul_op_ext
    test_gemm_transposeB_cpu,zdnn_matmul_op_ext

    # ==OP== GlobalAveragePool
    # ==LEVEL== z16,arch15
    # ==MIN== 1
    # ==LIM== - Input shape must be 4D tensor(NCHW).<br>- Dimensions in `H` and `W` must be static.
    test_globalaveragepool_cpu,zdnn_meanreduce2d,NO_DYNAMIC_SHAPE_TEST
    test_globalaveragepool_precomputed_cpu,zdnn_meanreduce2d,NO_DYNAMIC_SHAPE_TEST
    # GlobalMaxPool
    # test_globalmaxpool_cpu
    # test_globalmaxpool_precomputed_cpu

    # ==OP== GRU
    # ==LEVEL== z16,arch15
    # ==MIN== 7
    # ==LIM== - `direction` and `hidden_size` in `W` must have static dimensions.<br>- `R` must have static dimensions.<br>- If `B` and `initial_h` are given, they must have static dimensions.<br>- `sequence_lens` is not supported for bidirectional GRU.<br>- `activations` must be `["Sigmoid", "Tanh", "Tanh"]`.<br>- `clip` is not supported.<br>- `linear_before_reset` must be 1.<br>- `layout` is not supported.
    # test_gru_defaults_cpu
    # test_gru_seq_length_cpu
    # test_gru_with_initial_bias_cpu

    # ==OP== Log
    # ==LEVEL== z16,arch15
    # ==MIN== 6
    # ==LIM== Input tensor must have 4 dimensions.
    test_log_example_cpu,zdnn_log_ext
    test_log_cpu,zdnn_log_ext

    # ==OP== LogSoftmax
    # ==LEVEL== z16,arch15
    # ==MIN== 6
    # test_logsoftmax_axis_0_cpu
    # test_logsoftmax_axis_1_cpu
    # test_logsoftmax_axis_2_cpu,zdnn_log_ext
    test_logsoftmax_example_1_cpu,zdnn_softmax_ext
    # test_logsoftmax_default_axis_cpu
    # test_logsoftmax_negative_axis_cpu,zdnn_log_ext
    # test_logsoftmax_large_number_cpu #  accuracy error in test_logsoftmax_large_number_cpu

    # ==OP== LSTM
    # ==LEVEL== z16,arch15
    # ==MIN== 7
    # ==LIM== - `direction` and `hidden_size` in `W` must have static dimensions.<br>- `R` must have static dimensions.<br>- `B` and `initial_h` have static dimensions if given. `B`'s direction dim must be 1 or 2.<br>- `P`(peepholes), `activation_alpha`, and `activation_beta` are not supported.<br>- `activations` must be `["Sigmoid", "Tanh", "Tanh"]`.<br>- `clip` is not supported.<br>- `input_forget` must be default value(0).<br>- `layout` is not supported.
    test_lstm_defaults_cpu,zdnn_lstm
    test_lstm_with_initial_bias_cpu,zdnn_lstm
    # test_lstm_with_peepholes_cpu

    # ==OP== MatMul
    # ==LEVEL== z16,arch15
    # ==MIN== 1
    # ==LIM== Ranks of input tensors must be (Rank of A, Rank of B) = (M, N), where M >= 2 and N >= 2.
    test_matmul_2d_cpu,zdnn_matmul_op_ext
    test_matmul_3d_cpu,zdnn_matmul_op_ext
    test_matmul_4d_cpu,zdnn_matmul_op_ext,"0:0=a,1=b,2=c,3=d|1:0=a,1=b,2=d,3=c"

    # ==OP== Max
    # ==LEVEL== z16,arch15
    # ==MIN== 6
    # ==LIM== - Shape of input tensors must be the same since broadcasting is not supported.<br>- Input tensors must have static dimensions.
    # test_max_example_cpu
    #test_max_one_input_cpu
    test_max_two_inputs_cpu,zdnn_max_ext,"0:0=a|1:0=a"
    # test_max_float16_cpu
    test_max_float32_cpu,zdnn_max_ext,"0:0=a|1:0=a"
    # test_max_float64_cpu
    # test_max_int8_cpu
    # test_max_int16_cpu
    # test_max_int32_cpu
    # test_max_int64_cpu
    # test_max_uint8_cpu
    # test_max_uint16_cpu
    # test_max_uint32_cpu
    # test_max_uint64_cpu

    # ==OP== MaxPool
    # ==LEVEL== z16,arch15
    # ==MIN== 1
    # ==LIM== - `auto_pad` must be `NOTSET`, `VALID`, and `SAME_UPPER`. If `NOTSET` is used, `pads` must be set so that the padding valid type or same upper.<br>- `ceil_mode` must be default value(0) <br>- Input and output tensors must be 4D tensors(N x C x H x W).<br>- `kernel_shape` must be static.<br>- `ceil_mode` must be default value(0).<br>- `dilations` must be default value(1).
    # test_maxpool_1d_default_cpu
    # test_maxpool_2d_ceil_cpu
    test_maxpool_2d_default_cpu,zdnn_maxpool2d
    # test_maxpool_2d_dilations_cpu
    # test_maxpool_2d_pads_cpu
    test_maxpool_2d_precomputed_pads_cpu,zdnn_maxpool2d,NO_DYNAMIC_SHAPE_TEST
    test_maxpool_2d_precomputed_same_upper_cpu,zdnn_maxpool2d
    test_maxpool_2d_precomputed_strides_cpu,zdnn_maxpool2d
    # test_maxpool_2d_same_lower_cpu
    test_maxpool_2d_same_upper_cpu,zdnn_maxpool2d
    test_maxpool_2d_strides_cpu,zdnn_maxpool2d
    # test_maxpool_3d_default_cpu

    # ==OP== Min
    # ==LEVEL== z16,arch15
    # ==MIN== 6
    # ==LIM== - Shape of input tensors must be the same since broadcasting is not supported.<br>- Input tensors must have static dimensions.
    # test_min_example_cpu
    # test_min_one_input_cpu
    test_min_two_inputs_cpu,zdnn_min_ext,"0:0=a|1:0=a"
    # test_min_float16_cpu
    test_min_float32_cpu,zdnn_min_ext,"0:0=a|1:0=a"
    # test_min_float64_cpu
    # test_min_int8_cpu
    # test_min_int16_cpu
    # test_min_int32_cpu
    # test_min_int64_cpu
    # test_min_uint8_cpu
    # test_min_uint16_cpu
    # test_min_uint32_cpu
    # test_min_uint64_cpu

    # ==OP== Mul
    # ==LEVEL== z16,arch15
    # ==MIN== 6
    # ==LIM== - Shape of input tensors should be the same since broadcasting is not supported.<br>- Input tensors must have static dimensions.
    test_mul_cpu,zdnn_mul_ext,"0:0=a,1=b,2=c|1:0=a,1=b,2=c"
    # test_mul_bcast_cpu
    test_mul_example_cpu,zdnn_mul_ext,"0:0=a|1:0=a"

    # ==OP== Pow
    # ==LEVEL== z16,arch15
    # ==MIN== 7
    # ==LIM== - Exponent should be a scalar integer and less or equal to 64.
    test_pow_bcast_scalar_cpu

    # ==OP== ReduceMean
    # ==LEVEL== z16,arch15
    # ==MIN== 1
    # ==LIM== - `keepdims` must be 1.<br>- Input tensor must be 4D tensors and `axis` must be [2, 3].
    # test_reduce_mean_default_axes_keepdims_example_cpu
    # test_reduce_mean_default_axes_keepdims_random_cpu
    # test_reduce_mean_do_not_keepdims_example_cpu
    # test_reduce_mean_do_not_keepdims_random_cpu
    # test_reduce_mean_keepdims_example_cpu
    # test_reduce_mean_keepdims_random_cpu
    # test_reduce_mean_negative_axes_keepdims_example_cpu
    # test_reduce_mean_negative_axes_keepdims_random_cpu

    # ==OP== Relu
    # ==LEVEL== z16,arch15
    # ==MIN== 6
    # ==LIM== Input tensor must be less than or equal to 4 dimensions.
    test_relu_cpu,zdnn_relu_ext

    # ==OP== Softmax
    # ==LEVEL== z16,arch15
    # ==MIN== 1
    # ==LIM== - `axis` must be the last dimension, i.e. `rank - 1` or -1.
    # test_softmax_axis_0_cpu
    # test_softmax_axis_1_cpu
    # test_softmax_axis_2_cpu
    # test_softmax_default_axis_cpu
    test_softmax_example_cpu,zdnn_softmax_ext
    # test_softmax_large_number_cpu #  accuracy error

    # ==OP== Softplus
    # ==LEVEL== z16,arch15
    # ==MIN== 1
    # ==LIM== The operations immediately before and after the Softplus operation must be executed on the NNPA. Otherwise, Softplus is executed on the CPU. This limitation is set to avoid performance degradation.
    # Softplus op in following test cases doesn't run on NNPA because single Softplus op is included. Softplus is tested not by backend tests but by the TestSoftplus numerical test
    # test_softplus_cpu,zdnn_log
    # test_softplus_example_cpu,zdnn_log

    # ==OP== Sub
    # ==LEVEL== z16,arch15
    # ==MIN== 6
    # ==LIM== - Shape of input tensors should be the same since broadcasting is not supported.<br>- Input tensors must have static dimensions.
    test_sub_cpu,zdnn_sub_ext,"0:0=a,1=b,2=c|1:0=a,1=b,2=c"
    # test_sub_bcast_cpu
    test_sub_example_cpu,zdnn_sub_ext,"0:0=a|1:0=a"

    # ==OP== Sum
    # ==LEVEL== z16,arch15
    # ==MIN== 6
    # ==LIM== - All inputs must have the same static shape (Broadcasting not supported.)<br>- Single input not supported.
    test_sum_example_cpu,zdnn_add,"0:0=a|1:0=a|2:0=a"
    # test_sum_one_input_cpu
    test_sum_two_inputs_cpu,zdnn_add,"0:0=a|1:0=a"

    # ==OP== Tanh
    # ==LEVEL== z16,arch15
    # ==MIN== 6
    # ==LIM== Input tensor must be less than or equal to 4 dimensions.

    # ==OP== Sigmoid
    # ==LEVEL== z16,arch15
    # ==MIN== 6
    # ==LIM== Input tensor must be less than or equal to 4 dimensions.

    # Model
    test_densenet121_cpu,zdnn_conv2d
    # TODO re-enable below 2 tests
    # test_inception_v1_cpu,zdnn_conv2d
    # test_resnet50_cpu,zdnn_conv2d

    # test_shufflenet_cpu,zdnn_matmul_op_ext # got NaN results in check-onnx-backend-dynamic-jni-nnpa.
    # Got NaN results because the last Conv running on NNPA produces dlfloat16 out-of-range values that are represented as NaN.
    # test_squeezenet_cpu,zdnn_conv
    # TODO re-enable below test
    # test_vgg19_cpu,zdnn_conv
)

set(NNPA_TEST_LIST_ARCH_15

    # ==OP== Gelu
    # ==LEVEL== arch15
    # ==MIN== 20
    test_gelu_default_1_cpu,zdnn_gelu_ext
    test_gelu_default_2_cpu,zdnn_gelu_ext
    test_gelu_tanh_1_cpu,zdnn_gelu_ext
    test_gelu_tanh_2_cpu,zdnn_gelu_ext

    # Gemm Transpose
    test_gemm_transposeA_cpu,zdnn_matmul_transpose_op_ext
    test_gemm_transposeB_cpu,zdnn_matmul_transpose_op_ext

    # ==OP== LeakyRelu
    # ==LEVEL== arch15
    # ==MIN== 6
    # ==LIM== Input tensor must be less than or equal to 4 dimensions.
    test_leakyrelu_cpu,zdnn_leaky_relu_ext
    test_leakyrelu_default_cpu,zdnn_leaky_relu_ext
    test_leakyrelu_example_cpu,zdnn_leaky_relu_ext

    # ==OP== MatMulInteger
    # ==LEVEL== arch15
    # ==MIN== 10
    test_matmulinteger_cpu,zdnn_quantized_matmul_op

    # ==OP== QLinearMatMul
    # ==LEVEL== arch15
    # ==MIN== 10 
    test_qlinearmatmul_2D_uint8_float32_cpu,zdnn_quantized_matmul_op
    test_qlinearmatmul_3D_uint8_float32_cpu,zdnn_quantized_matmul_op
    # Error: at (1, 0) mismatch 0 (actual) vs 1 (reference)
    # test_qlinearmatmul_2D_int8_float32_cpu,zdnn_quantized_matmul_op
    # test_qlinearmatmul_3D_int8_float32_cpu,zdnn_quantized_matmul_op

    # ==OP== ReduceMax
    # ==LEVEL== arch15
    # ==MIN== 1
    # ==LIM== - We do not support `do_not_keepdims` backend tests. Only support reduction over the innermost dimension.
    # Currrently, there is no backend test in ONNX that does reduction on the innermost dimension.

    # ==OP== ReduceMin
    # ==LEVEL== arch15 
    # ==MIN== 1
    # ==LIM== - We do not support `do_not_keepdims` backend tests. Only support reduction over the innermost dimension.
    # Currrently, there is no backend test in ONNX that does reduction on the innermost dimension.

    # ==OP== Sqrt
    # ==LEVEL== arch15
    # ==MIN== 6
    test_sqrt_cpu,zdnn_sqrt_ext,zdnn_invsqrt_ext
    test_sqrt_example_cpu,zdnn_sqrt_ext,zdnn_invsqrt_ext
)

set(ENV_TEST_CASE_BY_USER_z16 "")
foreach(test_name IN LISTS NNPA_TEST_LIST_z16)
  set(ENV_TEST_CASE_BY_USER_z16 "${ENV_TEST_CASE_BY_USER_z16} ${test_name}")
endforeach()

set(ENV_TEST_CASE_BY_USER_ARCH_15 "")
foreach(test_name IN LISTS NNPA_TEST_LIST_ARCH_15)
  set(ENV_TEST_CASE_BY_USER_ARCH_15 "${ENV_TEST_CASE_BY_USER_ARCH_15} ${test_name}")
endforeach()

set(NNPA_TESTS_ENVS_z16 TEST_MARCH=z16 TEST_MACCEL=NNPA TEST_CASE_BY_USER=${ENV_TEST_CASE_BY_USER_z16} TEST_ATOL=0.01 TEST_RTOL=0.05)
set(NNPA_TESTS_ENVS_ARCH_15 TEST_MARCH=arch15 TEST_MACCEL=NNPA TEST_CASE_BY_USER=${ENV_TEST_CASE_BY_USER_ARCH_15} TEST_ATOL=0.01 TEST_RTOL=0.05)

set(ENV_TEST_CASE_BY_USER_DYNAMIC_z16 "")
foreach(test_name IN LISTS NNPA_TEST_LIST_z16)
  if(NOT ${test_name} MATCHES ",NO_DYNAMIC_SHAPE_TEST$")
    set(ENV_TEST_CASE_BY_USER_DYNAMIC_z16 "${ENV_TEST_CASE_BY_USER_DYNAMIC_z16} ${test_name}")
  endif()
endforeach()

set(ENV_TEST_CASE_BY_USER_DYNAMIC_ARCH_15 "")
foreach(test_name IN LISTS NNPA_TEST_LIST_ARCH_15)
  if(NOT ${test_name} MATCHES ",NO_DYNAMIC_SHAPE_TEST$")
    set(ENV_TEST_CASE_BY_USER_DYNAMIC_ARCH_15 "${ENV_TEST_CASE_BY_USER_DYNAMIC_ARCH_15} ${test_name}")
  endif()
endforeach()

set(NNPA_TESTS_ENVS_DYNAMIC_z16 TEST_MARCH=z16 TEST_MACCEL=NNPA TEST_CASE_BY_USER=${ENV_TEST_CASE_BY_USER_DYNAMIC_z16} TEST_ATOL=0.01 TEST_RTOL=0.05)
set(NNPA_TESTS_ENVS_DYNAMIC_ARCH_15 TEST_MARCH=arch15 TEST_MACCEL=NNPA TEST_CASE_BY_USER=${ENV_TEST_CASE_BY_USER_DYNAMIC_ARCH_15} TEST_ATOL=0.01 TEST_RTOL=0.05)

# ${ONNX_HOME} is the directory where onnx downloads real model files.
# Model files are saved under ${ONNX_HOME}/models/model_name/model.onnx.
# C/C++ and JNI tests run in parallel so they must use a different
# ONNX_HOME to avoid conflicts.
add_custom_target(check-onnx-backend-nnpa
  COMMAND
    TEST_INSTRUCTION_CHECK=true
    ONNX_HOME=${FILE_GENERATE_DIR}/check-onnx-backend-nnpa
    # Needed for convolution models to avoid NaN outputs.
    # Remove this if saturation is enabled by default.
    TEST_COMPILE_ARGS="--nnpa-saturation=true"
    ${NNPA_TESTS_ENVS_z16} ${BACKEND_TEST_COMMAND} ${BACKEND_TEST_ARGS} ${FILE_GENERATE_DIR}/test.py
  DEPENDS
    ${FILE_GENERATE_DIR}/test.py
    ${FILE_GENERATE_DIR}/test_config.py
   )

# Ensure check-onnx-backend-ARCH_15-nnpa is backwards compatible
add_custom_target(check-onnx-backend-arch15-nnpa
  COMMAND
    TEST_INSTRUCTION_CHECK=true
    ONNX_HOME=${FILE_GENERATE_DIR}/check-onnx-backend-arch15-nnpa
    # Needed for convolution models to avoid NaN outputs.
    # Remove this if saturation is enabled by default.
    TEST_COMPILE_ARGS="--nnpa-saturation=true"
    ${NNPA_TESTS_ENVS_z16} ${BACKEND_TEST_COMMAND} ${BACKEND_TEST_ARGS} ${FILE_GENERATE_DIR}/test.py
    && ${NNPA_TESTS_ENVS_ARCH_15} ${BACKEND_TEST_COMMAND} ${BACKEND_TEST_ARGS} ${FILE_GENERATE_DIR}/test.py
  DEPENDS
    ${FILE_GENERATE_DIR}/test.py
    ${FILE_GENERATE_DIR}/test_config.py
   )

add_custom_target(check-onnx-backend-dynamic-nnpa
  COMMAND
    ONNX_HOME=${FILE_GENERATE_DIR}/check-onnx-backend-dynamic-nnpa
    TEST_INSTRUCTION_CHECK=true
    TEST_DYNAMIC=true
    # Needed for convolution models to avoid NaN outputs.
    # Remove this if saturation is enabled by default.
    TEST_COMPILE_ARGS="--nnpa-saturation=true"
    ${NNPA_TESTS_ENVS_DYNAMIC_z16} ${BACKEND_TEST_COMMAND} ${BACKEND_TEST_ARGS} ${FILE_GENERATE_DIR}/test.py
  DEPENDS
    ${FILE_GENERATE_DIR}/test.py
    ${FILE_GENERATE_DIR}/test_config.py
  )

# Ensure check-onnx-backend-dynamic-arch15-nnpa is backwards compatible
add_custom_target(check-onnx-backend-dynamic-arch15-nnpa
  COMMAND
    ONNX_HOME=${FILE_GENERATE_DIR}/check-onnx-backend-dynamic-arch15-nnpa
    TEST_INSTRUCTION_CHECK=true
    TEST_DYNAMIC=true
    # Needed for convolution models to avoid NaN outputs.
    # Remove this if saturation is enabled by default.
    TEST_COMPILE_ARGS="--nnpa-saturation=true"
    ${NNPA_TESTS_ENVS_DYNAMIC_z16} ${BACKEND_TEST_COMMAND} ${BACKEND_TEST_ARGS} ${FILE_GENERATE_DIR}/test.py
    && ${NNPA_TESTS_ENVS_DYNAMIC_ARCH_15} ${BACKEND_TEST_COMMAND} ${BACKEND_TEST_ARGS} ${FILE_GENERATE_DIR}/test.py
  DEPENDS
    ${FILE_GENERATE_DIR}/test.py
    ${FILE_GENERATE_DIR}/test_config.py
  )

add_custom_target(check-onnx-backend-constant-nnpa
  COMMAND
    # In some test cases such as `test_add_cpu`, operations are removed by optimization and
    # instruction check fails. So, currently instruction check is disabled in constant test.
    # TEST_INSTRUCTION_CHECK=true
    ONNX_HOME=${FILE_GENERATE_DIR}/check-onnx-backend-constant-nnpa
    TEST_CONSTANT=true
    # Needed for convolution models to avoid NaN outputs.
    # Remove this if saturation is enabled by default.
    TEST_COMPILE_ARGS="--nnpa-saturation=true" 
    ${NNPA_TESTS_ENVS_z16} ${BACKEND_TEST_COMMAND} ${BACKEND_TEST_ARGS} ${FILE_GENERATE_DIR}/test.py
  DEPENDS
    ${FILE_GENERATE_DIR}/test.py
    ${FILE_GENERATE_DIR}/test_config.py
  )

add_custom_target(check-onnx-backend-compilerlib-nnpa
  COMMAND
    TEST_COMPILERLIB=true ONNX_HOME=${CMAKE_CURRENT_BINARY_DIR}
    # Needed for convolution models to avoid NaN outputs.
    # Remove this if saturation is enabled by default.
    TEST_COMPILE_ARGS="--nnpa-saturation=true" 
    ${NNPA_TESTS_ENVS_z16} ${BACKEND_TEST_COMMAND} ${BACKEND_TEST_ARGS} ${FILE_GENERATE_DIR}/test.py
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
add_dependencies(check-onnx-backend-nnpa PyRuntimeC)
add_dependencies(check-onnx-backend-arch15-nnpa onnx-mlir)
add_dependencies(check-onnx-backend-arch15-nnpa PyRuntimeC)
add_dependencies(check-onnx-backend-dynamic-nnpa onnx-mlir)
add_dependencies(check-onnx-backend-dynamic-nnpa PyRuntimeC)
add_dependencies(check-onnx-backend-dynamic-arch15-nnpa onnx-mlir)
add_dependencies(check-onnx-backend-dynamic-arch15-nnpa PyRuntimeC)
add_dependencies(check-onnx-backend-constant-nnpa onnx-mlir)
add_dependencies(check-onnx-backend-constant-nnpa PyRuntimeC)
add_dependencies(check-onnx-backend-compilerlib-nnpa CompilerLibTest)
add_dependencies(check-onnx-backend-compilerlib-nnpa PyRuntimeC)

add_dependencies(check-onnx-backend-numerical-nnpa check-onnx-backend-nnpa)
# TODO arch15: if (avail on test machines):
# In addition to testing arch14, also test arch15.
# add_dependencies(check-onnx-backend-numerical-nnpa check-onnx-backend-arch15-nnpa)
# end if.
add_dependencies(check-onnx-backend-numerical-nnpa check-onnx-backend-dynamic-nnpa)
add_dependencies(check-onnx-backend-numerical-nnpa check-onnx-backend-constant-nnpa)

if (ONNX_MLIR_ENABLE_JNI)
  message(STATUS "JNI backend tests        : ON")
  message(STATUS "JSONITER_JAR             : ${JSONITER_JAR}")
  add_custom_target(check-onnx-backend-jni-nnpa
    COMMAND
      ONNX_HOME=${FILE_GENERATE_DIR}/check-onnx-backend-jni-nnpa
      TEST_EMIT=jni JSONITER_JAR=${JSONITER_JAR}
      # Needed for convolution models to avoid NaN outputs.
      # Remove this if saturation is enabled by default.
      TEST_COMPILE_ARGS="--nnpa-saturation=true"
      ${NNPA_TESTS_ENVS_z16} ${BACKEND_TEST_COMMAND} ${BACKEND_TEST_ARGS} ${FILE_GENERATE_DIR}/test.py
    DEPENDS
      ${FILE_GENERATE_DIR}/test.py
      ${FILE_GENERATE_DIR}/test_config.py
    )

  add_custom_target(check-onnx-backend-dynamic-jni-nnpa
    COMMAND
      ONNX_HOME=${FILE_GENERATE_DIR}/check-onnx-backend-dynamic-jni-nnpa
      TEST_DYNAMIC=true TEST_EMIT=jni JSONITER_JAR=${JSONITER_JAR}
      # Needed for convolution models to avoid NaN outputs.
      # Remove this if saturation is enabled by default.
      TEST_COMPILE_ARGS="--nnpa-saturation=true"
      ${NNPA_TESTS_ENVS_DYNAMIC_z16} ${BACKEND_TEST_COMMAND} ${BACKEND_TEST_ARGS} ${FILE_GENERATE_DIR}/test.py
    DEPENDS
      ${FILE_GENERATE_DIR}/test.py
      ${FILE_GENERATE_DIR}/test_config.py
    )

  add_custom_target(check-onnx-backend-constant-jni-nnpa
    COMMAND
      ONNX_HOME=${FILE_GENERATE_DIR}/check-onnx-backend-constant-jni-nnpa
      TEST_CONSTANT=true TEST_EMIT=jni JSONITER_JAR=${JSONITER_JAR}
      # Needed for convolution models to avoid NaN outputs.
      # Remove this if saturation is enabled by default.
      TEST_COMPILE_ARGS="--nnpa-saturation=true"
      ${NNPA_TESTS_ENVS_z16} ${BACKEND_TEST_COMMAND} ${BACKEND_TEST_ARGS} ${FILE_GENERATE_DIR}/test.py
    DEPENDS
      ${FILE_GENERATE_DIR}/test.py
      ${FILE_GENERATE_DIR}/test_config.py
    )

  add_dependencies(check-onnx-backend-jni-nnpa onnx-mlir)
  add_dependencies(check-onnx-backend-jni-nnpa PyRuntimeC)
  add_dependencies(check-onnx-backend-jni-nnpa javaruntime)
  add_dependencies(check-onnx-backend-jni-nnpa jniruntime)
  add_dependencies(check-onnx-backend-dynamic-jni-nnpa onnx-mlir)
  add_dependencies(check-onnx-backend-dynamic-jni-nnpa PyRuntimeC)
  add_dependencies(check-onnx-backend-dynamic-jni-nnpa javaruntime)
  add_dependencies(check-onnx-backend-dynamic-jni-nnpa jniruntime)
  add_dependencies(check-onnx-backend-constant-jni-nnpa onnx-mlir)
  add_dependencies(check-onnx-backend-constant-jni-nnpa PyRuntimeC)
  add_dependencies(check-onnx-backend-constant-jni-nnpa javaruntime)
  add_dependencies(check-onnx-backend-constant-jni-nnpa jniruntime)

  # ONNX models failed with NaN results, so temporarily disable these.
  #add_dependencies(check-onnx-backend-numerical-nnpa check-onnx-backend-jni-nnpa)
  #add_dependencies(check-onnx-backend-numerical-nnpa check-onnx-backend-dynamic-jni-nnpa)
  #add_dependencies(check-onnx-backend-numerical-nnpa check-onnx-backend-constant-jni-nnpa)

else()
  message(STATUS "  JNI backend-nnpa tests         : OFF")
endif()

