from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from distutils.util import strtobool
import os
import sys
import unittest
import warnings
import onnx.backend.base
import onnx.backend.test

from onnx.backend.base import Device, DeviceType
import subprocess
import test_config
import tempfile
import argparse

# Casting with "bool" does not work well. When you specify VERBOSE=xxx,
# regardless of the value of xxx (e.g., true, false, y, n, etc.) the
# casted bool value will be true. Only if xxx is empty, the casted bool
# value will be false. This is a bit counter intuitive. So we use strtobool
# to do the conversion. But note that strtobool can't take an emtpy string.

VERBOSE = os.getenv("VERBOSE")
IMPORTER_FORCE_DYNAMIC = os.getenv("IMPORTER_FORCE_DYNAMIC")
TEST_DYNAMIC = os.getenv("TEST_DYNAMIC")

parser = argparse.ArgumentParser(description='with dynamic shape or not.')
parser.add_argument('--dynamic', action='store_true',
    default=(strtobool(TEST_DYNAMIC) if TEST_DYNAMIC else False),
    help='enable dynamic shape tests (default: false if TEST_DYNAMIC env var not set)')
parser.add_argument('-i', '--input', type=int,
    default=os.getenv("TEST_INPUT", -1),
    help='inputs whose dimensions to be changed to unknown (default: all inputs if TEST_INPUT env var not set)')
parser.add_argument('-d', '--dim', type=int,
    default=os.getenv("TEST_DIM", -1),
    help='dimensions to be changed to unknown (default: all dimensions if TEST_DIM env var not set)')
parser.add_argument('-v', '--verbose', action='store_true',
    default=(strtobool(VERBOSE) if VERBOSE else False),
    help='verbose output (default: false if VERBOSE env var not set)')
parser.add_argument('unittest_args', nargs='*')
args = parser.parse_args()
sys.argv[1:] = args.unittest_args

TEST_CASE_BY_USER = os.getenv("TEST_CASE_BY_USER")
if TEST_CASE_BY_USER is not None and TEST_CASE_BY_USER != "" :
    result_dir = "./"
else :
    tempdir = tempfile.TemporaryDirectory()
    result_dir = tempdir.name+"/"
print("temporary results are in dir "+result_dir)

CXX = test_config.CXX_PATH
TEST_DRIVER = os.path.join(test_config.TEST_DRIVER_BUILD_PATH, "bin",
                           test_config.TEST_DRIVER_COMMAND)
LLC = os.path.join(test_config.LLVM_PROJ_BUILD_PATH, "bin/llc")

# Make lib folder under build directory visible in PYTHONPATH
doc_check_base_dir = os.path.dirname(os.path.realpath(__file__))
RUNTIME_DIR = os.path.join(test_config.TEST_DRIVER_BUILD_PATH, "lib")
sys.path.append(RUNTIME_DIR)
from PyRuntime import ExecutionSession

# Test directories:
# https://github.com/onnx/onnx/tree/master/onnx/backend/test/data/node
# In our directories, the python files that generate the tests are found here
# onnx-mlir/third_party/onnx/onnx/backend/test/case/node

# Set value for each benchmark to: test_disabled, test_static, 
#   test_dynamic, test_static_dynamic, test_static_dynamicNA.
# The test_static_dynamicNA values indicates tests for which the dynamic test
# makes no sense, e.g. where we build an array of constant but we don't even
# know the rank of the constant array we are generating.
test_disabled = 0  # no tests
test_static = 1    # static test only (1st bit on).
test_dynamic = 2   # dynamic test only (2nd bit on).
test_static_dynamic = test_static + test_dynamic # both static & dynamic
test_static_dynamicNA = test_static # static tests for which dyn not available.

# For each benchmark, its value is a tuple of (test_type, dynamic_dict)
# - 'test_type' is one of test_disabled, test_static, test_dynamic, and
#     test_static_dynamic.
# - 'dynamic_dict' is a dict to define which inputs/dimensions are changed to
#     unknown, where its key is an input index and its value is a set of
#     dimension indices, e.g. {0:{0,1}, 1:{-1}, 2:{0}}
# If 'dynamic_dict' is not given, by default, all dimension of all inputs will
#   be changed to unknown. Use this script's arguments '--input' and '--dim' to
#   control the default values.
# Input and dimension indices start from 0. -1 means all inputs or all dimensions.

test_to_enable_static_dynamic = {

    ############################################################
    # Elementary ops, ordered alphabetically.

    # Abs
    "test_abs_cpu": (test_static_dynamic,),

    # Acos
    "test_acos_cpu": (test_static_dynamic,),
    "test_acos_example_cpu": (test_static_dynamic,),

    # Acosh
    "test_acosh_cpu": (test_static_dynamic,),
    "test_acosh_example_cpu": (test_static_dynamic,),

    # Adagrad

    # Adam

    # Add
    "test_add_cpu": (test_static_dynamic,),
    "test_add_bcast_cpu": (test_static_dynamic,),

    # And
    "test_and2d_cpu": (test_static_dynamic,),
    "test_and3d_cpu": (test_static_dynamic,),
    "test_and4d_cpu": (test_static_dynamic,),
    "test_and_bcast3v1d_cpu": (test_static_dynamic,),
    "test_and_bcast3v2d_cpu": (test_static_dynamic,),
    "test_and_bcast4v2d_cpu": (test_static_dynamic,),
    "test_and_bcast4v3d_cpu": (test_static_dynamic,),
    "test_and_bcast4v4d_cpu": (test_static_dynamic,),

    # Argmax
    "test_argmax_no_keepdims_example_cpu": (test_static_dynamic,),
    "test_argmax_keepdims_example_cpu": (test_static_dynamic,),
    "test_argmax_default_axis_example_cpu": (test_static_dynamic,),

    # Argmin

    # Asin
    "test_asin_cpu": (test_static_dynamic,),
    "test_asin_example_cpu": (test_static_dynamic,),

    # Asinh
    "test_asinh_cpu": (test_static_dynamic,),
    "test_asinh_example_cpu": (test_static_dynamic,),

    # Atan
    "test_atan_example_cpu": (test_static_dynamic,),
    "test_atan_cpu": (test_static_dynamic,),

    # Atanh
    "test_atanh_cpu": (test_static_dynamic,),
    "test_atanh_example_cpu": (test_static_dynamic,),

    # AveragePool
    "test_averagepool_1d_default_cpu": (test_static_dynamic,),
    "test_averagepool_2d_ceil_cpu": (test_static_dynamic,),
    "test_averagepool_2d_default_cpu": (test_static_dynamic,),
    "test_averagepool_2d_pads_count_include_pad_cpu": (test_static_dynamic,),
    "test_averagepool_2d_pads_cpu": (test_static_dynamic,),
    "test_averagepool_2d_precomputed_pads_count_include_pad_cpu": (test_static_dynamic,),
    "test_averagepool_2d_precomputed_pads_cpu": (test_static_dynamic,),
    "test_averagepool_2d_precomputed_same_upper_cpu": (test_static_dynamic,),
    "test_averagepool_2d_precomputed_strides_cpu": (test_static_dynamic,),
    "test_averagepool_2d_same_lower_cpu": (test_static_dynamic,),
    "test_averagepool_2d_same_upper_cpu": (test_static_dynamic,),
    "test_averagepool_2d_strides_cpu": (test_static_dynamic,),
    "test_averagepool_3d_default_cpu": (test_static_dynamic,),

    # BatchNormalization (test mode)
    "test_batchnorm_epsilon_cpu": (test_static_dynamic,),
    "test_batchnorm_example_cpu": (test_static_dynamic,),

    # Bitshift left/right

    # Cast
    "test_cast_FLOAT_to_DOUBLE_cpu": (test_static_dynamic,),
    "test_cast_DOUBLE_to_FLOAT_cpu": (test_static_dynamic,),
    "test_cast_FLOAT_to_FLOAT16_cpu": (test_disabled,), # appers unsupported at this time
    "test_cast_FLOAT16_to_FLOAT_cpu": (test_disabled,), # appers unsupported at this time
    "test_cast_FLOAT16_to_DOUBLE_cpu": (test_disabled,), # appers unsupported at this time
    "test_cast_DOUBLE_to_FLOAT16_cpu": (test_disabled,), # appers unsupported at this time
    "test_cast_FLOAT_to_STRING_cpu": (test_disabled,), # appers unsupported at this time
    "test_cast_STRING_to_FLOAT_cpu": (test_disabled,), # appers unsupported at this time

    # Ceil
    "test_ceil_example_cpu": (test_static_dynamic,),
    "test_ceil_cpu": (test_static_dynamic,),

    # Celu

    # Clip
    "test_clip_cpu": (test_static_dynamic,),
    "test_clip_example_cpu": (test_static_dynamic,),
    "test_clip_inbounds_cpu": (test_static_dynamic,),
    "test_clip_outbounds_cpu": (test_static_dynamic,),
    "test_clip_splitbounds_cpu": (test_static_dynamic,),
    "test_clip_default_min_cpu": (test_static_dynamic,),
    "test_clip_default_max_cpu": (test_static_dynamic,),
    "test_clip_cpu": (test_static_dynamic,),
    "test_clip_default_inbounds_cpu": (test_static_dynamic,),
    #"test_clip_default_int8_min_cpu": (test_static_dynamic,),

    # Compress

    # Concat
    "test_concat_1d_axis_0_cpu": (test_static_dynamic,{0:{0}}),
    "test_concat_2d_axis_0_cpu": (test_static_dynamic,{0:{0}}),
    "test_concat_2d_axis_1_cpu": (test_static_dynamic,{0:{1}}),
    "test_concat_3d_axis_0_cpu": (test_static_dynamic,{0:{0}}),
    "test_concat_3d_axis_1_cpu": (test_static_dynamic,{0:{1}}),
    "test_concat_3d_axis_2_cpu": (test_static_dynamic,{0:{2}}),
    "test_concat_1d_axis_negative_1_cpu": (test_static_dynamic,{0:{0}}),
    "test_concat_2d_axis_negative_1_cpu": (test_static_dynamic,{0:{1}}),
    "test_concat_2d_axis_negative_2_cpu": (test_static_dynamic,{0:{0}}),
    "test_concat_3d_axis_negative_1_cpu": (test_static_dynamic,{0:{2}}),
    "test_concat_3d_axis_negative_2_cpu": (test_static_dynamic,{0:{1}}),
    "test_concat_3d_axis_negative_3_cpu": (test_static_dynamic,{0:{0}}),

    # Constant (dynamic NA)
    "test_constant_cpu": (test_static_dynamicNA,),

    # ConstantOfShape (dynamic NA)
    "test_constantofshape_float_ones_cpu": (test_static_dynamicNA,),
    "test_constantofshape_int_zeros_cpu": (test_static_dynamicNA,),

    # Conv
    "test_basic_conv_without_padding_cpu": (test_static_dynamic,{0:{0}}),
    "test_conv_with_strides_no_padding_cpu": (test_static_dynamic,{0:{0}}),
    "test_conv_with_strides_padding_cpu": (test_static_dynamic,{0:{0}}),
    "test_conv_with_strides_and_asymmetric_padding_cpu": (test_static_dynamic,{0:{0}}),

    # ConvInteger

    # ConvTranspose

    # Cos
    "test_cos_example_cpu": (test_static_dynamic,),
    "test_cos_cpu": (test_static_dynamic,),

    # Cosh
    "test_cosh_cpu": (test_static_dynamic,),
    "test_cosh_example_cpu": (test_static_dynamic,),

    # CumSum

    # DepthOfSpace

    # DequatizeLinear

    # Det

    # Div
    "test_div_cpu": (test_static_dynamic,),
    "test_div_bcast_cpu": (test_static_dynamic,),
    "test_div_example_cpu": (test_static_dynamic,),

    # Dropout
    "test_dropout_default_cpu": (test_static_dynamic,),
    "test_dropout_default_ratio_cpu": (test_static_dynamic,),
    # Other dopout test case failed: implementation is missing
    # mask is not supported for inference
    #"test_dropout_default_mask_cpu": (test_static_dynamic,),
    #"test_dropout_default_mask_ratio_cpu": (test_static_dynamic,),

    # Error: input arrays contain a mixture of endianness configuration
    #"test_training_dropout_default_cpu": (test_static_dynamic,),

    #"test_training_dropout_default_mask_cpu": (test_static_dynamic,),

    # Error: input arrays contain a mixture of endianness configuration
    #"test_training_dropout_default_cpu": (test_static_dynamic,),

    #"test_training_dropout_mask_cpu": (test_static_dynamic,),

    # Error: input arrays contain a mixture of endianness configuration
    #"test_training_dropout_zero_ratio_cpu": (test_static_dynamic,),

    #"test_training_dropout_zero_ratio_mask_cpu": (test_static_dynamic,),

    # DynamicQuantizeLinear

    # Edge

    # EinSum

    # Elu
    "test_elu_cpu": (test_static_dynamic,),
    "test_elu_default_cpu": (test_static_dynamic,),
    "test_elu_example_cpu": (test_static_dynamic,),

    # Equal

    # Erf
    "test_erf_cpu": (test_static_dynamic,),

    # Exp
    "test_exp_cpu": (test_static_dynamic,),
    "test_exp_example_cpu": (test_static_dynamic,),

    # Expand

    # Eyelike

    # Flatten
    "test_flatten_axis0_cpu": (test_static_dynamic,),
    "test_flatten_axis1_cpu": (test_static_dynamic,),
    "test_flatten_axis2_cpu": (test_static_dynamic,),
    "test_flatten_axis3_cpu": (test_static_dynamic,),
    "test_flatten_default_axis_cpu": (test_static_dynamic,),
    "test_flatten_negative_axis1_cpu": (test_static_dynamic,),
    "test_flatten_negative_axis2_cpu": (test_static_dynamic,),
    "test_flatten_negative_axis3_cpu": (test_static_dynamic,),
    "test_flatten_negative_axis4_cpu": (test_static_dynamic,),

    # Floor
    "test_floor_example_cpu": (test_static_dynamic,),
    "test_floor_cpu": (test_static_dynamic,),
    
    # Gather
    "test_gather_0_cpu": (test_static_dynamic,),
    "test_gather_1_cpu": (test_static_dynamic,),
    "test_gather_negative_indices_cpu": (test_static_dynamic,),

    # Gemm
    "test_gemm_all_attributes_cpu": (test_static_dynamic,),
    "test_gemm_alpha_cpu": (test_static_dynamic,),
    "test_gemm_beta_cpu": (test_static_dynamic,),
    "test_gemm_default_matrix_bias_cpu": (test_static_dynamic,),
    "test_gemm_default_no_bias_cpu": (test_static_dynamic,),
    "test_gemm_default_scalar_bias_cpu": (test_static_dynamic,),
    "test_gemm_default_single_elem_vector_bias_cpu": (test_static_dynamic,),
    "test_gemm_default_vector_bias_cpu": (test_static_dynamic,),
    "test_gemm_default_zero_bias_cpu": (test_static_dynamic,),
    "test_gemm_transposeA_cpu": (test_static_dynamic,),
    "test_gemm_transposeB_cpu": (test_static_dynamic,),

    # Global Average Pool
    "test_globalaveragepool_cpu": (test_static_dynamic,),
    "test_globalaveragepool_precomputed_cpu": (test_static_dynamic,),

    # Global Max Pool
    "test_globalmaxpool_cpu": (test_static_dynamic,),
    "test_globalmaxpool_precomputed_cpu": (test_static_dynamic,),

    # Greater

    # GRU
    "test_gru_defaults_cpu": (test_static_dynamic,{0:{0,1,2}}),
    "test_gru_seq_length_cpu": (test_static_dynamic,{0:{0,1,2}}),
    "test_gru_with_initial_bias_cpu": (test_static_dynamic,{0:{0,1,2}}),

    # Hard Max

    # Hard Sigmoid
    "test_hardsigmoid_cpu": (test_static_dynamic,),
    "test_hardsigmoid_default_cpu": (test_static_dynamic,),
    "test_hardsigmoid_example_cpu": (test_static_dynamic,),

    # Identity
    "test_identity_cpu": (test_static_dynamic,),

    # Instance Norm

    # Is Inf Neg/Pos

    # Is Nan

    # Leaky Relu
    "test_leakyrelu_cpu": (test_static_dynamic,),
    "test_leakyrelu_default_cpu": (test_static_dynamic,),
    "test_leakyrelu_example_cpu": (test_static_dynamic,),

    # Less
    "test_less_cpu": (test_static_dynamic,),
    "test_less_bcast_cpu": (test_static_dynamic,),

    # Log
    "test_log_example_cpu": (test_static_dynamic,),
    "test_log_cpu": (test_static_dynamic,),

    # LogSoftmax
    "test_logsoftmax_axis_0_cpu": (test_static_dynamic,),
    "test_logsoftmax_axis_1_cpu": (test_static_dynamic,),
    "test_logsoftmax_axis_2_cpu": (test_static_dynamic,),
    "test_logsoftmax_example_1_cpu": (test_static_dynamic,),
    "test_logsoftmax_default_axis_cpu": (test_static_dynamic,),
    "test_logsoftmax_negative_axis_cpu": (test_static_dynamic,),
    "test_logsoftmax_large_number_cpu": (test_static_dynamic,),

    # LRN
    "test_lrn_cpu": (test_static_dynamic,),
    "test_lrn_default_cpu": (test_static_dynamic,),
    

    # LSTM
    "test_lstm_defaults_cpu": (test_static_dynamic,{0:{0,1,2}}),
    "test_lstm_with_initial_bias_cpu": (test_static_dynamic,{0:{0,1,2}}),
    "test_lstm_with_peepholes_cpu": (test_static_dynamic,{0:{0,1,2}}),

    # Matmul
    "test_matmul_2d_cpu": (test_static_dynamic,),
    "test_matmul_3d_cpu": (test_static_dynamic,),
    "test_matmul_4d_cpu": (test_static_dynamic,),

    # Matmul Integer

    # Max
    "test_max_example_cpu": (test_static_dynamic,),
    "test_max_one_input_cpu": (test_static_dynamic,),
    "test_max_two_inputs_cpu": (test_static_dynamic,),

    # MaxPoolSingleOut
    "test_maxpool_1d_default_cpu": (test_static_dynamic,),
    "test_maxpool_2d_ceil_cpu": (test_static_dynamic,),
    "test_maxpool_2d_default_cpu": (test_static_dynamic,),
    "test_maxpool_2d_dilations_cpu": (test_static_dynamic,),
    "test_maxpool_2d_pads_cpu": (test_static_dynamic,),
    "test_maxpool_2d_precomputed_pads_cpu": (test_static_dynamic,),
    "test_maxpool_2d_precomputed_same_upper_cpu": (test_static_dynamic,),
    "test_maxpool_2d_precomputed_strides_cpu": (test_static_dynamic,),
    "test_maxpool_2d_same_lower_cpu": (test_static_dynamic,),
    "test_maxpool_2d_same_upper_cpu": (test_static_dynamic,),
    "test_maxpool_2d_strides_cpu": (test_static_dynamic,),
    "test_maxpool_3d_default_cpu": (test_static_dynamic,),

    # Mean

    # Min
    "test_min_example_cpu": (test_static_dynamic,),
    "test_min_one_input_cpu": (test_static_dynamic,),
    "test_min_two_inputs_cpu": (test_static_dynamic,),

    # Mod

    # Momentum

    # Mul
    "test_mul_cpu": (test_static_dynamic,),
    "test_mul_bcast_cpu": (test_static_dynamic,),
    "test_mul_example_cpu": (test_static_dynamic,),

    # Multinomial (NMV)

    # Neg
    "test_neg_example_cpu": (test_static_dynamic,),
    "test_neg_cpu": (test_static_dynamic,),

    # Negative Log Likelihood Loss

    # Non Max Supression

    # Non Zero

    # Not

    # One Hot

    # Or
    "test_or2d_cpu": (test_static_dynamic,),
    "test_or3d_cpu": (test_static_dynamic,),
    "test_or4d_cpu": (test_static_dynamic,),
    "test_or_bcast3v1d_cpu": (test_static_dynamic,),
    "test_or_bcast3v2d_cpu": (test_static_dynamic,),
    "test_or_bcast4v2d_cpu": (test_static_dynamic,),
    "test_or_bcast4v3d_cpu": (test_static_dynamic,),
    "test_or_bcast4v4d_cpu": (test_static_dynamic,),

    # Pad (not working)
    #"test_constant_pad_cpu": test_static_dynamic,
    #"test_edge_pad_cpu": test_static_dynamic,
    #"test_reflect_pad_cpu": test_static_dynamic,

    # Pow
    "test_pow_cpu": (test_static_dynamic,),
    "test_pow_example_cpu": (test_static_dynamic,),
    "test_pow_bcast_scalar_cpu": (test_static_dynamic,),
    "test_pow_bcast_array_cpu": (test_static_dynamic,),
    # Does not support integer power yet

    # PRelu
    "test_prelu_example_cpu": (test_static_dynamic,),
    "test_prelu_broadcast_cpu": (test_static_dynamic,),

    # QLinear Conv

    # QLinear Matmul

    # Quantize Linear

    # Reciprocal Op:
    "test_reciprocal_cpu": (test_static_dynamic,),
    "test_reciprocal_example_cpu": (test_static_dynamic,),

    # ReduceL1
    "test_reduce_l1_default_axes_keepdims_example_cpu": (test_static_dynamic,),
    "test_reduce_l1_default_axes_keepdims_random_cpu": (test_static_dynamic,),
    "test_reduce_l1_do_not_keepdims_example_cpu": (test_static_dynamic,),
    "test_reduce_l1_do_not_keepdims_random_cpu": (test_static_dynamic,),
    "test_reduce_l1_keep_dims_example_cpu": (test_static_dynamic,),
    "test_reduce_l1_keep_dims_random_cpu": (test_static,),
    "test_reduce_l1_negative_axes_keep_dims_example_cpu": (test_static_dynamic,),
    "test_reduce_l1_negative_axes_keep_dims_random_cpu": (test_static_dynamic,),

    # ReduceL2
    "test_reduce_l2_default_axes_keepdims_example_cpu": (test_static_dynamic,),
    "test_reduce_l2_default_axes_keepdims_random_cpu": (test_static_dynamic,),
    "test_reduce_l2_do_not_keepdims_example_cpu": (test_static_dynamic,),
    "test_reduce_l2_do_not_keepdims_random_cpu": (test_static_dynamic,),
    "test_reduce_l2_keep_dims_example_cpu": (test_static_dynamic,),
    "test_reduce_l2_keep_dims_random_cpu": (test_static_dynamic,),
    "test_reduce_l2_negative_axes_keep_dims_example_cpu": (test_static_dynamic,),
    "test_reduce_l2_negative_axes_keep_dims_random_cpu": (test_static_dynamic,),

    # ReduceLogSum
    "test_reduce_log_sum_asc_axes_cpu": (test_static_dynamic,),
    "test_reduce_log_sum_cpu": (test_static_dynamic,),
    "test_reduce_log_sum_default_cpu": (test_static_dynamic,),
    "test_reduce_log_sum_desc_axes_cpu": (test_static_dynamic,),

    # ReduceLogSumExp
    "test_reduce_log_sum_exp_default_axes_keepdims_example_cpu": (test_static_dynamic,),
    "test_reduce_log_sum_exp_default_axes_keepdims_random_cpu": (test_static_dynamic,),
    "test_reduce_log_sum_exp_do_not_keepdims_example_cpu": (test_static_dynamic,),
    "test_reduce_log_sum_exp_do_not_keepdims_random_cpu": (test_static_dynamic,),
    "test_reduce_log_sum_exp_keepdims_example_cpu": (test_static_dynamic,),
    "test_reduce_log_sum_exp_keepdims_random_cpu": (test_static_dynamic,),
    "test_reduce_log_sum_exp_negative_axes_keepdims_example_cpu": (test_static_dynamic,),
    "test_reduce_log_sum_exp_negative_axes_keepdims_random_cpu": (test_static_dynamic,),
    "test_reduce_log_sum_negative_axes_cpu": (test_static_dynamic,),

    # ReduceMax
    "test_reduce_max_default_axes_keepdim_example_cpu": (test_static_dynamic,),
    "test_reduce_max_default_axes_keepdims_random_cpu": (test_static_dynamic,),
    "test_reduce_max_do_not_keepdims_example_cpu": (test_static_dynamic,),
    "test_reduce_max_do_not_keepdims_random_cpu": (test_static_dynamic,),
    "test_reduce_max_keepdims_example_cpu": (test_static_dynamic,),
    "test_reduce_max_keepdims_random_cpu": (test_static_dynamic,),
    "test_reduce_max_negative_axes_keepdims_example_cpu": (test_static_dynamic,),
    "test_reduce_max_negative_axes_keepdims_random_cpu": (test_static_dynamic,),

    # ReduceMean
    "test_reduce_mean_default_axes_keepdims_example_cpu": (test_static_dynamic,),
    "test_reduce_mean_default_axes_keepdims_random_cpu": (test_static_dynamic,),
    "test_reduce_mean_do_not_keepdims_example_cpu": (test_static_dynamic,),
    "test_reduce_mean_do_not_keepdims_random_cpu": (test_static_dynamic,),
    "test_reduce_mean_keepdims_example_cpu": (test_static_dynamic,),
    "test_reduce_mean_keepdims_random_cpu": (test_static_dynamic,),
    "test_reduce_mean_negative_axes_keepdims_example_cpu": (test_static_dynamic,),
    "test_reduce_mean_negative_axes_keepdims_random_cpu": (test_static_dynamic,),

    # ReduceMin
    "test_reduce_min_default_axes_keepdims_example_cpu": (test_static_dynamic,),
    "test_reduce_min_default_axes_keepdims_random_cpu": (test_static_dynamic,),
    "test_reduce_min_do_not_keepdims_example_cpu": (test_static_dynamic,),
    "test_reduce_min_do_not_keepdims_random_cpu": (test_static_dynamic,),
    "test_reduce_min_keepdims_example_cpu": (test_static_dynamic,),
    "test_reduce_min_keepdims_random_cpu": (test_static_dynamic,),
    "test_reduce_min_negative_axes_keepdims_example_cpu": (test_static_dynamic,),
    "test_reduce_min_negative_axes_keepdims_random_cpu": (test_static_dynamic,),

    # ReduceProd
    "test_reduce_prod_default_axes_keepdims_example_cpu": (test_static_dynamic,),
    "test_reduce_prod_default_axes_keepdims_random_cpu": (test_static_dynamic,),
    "test_reduce_prod_do_not_keepdims_example_cpu": (test_static_dynamic,),
    "test_reduce_prod_do_not_keepdims_random_cpu": (test_static_dynamic,),
    "test_reduce_prod_keepdims_example_cpu": (test_static_dynamic,),
    "test_reduce_prod_keepdims_random_cpu": (test_static_dynamic,),
    "test_reduce_prod_negative_axes_keepdims_example_cpu": (test_static_dynamic,),
    "test_reduce_prod_negative_axes_keepdims_random_cpu": (test_static_dynamic,),

    # ReduceSum
    "test_reduce_sum_default_axes_keepdims_example_cpu": (test_static_dynamic,),
    "test_reduce_sum_default_axes_keepdims_random_cpu": (test_static_dynamic,),
    "test_reduce_sum_do_not_keepdims_example_cpu": (test_static_dynamic,),
    "test_reduce_sum_do_not_keepdims_random_cpu": (test_static_dynamic,),
    "test_reduce_sum_keepdims_example_cpu": (test_static_dynamic,),
    "test_reduce_sum_keepdims_random_cpu": (test_static_dynamic,),
    "test_reduce_sum_negative_axes_keepdims_example_cpu": (test_static_dynamic,),
    "test_reduce_sum_negative_axes_keepdims_random_cpu": (test_static_dynamic,),

    # ReduceSumSquare
    "test_reduce_sum_square_default_axes_keepdims_example_cpu": (test_static_dynamic,),
    "test_reduce_sum_square_default_axes_keepdims_random_cpu": (test_static_dynamic,),
    "test_reduce_sum_square_do_not_keepdims_example_cpu": (test_static_dynamic,),
    "test_reduce_sum_square_do_not_keepdims_random_cpu": (test_static_dynamic,),
    "test_reduce_sum_square_keepdims_example_cpu": (test_static_dynamic,),
    "test_reduce_sum_square_keepdims_random_cpu": (test_static_dynamic,),
    "test_reduce_sum_square_negative_axes_keepdims_example_cpu": (test_static_dynamic,),
    "test_reduce_sum_square_negative_axes_keepdims_random_cpu": (test_static_dynamic,),

    # Relu
    "test_relu_cpu": (test_static_dynamic,),

    # Reshape
    "test_reshape_extended_dims_cpu": (test_static_dynamic,{0:{-1}}),
    "test_reshape_negative_dim_cpu": (test_static_dynamic,{0:{-1}}),
    "test_reshape_negative_extended_dims_cpu": (test_static_dynamic,{0:{-1}}),
    "test_reshape_one_dim_cpu": (test_static_dynamic,{0:{-1}}),
    "test_reshape_reduced_dims_cpu": (test_static_dynamic,{0:{-1}}),
    "test_reshape_reordered_all_dims_cpu": (test_static_dynamic,{0:{-1}}),
    "test_reshape_reordered_last_dims_cpu": (test_static_dynamic,{0:{-1}}),
    "test_reshape_zero_and_negative_dim_cpu": (test_static_dynamic,{0:{-1}}),
    "test_reshape_zero_dim_cpu": (test_static_dynamic,{0:{-1}}),

    # Resize

    # Reverse Sequence

    # RNN
    "test_rnn_seq_length_cpu": (test_static_dynamic,{0:{0,1,2}}),
    "test_simple_rnn_defaults_cpu": (test_static_dynamic,{0:{0,1,2}}),
    "test_simple_rnn_with_initial_bias_cpu": (test_static_dynamic,{0:{0,1,2}}),

    # Roi Align

    # Round

    # Scan

    # Scatter Element

    # Selu
    "test_selu_cpu": (test_static_dynamic,),
    "test_selu_default_cpu": (test_static_dynamic,),
    "test_selu_example_cpu": (test_static_dynamic,),

    # Shape
    "test_shape_cpu": (test_static_dynamic,), 
    "test_shape_example_cpu": (test_static_dynamic,), 

    # Shrink

    # Sigmoid
    "test_sigmoid_cpu": (test_static_dynamic,),
    "test_sigmoid_example_cpu": (test_static_dynamic,),

    # Sign
    "test_sign_cpu": (test_static_dynamic,),

    # Sin
    "test_sin_example_cpu": (test_static_dynamic,),
    "test_sin_cpu": (test_static_dynamic,),

    # Sinh
    "test_sinh_cpu": (test_static_dynamic,),
    "test_sinh_example_cpu": (test_static_dynamic,),

    # Size
    "test_size_cpu": (test_static_dynamic,),
    "test_size_example_cpu": (test_static_dynamic,),

    # Slice (makes Axis a runtime argument, which is not supported).

    # Softmax
    "test_softmax_axis_0_cpu": (test_static_dynamic,),
    "test_softmax_axis_1_cpu": (test_static_dynamic,),
    "test_softmax_axis_2_cpu": (test_static_dynamic,),
    "test_softmax_default_axis_cpu": (test_static_dynamic,),
    "test_softmax_example_cpu": (test_static_dynamic,),
    "test_softmax_large_number_cpu": (test_static_dynamic,),

    # Softplus
    "test_softplus_cpu": (test_static_dynamic,),
    "test_softplus_example_cpu": (test_static_dynamic,),

    # Softsign
    "test_softsign_cpu": (test_static_dynamic,),
    "test_softsign_example_cpu": (test_static_dynamic,),

    # Split
    "test_split_equal_parts_1d_cpu": (test_static_dynamic,),
    "test_split_equal_parts_2d_cpu": (test_static_dynamic,),
    "test_split_equal_parts_default_axis_cpu": (test_static_dynamic,),
    "test_split_variable_parts_1d_cpu": (test_static_dynamic,),
    "test_split_variable_parts_2d_cpu": (test_static_dynamic,),
    "test_split_variable_parts_default_axis_cpu": (test_static_dynamic,),
    
    # Sqrt
    "test_sqrt_cpu": (test_static_dynamic,),
    "test_sqrt_example_cpu": (test_static_dynamic,),

    # Squeeze
    "test_squeeze_cpu": (test_static_dynamic,),
    "test_squeeze_negative_axes_cpu": (test_static_dynamic,),

    # Str Normalizer

    # Sub
    "test_sub_cpu": (test_static_dynamic,),
    "test_sub_bcast_cpu": (test_static_dynamic,),
    "test_sub_example_cpu": (test_static_dynamic,),

    # Sum
    "test_sum_example_cpu": (test_static_dynamic,),
    "test_sum_one_input_cpu": (test_static_dynamic,),
    "test_sum_two_inputs_cpu": (test_static_dynamic,),

    # Tan
    "test_tan_cpu": (test_static_dynamic,),
    "test_tan_example_cpu": (test_static_dynamic,),

    # Tanh
    "test_tanh_cpu": (test_static_dynamic,),
    "test_tanh_example_cpu": (test_static_dynamic,),

    # Tfdf Vectorizer

    # Threshold Relu

    # Tile
    "test_tile_cpu": (test_static_dynamic,),
    "test_tile_precomputed_cpu": (test_static_dynamic,),

    # TopK

    # Training Dropout

    # Transpose
    "test_transpose_default_cpu": (test_static_dynamic,),
    "test_transpose_all_permutations_0_cpu": (test_static_dynamic,),
    "test_transpose_all_permutations_1_cpu": (test_static_dynamic,),
    "test_transpose_all_permutations_2_cpu": (test_static_dynamic,),
    "test_transpose_all_permutations_3_cpu": (test_static_dynamic,),
    "test_transpose_all_permutations_4_cpu": (test_static_dynamic,),
    "test_transpose_all_permutations_5_cpu": (test_static_dynamic,),

    # Unique

    # Unsqueeze
    "test_unsqueeze_axis_0_cpu": (test_static_dynamic,),
    "test_unsqueeze_axis_1_cpu": (test_static_dynamic,),
    "test_unsqueeze_axis_2_cpu": (test_static_dynamic,),
    "test_unsqueeze_axis_3_cpu": (test_static_dynamic,),
    "test_unsqueeze_negative_axes_cpu": (test_static_dynamic,),
    "test_unsqueeze_three_axes_cpu": (test_static_dynamic,),
    "test_unsqueeze_two_axes_cpu": (test_static_dynamic,),
    "test_unsqueeze_unsorted_axes_cpu": (test_static_dynamic,),

    # Upsample

    # Where

    # Xor
    "test_xor2d_cpu": (test_static_dynamic,),
    "test_xor3d_cpu": (test_static_dynamic,),
    "test_xor4d_cpu": (test_static_dynamic,),
    "test_xor_bcast3v1d_cpu": (test_static_dynamic,),
    "test_xor_bcast3v2d_cpu": (test_static_dynamic,),
    "test_xor_bcast4v2d_cpu": (test_static_dynamic,),
    "test_xor_bcast4v3d_cpu": (test_static_dynamic,),
    "test_xor_bcast4v4d_cpu": (test_static_dynamic,),

    ############################################################
    # Model (alphabetical order)

    "test_shufflenet_cpu": (test_static,),
    "test_resnet50_cpu": (test_static,),
    "test_vgg19_cpu": (test_static,),
    "test_densenet121_cpu": (test_static,),
    "test_inception_v1_cpu": (test_static,),
}

# test for static
test_to_enable = [ key for (key, value) in test_to_enable_static_dynamic.items() if value[0] & test_static ]

# Specify the test cases which currently can not pass for dynamic shape
# Presumably, this list should be empty
# Except for some operation too difficult to handle for dynamic shape
# or big models
test_for_dynamic = [ key for (key, value) in test_to_enable_static_dynamic.items() if value[0] & test_dynamic ]

if args.dynamic :
    print("dynamic shape is enabled")
    test_to_enable = test_for_dynamic 

# User case specify one test case with BCKEND_TEST env
if TEST_CASE_BY_USER is not None and TEST_CASE_BY_USER != "" :
    test_to_enable = TEST_CASE_BY_USER.split()

# determine the dynamic input and dim
def determine_dynamic_parameters(test_name):
    if not args.dynamic :
        return None
    # set default value: all inputs, first dimension.
    # Use this script's arguments '--input' and '--dim' or environment variables
    # TEST_INPUT and TEST_DIM to control the values.
    selected_list = {args.input: {args.dim}}
    test_name_cpu = test_name + "_cpu"
    if test_name_cpu in test_for_dynamic:
        if len(test_to_enable_static_dynamic[test_name_cpu]) > 1:
            selected_list = test_to_enable_static_dynamic[test_name_cpu][1]
    return selected_list 

def execute_commands(cmds, dynamic_inputs_dims):
    if (args.verbose):
        print(" ".join(cmds))
        print("IMPORTER FORCE DYNAMIC ", dynamic_inputs_dims)
    my_env = os.environ.copy();
    env_string = ""
    if dynamic_inputs_dims is not None:
        first_input = True;
        for (input_index, dim_indices) in dynamic_inputs_dims.items():
            if first_input:
                env_string += str(input_index)
                first_input = False
            else:
                env_string += "|" + str(input_index)
            first_dim = True
            for dim_index in dim_indices:
                if first_dim:
                   env_string +=  ":" + str(dim_index)
                   first_dim = False
                else:
                   env_string += "," + str(dim_index)
        my_env["IMPORTER_FORCE_DYNAMIC"] = env_string 
    subprocess.run(cmds, env=my_env)


# There are two issues, which necessitates the adoption of this endianness
# aware wrapper around Execution Session:
# 1. Input arrays are given sometimes in native byte order, sometime in
#    LE byte order, and as soon as the python array enters into py::array
#    C++ objects through pybind, we will no longer be able to query their
#    endianness. So we must intercept the inputs and convert them into
#    native endianness.
# 2. Output arrays are compared with reference outputs, the comparison
#    unfortunately includes checking that our outputs and reference outputs
#    share the same endianness. So we try to figure out what is the desired
#    reference output endianness, and convert our outputs to this desired
#    endianness.
class EndiannessAwareExecutionSession(ExecutionSession):
    def __init__(self, path, entry_point):
        super().__init__(path, entry_point)

    def is_input_le(self, inputs):
        inputs_endianness = list(map(lambda x: x.dtype.byteorder, inputs))
        endianness_is_consistent = len(set(inputs_endianness)) <= 1
        assert endianness_is_consistent, \
            "Input arrays contain a mixture of endianness configuration."

        sys_is_le = sys.byteorder == 'little'
        # To interpret character symbols indicating endianness:
        # https://numpy.org/doc/stable/reference/generated/numpy.dtype.byteorder.html
        explicitly_le = inputs_endianness[0] == "<"
        implicitly_le = (inputs_endianness[0] == "=" and sys_is_le)
        return explicitly_le or implicitly_le

    def run(self, inputs, **kwargs):
        if len(inputs):
            # Deduce desired endianness of output from inputs.
            sys_is_le = sys.byteorder == 'little'
            inp_is_le = self.is_input_le(inputs)
            if (sys_is_le != inp_is_le):
                inputs = list(
                    map(lambda x: x.byteswap().newbyteorder(), inputs))
            outputs = super().run(inputs)
            if (sys_is_le != inp_is_le):
                outputs = list(
                    map(lambda x: x.byteswap().newbyteorder(), outputs))
            return outputs
        else:
            # Can't deduce desired output endianess, fingers crossed.
            warnings.warn(
                "Cannot deduce desired output endianness, using native endianness by default."
            )
            return super().run(inputs)


class DummyBackend(onnx.backend.base.Backend):
    @classmethod
    def prepare(cls, model, device='CPU', **kwargs):
        super(DummyBackend, cls).prepare(model, device, **kwargs)
        name = model.graph.name
        model_name = result_dir+name+".onnx"
        exec_name = result_dir+name + ".so"
        # Clean the temporary files in case
        # Save model to disk as temp_model.onnx.
        onnx.save(model, model_name)
        if not os.path.exists(model_name) :
            print("Failed save model: "+ name)
        print(name)

        # Call frontend to process temp_model.onnx, bit code will be generated.
        dynamic_inputs_dims = determine_dynamic_parameters(name)
        execute_commands([TEST_DRIVER, model_name], dynamic_inputs_dims)
        if not os.path.exists(exec_name) :
            print("Failed " + test_config.TEST_DRIVER_COMMAND + ": " + name)
        return EndiannessAwareExecutionSession(exec_name,
                                                   "run_main_graph")

    @classmethod
    def supports_device(cls, device):
        d = Device(device)
        if d.type == DeviceType.CPU:
            return True
        return False


backend_test = onnx.backend.test.BackendTest(DummyBackend, __name__)

# Extract name of all test cases.
import inspect
all_tests = []
all_tests += inspect.getmembers(
    backend_test.test_cases["OnnxBackendRealModelTest"])
all_tests += inspect.getmembers(
    backend_test.test_cases["OnnxBackendNodeModelTest"])
all_test_names = list(map(lambda x: x[0], all_tests))

# Ensure that test names specified in test_to_enable actually exist.
for test_name in test_to_enable:
    assert test_name in all_test_names, """test name {} not found, it is likely
    that you may have misspelled the test name or the specified test does not
    exist in the version of onnx package you installed.""".format(test_name)
    backend_test.include(r"^{}$".format(test_name))

# import all test cases at global scope to make them visible to python.unittest
globals().update(backend_test.test_cases)

if __name__ == '__main__':

    unittest.main()
