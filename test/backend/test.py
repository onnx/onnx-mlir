from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from distutils.util import strtobool
import os
import sys
import onnx
import unittest
import warnings
import onnx.backend.base
import onnx.backend.test
from onnx import numpy_helper

from onnx.backend.base import Device, DeviceType
import subprocess
import tempfile
import argparse

# Casting with "bool" does not work well. When you specify VERBOSE=xxx,
# regardless of the value of xxx (e.g., true, false, y, n, etc.) the
# casted bool value will be true. Only if xxx is empty, the casted bool
# value will be false. This is a bit counter intuitive. So we use strtobool
# to do the conversion. But note that strtobool can't take an emtpy string.

VERBOSE = os.getenv("VERBOSE")
INVOKECONVERTER = os.getenv("INVOKECONVERTER")
IMPORTER_FORCE_DYNAMIC = os.getenv("IMPORTER_FORCE_DYNAMIC")
# Force input tensors to constants. Set this to a list of input indices.
# E.g.
#   - "0, 2" for the first and third input tensors.
#   - "-1" for all the input tensors.
IMPORTER_FORCE_CONSTANT = os.getenv("IMPORTER_FORCE_CONSTANT")
TEST_DYNAMIC = os.getenv("TEST_DYNAMIC")
TEST_CONSTANT = os.getenv("TEST_CONSTANT")
TEST_COMPILERLIB = os.getenv("TEST_COMPILERLIB")

parser = argparse.ArgumentParser(description='with dynamic shape or not.')
parser.add_argument('--dynamic', action='store_true',
    default=(strtobool(TEST_DYNAMIC) if TEST_DYNAMIC else False),
    help='enable dynamic shape tests (default: false if TEST_DYNAMIC env var not set)')
parser.add_argument('--constant', action='store_true',
    default=(strtobool(TEST_CONSTANT) if TEST_CONSTANT else False),
    help='enable constant input tests (default: false if TEST_CONSTANT env var not set)')
parser.add_argument('--compilerlib', action='store_true',
    default=(strtobool(TEST_COMPILERLIB) if TEST_COMPILERLIB else False),
    help='enable compiler lib tests (default: false if TEST_COMPILERLIB env var not set)')
parser.add_argument('-i', '--input', type=int,
    default=os.getenv("TEST_INPUT", -1),
    help='inputs whose dimensions to be changed to unknown (default: all inputs if TEST_INPUT env var not set)')
parser.add_argument('-d', '--dim', type=int,
    default=os.getenv("TEST_DIM", -1),
    help='dimensions to be changed to unknown (default: all dimensions if TEST_DIM env var not set)')
parser.add_argument('-v', '--verbose', action='store_true',
    default=(strtobool(VERBOSE) if VERBOSE else False),
    help='verbose output (default: false if VERBOSE env var not set)')
parser.add_argument('--mtriple', type=str, default=os.getenv("TEST_MTRIPLE", ""),
    help='triple to pass to the compiler')
parser.add_argument('--mcpu', type=str, default=os.getenv("TEST_MCPU", ""),
    help='target a specific cpu, passed to the compiler')
parser.add_argument('--converter', action='store_true',
    default=(strtobool(INVOKECONVERTER) if INVOKECONVERTER else False),
    help='invoke version converter (default: false if INVOKECONVERTER env var not set)')
parser.add_argument('unittest_args', nargs='*')
args = parser.parse_args()
sys.argv[1:] = args.unittest_args

TEST_CASE_BY_USER = os.getenv("TEST_CASE_BY_USER")
if TEST_CASE_BY_USER is not None and TEST_CASE_BY_USER != "" :
    result_dir = "./"
else :
    tempdir = tempfile.TemporaryDirectory()
    result_dir = tempdir.name+"/"
print("Test info:")
print("  temporary results are in dir:"+result_dir)
if args.mcpu:
    print("  targeting cpu:", args.mcpu)
if args.mtriple:
    print("  targeting triple:", args.mtriple)

if args.compilerlib:
    import test_config_compilerlib
    CXX = test_config_compilerlib.CXX_PATH
    LLC = test_config_compilerlib.LLC_PATH
    RUNTIME_DIR = test_config_compilerlib.TEST_DRIVER_RUNTIME_PATH
    TEST_DRIVER = test_config_compilerlib.TEST_DRIVER_PATH
else:
    import test_config
    CXX = test_config.CXX_PATH
    LLC = test_config.LLC_PATH
    RUNTIME_DIR = test_config.TEST_DRIVER_RUNTIME_PATH
    TEST_DRIVER = test_config.TEST_DRIVER_PATH


# Make lib folder under build directory visible in PYTHONPATH
doc_check_base_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(RUNTIME_DIR)
from PyRuntime import ExecutionSession

# Test directories:
# https://github.com/onnx/onnx/tree/master/onnx/backend/test/data/node
# In our directories, the python files that generate the tests are found here
# onnx-mlir/third_party/onnx/onnx/backend/test/case/node

# Each benchmark is defined by a dictionary element: `key:value`, where
# - key: the ONNX testname
# - value: a dictionary accepting only three key "static", "dynamic", and
#          "constant" to enable testings for static, dynamic, and constant
#          inputs, respectively.
# When a dynamic or constant testing is enabled, we must enter the indices of
# the tensor expected to be dynamic or constant.
# Indices start from 0. -1 means all inputs or all dimensions.
#
# Value for "static" key is not taken into account. So empty set {} is O.K.
#
# Value for "dynamic" key is a dict to define which inputs/dimensions are changed
# to unknown, where its key is an input index and its value is a set of
# dimension indices, e.g. {0:{0,1}, 1:{-1}, 2:{0}}
#
# Value for "constant" key is set of indices, e.g. {0, 2, 3}

STATIC_SHAPE="static"
DYNAMIC_SHAPE="dynamic"
CONSTANT_INPUT="constant"
test_to_enable_dict = {

    ############################################################
    # Elementary ops, ordered alphabetically.

    # Abs
    "test_abs_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},

    # Acos
    "test_acos_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},
    "test_acos_example_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},

    # Acosh
    "test_acosh_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},
    "test_acosh_example_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},

    # Adagrad

    # Adam

    # Add
    "test_add_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},
    "test_add_bcast_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},

    # And
    "test_and2d_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},
    "test_and3d_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},
    "test_and4d_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},
    "test_and_bcast3v1d_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},
    "test_and_bcast3v2d_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},
    "test_and_bcast4v2d_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},
    "test_and_bcast4v3d_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},
    "test_and_bcast4v4d_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},

    # Argmax
    "test_argmax_no_keepdims_example_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},
    "test_argmax_keepdims_example_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},
    "test_argmax_default_axis_example_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},

    # Argmin

    # Asin
    "test_asin_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},
    "test_asin_example_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},

    # Asinh
    "test_asinh_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},
    "test_asinh_example_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},

    # Atan
    "test_atan_example_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},
    "test_atan_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},

    # Atanh
    "test_atanh_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},
    "test_atanh_example_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},

    # AveragePool: same_upper/lower dyn padding-shapes not supported.
    "test_averagepool_1d_default_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},
    "test_averagepool_2d_ceil_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},
    "test_averagepool_2d_default_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},
    "test_averagepool_2d_pads_count_include_pad_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},
    "test_averagepool_2d_pads_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},
    "test_averagepool_2d_precomputed_pads_count_include_pad_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},
    "test_averagepool_2d_precomputed_pads_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},
    "test_averagepool_2d_precomputed_same_upper_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},
    "test_averagepool_2d_precomputed_strides_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},
    "test_averagepool_2d_same_lower_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},
    "test_averagepool_2d_same_upper_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},
    "test_averagepool_2d_strides_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},
    "test_averagepool_3d_default_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},

    # BatchNormalization (test mode)
    "test_batchnorm_epsilon_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},
    "test_batchnorm_example_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},

    # Bitshift left/right

    # Cast
    "test_cast_FLOAT_to_DOUBLE_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},
    "test_cast_DOUBLE_to_FLOAT_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},
    "test_cast_FLOAT_to_FLOAT16_cpu": {}, # appers unsupported at this time
    "test_cast_FLOAT16_to_FLOAT_cpu": {}, # appers unsupported at this time
    "test_cast_FLOAT16_to_DOUBLE_cpu": {}, # appers unsupported at this time
    "test_cast_DOUBLE_to_FLOAT16_cpu": {}, # appers unsupported at this time
    "test_cast_FLOAT_to_STRING_cpu": {}, # appers unsupported at this time
    "test_cast_STRING_to_FLOAT_cpu": {}, # appers unsupported at this time

    # Ceil
    "test_ceil_example_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},
    "test_ceil_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},

    # Celu

    # Clip
    "test_clip_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},
    "test_clip_example_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},
    "test_clip_inbounds_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},
    "test_clip_outbounds_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},
    "test_clip_splitbounds_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},
    "test_clip_default_min_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},
    "test_clip_default_max_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},
    "test_clip_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},
    "test_clip_default_inbounds_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},
    #"test_clip_default_int8_min_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}},

    # Compress
    "test_compress_0_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},
    "test_compress_1_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},
    "test_compress_default_axis_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},
    "test_compress_negative_axis_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},

    # Concat
    "test_concat_1d_axis_0_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{0:{0}}, CONSTANT_INPUT:{-1}},
    "test_concat_2d_axis_0_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{0:{0}}, CONSTANT_INPUT:{-1}},
    "test_concat_2d_axis_1_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{0:{1}}, CONSTANT_INPUT:{-1}},
    "test_concat_3d_axis_0_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{0:{0}}, CONSTANT_INPUT:{-1}},
    "test_concat_3d_axis_1_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{0:{1}}, CONSTANT_INPUT:{-1}},
    "test_concat_3d_axis_2_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{0:{2}}, CONSTANT_INPUT:{-1}},
    "test_concat_1d_axis_negative_1_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{0:{0}}, CONSTANT_INPUT:{-1}},
    "test_concat_2d_axis_negative_1_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{0:{1}}, CONSTANT_INPUT:{-1}},
    "test_concat_2d_axis_negative_2_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{0:{0}}, CONSTANT_INPUT:{-1}},
    "test_concat_3d_axis_negative_1_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{0:{2}}, CONSTANT_INPUT:{-1}},
    "test_concat_3d_axis_negative_2_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{0:{1}}, CONSTANT_INPUT:{-1}},
    "test_concat_3d_axis_negative_3_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{0:{0}}, CONSTANT_INPUT:{-1}},

    # Constant (dynamic NA)
    "test_constant_cpu": {STATIC_SHAPE:{}},

    # ConstantOfShape (dynamic NA)
    "test_constantofshape_float_ones_cpu": {STATIC_SHAPE:{}},
    "test_constantofshape_int_zeros_cpu": {STATIC_SHAPE:{}},

    # Conv.
    # CONSTANT_INPUT for weight.
    "test_basic_conv_with_padding_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{1}},
    "test_basic_conv_without_padding_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{1}},
    "test_conv_with_autopad_same_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{1}},
    "test_conv_with_strides_no_padding_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{1}},
    "test_conv_with_strides_padding_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{1}},
    "test_conv_with_strides_and_asymmetric_padding_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{1}},

    # ConvInteger

    # ConvTranspose

    # Cos
    "test_cos_example_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},
    "test_cos_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},

    # Cosh
    "test_cosh_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},
    "test_cosh_example_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},

    # CumSum
    "test_cumsum_1d_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},
    "test_cumsum_1d_exclusive_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},
    "test_cumsum_1d_reverse_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},
    "test_cumsum_1d_reverse_exclusive_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},
    "test_cumsum_2d_axis_0_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},
    "test_cumsum_2d_axis_1_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},
    "test_cumsum_2d_negative_axis_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},

    # DepthOfSpace
    "test_depthtospace_example_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},
    "test_depthtospace_crd_mode_example_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},

    # DequatizeLinear

    # Det

    # Div
    "test_div_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},
    "test_div_bcast_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},
    "test_div_example_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},

    # Dropout
    "test_dropout_default_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},
    "test_dropout_default_ratio_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},
    # Other dopout test case failed: implementation is missing
    # mask is not supported for inference
    #"test_dropout_default_mask_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}},
    #"test_dropout_default_mask_ratio_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}},

    # Error: input arrays contain a mixture of endianness configuration
    #"test_training_dropout_default_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}},

    #"test_training_dropout_default_mask_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}},

    # Error: input arrays contain a mixture of endianness configuration
    #"test_training_dropout_default_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}},

    #"test_training_dropout_mask_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}},

    # Error: input arrays contain a mixture of endianness configuration
    #"test_training_dropout_zero_ratio_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}},

    #"test_training_dropout_zero_ratio_mask_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}},

    # DynamicQuantizeLinear

    # Edge

    # EinSum

    # Elu
    "test_elu_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},
    "test_elu_default_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},
    "test_elu_example_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},

    # Equal
    "test_equal_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},
    "test_equal_bcast_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},

    # Erf
    "test_erf_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},

    # Exp
    "test_exp_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},
    "test_exp_example_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},

    # Expand
    "test_expand_dim_changed_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{0:{-1}}, CONSTANT_INPUT:{-1}},
    "test_expand_dim_unchanged_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{0:{-1}}, CONSTANT_INPUT:{-1}},

    # Eyelike

    # Flatten
    "test_flatten_axis0_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},
    "test_flatten_axis1_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},
    "test_flatten_axis2_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},
    "test_flatten_axis3_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},
    "test_flatten_default_axis_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},
    "test_flatten_negative_axis1_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},
    "test_flatten_negative_axis2_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},
    "test_flatten_negative_axis3_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},
    "test_flatten_negative_axis4_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},

    # Floor
    "test_floor_example_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},
    "test_floor_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},

    # Gather
    "test_gather_0_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},
    "test_gather_1_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},
    "test_gather_negative_indices_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},

    # Gemm
    "test_gemm_all_attributes_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},
    "test_gemm_alpha_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},
    "test_gemm_beta_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},
    "test_gemm_default_matrix_bias_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},
    "test_gemm_default_no_bias_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},
    "test_gemm_default_scalar_bias_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},
    "test_gemm_default_single_elem_vector_bias_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},
    "test_gemm_default_vector_bias_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},
    "test_gemm_default_zero_bias_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},
    "test_gemm_transposeA_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},
    "test_gemm_transposeB_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},

    # Global Average Pool
    "test_globalaveragepool_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},
    "test_globalaveragepool_precomputed_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},

    # Global Max Pool
    "test_globalmaxpool_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},
    "test_globalmaxpool_precomputed_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},

    # Greater
    "test_greater_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},
    "test_greater_bcast_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},
    "test_greater_equal_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},
    "test_greater_equal_bcast_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},
    "test_greater_equal_bcast_expanded_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},
    "test_greater_equal_expanded_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},

    # GRU
    # CONSTANT_INPUT for W and R.
    "test_gru_defaults_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{0:{0,1,2}}, CONSTANT_INPUT:{1,2}},
    "test_gru_seq_length_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{0:{0,1,2}}, CONSTANT_INPUT:{1,2}},
    "test_gru_with_initial_bias_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{0:{0,1,2}}, CONSTANT_INPUT:{1,2}},

    # Hard Max
    "test_hardmax_axis_0_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},
    "test_hardmax_axis_2_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},
    "test_hardmax_example_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},
    "test_hardmax_one_hot_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},
    "test_hardmax_axis_1_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},
    "test_hardmax_default_axis_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},
    "test_hardmax_negative_axis_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},

    # Hard Sigmoid
    "test_hardsigmoid_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},
    "test_hardsigmoid_default_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},
    "test_hardsigmoid_example_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},

    # Identity
    "test_identity_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},

    # Instance Norm
    "test_instancenorm_example_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},
    "test_instancenorm_epsilon_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},

    # Is Inf Neg/Pos

    # Is Nan

    # Leaky Relu
    "test_leakyrelu_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},
    "test_leakyrelu_default_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},
    "test_leakyrelu_example_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},

    # Less
    "test_less_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},
    "test_less_bcast_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},
    "test_less_equal_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},
    "test_less_equal_bcast_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},
    "test_less_equal_bcast_expanded_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},
    "test_less_equal_expanded_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},

    # Log
    "test_log_example_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},
    "test_log_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},

    # LogSoftmax
    # Temporally removed due to changes in onnx 1.8.1
    # "test_logsoftmax_axis_0_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},
    # "test_logsoftmax_axis_1_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},
    "test_logsoftmax_axis_2_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},
    "test_logsoftmax_example_1_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},
    # "test_logsoftmax_default_axis_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},
    "test_logsoftmax_negative_axis_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},
    "test_logsoftmax_large_number_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},

    # LRN
    "test_lrn_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},
    "test_lrn_default_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},


    # LSTM
    # CONSTANT_INPUT for W and R.
    "test_lstm_defaults_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{0:{0,1,2}}, CONSTANT_INPUT:{1,2}},
    "test_lstm_with_initial_bias_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{0:{0,1,2}}, CONSTANT_INPUT:{1,2}},
    "test_lstm_with_peepholes_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{0:{0,1,2}}, CONSTANT_INPUT:{1,2}},

    # Matmul
    "test_matmul_2d_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},
    "test_matmul_3d_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},
    "test_matmul_4d_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},

    # Matmul Integer

    # Max
    "test_max_example_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},
    "test_max_one_input_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},
    "test_max_two_inputs_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},
    # float16 failed on Z. It seems LLVM on Z does not have fp16 simulation.
    # "test_max_float16_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},
    "test_max_float32_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},
    "test_max_float64_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},
    "test_max_int8_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},
    "test_max_int16_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},
    "test_max_int32_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},
    "test_max_int64_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},
    # loc("onnx.Max"): error: 'std.cmpi' op operand #0 must be signless-integer-like, but got 'ui8'
    # MLIR integers are curretnly signless.
    # "test_max_uint8_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},
    # "test_max_uint16_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},
    # "test_max_uint32_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},
    # "test_max_uint64_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},

    # MaxPoolSingleOut: same_upper/lower dyn padding-shapes not supported.
    "test_maxpool_1d_default_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},
    "test_maxpool_2d_ceil_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},
    "test_maxpool_2d_default_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},
    "test_maxpool_2d_dilations_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},
    "test_maxpool_2d_pads_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},
    "test_maxpool_2d_precomputed_pads_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},
    "test_maxpool_2d_precomputed_same_upper_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},
    "test_maxpool_2d_precomputed_strides_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},
    "test_maxpool_2d_same_lower_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},
    "test_maxpool_2d_same_upper_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},
    "test_maxpool_2d_strides_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},
    "test_maxpool_3d_default_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},

    # Mean
    "test_mean_example_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},
    "test_mean_one_input_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},
    "test_mean_two_inputs_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},

    # Min
    "test_min_example_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},
    "test_min_one_input_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},
    "test_min_two_inputs_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},
    # float16 failed on Z. It seems LLVM on Z does not have fp16 simulation.
    # "test_min_float16_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},
    "test_min_float32_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},
    "test_min_float64_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},
    "test_min_int8_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},
    "test_min_int16_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},
    "test_min_int32_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},
    "test_min_int64_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},
    # loc("onnx.Min"): error: 'std.cmpi' op operand #0 must be signless-integer-like, but got 'ui8'
    # MLIR integers are curretnly signless.
    # "test_min_uint8_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},
    # "test_min_uint16_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},
    # "test_min_uint32_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},
    # "test_min_uint64_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},
    
    # Mod
    "test_mod_mixed_sign_float32_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},
    "test_mod_mixed_sign_float64_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},
    # float16 failed on Z. It seems LLVM on Z does not have fp16 simulation.
    # "test_mod_mixed_sign_float16_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},
    # Not yet support integers since MLIR integers are signless.
    # "test_mod_broadcast_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},
    # "test_mod_int64_fmod_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},
    # "test_mod_mixed_sign_int16_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},
    # "test_mod_mixed_sign_int32_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},
    # "test_mod_mixed_sign_int64_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},
    # "test_mod_mixed_sign_int8_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},
    # "test_mod_uint16_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},
    # "test_mod_uint32_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},
    # "test_mod_uint64_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},
    # "test_mod_uint8_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},

    # Momentum

    # Mul
    "test_mul_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},
    "test_mul_bcast_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},
    "test_mul_example_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},

    # Multinomial (NMV)

    # Neg
    "test_neg_example_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},
    "test_neg_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},

    # Negative Log Likelihood Loss

    # Non Max Supression
    "test_nonmaxsuppression_center_point_box_format_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},
    "test_nonmaxsuppression_flipped_coordinates_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},
    "test_nonmaxsuppression_identical_boxes_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},
    "test_nonmaxsuppression_limit_output_size_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},
    "test_nonmaxsuppression_single_box_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},
    "test_nonmaxsuppression_suppress_by_IOU_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},
    "test_nonmaxsuppression_suppress_by_IOU_and_scores_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},
    "test_nonmaxsuppression_two_batches_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},
    "test_nonmaxsuppression_two_classes_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},

    # Non Zero
    "test_nonzero_example_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},

    # Not
    "test_not_2d_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},
    "test_not_3d_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},
    "test_not_4d_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},

    # One Hot
    "test_onehot_without_axis_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},
    "test_onehot_with_axis_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},
    "test_onehot_negative_indices_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},
    "test_onehot_with_negative_axis_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},

    # Or
    "test_or2d_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},
    "test_or3d_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},
    "test_or4d_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},
    "test_or_bcast3v1d_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},
    "test_or_bcast3v2d_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},
    "test_or_bcast4v2d_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},
    "test_or_bcast4v3d_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},
    "test_or_bcast4v4d_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},

    # Pad
    "test_constant_pad_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},
    "test_edge_pad_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},
    "test_reflect_pad_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},

    # Pow
    "test_pow_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},
    "test_pow_example_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},
    "test_pow_bcast_scalar_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},
    "test_pow_bcast_array_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},
    # Does not support integer power yet

    # PRelu
    "test_prelu_example_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},
    "test_prelu_broadcast_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},

    # QLinear Conv

    # QLinear Matmul

    # Quantize Linear

    # Range
    "test_range_float_type_positive_delta_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},
    "test_range_int32_type_negative_delta_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},

    # Reciprocal Op:
    "test_reciprocal_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},
    "test_reciprocal_example_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},

    # ReduceL1
    "test_reduce_l1_default_axes_keepdims_example_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},
    "test_reduce_l1_default_axes_keepdims_random_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},
    "test_reduce_l1_do_not_keepdims_example_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},
    "test_reduce_l1_do_not_keepdims_random_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},
    "test_reduce_l1_keep_dims_example_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},
    "test_reduce_l1_keep_dims_random_cpu": {STATIC_SHAPE:{}},
    "test_reduce_l1_negative_axes_keep_dims_example_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},
    "test_reduce_l1_negative_axes_keep_dims_random_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},

    # ReduceL2
    "test_reduce_l2_default_axes_keepdims_example_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},
    "test_reduce_l2_default_axes_keepdims_random_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},
    "test_reduce_l2_do_not_keepdims_example_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},
    "test_reduce_l2_do_not_keepdims_random_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},
    "test_reduce_l2_keep_dims_example_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},
    "test_reduce_l2_keep_dims_random_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},
    "test_reduce_l2_negative_axes_keep_dims_example_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},
    "test_reduce_l2_negative_axes_keep_dims_random_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},

    # ReduceLogSum
    "test_reduce_log_sum_asc_axes_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},
    "test_reduce_log_sum_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},
    "test_reduce_log_sum_default_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},
    "test_reduce_log_sum_desc_axes_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},

    # ReduceLogSumExp
    "test_reduce_log_sum_exp_default_axes_keepdims_example_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},
    "test_reduce_log_sum_exp_default_axes_keepdims_random_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},
    "test_reduce_log_sum_exp_do_not_keepdims_example_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},
    "test_reduce_log_sum_exp_do_not_keepdims_random_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},
    "test_reduce_log_sum_exp_keepdims_example_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},
    "test_reduce_log_sum_exp_keepdims_random_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},
    "test_reduce_log_sum_exp_negative_axes_keepdims_example_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},
    "test_reduce_log_sum_exp_negative_axes_keepdims_random_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},
    "test_reduce_log_sum_negative_axes_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},

    # ReduceMax
    "test_reduce_max_default_axes_keepdim_example_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},
    "test_reduce_max_default_axes_keepdims_random_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},
    "test_reduce_max_do_not_keepdims_example_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},
    "test_reduce_max_do_not_keepdims_random_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},
    "test_reduce_max_keepdims_example_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},
    "test_reduce_max_keepdims_random_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},
    "test_reduce_max_negative_axes_keepdims_example_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},
    "test_reduce_max_negative_axes_keepdims_random_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},

    # ReduceMean
    "test_reduce_mean_default_axes_keepdims_example_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},
    "test_reduce_mean_default_axes_keepdims_random_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},
    "test_reduce_mean_do_not_keepdims_example_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},
    "test_reduce_mean_do_not_keepdims_random_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},
    "test_reduce_mean_keepdims_example_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},
    "test_reduce_mean_keepdims_random_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},
    "test_reduce_mean_negative_axes_keepdims_example_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},
    "test_reduce_mean_negative_axes_keepdims_random_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},

    # ReduceMin
    "test_reduce_min_default_axes_keepdims_example_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},
    "test_reduce_min_default_axes_keepdims_random_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},
    "test_reduce_min_do_not_keepdims_example_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},
    "test_reduce_min_do_not_keepdims_random_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},
    "test_reduce_min_keepdims_example_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},
    "test_reduce_min_keepdims_random_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},
    "test_reduce_min_negative_axes_keepdims_example_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},
    "test_reduce_min_negative_axes_keepdims_random_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},

    # ReduceProd
    "test_reduce_prod_default_axes_keepdims_example_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},
    "test_reduce_prod_default_axes_keepdims_random_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},
    "test_reduce_prod_do_not_keepdims_example_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},
    "test_reduce_prod_do_not_keepdims_random_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},
    "test_reduce_prod_keepdims_example_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},
    "test_reduce_prod_keepdims_random_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},
    "test_reduce_prod_negative_axes_keepdims_example_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},
    "test_reduce_prod_negative_axes_keepdims_random_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},

    # ReduceSum
    # Temporally removed due to changes in onnx 1.8.1
    #"test_reduce_sum_default_axes_keepdims_example_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},
    #"test_reduce_sum_default_axes_keepdims_random_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},
    #"test_reduce_sum_do_not_keepdims_example_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},
    #"test_reduce_sum_do_not_keepdims_random_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},
    "test_reduce_sum_keepdims_example_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},
    "test_reduce_sum_keepdims_random_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},
    "test_reduce_sum_negative_axes_keepdims_example_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{0}},
    "test_reduce_sum_negative_axes_keepdims_random_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{0}},

    # ReduceSumSquare
    "test_reduce_sum_square_default_axes_keepdims_example_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},
    "test_reduce_sum_square_default_axes_keepdims_random_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},
    "test_reduce_sum_square_do_not_keepdims_example_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},
    "test_reduce_sum_square_do_not_keepdims_random_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},
    "test_reduce_sum_square_keepdims_example_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},
    "test_reduce_sum_square_keepdims_random_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},
    "test_reduce_sum_square_negative_axes_keepdims_example_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},
    "test_reduce_sum_square_negative_axes_keepdims_random_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},

    # Relu
    "test_relu_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},

    # Reshape
    "test_reshape_extended_dims_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{0:{-1}}, CONSTANT_INPUT:{-1}},
    "test_reshape_negative_dim_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{0:{-1}}, CONSTANT_INPUT:{-1}},
    "test_reshape_negative_extended_dims_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{0:{-1}}, CONSTANT_INPUT:{-1}},
    "test_reshape_one_dim_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{0:{-1}}, CONSTANT_INPUT:{-1}},
    "test_reshape_reduced_dims_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{0:{-1}}, CONSTANT_INPUT:{-1}},
    "test_reshape_reordered_all_dims_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{0:{-1}}, CONSTANT_INPUT:{-1}},
    "test_reshape_reordered_last_dims_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{0:{-1}}, CONSTANT_INPUT:{-1}},
    "test_reshape_zero_and_negative_dim_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{0:{-1}}, CONSTANT_INPUT:{-1}},
    "test_reshape_zero_dim_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{0:{-1}}, CONSTANT_INPUT:{-1}},

    # Resize
    "test_resize_upsample_scales_nearest_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE: {0:{-1}}, CONSTANT_INPUT:{-1}},
    "test_resize_downsample_scales_nearest_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE: {0:{-1}}, CONSTANT_INPUT:{-1}},
    "test_resize_upsample_sizes_nearest_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE: {0:{-1}}, CONSTANT_INPUT:{-1}},
    "test_resize_downsample_sizes_nearest_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE: {0:{-1}}, CONSTANT_INPUT:{-1}},

    # Reverse Sequence
    "test_reversesequence_time_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},
    "test_reversesequence_batch_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},

    # RNN
    # CONSTANT_INPUT for W and R.
    "test_rnn_seq_length_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{0:{0,1,2}}, CONSTANT_INPUT:{1,2}},
    "test_simple_rnn_defaults_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{0:{0,1,2}}, CONSTANT_INPUT:{1,2}},
    "test_simple_rnn_with_initial_bias_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{0:{0,1,2}}, CONSTANT_INPUT:{1,2}},

    # Roi Align

    # Round
    "test_round_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE: {0:{-1}}, CONSTANT_INPUT:{-1}},

    # Scan
    "test_scan9_sum_cpu": {STATIC_SHAPE:{}},

    # Scatter Element

    # Selu
    "test_selu_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},
    "test_selu_default_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},
    "test_selu_example_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},

    # Shape
    "test_shape_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},
    "test_shape_example_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},

    # Shrink

    # Sigmoid
    "test_sigmoid_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},
    "test_sigmoid_example_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},

    # Sign
    "test_sign_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},

    # Sin
    "test_sin_example_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},
    "test_sin_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},

    # Sinh
    "test_sinh_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},
    "test_sinh_example_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},

    # Size
    "test_size_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},
    "test_size_example_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},

    # Slice (makes Axis a runtime argument, which is not supported).

    # Softmax
    "test_softmax_axis_0_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},
    "test_softmax_axis_1_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},
    "test_softmax_axis_2_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},
    "test_softmax_default_axis_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},
    "test_softmax_example_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},
    "test_softmax_large_number_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},

    # Softplus
    "test_softplus_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},
    "test_softplus_example_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},

    # Softsign
    "test_softsign_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},
    "test_softsign_example_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},

    # Split
    # Temporally removed due to changes in onnx 1.8.1
    # "test_split_equal_parts_1d_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},
    # "test_split_equal_parts_2d_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},
    # "test_split_equal_parts_default_axis_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},
    # "test_split_variable_parts_1d_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},
    # "test_split_variable_parts_2d_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},
    # "test_split_variable_parts_default_axis_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},
    # Enabled to test for constant splits
    "test_split_equal_parts_1d_cpu": {CONSTANT_INPUT:{-1}},
    "test_split_equal_parts_2d_cpu": {CONSTANT_INPUT:{-1}},
    "test_split_equal_parts_default_axis_cpu": {CONSTANT_INPUT:{-1}},
    "test_split_variable_parts_1d_cpu": {CONSTANT_INPUT:{1}},
    "test_split_variable_parts_2d_cpu": {CONSTANT_INPUT:{1}},
    "test_split_variable_parts_default_axis_cpu": {CONSTANT_INPUT:{1}},

    # Sqrt
    "test_sqrt_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},
    "test_sqrt_example_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},

    # Squeeze
    # Temporally removed due to changes in onnx 1.8.1
    #"test_squeeze_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},
    #"test_squeeze_negative_axes_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},
    # Enabled to test for constant axes
    "test_squeeze_cpu": {CONSTANT_INPUT:{1}},
    "test_squeeze_negative_axes_cpu": {CONSTANT_INPUT:{1}},

    # Str Normalizer

    # Sub
    "test_sub_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},
    "test_sub_bcast_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},
    "test_sub_example_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},

    # Sum
    "test_sum_example_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},
    "test_sum_one_input_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},
    "test_sum_two_inputs_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},

    # Tan
    "test_tan_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},
    "test_tan_example_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},

    # Tanh
    "test_tanh_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},
    "test_tanh_example_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},

    # Tfdf Vectorizer

    # Threshold Relu

    # Tile
    "test_tile_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},
    "test_tile_precomputed_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},

    # TopK
    "test_top_k_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},
    "test_top_k_smallest_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},
    "test_top_k_negative_axis_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},

    # Training Dropout

    # Transpose
    "test_transpose_default_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},
    "test_transpose_all_permutations_0_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},
    "test_transpose_all_permutations_1_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},
    "test_transpose_all_permutations_2_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},
    "test_transpose_all_permutations_3_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},
    "test_transpose_all_permutations_4_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},
    "test_transpose_all_permutations_5_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},

    # Unique

    # Unsqueeze
    # Temporally removed due to changes in onnx 1.8.1
    # "test_unsqueeze_axis_0_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},
    # "test_unsqueeze_axis_1_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},
    # "test_unsqueeze_axis_2_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},
    # "test_unsqueeze_axis_3_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},
    # "test_unsqueeze_negative_axes_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},
    # "test_unsqueeze_three_axes_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},
    # "test_unsqueeze_two_axes_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},
    # "test_unsqueeze_unsorted_axes_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},
    # Enabled to test for constant axes
    "test_unsqueeze_axis_0_cpu": {CONSTANT_INPUT:{1}},
    "test_unsqueeze_axis_1_cpu": {CONSTANT_INPUT:{1}},
    "test_unsqueeze_axis_2_cpu": {CONSTANT_INPUT:{1}},
    # Using Opset v11 still
    "test_unsqueeze_axis_3_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}},
    "test_unsqueeze_negative_axes_cpu": {CONSTANT_INPUT:{1}},
    "test_unsqueeze_three_axes_cpu": {CONSTANT_INPUT:{1}},
    "test_unsqueeze_two_axes_cpu": {CONSTANT_INPUT:{1}},
    "test_unsqueeze_unsorted_axes_cpu": {CONSTANT_INPUT:{1}},

    # Upsample
    "test_upsample_nearest_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE: {0:{-1}}, CONSTANT_INPUT:{-1}},

    # Where
    "test_where_example_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},
    "test_where_long_example_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},

    # Xor
    "test_xor2d_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},
    "test_xor3d_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},
    "test_xor4d_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},
    "test_xor_bcast3v1d_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},
    "test_xor_bcast3v2d_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},
    "test_xor_bcast4v2d_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},
    "test_xor_bcast4v3d_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},
    "test_xor_bcast4v4d_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},

    ############################################################
    # Model (alphabetical order)

    "test_densenet121_cpu": {STATIC_SHAPE:{}},
    "test_inception_v1_cpu": {STATIC_SHAPE:{}},
    "test_resnet50_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{0:{-1}}},
    "test_shufflenet_cpu": {STATIC_SHAPE:{}},
    "test_squeezenet_cpu": {STATIC_SHAPE:{}},
    "test_vgg19_cpu": {STATIC_SHAPE:{}},
}

# Test for static inputs.
test_to_enable = [ key for (key, value) in test_to_enable_dict.items() if STATIC_SHAPE in value ]

# Test for dynamic inputs.
# Specify the test cases which currently can not pass for dynamic shape
# Presumably, this list should be empty
# Except for some operation too difficult to handle for dynamic shape
# or big models
test_for_dynamic = [ key for (key, value) in test_to_enable_dict.items() if DYNAMIC_SHAPE in value]

# Test for constant inputs.
test_for_constant = [ key for (key, value) in test_to_enable_dict.items() if CONSTANT_INPUT in value]

# Specify the test cases which need version converter
test_need_converter = []

if args.dynamic :
    print("dynamic shape is enabled")
    test_to_enable = test_for_dynamic

if args.constant:
    print("constant input is enabled")
    test_to_enable = test_for_constant

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
        if len(test_to_enable_dict[test_name_cpu]) > 1:
            selected_list = test_to_enable_dict[test_name_cpu].get(DYNAMIC_SHAPE)
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
class EndiannessAwareExecutionSession:
    def __init__(self, model):
        self.model = model 
        self.entry_point = "run_main_graph"
        self.exec_name = None
        # Compiling the model in advance if not testing constants, so that
        # the model is compiled once and used multiple times.
        if not args.constant:
            self.exec_name = self.compile_model()

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

    def is_not_relevant_endian(self, inputs):
        inputs_endianness = list(map(lambda x: x.dtype.byteorder, inputs))
        endianness_is_consistent = len(set(inputs_endianness)) <= 1
        assert endianness_is_consistent, \
            "Input arrays contain a mixture of endianness configuration."

        sys_is_le = sys.byteorder == 'little'
        # To interpret character symbols indicating endianness:
        # https://numpy.org/doc/stable/reference/generated/numpy.dtype.byteorder.html
        i1_not_relevant_endian = inputs_endianness[0] == "|"
        return i1_not_relevant_endian

    def compile_model(self):
        name = self.model.graph.name
        model_name = result_dir+name+".onnx"
        exec_name = result_dir+name + ".so"
        # Clean the temporary files in case
        # Save model to disk as temp_model.onnx.
        onnx.save(self.model, model_name)
        if not os.path.exists(model_name) :
            print("Failed save model: "+ name)
        print(name)
        # Command
        command_list = [TEST_DRIVER]
        if args.mcpu:
            command_list.append("--mcpu="+args.mcpu)
        if args.mtriple:
            command_list.append("--mtriple="+args.mtriple)
        if args.converter or name in test_need_converter :
            command_list.append("--invokeOnnxVersionConverter=true")
        command_list.append(model_name)
        # Call frontend to process temp_model.onnx, bit code will be generated.
        dynamic_inputs_dims = determine_dynamic_parameters(name)
        execute_commands(command_list, dynamic_inputs_dims)
        if not os.path.exists(exec_name) :
            print("Failed " + TEST_DRIVER + ": " + name)
        return exec_name

    def turn_model_input_to_constant(self, inputs):
        # If IMPORTER_FORCE_CONSTANT is set, get input indices from it. 
        # Otherwise, get from test_to_enable_dict.
        input_indices = {} 
        if IMPORTER_FORCE_CONSTANT:
            input_indices = set(map(lambda x: int(x.strip()),
                                    IMPORTER_FORCE_CONSTANT.split(",")))
        else:
            test_name_cpu = self.model.graph.name + "_cpu"
            if test_name_cpu in test_for_constant:
                test_info = test_to_enable_dict[test_name_cpu]
                input_indices = test_info.get(CONSTANT_INPUT)

        # Change the model by turning input tensors to initializers with the
        # same name, so that the inputs will be constants at compile time.
        # This is for testing a model when its inputs are constants.
        num_of_inputs = len(inputs)
        if -1 in input_indices:
            input_indices = range(num_of_inputs)
        # Create initializers that have the same name as inputs.
        for idx in input_indices:
            tensor = inputs[idx]
            tensor = numpy_helper.from_array(tensor,
                                             self.model.graph.input[idx].name)
            self.model.graph.initializer.extend([tensor])
        # Remove inputs that were turned to constants.
        new_inputs = []
        for idx in range(num_of_inputs):
            if idx not in input_indices:
                new_inputs.append(inputs[idx])
        return new_inputs

    def run(self, inputs, **kwargs):
        if len(inputs):
            inputs_endianness = list(map(lambda x: x.dtype.byteorder, inputs))
            endianness_is_consistent = len(set(inputs_endianness)) <= 1
            # Deduce desired endianness of output from inputs.
            # Only possible if all inputs are consistent in endiannness.
            if endianness_is_consistent:
                sys_is_le = sys.byteorder == 'little'
                inp_is_le = self.is_input_le(inputs)
                inp_is_not_relevant_endian = self.is_not_relevant_endian(inputs)
                if (not inp_is_not_relevant_endian and sys_is_le != inp_is_le):
                    inputs = list(
                        map(lambda x: x.byteswap().newbyteorder(), inputs))
            # If constant test, change the model inputs to constants.
            if args.constant:
                inputs = self.turn_model_input_to_constant(inputs)
                self.exec_name = self.compile_model()
            session = ExecutionSession(self.exec_name, self.entry_point)
            outputs = session.run(inputs)
            if (endianness_is_consistent and not inp_is_not_relevant_endian and
                sys_is_le != inp_is_le):
                outputs = list(
                    map(lambda x: x.byteswap().newbyteorder(), outputs))
            return outputs
        else:
            # Can't deduce desired output endianess, fingers crossed.
            warnings.warn(
                "Cannot deduce desired output endianness, using native endianness by default."
            )
            session = ExecutionSession(self.exec_name, self.entry_point)
            return session.run(inputs)


class DummyBackend(onnx.backend.base.Backend):
    @classmethod
    def prepare(cls, model, device='CPU', **kwargs):
        super(DummyBackend, cls).prepare(model, device, **kwargs)
        return EndiannessAwareExecutionSession(model)

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
