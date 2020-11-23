from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

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

VERBOSE = bool(os.environ.get("VERBOSE"))
TEST_DYNAMIC = os.environ.get("IMPORTER_FORCE_DYNAMIC")
    
parser = argparse.ArgumentParser(description='with dynamic shape or not.')
parser.add_argument('--dynamic', action='store_true',
    help='enable dynamic (default: false)')
parser.add_argument('unittest_args', nargs='*')
args = parser.parse_args()
sys.argv[1:] = args.unittest_args

TEST_CASE_BY_USER = os.environ.get("BACKEND_TEST")
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

dynamic_input_dict = {
    "test_reshape_extended_dims":0,
    "test_reshape_negative_dim":0,
    "test_reshape_negative_extended_dims":0,
    "test_reshape_one_dim":0,
    "test_reshape_reduced_dims":0,
    "test_reshape_reordered_all_dims":0,
    "test_reshape_reordered_last_dims":0,
    "test_reshape_zero_and_negative_dim":0,
    "test_reshape_zero_dim":0,
    "test_basic_conv_without_padding":0,
    "test_conv_with_strides_no_padding":0,
}

dynamic_dim_dict = {
    "test_concat_1d_axis_0":0,
    "test_concat_2d_axis_0":0,
    "test_concat_2d_axis_1":1,
    "test_concat_3d_axis_0":0,
    "test_concat_3d_axis_1":1,
    "test_concat_3d_axis_2":2,
    "test_concat_1d_axis_negative_1":0,
    "test_concat_2d_axis_negative_1":1,
    "test_concat_2d_axis_negative_2":0,
    "test_concat_3d_axis_negative_1":2,
    "test_concat_3d_axis_negative_2":1,
    "test_concat_3d_axis_negative_3":0,
}

# determine the dynamic input and dim
def determine_dynamic_parameters(test_name):
    if not args.dynamic :
        return [None, None]
    # set default value
    selected_input = -1
    selected_dim = 0
    if test_name in dynamic_input_dict :
        selected_input = dynamic_input_dict[test_name]
    if test_name in dynamic_dim_dict :
        selected_dim = dynamic_dim_dict[test_name]
    return [selected_input, selected_dim]

def execute_commands(cmds, dynamic_input, dynamic_dim):
    if (VERBOSE):
        print(" ".join(cmds))
        print("IMPORTER FORCE DYNAMIC ", dynamic_input, dynamic_dim)
    my_env = os.environ.copy();
    if dynamic_input is not None:
        my_env["IMPORTER_FORCE_DYNAMIC"] = str(dynamic_input) 
    if dynamic_dim is not None:
        my_env["IMPORTER_FORCE_DYNAMIC_DIM"] = str(dynamic_dim) 
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

        # Call frontend to process temp_model.onnx, bit code will be generated.
        my_input, my_dim = determine_dynamic_parameters(name)
        execute_commands([TEST_DRIVER, model_name], my_input, my_dim)
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

# Test directories:
# https://github.com/onnx/onnx/tree/master/onnx/backend/test/data/node

# Set value for each benchmark to: test_disabled, test_static, 
#   test_dynamic, or test_static_dynamic.
test_disabled = 0
test_static = 1
test_dynamic = 2
test_static_dynamic = test_static + test_dynamic

test_to_enable_static_dynamic = {

    ############################################################
    # Elementary ops, ordered alphabetically.

    # Abs
    "test_abs_cpu": test_static_dynamic,

    # Acos

    # Acosh

    # Adagrad

    # Adam

    # Add
    "test_add_cpu": test_static_dynamic,
    "test_add_bcast_cpu": test_static_dynamic,

    # And

    # Argmax

    # Argmin

    # Asin

    # Asinh

    # Atan

    # Atanh


    # AveragePool
    "test_averagepool_1d_default_cpu": test_static_dynamic,
    "test_averagepool_2d_ceil_cpu": test_static_dynamic,
    "test_averagepool_2d_default_cpu": test_static_dynamic,
    "test_averagepool_2d_pads_count_include_pad_cpu": test_static_dynamic,
    "test_averagepool_2d_pads_cpu": test_static_dynamic,
    "test_averagepool_2d_precomputed_pads_count_include_pad_cpu": test_static_dynamic,
    "test_averagepool_2d_precomputed_pads_cpu": test_static_dynamic,
    "test_averagepool_2d_precomputed_same_upper_cpu": test_static_dynamic,
    "test_averagepool_2d_precomputed_strides_cpu": test_static_dynamic,
    "test_averagepool_2d_same_lower_cpu": test_static_dynamic,
    "test_averagepool_2d_same_upper_cpu": test_static_dynamic,
    "test_averagepool_2d_strides_cpu": test_static_dynamic,
    "test_averagepool_3d_default_cpu": test_static_dynamic,

    # BatchNormalization (test mode)
    #"test_batchnorm_epsilon_cpu": test_static,
    #"test_batchnorm_example_cpu": test_static,

    # Bitshift left/right

    # Cast

    # Ceil

    # Celu

    # Clip

    # Compress

    # Concat
    "test_concat_1d_axis_0_cpu": test_static_dynamic,
    "test_concat_2d_axis_0_cpu": test_static_dynamic,
    "test_concat_2d_axis_1_cpu": test_static_dynamic,
    "test_concat_3d_axis_0_cpu": test_static_dynamic,
    "test_concat_3d_axis_1_cpu": test_static,
    "test_concat_3d_axis_2_cpu": test_static,
    "test_concat_1d_axis_negative_1_cpu": test_static_dynamic,
    "test_concat_2d_axis_negative_1_cpu": test_static_dynamic,
    "test_concat_2d_axis_negative_2_cpu": test_static_dynamic,
    "test_concat_3d_axis_negative_1_cpu": test_static,
    "test_concat_3d_axis_negative_2_cpu": test_static,
    "test_concat_3d_axis_negative_3_cpu": test_static_dynamic,

    # Constant

    # ConstantOfShape
    "test_constantofshape_float_ones_cpu": test_static,
    "test_constantofshape_int_zeros_cpu": test_static,

    # Conv
    "test_basic_conv_without_padding_cpu": test_static_dynamic,
    "test_conv_with_strides_no_padding_cpu": test_static_dynamic,

    # ConvInteger

    # ConvTranspose

    # Cos

    # Cosh
    "test_cosh_cpu": test_static_dynamic,
    "test_cosh_example_cpu": test_static_dynamic,

    # CumSum

    # DepthOfSpace

    # DequatizeLinear

    # Det

    # Div
    "test_div_cpu": test_static_dynamic,
    "test_div_bcast_cpu": test_static_dynamic,
    "test_div_example_cpu": test_static_dynamic,

    # Dropout

    # DynamicQuantizeLinear

    # Edge

    # EinSum

    # Elu
    "test_elu_cpu": test_static_dynamic,
    "test_elu_default_cpu": test_static_dynamic,
    "test_elu_example_cpu": test_static_dynamic,

    # Equal

    # Exp
    "test_exp_cpu": test_static_dynamic,
    "test_exp_example_cpu": test_static_dynamic,

    # Expand

    # Eyelike

    # Flatten
    "test_flatten_axis0_cpu": test_static_dynamic,
    "test_flatten_axis1_cpu": test_static_dynamic,
    "test_flatten_axis2_cpu": test_static_dynamic,
    "test_flatten_axis3_cpu": test_static_dynamic,
    "test_flatten_default_axis_cpu": test_static_dynamic,
    "test_flatten_negative_axis1_cpu": test_static_dynamic,
    "test_flatten_negative_axis2_cpu": test_static_dynamic,
    "test_flatten_negative_axis3_cpu": test_static_dynamic,
    "test_flatten_negative_axis4_cpu": test_static_dynamic,

    # Floor
    
    # Gather
    "test_gather_0_cpu": test_static_dynamic,
    "test_gather_1_cpu": test_static_dynamic,
    "test_gather_negative_indices_cpu": test_static_dynamic,

    # Gemm
    "test_gemm_all_attributes_cpu": test_static_dynamic,
    "test_gemm_alpha_cpu": test_static_dynamic,
    "test_gemm_beta_cpu": test_static_dynamic,
    "test_gemm_default_matrix_bias_cpu": test_static_dynamic,
    "test_gemm_default_no_bias_cpu": test_static_dynamic,
    "test_gemm_default_scalar_bias_cpu": test_static_dynamic,
    "test_gemm_default_single_elem_vector_bias_cpu": test_static_dynamic,
    "test_gemm_default_vector_bias_cpu": test_static_dynamic,
    "test_gemm_default_zero_bias_cpu": test_static_dynamic,
    "test_gemm_transposeA_cpu": test_static_dynamic,
    "test_gemm_transposeB_cpu": test_static_dynamic,

    # Global Average Pool

    # Global Max Pool

    # Greater

    # GRU
    "test_gru_defaults_cpu": test_static,
    "test_gru_seq_length_cpu": test_static,
    "test_gru_with_initial_bias_cpu": test_static,

    # Hard Max

    # Hard Sigmoid
    "test_hardsigmoid_cpu": test_static_dynamic,
    "test_hardsigmoid_default_cpu": test_static_dynamic,
    "test_hardsigmoid_example_cpu": test_static_dynamic,

    # Identity

    # Instance Norm

    # Is Inf Neg/Pos

    # Is Nan

    # Leaky Relu
    "test_leakyrelu_cpu": test_static_dynamic,
    "test_leakyrelu_default_cpu": test_static_dynamic,
    "test_leakyrelu_example_cpu": test_static_dynamic,

    # Less
    "test_less_cpu": test_static_dynamic,
    "test_less_bcast_cpu": test_static_dynamic,

    # Log

    # LogSoftmax
    "test_logsoftmax_axis_0_cpu": test_static,
    "test_logsoftmax_axis_1_cpu": test_static,
    "test_logsoftmax_axis_2_cpu": test_static,
    "test_logsoftmax_example_1_cpu": test_static,
    "test_logsoftmax_default_axis_cpu": test_static,
    "test_logsoftmax_negative_axis_cpu": test_static,
    "test_logsoftmax_large_number_cpu": test_static,

    # LRN

    # LSTM
    "test_lstm_defaults_cpu": test_static,
    "test_lstm_with_initial_bias_cpu": test_static,
    "test_lstm_with_peepholes_cpu": test_static,

    # Matmul
    "test_matmul_2d_cpu": test_static_dynamic,
    "test_matmul_3d_cpu": test_static_dynamic,
    "test_matmul_4d_cpu": test_static_dynamic,

    # Matmul Integer

    # Max
    "test_max_example_cpu": test_static_dynamic,
    "test_max_one_input_cpu": test_static_dynamic,
    "test_max_two_inputs_cpu": test_static_dynamic,

    # MaxPoolSingleOut
    "test_maxpool_1d_default_cpu": test_static_dynamic,
    "test_maxpool_2d_ceil_cpu": test_static_dynamic,
    "test_maxpool_2d_default_cpu": test_static_dynamic,
    "test_maxpool_2d_dilations_cpu": test_static_dynamic,
    "test_maxpool_2d_pads_cpu": test_static_dynamic,
    "test_maxpool_2d_precomputed_pads_cpu": test_static_dynamic,
    "test_maxpool_2d_precomputed_same_upper_cpu": test_static_dynamic,
    "test_maxpool_2d_precomputed_strides_cpu": test_static_dynamic,
    "test_maxpool_2d_same_lower_cpu": test_static_dynamic,
    "test_maxpool_2d_same_upper_cpu": test_static_dynamic,
    "test_maxpool_2d_strides_cpu": test_static_dynamic,
    "test_maxpool_3d_default_cpu": test_static_dynamic,

    # Mean

    # Min
    "test_min_example_cpu": test_static_dynamic,
    "test_min_one_input_cpu": test_static_dynamic,
    "test_min_two_inputs_cpu": test_static_dynamic,

    # Mod

    # Momentum

    # Mul
    "test_mul_cpu": test_static_dynamic,
    "test_mul_bcast_cpu": test_static_dynamic,
    "test_mul_example_cpu": test_static_dynamic,

    # Multinomial (NMV)

    # Neg

    # Negative Log Likelihood Loss

    # Non Max Supression

    # Non Zero

    # Not

    # One Hot

    # Or

    # Pow

    # PRelu

    # QLinear Conv

    # QLinear Matmul

    # Quantize Lienar

    # Reciprocal Op:
    "test_reciprocal_cpu": test_static_dynamic,
    "test_reciprocal_example_cpu": test_static_dynamic,

    # ReduceL1
    "test_reduce_l1_default_axes_keepdims_example_cpu": test_static,
    "test_reduce_l1_default_axes_keepdims_random_cpu": test_static,
    "test_reduce_l1_do_not_keepdims_example_cpu": test_static,
    "test_reduce_l1_do_not_keepdims_random_cpu": test_static,
    "test_reduce_l1_keep_dims_example_cpu": test_static,
    "test_reduce_l1_keep_dims_random_cpu": test_static,
    "test_reduce_l1_negative_axes_keep_dims_example_cpu": test_static,
    "test_reduce_l1_negative_axes_keep_dims_random_cpu": test_static,

    # ReduceL2
    "test_reduce_l2_default_axes_keepdims_example_cpu": test_static,
    "test_reduce_l2_default_axes_keepdims_random_cpu": test_static,
    "test_reduce_l2_do_not_keepdims_example_cpu": test_static,
    "test_reduce_l2_do_not_keepdims_random_cpu": test_static,
    "test_reduce_l2_keep_dims_example_cpu": test_static,
    "test_reduce_l2_keep_dims_random_cpu": test_static,
    "test_reduce_l2_negative_axes_keep_dims_example_cpu": test_static,
    "test_reduce_l2_negative_axes_keep_dims_random_cpu": test_static,

    # ReduceLogSum
    "test_reduce_log_sum_asc_axes_cpu": test_static,
    "test_reduce_log_sum_cpu": test_static,
    "test_reduce_log_sum_default_cpu": test_static,
    "test_reduce_log_sum_desc_axes_cpu": test_static,

    # ReduceLogSumExp
    "test_reduce_log_sum_exp_default_axes_keepdims_example_cpu": test_static,
    "test_reduce_log_sum_exp_default_axes_keepdims_random_cpu": test_static,
    "test_reduce_log_sum_exp_do_not_keepdims_example_cpu": test_static,
    "test_reduce_log_sum_exp_do_not_keepdims_random_cpu": test_static,
    "test_reduce_log_sum_exp_keepdims_example_cpu": test_static,
    "test_reduce_log_sum_exp_keepdims_random_cpu": test_static,
    "test_reduce_log_sum_exp_negative_axes_keepdims_example_cpu": test_static,
    "test_reduce_log_sum_exp_negative_axes_keepdims_random_cpu": test_static,
    "test_reduce_log_sum_negative_axes_cpu": test_static,

    # ReduceMax
    "test_reduce_max_default_axes_keepdim_example_cpu": test_static,
    "test_reduce_max_default_axes_keepdims_random_cpu": test_static,
    "test_reduce_max_do_not_keepdims_example_cpu": test_static,
    "test_reduce_max_do_not_keepdims_random_cpu": test_static,
    "test_reduce_max_keepdims_example_cpu": test_static,
    "test_reduce_max_keepdims_random_cpu": test_static,
    "test_reduce_max_negative_axes_keepdims_example_cpu": test_static,
    "test_reduce_max_negative_axes_keepdims_random_cpu": test_static,

    # ReduceMean
    "test_reduce_mean_default_axes_keepdims_example_cpu": test_static,
    "test_reduce_mean_default_axes_keepdims_random_cpu": test_static,
    "test_reduce_mean_do_not_keepdims_example_cpu": test_static,
    "test_reduce_mean_do_not_keepdims_random_cpu": test_static,
    "test_reduce_mean_keepdims_example_cpu": test_static,
    "test_reduce_mean_keepdims_random_cpu": test_static,
    "test_reduce_mean_negative_axes_keepdims_example_cpu": test_static,
    "test_reduce_mean_negative_axes_keepdims_random_cpu": test_static,

    # ReduceMin
    "test_reduce_min_default_axes_keepdims_example_cpu": test_static,
    "test_reduce_min_default_axes_keepdims_random_cpu": test_static,
    "test_reduce_min_do_not_keepdims_example_cpu": test_static,
    "test_reduce_min_do_not_keepdims_random_cpu": test_static,
    "test_reduce_min_keepdims_example_cpu": test_static,
    "test_reduce_min_keepdims_random_cpu": test_static,
    "test_reduce_min_negative_axes_keepdims_example_cpu": test_static,
    "test_reduce_min_negative_axes_keepdims_random_cpu": test_static,

    # ReduceProd
    "test_reduce_prod_default_axes_keepdims_example_cpu": test_static,
    "test_reduce_prod_default_axes_keepdims_random_cpu": test_static,
    "test_reduce_prod_do_not_keepdims_example_cpu": test_static,
    "test_reduce_prod_do_not_keepdims_random_cpu": test_static,
    "test_reduce_prod_keepdims_example_cpu": test_static,
    "test_reduce_prod_keepdims_random_cpu": test_static,
    "test_reduce_prod_negative_axes_keepdims_example_cpu": test_static,
    "test_reduce_prod_negative_axes_keepdims_random_cpu": test_static,

    # ReduceSum
    "test_reduce_sum_default_axes_keepdims_example_cpu": test_static,
    "test_reduce_sum_default_axes_keepdims_random_cpu": test_static,
    "test_reduce_sum_do_not_keepdims_example_cpu": test_static,
    "test_reduce_sum_do_not_keepdims_random_cpu": test_static,
    "test_reduce_sum_keepdims_example_cpu": test_static,
    "test_reduce_sum_keepdims_random_cpu": test_static,
    "test_reduce_sum_negative_axes_keepdims_example_cpu": test_static,
    "test_reduce_sum_negative_axes_keepdims_random_cpu": test_static,

    # ReduceSumSquare
    "test_reduce_sum_square_default_axes_keepdims_example_cpu": test_static,
    "test_reduce_sum_square_default_axes_keepdims_random_cpu": test_static,
    "test_reduce_sum_square_do_not_keepdims_example_cpu": test_static,
    "test_reduce_sum_square_do_not_keepdims_random_cpu": test_static,
    "test_reduce_sum_square_keepdims_example_cpu": test_static,
    "test_reduce_sum_square_keepdims_random_cpu": test_static,
    "test_reduce_sum_square_negative_axes_keepdims_example_cpu": test_static,
    "test_reduce_sum_square_negative_axes_keepdims_random_cpu": test_static,

    # Relu
    "test_relu_cpu": test_static,

    # Reshape
    "test_reshape_extended_dims_cpu": test_static_dynamic,
    "test_reshape_negative_dim_cpu": test_static_dynamic,
    "test_reshape_negative_extended_dims_cpu": test_static_dynamic,
    "test_reshape_one_dim_cpu": test_static_dynamic,
    "test_reshape_reduced_dims_cpu": test_static_dynamic,
    "test_reshape_reordered_all_dims_cpu": test_static_dynamic,
    "test_reshape_reordered_last_dims_cpu": test_static_dynamic,
    "test_reshape_zero_and_negative_dim_cpu": test_static_dynamic,
    "test_reshape_zero_dim_cpu": test_static_dynamic,

    # Resize

    # Reverse Sequence

    # RNN
    "test_rnn_seq_length_cpu": test_static,
    "test_simple_rnn_defaults_cpu": test_static,
    "test_simple_rnn_with_initial_bias_cpu": test_static,

    # Roi Align

    # Round

    # Scan

    # Scatter Element

    # Selu
    "test_selu_cpu": test_static_dynamic,
    "test_selu_default_cpu": test_static_dynamic,
    "test_selu_example_cpu": test_static_dynamic,

    # Shape

    # Shrink

    # Sigmoid
    "test_sigmoid_cpu": test_static_dynamic,
    "test_sigmoid_example_cpu": test_static_dynamic,

    # Sign
    #"test_sign_cpu": test_static,

    # Sin

    # Sinh

    # Size
    # TODO(tjingrant): fix unit test for size ops.
    # "test_size_cpu": test_static,
    # "test_size_example_cpu": test_static,

    # Slice
    # Slice makes Axis a runtime argument, which is not supported.

    # Softmax
    "test_softmax_axis_0_cpu": test_static_dynamic,
    "test_softmax_axis_1_cpu": test_static_dynamic,
    "test_softmax_axis_2_cpu": test_static_dynamic,
    "test_softmax_default_axis_cpu": test_static_dynamic,
    "test_softmax_example_cpu": test_static_dynamic,
    "test_softmax_large_number_cpu": test_static_dynamic,

    # Softplus
    "test_softplus_cpu": test_static_dynamic,
    "test_softplus_example_cpu": test_static_dynamic,

    # Softsign
    "test_softsign_cpu": test_static_dynamic,
    "test_softsign_example_cpu": test_static_dynamic,

    # Split
    "test_split_equal_parts_1d_cpu": test_static_dynamic,
    "test_split_equal_parts_2d_cpu": test_static_dynamic,
    "test_split_equal_parts_default_axis_cpu": test_static_dynamic,
    "test_split_variable_parts_1d_cpu": test_static_dynamic,
    "test_split_variable_parts_2d_cpu": test_static_dynamic,
    "test_split_variable_parts_default_axis_cpu": test_static_dynamic,
    
    # Sqrt
    "test_sqrt_cpu": test_static_dynamic,
    "test_sqrt_example_cpu": test_static_dynamic,

    # Squeeze
    "test_squeeze_cpu": test_static_dynamic,
    "test_squeeze_negative_axes_cpu": test_static_dynamic,

    # Str Normalizer

    # Sub
    "test_sub_cpu": test_static_dynamic,
    "test_sub_bcast_cpu": test_static_dynamic,
    "test_sub_example_cpu": test_static_dynamic,

    # Sum
    "test_sum_example_cpu": test_static_dynamic,
    "test_sum_one_input_cpu": test_static_dynamic,
    "test_sum_two_inputs_cpu": test_static_dynamic,

    # Tan

    # Tanh
    "test_tanh_cpu": test_static_dynamic,
    "test_tanh_example_cpu": test_static_dynamic,

    # Tfdf Vectorizer

    # Threshold Relu

    # Tile
    "test_tile_cpu": test_static_dynamic,
    "test_tile_precomputed_cpu": test_static_dynamic,

    # TopK

    # Training Dropout

    # Transpose
    "test_transpose_default_cpu": test_static,
    "test_transpose_all_permutations_0_cpu": test_static_dynamic,
    "test_transpose_all_permutations_1_cpu": test_static_dynamic,
    "test_transpose_all_permutations_2_cpu": test_static,
    "test_transpose_all_permutations_3_cpu": test_static,
    "test_transpose_all_permutations_4_cpu": test_static,
    "test_transpose_all_permutations_5_cpu": test_static,

    # Unique

    # Unsqueeze
    "test_unsqueeze_axis_0_cpu": test_static,
    "test_unsqueeze_axis_1_cpu": test_static,
    "test_unsqueeze_axis_2_cpu": test_static,
    "test_unsqueeze_axis_3_cpu": test_static,
    "test_unsqueeze_negative_axes_cpu": test_static,
    "test_unsqueeze_three_axes_cpu": test_static,
    "test_unsqueeze_two_axes_cpu": test_static,
    # "test_unsqueeze_unsorted_axes_cpu": test_static,

    # Upsample

    # Where

    # Xor


    ############################################################
    # Model (alphabetical order)

    "test_shufflenet_cpu": test_static,
    "test_resnet50_cpu": test_static,
    "test_vgg19_cpu": test_static,
}

# test for static
test_to_enable = [ key for (key, value) in test_to_enable_static_dynamic.items() if value & test_static ]

# Specify the test cases which currently can not pass for dynamic shape
# Presumably, this list should be empty
# Except for some operation too difficult to handle for dynamic shape
# or big models
test_not_for_dynamic = [ key for (key, value) in test_to_enable_static_dynamic.items() if value == test_static ]


if args.dynamic :
    print("dynamic shape is enabled")
    test_to_enable = [case for case in test_to_enable if case not in test_not_for_dynamic]
    

# User case specify one test case with BCKEND_TEST env
if TEST_CASE_BY_USER is not None and TEST_CASE_BY_USER != "" :
    test_to_enable = [TEST_CASE_BY_USER]

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
