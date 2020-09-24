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

VERBOSE = bool(os.environ.get("VERBOSE"))

CXX = test_config.CXX_PATH
ONNX_MLIR = os.path.join(test_config.ONNX_MLIR_BUILD_PATH, "bin/onnx-mlir")
LLC = os.path.join(test_config.LLVM_PROJ_BUILD_PATH, "bin/llc")

# Make lib folder under build directory visible in PYTHONPATH
doc_check_base_dir = os.path.dirname(os.path.realpath(__file__))
RUNTIME_DIR = os.path.join(test_config.ONNX_MLIR_BUILD_PATH, "lib")
sys.path.append(RUNTIME_DIR)
from PyRuntime import ExecutionSession


def execute_commands(cmds):
    if (VERBOSE):
        print(" ".join(cmds))
    subprocess.run(cmds)


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
        # Save model to disk as temp_model.onnx.
        onnx.save(model, "temp_model.onnx")
        # Call frontend to process temp_model.onnx, bit code will be generated.
        execute_commands([ONNX_MLIR, "temp_model.onnx"])
        return EndiannessAwareExecutionSession("./temp_model.so",
                                               "_dyn_entry_point_main_graph")

    @classmethod
    def supports_device(cls, device):
        d = Device(device)
        if d.type == DeviceType.CPU:
            return True
        return False


backend_test = onnx.backend.test.BackendTest(DummyBackend, __name__)

# Test directories:
# https://github.com/onnx/onnx/tree/master/onnx/backend/test/data/node
test_to_enable = [
    # Abs Op:
    "test_abs_cpu",

    # Add Op:
    "test_add_cpu",
    "test_add_bcast_cpu",

    # And Op:

    # Sub Op:
    "test_sub_cpu",
    "test_sub_bcast_cpu",
    "test_sub_example_cpu",

    # Cosh Op:
    "test_cosh_cpu",
    "test_cosh_example_cpu",

    # Concat
    "test_concat_1d_axis_0_cpu",
    "test_concat_2d_axis_0_cpu",
    "test_concat_2d_axis_1_cpu",
    "test_concat_3d_axis_0_cpu",
    "test_concat_3d_axis_1_cpu",
    "test_concat_3d_axis_2_cpu",
    "test_concat_1d_axis_negative_1_cpu",
    "test_concat_2d_axis_negative_1_cpu",
    "test_concat_2d_axis_negative_2_cpu",
    "test_concat_3d_axis_negative_1_cpu",
    "test_concat_3d_axis_negative_2_cpu",
    "test_concat_3d_axis_negative_3_cpu",

    # Tanh:
    "test_tanh_cpu",
    "test_tanh_example_cpu",

    # Div Op:
    "test_div_cpu",
    "test_div_bcast_cpu",
    "test_div_example_cpu",

    # Elu Op:
    "test_elu_cpu",
    "test_elu_default_cpu",
    "test_elu_example_cpu",

    # Exp Op:
    "test_exp_cpu",
    "test_exp_example_cpu",

    # Gather Op:
    "test_gather_0_cpu",
    "test_gather_1_cpu",
    "test_gather_negative_indices_cpu",

    # Gemm Op:
    "test_gemm_all_attributes_cpu",
    "test_gemm_alpha_cpu",
    "test_gemm_beta_cpu",
    "test_gemm_default_matrix_bias_cpu",
    "test_gemm_default_no_bias_cpu",
    "test_gemm_default_scalar_bias_cpu",
    "test_gemm_default_single_elem_vector_bias_cpu",
    "test_gemm_default_vector_bias_cpu",
    "test_gemm_default_zero_bias_cpu",
    "test_gemm_transposeA_cpu",
    "test_gemm_transposeB_cpu",

    # Hard Sigmoid Op:
    "test_hardsigmoid_cpu",
    "test_hardsigmoid_default_cpu",
    "test_hardsigmoid_example_cpu",

    # Leaky Relu Op:
    "test_leakyrelu_cpu",
    "test_leakyrelu_default_cpu",
    "test_leakyrelu_example_cpu",

    # Max Op:
    "test_max_example_cpu",
    "test_max_one_input_cpu",
    "test_max_two_inputs_cpu",

    # Min Op:
    "test_min_example_cpu",
    "test_min_one_input_cpu",
    "test_min_two_inputs_cpu",

    # Mul Op:
    "test_mul_cpu",
    "test_mul_bcast_cpu",
    "test_mul_example_cpu",

    # Relu Op:
    "test_relu_cpu",

    # ReduceMax Op:
    "test_reduce_max_default_axes_keepdim_example_cpu",
    "test_reduce_max_default_axes_keepdims_random_cpu",
    "test_reduce_max_do_not_keepdims_example_cpu",
    "test_reduce_max_do_not_keepdims_random_cpu",
    "test_reduce_max_keepdims_example_cpu",
    "test_reduce_max_keepdims_random_cpu",
    "test_reduce_max_negative_axes_keepdims_example_cpu",
    "test_reduce_max_negative_axes_keepdims_random_cpu",

    # ReduceMin Op:
    "test_reduce_min_default_axes_keepdims_example_cpu",
    "test_reduce_min_default_axes_keepdims_random_cpu",
    "test_reduce_min_do_not_keepdims_example_cpu",
    "test_reduce_min_do_not_keepdims_random_cpu",
    "test_reduce_min_keepdims_example_cpu",
    "test_reduce_min_keepdims_random_cpu",
    "test_reduce_min_negative_axes_keepdims_example_cpu",
    "test_reduce_min_negative_axes_keepdims_random_cpu",

    # ReduceProd Op:
    "test_reduce_prod_default_axes_keepdims_example_cpu",
    "test_reduce_prod_default_axes_keepdims_random_cpu",
    "test_reduce_prod_do_not_keepdims_example_cpu",
    "test_reduce_prod_do_not_keepdims_random_cpu",
    "test_reduce_prod_keepdims_example_cpu",
    "test_reduce_prod_keepdims_random_cpu",
    "test_reduce_prod_negative_axes_keepdims_example_cpu",
    "test_reduce_prod_negative_axes_keepdims_random_cpu",

    # ReduceSum Op:
    "test_reduce_sum_default_axes_keepdims_example_cpu",
    "test_reduce_sum_default_axes_keepdims_random_cpu",
    "test_reduce_sum_do_not_keepdims_example_cpu",
    "test_reduce_sum_do_not_keepdims_random_cpu",
    "test_reduce_sum_keepdims_example_cpu",
    "test_reduce_sum_keepdims_random_cpu",
    "test_reduce_sum_negative_axes_keepdims_example_cpu",
    "test_reduce_sum_negative_axes_keepdims_random_cpu",

    # ReduceL1
    "test_reduce_l1_default_axes_keepdims_example_cpu",
    "test_reduce_l1_default_axes_keepdims_random_cpu",
    "test_reduce_l1_do_not_keepdims_example_cpu",
    "test_reduce_l1_do_not_keepdims_random_cpu",
    "test_reduce_l1_keep_dims_example_cpu",
    "test_reduce_l1_keep_dims_random_cpu",
    "test_reduce_l1_negative_axes_keep_dims_example_cpu",
    "test_reduce_l1_negative_axes_keep_dims_random_cpu",

    # ReduceL2
    "test_reduce_l2_default_axes_keepdims_example_cpu",
    "test_reduce_l2_default_axes_keepdims_random_cpu",
    "test_reduce_l2_do_not_keepdims_example_cpu",
    "test_reduce_l2_do_not_keepdims_random_cpu",
    "test_reduce_l2_keep_dims_example_cpu",
    "test_reduce_l2_keep_dims_random_cpu",
    "test_reduce_l2_negative_axes_keep_dims_example_cpu",
    "test_reduce_l2_negative_axes_keep_dims_random_cpu",

    # ReduceLogSum
    "test_reduce_log_sum_asc_axes_cpu",
    "test_reduce_log_sum_cpu",
    "test_reduce_log_sum_default_cpu",
    "test_reduce_log_sum_desc_axes_cpu",

    # ReduceLogSumExp
    "test_reduce_log_sum_exp_default_axes_keepdims_example_cpu",
    "test_reduce_log_sum_exp_default_axes_keepdims_random_cpu",
    "test_reduce_log_sum_exp_do_not_keepdims_example_cpu",
    "test_reduce_log_sum_exp_do_not_keepdims_random_cpu",
    "test_reduce_log_sum_exp_keepdims_example_cpu",
    "test_reduce_log_sum_exp_keepdims_random_cpu",
    "test_reduce_log_sum_exp_negative_axes_keepdims_example_cpu",
    "test_reduce_log_sum_exp_negative_axes_keepdims_random_cpu",
    "test_reduce_log_sum_negative_axes_cpu",

    # ReduceSumSquare
    "test_reduce_sum_square_default_axes_keepdims_example_cpu",
    "test_reduce_sum_square_default_axes_keepdims_random_cpu",
    "test_reduce_sum_square_do_not_keepdims_example_cpu",
    "test_reduce_sum_square_do_not_keepdims_random_cpu",
    "test_reduce_sum_square_keepdims_example_cpu",
    "test_reduce_sum_square_keepdims_random_cpu",
    "test_reduce_sum_square_negative_axes_keepdims_example_cpu",
    "test_reduce_sum_square_negative_axes_keepdims_random_cpu",

    # Selu Op:
    "test_selu_cpu",
    "test_selu_default_cpu",
    "test_selu_example_cpu",

    # Sigmoid Op:
    "test_sigmoid_cpu",
    "test_sigmoid_example_cpu",

    # Softmax Op:
    "test_softmax_axis_0_cpu",
    "test_softmax_axis_1_cpu",
    "test_softmax_axis_2_cpu",
    "test_softmax_default_axis_cpu",
    "test_softmax_example_cpu",
    "test_softmax_large_number_cpu",

    # Sqrt Op:
    "test_sqrt_cpu",
    "test_sqrt_example_cpu",

    # Sum Op:
    "test_sum_example_cpu",
    "test_sum_one_input_cpu",
    "test_sum_two_inputs_cpu",

    # Unsqueeze Op:
    "test_unsqueeze_axis_0_cpu",
    "test_unsqueeze_axis_1_cpu",
    "test_unsqueeze_axis_2_cpu",
    "test_unsqueeze_axis_3_cpu",
    "test_unsqueeze_negative_axes_cpu",
    "test_unsqueeze_three_axes_cpu",
    "test_unsqueeze_two_axes_cpu",
    # "test_unsqueeze_unsorted_axes_cpu",

    # Reciprocal Op:
    "test_reciprocal_cpu",
    "test_reciprocal_example_cpu",

    # SoftplusOp:
    "test_softplus_cpu",
    "test_softplus_example_cpu",

    # SoftsignOp:
    "test_softsign_cpu",
    "test_softsign_example_cpu",

    # ReshapeOp:
    "test_reshape_extended_dims_cpu",
    "test_reshape_negative_dim_cpu",
    "test_reshape_negative_extended_dims_cpu",
    "test_reshape_one_dim_cpu",
    "test_reshape_reduced_dims_cpu",
    "test_reshape_reordered_all_dims_cpu",
    "test_reshape_reordered_last_dims_cpu",
    "test_reshape_zero_and_negative_dim_cpu",
    "test_reshape_zero_dim_cpu",

    # Transpose
    "test_transpose_default_cpu",
    "test_transpose_all_permutations_0_cpu",
    "test_transpose_all_permutations_1_cpu",
    "test_transpose_all_permutations_2_cpu",
    "test_transpose_all_permutations_3_cpu",
    "test_transpose_all_permutations_4_cpu",
    "test_transpose_all_permutations_5_cpu",

    # Conv
    "test_basic_conv_without_padding_cpu",
    "test_conv_with_strides_no_padding_cpu",

    # Sign Op:
    "test_sign_cpu",

    # MatmulOp
    "test_matmul_2d_cpu",
    "test_matmul_3d_cpu",
    "test_matmul_4d_cpu",

    # BatchNormalization (test mode)
    "test_batchnorm_epsilon_cpu",
    "test_batchnorm_example_cpu",

    # MaxPoolSingleOut
    "test_maxpool_1d_default_cpu",
    "test_maxpool_2d_ceil_cpu",
    "test_maxpool_2d_default_cpu",
    "test_maxpool_2d_dilations_cpu",
    "test_maxpool_2d_pads_cpu",
    "test_maxpool_2d_precomputed_pads_cpu",
    "test_maxpool_2d_precomputed_same_upper_cpu",
    "test_maxpool_2d_precomputed_strides_cpu",
    "test_maxpool_2d_same_lower_cpu",
    "test_maxpool_2d_same_upper_cpu",
    "test_maxpool_2d_strides_cpu",
    "test_maxpool_3d_default_cpu",

    # AveragePool
    "test_averagepool_1d_default_cpu",
    "test_averagepool_2d_ceil_cpu",
    "test_averagepool_2d_default_cpu",
    "test_averagepool_2d_pads_count_include_pad_cpu",
    "test_averagepool_2d_pads_cpu",
    "test_averagepool_2d_precomputed_pads_count_include_pad_cpu",
    "test_averagepool_2d_precomputed_pads_cpu",
    "test_averagepool_2d_precomputed_same_upper_cpu",
    "test_averagepool_2d_precomputed_strides_cpu",
    "test_averagepool_2d_same_lower_cpu",
    "test_averagepool_2d_same_upper_cpu",
    "test_averagepool_2d_strides_cpu",
    "test_averagepool_3d_default_cpu",

    # LSTM
    "test_lstm_defaults_cpu",
    "test_lstm_with_initial_bias_cpu",
    "test_lstm_with_peepholes_cpu",

    # Squeeze
    "test_squeeze_cpu",
    "test_squeeze_negative_axes_cpu",

    # Split
    "test_split_equal_parts_1d_cpu",
    "test_split_equal_parts_2d_cpu",
    "test_split_equal_parts_default_axis_cpu",
    "test_split_variable_parts_1d_cpu",
    "test_split_variable_parts_2d_cpu",
    "test_split_variable_parts_default_axis_cpu",

    # ConstantOfShape
    "test_constantofshape_float_ones_cpu",
    
    # Size
    # TODO(tjingrant): fix unit test for size ops.
    # "test_size_cpu",
    # "test_size_example_cpu",
    
    # Error:
    #    Items are not equal:
    #     ACTUAL: dtype('int32')
    #     DESIRED: dtype('uint8')
    # In this test, 'int32' was specified for value attribute as in
    # onnx/onnx/backend/test/case/node/constantofshape.py
    # and onnx-mlir correctly imported and converted the model.
    # It is unknown why 'uint8' came from.
    #"test_constantofshape_int_zeros_cpu",

    # Model
    "test_resnet50_cpu",
    "test_vgg19_cpu",
    "test_shufflenet_cpu",
]


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
