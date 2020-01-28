from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os
import sys
import unittest
import onnx.backend.base
import onnx.backend.test

from onnx.backend.base import Device, DeviceType
import subprocess
import test_config

VERBOSE = bool(os.environ.get("VERBOSE"))

CXX = test_config.CXX_PATH
ONNF = os.path.join(test_config.ONNF_BUILD_PATH, "bin/onnf")
LLC = os.path.join(test_config.LLVM_PROJ_BUILD_PATH, "bin/llc")

# Make lib folder under build directory visible in PYTHONPATH
doc_check_base_dir = os.path.dirname(os.path.realpath(__file__))
RUNTIME_DIR = os.path.join(test_config.ONNF_BUILD_PATH, "lib")
sys.path.append(RUNTIME_DIR)
from pyruntime import ExecutionSession


def execute_commands(cmds):
    if (VERBOSE):
        print(" ".join(cmds))
    subprocess.run(cmds, stdout=subprocess.PIPE)


class DummyBackend(onnx.backend.base.Backend):
    @classmethod
    def prepare(cls, model, device='CPU', **kwargs):
        super(DummyBackend, cls).prepare(model, device, **kwargs)
        # Save model to disk as temp_model.onnx.
        onnx.save(model, "temp_model.onnx")
        # Call frontend to process temp_model.onnx, bit code will be generated.
        execute_commands([ONNF, "temp_model.onnx"])
        # Call llc to generate object file from bitcode.
        execute_commands(
            [LLC, "-filetype=obj", "-relocation-model=pic", "model.bc"])
        # Generate shared library from object file, linking with c runtime.
        execute_commands([
            CXX, "-shared", "-fPIC", "model.o", "-o", "model.so",
            "-L" + RUNTIME_DIR, "-lcruntime"
        ])
        return ExecutionSession("./model.so", "_dyn_entry_point_main_graph")

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

    # Sum Op:
    "test_sum_example_cpu",
    "test_sum_one_input_cpu",
    "test_sum_two_inputs_cpu",

    # Reciprocal Op:
    "test_reciprocal_cpu",
    "test_reciprocal_example_cpu",

    # SoftplusOp:
    "test_softplus_cpu",
    "test_softplus_example_cpu",

    # SoftsignOp:
    "test_softsign_cpu",
    "test_softsign_example_cpu",
]

# Extract name of all test cases.
import inspect
all_tests = inspect.getmembers(
    backend_test.test_cases["OnnxBackendNodeModelTest"])
all_test_names = list(map(lambda x: x[0], all_tests))

# Ensure that test names specified in test_to_enable actually exist.
for test_name in test_to_enable:
    assert test_name in all_test_names, "test name {} not found, it is likely "
    "that you may have misspelled the test name or the specified test does not "
    "exist in the version of onnx package you installed.".format(
        test_name)
    backend_test.include(r"^{}$".format(test_name))

# import all test cases at global scope to make them visible to python.unittest
globals().update(backend_test.test_cases)

if __name__ == '__main__':
    unittest.main()
