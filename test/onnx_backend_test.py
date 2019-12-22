from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import itertools
import os
import unittest
import onnx.backend.base
import onnx.backend.test

from onnx.backend.base import Device, DeviceType
import onnx.shape_inference
import onnx.version_converter
from typing import Optional, Text, Any, Tuple, Sequence
from onnx import NodeProto, ModelProto, TensorProto
import numpy  # type: ignore
import subprocess
from pyruntime import ExecutionSession

CXX = os.getenv('CXX')
FE = os.getenv('FE')
LLC = os.getenv('LLC')
RT_DIR = os.getenv('RT_DIR')
assert CXX and FE and LLC and RT_DIR, "tools path not set"

class DummyBackend(onnx.backend.base.Backend):
    @classmethod
    def prepare(
            cls,
            model,
            device='CPU',
            **kwargs
    ):
        super(DummyBackend, cls).prepare(model, device, **kwargs)
        # Save model to disk as temp_model.onnx.
        onnx.save(model, "temp_model.onnx")
        # Call frontend to process temp_model.onnx, bit code will be generated.
        subprocess.run([FE, "temp_model.onnx"], stdout=subprocess.PIPE)
        # Call llc to generate object file from bitcode.
        subprocess.run([LLC, "-filetype=obj", "model.bc"],
                       stdout=subprocess.PIPE)
        # Generate shared library from object file, linking with c runtime.
        subprocess.run([
            CXX, "-shared", "model.o", "-o", "model.so", "-L" + RT_DIR,
            "-lcruntime"
        ],
                       stdout=subprocess.PIPE)
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
    # "test_max_example_cpu", <- error
    "test_max_one_input_cpu",
    # "test_max_two_inputs_cpu", <- error

    # Min Op:
    # "test_min_example_cpu", <- error
    "test_min_one_input_cpu",
    # "test_min_two_inputs_cpu", <- error

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
    
    # Sum Op:
    #"test_sum_example_cpu", <- error
    "test_sum_one_input_cpu",
    #"test_sum_two_inputs_cpu", <- error

    # Reciprocal Op:
    #"test_reciprocal_cpu", <- error on shape inference.
    #"test_reciprocal_example_cpu", <- error on shape inference.
]

# Extract name of all test cases.
import inspect
all_tests = inspect.getmembers(
    backend_test.test_cases["OnnxBackendNodeModelTest"])
all_test_names = list(map(lambda x: x[0], all_tests))
for test_name in test_to_enable:
    assert test_name in all_test_names, "test name {} not found".format(test_name)
    backend_test.include(r"^{}$".format(test_name))


def tearDownModule():
    print()
    print("*" * 40)
    print("A total of {} tests should have run".format(len(test_to_enable)))
    print("*" * 40)


# import all test cases at global scope to make them visible to python.unittest
globals().update(backend_test.test_cases)

if __name__ == '__main__':
    unittest.main()
