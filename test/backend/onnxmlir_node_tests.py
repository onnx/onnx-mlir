# SPDX-License-Identifier: Apache-2.0

##################### onnxmlir_node_tests.py ###################################
#
# Extra backend node tests beyond those in
# third_party/onnx/onnx/backend/test/data/node
#
################################################################################

import numpy as np
import onnx
import onnx.parser
from collections import namedtuple

OnnxMlirNodeTestCase = namedtuple(
    "OnnxMlirTestCase", ["model", "inputs", "outputs", "rtol", "atol"]
)

# Graph names must start with "test_" and we also prefix with
# "onnxmlir_" to avoid name clashes with onnx node tests.
test_onnxmlir_top_k_float16 = """
test_onnxmlir_top_k_float16
(float16[3,4] x, int64[1] k) => (float16[3,3] values, int64[3,3] indices) {
    values, indices = TopK <axis = 1> (x, k)
}
"""
test_onnxmlir_top_k_smallest_float16 = """
test_onnxmlir_top_k_smallest_float16
(float16[3,4] x, int64[1] k) => (float16[3,3] values, int64[3,3] indices) {
    values, indices = TopK <largest = 0> (x, k)
}
"""


def load_onnxmlir_node_tests():
    # rtol, atol defaults from onnx.backend.test.loader.load_model_tests
    def make_onnxmlir_node_test(text, inputs, outputs, rtol=1e-3, atol=1e-7):
        graph = onnx.parser.parse_graph(text)
        model = onnx.helper.make_model(graph, producer_name="onnx-mlir")
        return OnnxMlirNodeTestCase(model, inputs, outputs, rtol, atol)

    return [
        make_onnxmlir_node_test(
            test_onnxmlir_top_k_float16,
            [
                np.array([[1, 3, 2, 0], [1, 0, 1, 0], [0, 1, 2, 3]], np.float16),
                np.array([3], np.int64),
            ],
            [
                np.array([[3, 2, 1], [1, 1, 0], [3, 2, 1]], np.float16),
                np.array([[1, 2, 0], [0, 2, 1], [3, 2, 1]], np.int64),
            ],
        ),
        make_onnxmlir_node_test(
            test_onnxmlir_top_k_smallest_float16,
            [
                np.array([[1, 3, 2, 0], [1, 0, 1, 0], [0, 1, 2, 3]], np.float16),
                np.array([3], np.int64),
            ],
            [
                np.array([[0, 1, 2], [0, 0, 1], [0, 1, 2]], np.float16),
                np.array([[3, 0, 2], [1, 3, 0], [0, 1, 2]], np.int64),
            ],
        ),
        # add more onnxmlir node tests here
    ]
