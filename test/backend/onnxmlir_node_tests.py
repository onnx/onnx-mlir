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
        # GridSample tests
        make_onnxmlir_node_test(
            """
test_onnxmlir_gridsample_2d_bilinear
(float[1,1,4,4] X, float[1,2,2,2] grid) => (float[1,1,2,2] Y) {
    Y = GridSample <mode = "linear", padding_mode = "zeros", align_corners = 0> (X, grid)
}
""",
            [
                np.array([[[[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]]]], np.float32),
                np.array([[[[-0.5, -0.5], [0.5, -0.5]], [[-0.5, 0.5], [0.5, 0.5]]]], np.float32),
            ],
            [
                np.array([[[[3.5, 4.5], [7.5, 8.5]]]], np.float32),
            ],
        ),
        make_onnxmlir_node_test(
            """
test_onnxmlir_gridsample_2d_nearest
(float[1,1,4,4] X, float[1,2,2,2] grid) => (float[1,1,2,2] Y) {
    Y = GridSample <mode = "nearest", padding_mode = "zeros", align_corners = 0> (X, grid)
}
""",
            [
                np.array([[[[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]]]], np.float32),
                np.array([[[[-0.5, -0.5], [0.5, -0.5]], [[-0.5, 0.5], [0.5, 0.5]]]], np.float32),
            ],
            [
                np.array([[[[6, 6], [10, 10]]]], np.float32),
            ],
        ),
        make_onnxmlir_node_test(
            """
test_onnxmlir_gridsample_2d_bicubic
(float[1,1,4,4] X, float[1,2,2,2] grid) => (float[1,1,2,2] Y) {
    Y = GridSample <mode = "cubic", padding_mode = "zeros", align_corners = 0> (X, grid)
}
""",
            [
                np.array([[[[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]]]], np.float32),
                np.array([[[[-0.5, -0.5], [0.5, -0.5]], [[-0.5, 0.5], [0.5, 0.5]]]], np.float32),
            ],
            [
                np.array([[[[3.5, 4.5], [7.5, 8.5]]]], np.float32),
            ],
            rtol=1e-2,
        ),
        make_onnxmlir_node_test(
            """
test_onnxmlir_gridsample_2d_border
(float[1,1,4,4] X, float[1,2,2,2] grid) => (float[1,1,2,2] Y) {
    Y = GridSample <mode = "linear", padding_mode = "border", align_corners = 0> (X, grid)
}
""",
            [
                np.array([[[[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]]]], np.float32),
                np.array([[[[-1.5, -1.5], [1.5, -1.5]], [[-1.5, 1.5], [1.5, 1.5]]]], np.float32),
            ],
            [
                np.array([[[[1.0, 4.0], [13.0, 16.0]]]], np.float32),
            ],
        ),
        make_onnxmlir_node_test(
            """
test_onnxmlir_gridsample_align_corners
(float[1,1,4,4] X, float[1,2,2,2] grid) => (float[1,1,2,2] Y) {
    Y = GridSample <mode = "linear", padding_mode = "zeros", align_corners = 1> (X, grid)
}
""",
            [
                np.array([[[[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]]]], np.float32),
                np.array([[[[-1.0, -1.0], [1.0, -1.0]], [[-1.0, 1.0], [1.0, 1.0]]]], np.float32),
            ],
            [
                np.array([[[[1.0, 4.0], [13.0, 16.0]]]], np.float32),
            ],
        ),
        # add more onnxmlir node tests here
    ]
