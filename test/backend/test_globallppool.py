#!/usr/bin/env python3

"""
Test GlobalLpPool operation against a reference implementation.

ONNX's own backend test suite ships no test cases for GlobalLpPool. This
builds a small ONNX model, computes the expected output with a plain numpy
reference, then actually compiles and runs the model through onnx-mlir (via
utils/onnxmlirrun.py) and asserts the compiled output matches.

GlobalLpPool(X, p) == (sum over spatial dims of |X|^p)) ^ (1/p), which is
what the ONNXToKrnl-level canonicalization pattern rewrites it into
(Abs -> Pow -> ReduceSum -> Pow), so this is an end-to-end, numeric check of
that rewrite (the lit tests only check the rewritten IR shape, not the
actual output values).

Requires the PyRuntimeC target to be built and ONNX_MLIR_HOME set to the
build directory (e.g. onnx-mlir/build/Release or .../Debug), same
requirement as utils/onnxmlirrun.py itself.
"""

import os
import sys

import numpy as np
import onnx
from onnx import helper, TensorProto

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "..", "utils"))
import onnxmlirrun


def global_lp_pool_reference(x, p):
    """Plain numpy reference for GlobalLpPool: reduce over all spatial dims
    (i.e. every axis after the leading N, C), keeping them as size-1 dims."""
    spatial_axes = tuple(range(2, x.ndim))
    return np.power(
        np.sum(np.power(np.abs(x), p), axis=spatial_axes, keepdims=True), 1.0 / p
    )


def create_global_lp_pool_model(input_shape, p=2):
    """Create an ONNX model with a single GlobalLpPool node."""
    output_shape = list(input_shape[:2]) + [1] * (len(input_shape) - 2)

    X = helper.make_tensor_value_info("X", TensorProto.FLOAT, input_shape)
    Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, output_shape)

    node = helper.make_node(
        "GlobalLpPool",
        inputs=["X"],
        outputs=["Y"],
        p=p,
    )

    graph = helper.make_graph([node], "global_lp_pool_test", [X], [Y])
    model = helper.make_model(graph, producer_name="global_lp_pool_test")
    model.opset_import[0].version = 2  # GlobalLpPool's p-as-int64 schema.

    return model


def run_and_check(name, input_data, p):
    """Build the model, compile+run it with onnx-mlir, and assert the
    compiled output matches the numpy reference. Returns True/False."""
    print(f"\n=== Test GlobalLpPool {name} (p={p}, input shape {input_data.shape}) ===")

    expected = global_lp_pool_reference(input_data, p)
    print(f"Expected output shape: {expected.shape}")
    print(f"Expected output:\n{expected}")

    model = create_global_lp_pool_model(input_shape=list(input_data.shape), p=p)
    model_path = f"/tmp/globallppool_{name}.onnx"
    onnx.save(model, model_path)
    print(f"Model saved to {model_path}")

    session = onnxmlirrun.InferenceSession(model_path)
    (actual,) = session.run(None, {"X": input_data})

    print(f"Actual output:\n{actual}")

    if not np.allclose(actual, expected, rtol=1e-4, atol=1e-5):
        print(f"FAIL: {name} - output does not match reference")
        print(f"Max abs diff: {np.max(np.abs(actual - expected))}")
        return False

    print(f"PASS: {name}")
    return True


def test_global_lp_pool_default():
    """Test GlobalLpPool with the default p=2 (L2 norm) on a 4D NCHW input."""
    rng = np.random.default_rng(0)
    input_data = rng.standard_normal((1, 3, 5, 5)).astype(np.float32)
    return run_and_check("default", input_data, p=2)


def test_global_lp_pool_1d_p3():
    """Test GlobalLpPool with p=3 (L3 norm) on a 1D-spatial (rank-3) input."""
    rng = np.random.default_rng(1)
    input_data = rng.standard_normal((2, 4, 7)).astype(np.float32)
    return run_and_check("1d_p3", input_data, p=3)


def test_global_lp_pool_3d():
    """Test GlobalLpPool (default p=2) on a 3D-spatial (rank-5) input."""
    rng = np.random.default_rng(2)
    input_data = rng.standard_normal((1, 2, 3, 4, 4)).astype(np.float32)
    return run_and_check("3d", input_data, p=2)


def test_global_lp_pool_p1():
    """Test GlobalLpPool with p=1 (L1 norm), which takes a distinct rewrite
    path (Abs -> ReduceSum, no Pow) compared to p>1."""
    rng = np.random.default_rng(3)
    input_data = rng.standard_normal((1, 3, 5, 5)).astype(np.float32)
    return run_and_check("p1", input_data, p=1)


if __name__ == "__main__":
    print("GlobalLpPool onnx-mlir Numeric Verification Tests")
    print("=" * 50)

    results = [
        test_global_lp_pool_default(),
        test_global_lp_pool_1d_p3(),
        test_global_lp_pool_3d(),
        test_global_lp_pool_p1(),
    ]

    print("\n" + "=" * 50)
    if all(results):
        print("All tests passed!")
        sys.exit(0)
    else:
        print(f"{results.count(False)} of {len(results)} test(s) FAILED")
        sys.exit(1)
