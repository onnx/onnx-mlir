# SPDX-License-Identifier: Apache-2.0

##################### test_hash_key.py #########################################
#
# Copyright 2026 The IBM Research Authors.
#
################################################################################
#
# Unit tests for generate_hash_key() and its helpers (_stable_args_str,
# _tensor_sha256).
#
# These tests run without an onnx-mlir compiler or InferenceSession: they only
# need torch and the torch_onnxmlir package to be installed.  They specifically
# cover the two gaps fixed by f026:
#
#   1. Literal constants in node.args/node.kwargs were invisible to the
#      lightweight hash — two graphs with identical topology but different
#      constant values (reshape dims, axis index, scalar threshold …) produced
#      the same cache key.
#
#   2. Parameter tensors were represented by only the first 3 sampled values —
#      graphs that share the first 3 weight values but differ elsewhere produced
#      the same cache key.
#
################################################################################

import hashlib
import unittest

import torch
import torch.nn as nn

from torch_onnxmlir.backend import (
    _stable_args_str,
    _tensor_sha256,
    generate_hash_key,
)

# ---------------------------------------------------------------------------
# Minimal helpers to build symbolic FX graphs without a compiler
# ---------------------------------------------------------------------------


def _trace(model: nn.Module, *example_inputs):
    """Return a symbolic FX GraphModule via torch.fx.symbolic_trace."""
    return torch.fx.symbolic_trace(model)


def _compile_options():
    return {"compile_options": "-O3", "compiler_path": "/unused"}


# ---------------------------------------------------------------------------
# Tests for _stable_args_str
# ---------------------------------------------------------------------------


class TestStableArgsStr(unittest.TestCase):
    """_stable_args_str must produce stable, type-faithful strings."""

    def test_primitives(self):
        self.assertEqual(_stable_args_str(42), "42")
        self.assertEqual(_stable_args_str(3.14), "3.14")
        self.assertEqual(_stable_args_str("axis"), "'axis'")
        self.assertEqual(_stable_args_str(True), "True")
        self.assertEqual(_stable_args_str(None), "None")

    def test_different_ints_differ(self):
        self.assertNotEqual(_stable_args_str(0), _stable_args_str(1))

    def test_different_floats_differ(self):
        self.assertNotEqual(_stable_args_str(1.0), _stable_args_str(2.0))

    def test_tuple_stable(self):
        s1 = _stable_args_str((1, 2, 3))
        s2 = _stable_args_str((1, 2, 3))
        self.assertEqual(s1, s2)

    def test_tuple_order_matters(self):
        self.assertNotEqual(_stable_args_str((1, 2)), _stable_args_str((2, 1)))

    def test_list_vs_tuple_differ(self):
        # Lists and tuples produce different bracket characters.
        self.assertNotEqual(_stable_args_str([1, 2]), _stable_args_str((1, 2)))

    def test_nested(self):
        s = _stable_args_str((1, [2, 3], None))
        self.assertIn("1", s)
        self.assertIn("2", s)
        self.assertIn("3", s)
        self.assertIn("None", s)

    def test_dict_sorted(self):
        s1 = _stable_args_str({"b": 2, "a": 1})
        s2 = _stable_args_str({"a": 1, "b": 2})
        self.assertEqual(s1, s2)

    def test_tensor_uses_metadata_not_values(self):
        t1 = torch.tensor([1.0, 2.0])
        t2 = torch.tensor([9.0, 9.0])  # different values, same shape/dtype
        self.assertEqual(_stable_args_str(t1), _stable_args_str(t2))

    def test_tensor_shape_differences_matter(self):
        t1 = torch.zeros(2, 3)
        t2 = torch.zeros(3, 2)
        self.assertNotEqual(_stable_args_str(t1), _stable_args_str(t2))


# ---------------------------------------------------------------------------
# Tests for _tensor_sha256
# ---------------------------------------------------------------------------


class TestTensorSha256(unittest.TestCase):
    """_tensor_sha256 must be a stable, value-sensitive digest."""

    def test_deterministic(self):
        t = torch.tensor([1.0, 2.0, 3.0])
        self.assertEqual(_tensor_sha256(t), _tensor_sha256(t))

    def test_same_values_same_hash(self):
        t1 = torch.tensor([1.0, 2.0, 3.0])
        t2 = torch.tensor([1.0, 2.0, 3.0])
        self.assertEqual(_tensor_sha256(t1), _tensor_sha256(t2))

    def test_different_values_differ(self):
        t1 = torch.tensor([1.0, 2.0, 3.0])
        t2 = torch.tensor([1.0, 2.0, 9.9])
        self.assertNotEqual(_tensor_sha256(t1), _tensor_sha256(t2))

    def test_only_first_3_values_same_still_differs(self):
        """
        f026 gap: sampling only 3 values meant this pair produced the same key.
        Full-hash must distinguish them.
        """
        base = [1.0, 2.0, 3.0]
        t1 = torch.tensor(base + [4.0, 5.0])
        t2 = torch.tensor(base + [99.0, 99.0])
        self.assertNotEqual(_tensor_sha256(t1), _tensor_sha256(t2))

    def test_returns_hex_string(self):
        t = torch.zeros(4)
        h = _tensor_sha256(t)
        self.assertIsInstance(h, str)
        self.assertEqual(len(h), 64)  # SHA-256 hex digest is always 64 chars

    def test_parameter_same_as_tensor(self):
        data = torch.tensor([0.5, 1.5])
        param = nn.Parameter(data.clone())
        self.assertEqual(_tensor_sha256(data), _tensor_sha256(param))


# ---------------------------------------------------------------------------
# Tests for generate_hash_key — literal-args gap (f026 fix 1)
# ---------------------------------------------------------------------------


class TestHashKeyLiteralArgs(unittest.TestCase):
    """
    Two graphs with identical topology but different literal constants in
    node.args must produce different cache keys.
    """

    def _make_reshape_model(self, target_shape):
        """Return a GraphModule whose forward reshapes input to target_shape."""

        class ReshapeModel(nn.Module):
            def __init__(self, shape):
                super().__init__()
                self._shape = shape

            def forward(self, x):
                return x.reshape(self._shape)

        return torch.fx.symbolic_trace(ReshapeModel(target_shape))

    def test_different_reshape_dims_differ(self):
        gm1 = self._make_reshape_model((4, 8))
        gm2 = self._make_reshape_model((8, 4))
        inputs = [torch.zeros(32)]
        opts = _compile_options()
        k1 = generate_hash_key(gm1, inputs, opts)
        k2 = generate_hash_key(gm2, inputs, opts)
        self.assertNotEqual(
            k1, k2, "Different reshape dims must yield different cache keys"
        )

    def _make_add_scalar_model(self, scalar):
        class AddScalarModel(nn.Module):
            def __init__(self, s):
                super().__init__()
                self._s = s

            def forward(self, x):
                return x + self._s

        return torch.fx.symbolic_trace(AddScalarModel(scalar))

    def test_different_scalar_constants_differ(self):
        gm1 = self._make_add_scalar_model(1)
        gm2 = self._make_add_scalar_model(2)
        inputs = [torch.zeros(4)]
        opts = _compile_options()
        k1 = generate_hash_key(gm1, inputs, opts)
        k2 = generate_hash_key(gm2, inputs, opts)
        self.assertNotEqual(
            k1, k2, "Different scalar constants must yield different cache keys"
        )

    def test_same_graph_same_key(self):
        """Sanity: same graph produces the same key across two calls."""
        gm = self._make_reshape_model((2, 16))
        inputs = [torch.zeros(32)]
        opts = _compile_options()
        k1 = generate_hash_key(gm, inputs, opts)
        k2 = generate_hash_key(gm, inputs, opts)
        self.assertEqual(k1, k2)


# ---------------------------------------------------------------------------
# Tests for generate_hash_key — parameter-sampling gap (f026 fix 2)
# ---------------------------------------------------------------------------


class TestHashKeyParameterValues(unittest.TestCase):
    """
    Two graphs that share the first N parameter values but differ beyond that
    must produce different cache keys (full-bytes hash, not sampled values).
    """

    def _make_linear_model(self, weight: torch.Tensor) -> torch.fx.GraphModule:
        """
        Return a GraphModule for a single Linear layer with a given weight.
        The bias is zero.  torch.fx.symbolic_trace captures the weight as a
        get_attr + parameter node, exactly as the real backend sees it.
        """

        class LinearModel(nn.Module):
            def __init__(self, w):
                super().__init__()
                self.linear = nn.Linear(w.shape[1], w.shape[0], bias=False)
                with torch.no_grad():
                    self.linear.weight.copy_(w)

            def forward(self, x):
                return self.linear(x)

        model = LinearModel(weight)
        model.eval()
        return torch.fx.symbolic_trace(model)

    def _shared_prefix_weights(self):
        """
        Return two weight tensors that agree on the first 3 values but differ
        on the 4th and beyond — the exact scenario f026 allowed to collide.
        """
        base = [0.1, 0.2, 0.3]
        w1 = torch.tensor([base + [0.4, 0.5]], dtype=torch.float32)  # (1, 5)
        w2 = torch.tensor([base + [9.9, 9.9]], dtype=torch.float32)  # (1, 5)
        return w1, w2

    def test_params_differing_beyond_first_3_produce_different_keys(self):
        w1, w2 = self._shared_prefix_weights()
        gm1 = self._make_linear_model(w1)
        gm2 = self._make_linear_model(w2)
        inputs = [torch.zeros(1, 5)]
        opts = _compile_options()
        k1 = generate_hash_key(gm1, inputs, opts)
        k2 = generate_hash_key(gm2, inputs, opts)
        self.assertNotEqual(
            k1,
            k2,
            "Parameters differing beyond the first 3 values must yield "
            "different cache keys (full-hash, not 3-sample)",
        )

    def test_identical_params_produce_same_key(self):
        """Sanity: identical weights produce the same key."""
        w = torch.rand(1, 5)
        gm1 = self._make_linear_model(w.clone())
        gm2 = self._make_linear_model(w.clone())
        inputs = [torch.zeros(1, 5)]
        opts = _compile_options()
        k1 = generate_hash_key(gm1, inputs, opts)
        k2 = generate_hash_key(gm2, inputs, opts)
        self.assertEqual(k1, k2)

    def test_completely_different_params_differ(self):
        w1 = torch.zeros(1, 4)
        w2 = torch.ones(1, 4)
        gm1 = self._make_linear_model(w1)
        gm2 = self._make_linear_model(w2)
        inputs = [torch.zeros(1, 4)]
        opts = _compile_options()
        k1 = generate_hash_key(gm1, inputs, opts)
        k2 = generate_hash_key(gm2, inputs, opts)
        self.assertNotEqual(k1, k2)


if __name__ == "__main__":
    unittest.main()
