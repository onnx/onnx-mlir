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
    _PARAM_FULL_HASH_LIMIT,
    _stable_args_str,
    _tensor_sha256,
    generate_hash_key,
)

# ---------------------------------------------------------------------------
# Minimal helpers to build FX graphs without a compiler
# ---------------------------------------------------------------------------

# Two graph-building helpers are provided because they exercise different code
# paths in generate_hash_key:
#
# _make_gm_with_param(weight)
#   Builds a GraphModule with explicit get_attr + call_function nodes — the
#   same structure torch.compile (with prepare_freezing=1) produces.  This is
#   what the production backend actually sees, and it is what exercises the
#   parameter-hashing loop in generate_hash_key.
#
# _make_linear_model(weight)
#   Builds a GraphModule via torch.fx.symbolic_trace.  symbolic_trace wraps
#   nn.Linear as a single call_module node — the weight never surfaces as a
#   get_attr node and is therefore invisible to the parameter-hashing loop.
#   Tests using this helper verify graph-structure hashing (node names, ops)
#   and document the known limitation: call_module graphs are not
#   parameter-value-sensitive.


def _compile_options():
    return {"compile_options": "-O3", "compiler_path": "/unused"}


def _make_gm_with_param(weight: torch.Tensor) -> torch.fx.GraphModule:
    """
    Build a GraphModule with a get_attr node for *weight* feeding a
    call_function node — the same structure torch.compile (prepare_freezing)
    produces for a single-linear-layer model.

    This is the correct helper for tests that verify parameter-value hashing.
    """
    gm = torch.fx.GraphModule({}, torch.fx.Graph())
    gm.register_parameter("weight", nn.Parameter(weight.clone()))
    with gm.graph.inserting_before(None):
        x = gm.graph.placeholder("x")
        w = gm.graph.get_attr("weight")
        out = gm.graph.call_function(torch.nn.functional.linear, (x, w))
        gm.graph.output(out)
    gm.recompile()
    return gm


def _make_linear_model(weight: torch.Tensor) -> torch.fx.GraphModule:
    """
    Build a GraphModule via torch.fx.symbolic_trace for a single nn.Linear
    layer with the given weight (no bias).

    symbolic_trace produces a call_module node for nn.Linear — the weight
    does NOT appear as a get_attr node.  Use this helper only for tests that
    check graph-structure hashing, not parameter-value sensitivity.
    """

    class _LinearModel(nn.Module):
        def __init__(self, w):
            super().__init__()
            self.linear = nn.Linear(w.shape[1], w.shape[0], bias=False)
            with torch.no_grad():
                self.linear.weight.copy_(w)

        def forward(self, x):
            return self.linear(x)

    model = _LinearModel(weight)
    model.eval()
    return torch.fx.symbolic_trace(model)


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
    """
    _tensor_sha256 must be stable, value-sensitive, and bounded in cost.

    Two code paths:
      - small tensors (numel <= _PARAM_FULL_HASH_LIMIT): full SHA-256 of all bytes
      - large tensors (numel >  _PARAM_FULL_HASH_LIMIT): strided-sample SHA-256
    Both paths are tested for determinism, value-sensitivity, and output format.
    """

    # ------------------------------------------------------------------ small
    def test_small_deterministic(self):
        t = torch.tensor([1.0, 2.0, 3.0])
        self.assertEqual(_tensor_sha256(t), _tensor_sha256(t))

    def test_small_same_values_same_hash(self):
        t1 = torch.tensor([1.0, 2.0, 3.0])
        t2 = torch.tensor([1.0, 2.0, 3.0])
        self.assertEqual(_tensor_sha256(t1), _tensor_sha256(t2))

    def test_small_different_values_differ(self):
        t1 = torch.tensor([1.0, 2.0, 3.0])
        t2 = torch.tensor([1.0, 2.0, 9.9])
        self.assertNotEqual(_tensor_sha256(t1), _tensor_sha256(t2))

    def test_small_only_first_3_values_same_still_differs(self):
        """f026: fixing first-3-value prefix must not defeat the hash."""
        base = [1.0, 2.0, 3.0]
        t1 = torch.tensor(base + [4.0, 5.0])
        t2 = torch.tensor(base + [99.0, 99.0])
        self.assertNotEqual(_tensor_sha256(t1), _tensor_sha256(t2))

    def test_small_returns_hex_string(self):
        t = torch.zeros(4)
        h = _tensor_sha256(t)
        self.assertIsInstance(h, str)
        self.assertEqual(len(h), 64)

    def test_small_parameter_same_as_tensor(self):
        data = torch.tensor([0.5, 1.5])
        param = nn.Parameter(data.clone())
        self.assertEqual(_tensor_sha256(data), _tensor_sha256(param))

    def test_small_different_shapes_differ(self):
        """Shape is included in the header, so shape differences must matter."""
        t1 = torch.ones(4)
        t2 = torch.ones(2, 2)  # same values, different shape
        self.assertNotEqual(_tensor_sha256(t1), _tensor_sha256(t2))

    # ------------------------------------------------------------------ large
    def _large_tensor(self, fill_value=1.0):
        """Return a tensor with numel > _PARAM_FULL_HASH_LIMIT."""
        return torch.full((_PARAM_FULL_HASH_LIMIT + 1,), fill_value)

    def test_large_deterministic(self):
        t = self._large_tensor()
        self.assertEqual(_tensor_sha256(t), _tensor_sha256(t))

    def test_large_same_values_same_hash(self):
        t1 = self._large_tensor(1.0)
        t2 = self._large_tensor(1.0)
        self.assertEqual(_tensor_sha256(t1), _tensor_sha256(t2))

    def test_large_different_values_differ(self):
        """Values beyond the first 3 must still produce different hashes."""
        numel = _PARAM_FULL_HASH_LIMIT + 64
        t1 = torch.zeros(numel)
        t2 = torch.zeros(numel)
        # Change the very last element — far beyond any fixed 3-value prefix.
        t2[-1] = 99.0
        self.assertNotEqual(
            _tensor_sha256(t1),
            _tensor_sha256(t2),
            "Strided sampling must detect changes beyond the first 3 elements",
        )

    def test_large_returns_hex_string(self):
        t = self._large_tensor()
        h = _tensor_sha256(t)
        self.assertIsInstance(h, str)
        self.assertEqual(len(h), 64)

    def test_large_different_shapes_differ(self):
        """Shape header must distinguish same-content, different-shape tensors."""
        numel = _PARAM_FULL_HASH_LIMIT + 1
        t1 = torch.ones(numel)
        t2 = torch.ones(1, numel)  # same values, different shape
        self.assertNotEqual(_tensor_sha256(t1), _tensor_sha256(t2))


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

    Uses _make_gm_with_param() which builds a GraphModule with explicit
    get_attr + call_function nodes — the same structure torch.compile produces
    with prepare_freezing=1.  torch.fx.symbolic_trace must NOT be used here
    because it produces call_module nodes that keep the weight opaque (no
    get_attr node), making the parameter invisible to the hash loop.
    """

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
        gm1 = _make_gm_with_param(w1)
        gm2 = _make_gm_with_param(w2)
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
        gm1 = _make_gm_with_param(w.clone())
        gm2 = _make_gm_with_param(w.clone())
        inputs = [torch.zeros(1, 5)]
        opts = _compile_options()
        k1 = generate_hash_key(gm1, inputs, opts)
        k2 = generate_hash_key(gm2, inputs, opts)
        self.assertEqual(k1, k2)

    def test_completely_different_params_differ(self):
        w1 = torch.zeros(1, 4)
        w2 = torch.ones(1, 4)
        gm1 = _make_gm_with_param(w1)
        gm2 = _make_gm_with_param(w2)
        inputs = [torch.zeros(1, 4)]
        opts = _compile_options()
        k1 = generate_hash_key(gm1, inputs, opts)
        k2 = generate_hash_key(gm2, inputs, opts)
        self.assertNotEqual(k1, k2)


# ---------------------------------------------------------------------------
# Tests for generate_hash_key — call_module graphs (symbolic_trace)
# ---------------------------------------------------------------------------


class TestHashKeyCallModuleGraphs(unittest.TestCase):
    """
    Graphs produced by torch.fx.symbolic_trace use call_module nodes for
    nn.Linear — the weight never appears as a get_attr node and is therefore
    NOT captured by the parameter-hashing loop.

    These tests document that behaviour: two call_module graphs with identical
    structure but different weights produce the same key.  This is a known
    limitation of symbolic_trace graphs; the production backend (torch.compile
    with prepare_freezing=1) always produces get_attr graphs and is not
    affected.
    """

    def test_same_structure_same_key_regardless_of_weights(self):
        """
        call_module graphs: identical structure → same key even if weights differ.
        This is the expected (and documented) behaviour for symbolic_trace output.
        """
        w1 = torch.zeros(1, 4)
        w2 = torch.ones(1, 4)
        gm1 = _make_linear_model(w1)
        gm2 = _make_linear_model(w2)
        inputs = [torch.zeros(1, 4)]
        opts = _compile_options()
        k1 = generate_hash_key(gm1, inputs, opts)
        k2 = generate_hash_key(gm2, inputs, opts)
        # call_module graphs hash by structure only — weight difference is invisible.
        self.assertEqual(
            k1,
            k2,
            "call_module graphs with the same structure must produce the same "
            "key regardless of weight values (parameter hashing only applies "
            "to get_attr graphs produced by torch.compile)",
        )

    def test_different_op_name_different_key(self):
        """
        call_module graphs with a structurally different node sequence (different
        call_module target name) must produce different keys.
        """

        class _ModelA(nn.Module):
            def __init__(self):
                super().__init__()
                self.layer_a = nn.Linear(4, 4, bias=False)

            def forward(self, x):
                return self.layer_a(x)

        class _ModelB(nn.Module):
            def __init__(self):
                super().__init__()
                self.layer_b = nn.Linear(4, 4, bias=False)  # different attr name

            def forward(self, x):
                return self.layer_b(x)

        gm1 = torch.fx.symbolic_trace(_ModelA())
        gm2 = torch.fx.symbolic_trace(_ModelB())
        inputs = [torch.zeros(1, 4)]
        opts = _compile_options()
        k1 = generate_hash_key(gm1, inputs, opts)
        k2 = generate_hash_key(gm2, inputs, opts)
        # Different call_module target names → different graph_str → different key.
        self.assertNotEqual(k1, k2)


if __name__ == "__main__":
    unittest.main()
