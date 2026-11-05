# SPDX-License-Identifier: Apache-2.0

##################### test_profiling.py ########################################
#
# Copyright 2026 The IBM Research Authors.
#
################################################################################
#
# Integration tests for the profiling feature with explain_context().
# These tests require a real onnx-mlir compiler.
#
################################################################################

import unittest
import json
import torch
import torch.nn as nn
import torch_onnxmlir
from torch_onnxmlir import explain, metrics
from utils import TorchOMTestCase, COMPILER_IMAGE_NAME, COMPILER_PATH


class SimpleModel(nn.Module):
    def forward(self, x):
        return x + 1


model = SimpleModel()
model.eval()

OM_OPTIONS = {
    "compiler_image_name": COMPILER_IMAGE_NAME,
    "compiler_path": COMPILER_PATH,
    "compile_options": "-O3",
}


class TestProfilingIntegration(TorchOMTestCase):

    def test_explain_context_with_profiling_table(self):
        """Test that profiling=True produces per-op data in table format."""
        compiled_model = torch.compile(model, backend="onnxmlir", options=OM_OPTIONS)
        with torch_onnxmlir.explain_context(profiling=True) as ctx:
            x = torch.randn(2, 3)
            output = compiled_model(x)

        self.assertIsNotNone(ctx.metrics)
        self.assertIsInstance(ctx.data, str)
        self.assertIn("PER-OP PROFILING DATA", ctx.data)

    def test_explain_context_with_profiling_dict(self):
        """Test that profiling=True produces per-op data in dict format."""
        compiled_model = torch.compile(model, backend="onnxmlir", options=OM_OPTIONS)
        with torch_onnxmlir.explain_context(profiling=True, format="dict") as ctx:
            x = torch.randn(2, 3)
            output = compiled_model(x)

        self.assertIsNotNone(ctx.data)
        self.assertIsInstance(ctx.data, dict)
        self.assertIn("profiling", ctx.data)
        prof = ctx.data["profiling"]
        self.assertGreater(len(prof), 0)
        for key, entry in prof.items():
            self.assertIn("total_time", entry)
            self.assertIn("ops", entry)
            self.assertGreater(len(entry["ops"]), 0)

    def test_explain_context_with_profiling_json(self):
        """Test that profiling=True produces valid JSON with profiling data."""
        compiled_model = torch.compile(model, backend="onnxmlir", options=OM_OPTIONS)
        with torch_onnxmlir.explain_context(profiling=True, format="json") as ctx:
            x = torch.randn(2, 3)
            output = compiled_model(x)

        self.assertIsInstance(ctx.data, str)
        data = json.loads(ctx.data)
        self.assertIn("profiling", data)

    def test_explain_context_without_profiling(self):
        """Test that profiling=False (default) has no profiling section."""
        compiled_model = torch.compile(model, backend="onnxmlir", options=OM_OPTIONS)
        with torch_onnxmlir.explain_context(format="dict") as ctx:
            x = torch.randn(2, 3)
            output = compiled_model(x)

        self.assertNotIn("profiling", ctx.data)

    def test_profiling_state_restoration(self):
        """Test that profiling state is properly restored after context."""
        from torch_onnxmlir import config

        self.assertFalse(config._profiling_enabled)

        compiled_model = torch.compile(model, backend="onnxmlir", options=OM_OPTIONS)
        with torch_onnxmlir.explain_context(profiling=True) as ctx:
            self.assertTrue(config._profiling_enabled)
            x = torch.randn(2, 3)
            output = compiled_model(x)

        self.assertFalse(config._profiling_enabled)

    def test_profiling_multiple_inferences(self):
        """Test profiling with multiple inference calls."""
        compiled_model = torch.compile(model, backend="onnxmlir", options=OM_OPTIONS)
        with torch_onnxmlir.explain_context(profiling=True, format="dict") as ctx:
            for i in range(3):
                x = torch.randn(2, 3)
                output = compiled_model(x)

        self.assertIn("profiling", ctx.data)
        prof = ctx.data["profiling"]
        for key, entry in prof.items():
            self.assertGreater(entry["num_measurements"], 0)

    def test_profiling_repr(self):
        """Test that repr includes profiling flag."""
        ctx = torch_onnxmlir.explain_context(profiling=True)
        self.assertIn("profiling=True", repr(ctx))

        ctx2 = torch_onnxmlir.explain_context()
        self.assertIn("profiling=False", repr(ctx2))

    def test_profiling_dict_config_overrides(self):
        """Test that profiling=dict overrides detail/warmup/reporting."""
        ctx = torch_onnxmlir.explain_context(
            profiling={"detail": 2, "warmup": 1, "reporting": "sum"}
        )
        self.assertTrue(ctx.profiling)
        self.assertEqual(ctx.profiling_detail, 2)
        self.assertEqual(ctx.profiling_warmup, 1)
        self.assertEqual(ctx.profiling_reporting, "sum")

    def test_profiling_dict_defaults_from_config(self):
        """Test that omitted dict keys fall back to config defaults."""
        from torch_onnxmlir import config

        ctx = torch_onnxmlir.explain_context(profiling={})
        self.assertTrue(ctx.profiling)
        self.assertEqual(ctx.profiling_detail, config.profiling_detail)
        self.assertEqual(ctx.profiling_warmup, config.profiling_warmup)
        self.assertEqual(ctx.profiling_reporting, config.profiling_reporting)

    def test_profiling_dict_partial_override(self):
        """Test that only specified dict keys override, rest use defaults."""
        from torch_onnxmlir import config

        ctx = torch_onnxmlir.explain_context(profiling={"detail": 1})
        self.assertTrue(ctx.profiling)
        self.assertEqual(ctx.profiling_detail, 1)
        self.assertEqual(ctx.profiling_warmup, config.profiling_warmup)
        self.assertEqual(ctx.profiling_reporting, config.profiling_reporting)

    def test_profiling_dict_with_warmup(self):
        """Test profiling=dict with warmup skips initial runs."""
        compiled_model = torch.compile(model, backend="onnxmlir", options=OM_OPTIONS)
        with torch_onnxmlir.explain_context(
            profiling={"warmup": 1}, format="dict"
        ) as ctx:
            for _ in range(3):
                x = torch.randn(2, 3)
                output = compiled_model(x)

        self.assertIn("profiling", ctx.data)
        for key, entry in ctx.data["profiling"].items():
            self.assertEqual(entry["reporting_mode"], "all")

    def test_profiling_dict_detail_signatures_in_dict(self):
        """Test profiling=dict with detail >= 1 includes signatures key."""
        compiled_model = torch.compile(model, backend="onnxmlir", options=OM_OPTIONS)
        with torch_onnxmlir.explain_context(
            profiling={"detail": 1}, format="dict"
        ) as ctx:
            x = torch.randn(2, 3)
            output = compiled_model(x)

        self.assertIn("profiling", ctx.data)
        for key, entry in ctx.data["profiling"].items():
            for op in entry["ops"]:
                self.assertIn("signatures", op)

    def test_profiling_dict_detail_nodes_in_dict(self):
        """Test profiling=dict with detail >= 2 includes node names."""
        compiled_model = torch.compile(model, backend="onnxmlir", options=OM_OPTIONS)
        with torch_onnxmlir.explain_context(
            profiling={"detail": 2}, format="dict"
        ) as ctx:
            x = torch.randn(2, 3)
            output = compiled_model(x)

        self.assertIn("profiling", ctx.data)
        for key, entry in ctx.data["profiling"].items():
            for op in entry["ops"]:
                self.assertIn("signatures", op)
                for sig in op["signatures"]:
                    self.assertIn("nodes", sig)

    def test_profiling_dict_detail_0_no_signatures(self):
        """Test profiling=dict with detail=0 omits signatures key."""
        compiled_model = torch.compile(model, backend="onnxmlir", options=OM_OPTIONS)
        with torch_onnxmlir.explain_context(
            profiling={"detail": 0}, format="dict"
        ) as ctx:
            x = torch.randn(2, 3)
            output = compiled_model(x)

        self.assertIn("profiling", ctx.data)
        for key, entry in ctx.data["profiling"].items():
            for op in entry["ops"]:
                self.assertNotIn("signatures", op)

    def test_profiling_dict_sum_reporting(self):
        """Test profiling=dict with reporting=sum produces cumulative time."""
        compiled_model = torch.compile(model, backend="onnxmlir", options=OM_OPTIONS)
        with torch_onnxmlir.explain_context(
            profiling={"reporting": "sum"}, format="dict"
        ) as ctx:
            for _ in range(3):
                x = torch.randn(2, 3)
                output = compiled_model(x)

        self.assertIn("profiling", ctx.data)
        for key, entry in ctx.data["profiling"].items():
            self.assertEqual(entry["reporting_mode"], "sum")

    def test_profiling_dict_quartile_reporting(self):
        """Test profiling=dict with reporting=quartile."""
        compiled_model = torch.compile(model, backend="onnxmlir", options=OM_OPTIONS)
        with torch_onnxmlir.explain_context(
            profiling={"reporting": "quartile"}, format="dict"
        ) as ctx:
            for _ in range(5):
                x = torch.randn(2, 3)
                output = compiled_model(x)

        self.assertIn("profiling", ctx.data)
        for key, entry in ctx.data["profiling"].items():
            self.assertEqual(entry["reporting_mode"], "quartile")

    def test_profiling_dict_table_format(self):
        """Test profiling=dict config works with table format output."""
        compiled_model = torch.compile(model, backend="onnxmlir", options=OM_OPTIONS)
        with torch_onnxmlir.explain_context(
            profiling={"detail": 1, "reporting": "all"}
        ) as ctx:
            x = torch.randn(2, 3)
            output = compiled_model(x)

        self.assertIsInstance(ctx.data, str)
        self.assertIn("PER-OP PROFILING DATA", ctx.data)
        self.assertIn("Per-node breakdown:", ctx.data)

    def test_profiling_dict_state_restoration(self):
        """Test that profiling state is restored after dict config context."""
        from torch_onnxmlir import config

        self.assertFalse(config._profiling_enabled)

        compiled_model = torch.compile(model, backend="onnxmlir", options=OM_OPTIONS)
        with torch_onnxmlir.explain_context(
            profiling={"detail": 2, "warmup": 1, "reporting": "sum"}
        ) as ctx:
            self.assertTrue(config._profiling_enabled)
            x = torch.randn(2, 3)
            output = compiled_model(x)

        self.assertFalse(config._profiling_enabled)


if __name__ == "__main__":
    unittest.main()
