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
        import os
        from torch_onnxmlir import config

        self.assertFalse(config._profiling_enabled)
        self.assertIsNone(config._profiling_file_path)
        self.assertNotIn("ONNX_MLIR_INSTRUMENT_FILE", os.environ)

        compiled_model = torch.compile(model, backend="onnxmlir", options=OM_OPTIONS)
        with torch_onnxmlir.explain_context(profiling=True) as ctx:
            self.assertTrue(config._profiling_enabled)
            self.assertIsNotNone(config._profiling_file_path)
            self.assertIn("ONNX_MLIR_INSTRUMENT_FILE", os.environ)
            x = torch.randn(2, 3)
            output = compiled_model(x)

        self.assertFalse(config._profiling_enabled)
        self.assertIsNone(config._profiling_file_path)
        self.assertNotIn("ONNX_MLIR_INSTRUMENT_FILE", os.environ)

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


if __name__ == "__main__":
    unittest.main()
