# SPDX-License-Identifier: Apache-2.0

##################### test_explain_context.py ##################################
#
# Copyright 2026 The IBM Research Authors.
#
################################################################################
#
# Tests for explain_context() context manager functionality.
#
################################################################################

import unittest
import time
import torch
import torch.nn as nn
import torch_onnxmlir
import tempfile
import os
import json
from torch_onnxmlir import explain, metrics
from utils import TorchOMTestCase, COMPILER_IMAGE_NAME, COMPILER_PATH


class SimpleModel(nn.Module):
    def forward(self, x):
        return x + 1


model = SimpleModel()
model.eval()

compiled_model = torch.compile(
    model,
    backend="onnxmlir",
    options={
        "compiler_image_name": COMPILER_IMAGE_NAME,
        "compiler_path": COMPILER_PATH,
        "compile_options": "-O3",
    },
)


class TestExplainContext(TorchOMTestCase):

    def test_context_manager_basic(self):
        """Test basic context manager usage with default table format."""

        with torch_onnxmlir.explain_context() as ctx:
            x = torch.randn(2, 3)
            output = compiled_model(x)

        # Metrics should be captured in table format (default).
        self.assertIsNotNone(ctx.metrics)
        self.assertIsInstance(ctx.data, str)
        self.assertIn("TORCH_ONNXMLIR PERFORMANCE METRICS", ctx.data)

    def test_context_manager_formats(self):
        """Test different output formats."""

        x = torch.randn(2, 3)
        # Test dict format.
        with torch_onnxmlir.explain_context(format="dict") as ctx:
            output = compiled_model(x)
        self.assertIsInstance(ctx.data, dict)

        # Test table format.
        with torch_onnxmlir.explain_context(format="table") as ctx:
            output = compiled_model(x)
        self.assertIsInstance(ctx.data, str)
        self.assertIn("TORCH_ONNXMLIR PERFORMANCE METRICS", ctx.data)

        # Test json format.
        with torch_onnxmlir.explain_context(format="json") as ctx:
            output = compiled_model(x)
        self.assertIsInstance(ctx.data, str)
        json.loads(ctx.data)  # Should not raise.

    def test_context_manager_sort_by(self):
        """Test different sort options."""

        x = torch.randn(2, 3)
        for sort_by in ["time", "calls", "order"]:
            with torch_onnxmlir.explain_context(sort_by=sort_by) as ctx:
                output = compiled_model(x)
            self.assertIsNotNone(ctx.data)

    def test_context_manager_state_restoration(self):
        """Test that enable_explain state is restored."""
        # Get initial state (should be False by default).
        initial_state = torch_onnxmlir.explain.is_enabled()
        self.assertFalse(initial_state)

        x = torch.randn(2, 3)
        with torch_onnxmlir.explain_context() as ctx:
            # Should be enabled inside context.
            self.assertTrue(torch_onnxmlir.explain.is_enabled())
            output = compiled_model(x)

        # Should be restored to initial state after context.
        self.assertEqual(torch_onnxmlir.explain.is_enabled(), initial_state)

    def test_context_manager_str_repr(self):
        """Test string representation."""
        ctx = torch_onnxmlir.explain_context(format="table", sort_by="calls")

        # Before entering context.
        repr_str = repr(ctx)
        self.assertIn("ExplainContext", repr_str)
        self.assertIn("table", repr_str)
        self.assertIn("calls", repr_str)

        str_before = str(ctx)
        self.assertIn("No metrics collected", str_before)

    def test_context_manager_exception_handling(self):
        """Test that exceptions are not suppressed."""

        class ErrorModel(nn.Module):
            def forward(self, x):
                raise ValueError("Test error")

        error_model = ErrorModel()
        x = torch.randn(2, 3)

        # Exception should propagate.
        with self.assertRaises(ValueError):
            with torch_onnxmlir.explain_context() as ctx:
                # This will raise ValueError.
                error_model(x)

        # Metrics should still be captured (even though no compilation happened).
        self.assertIsNotNone(ctx.metrics)

    def test_context_manager_multiple_inferences(self):
        """Test context manager with multiple inferences."""

        with torch_onnxmlir.explain_context(format="dict") as ctx:
            for i in range(3):
                x = torch.randn(2, 3)
                output = compiled_model(x)

        # Should have metrics for all inferences.
        self.assertIsNotNone(ctx.data)
        self.assertGreaterEqual(ctx.data["summary"]["total_inference_calls"], 3)

    def test_context_manager_export_to_file(self):
        """Test automatic export to file."""

        with tempfile.TemporaryDirectory() as tmpdir:
            export_path = os.path.join(tmpdir, "metrics.json")

            with torch_onnxmlir.explain_context(
                format="json", export_to=export_path
            ) as ctx:
                x = torch.randn(2, 3)
                output = compiled_model(x)

            # File should exist.
            self.assertTrue(os.path.exists(export_path))

            # File should contain valid JSON.
            with open(export_path, "r") as f:
                data = json.load(f)
            self.assertIn("summary", data)

    def test_context_manager_export_nested_dirs(self):
        """Test export with nested directory creation."""

        with tempfile.TemporaryDirectory() as tmpdir:
            export_path = os.path.join(tmpdir, "logs", "run1", "metrics.json")

            with torch_onnxmlir.explain_context(export_to=export_path) as ctx:
                x = torch.randn(2, 3)
                output = compiled_model(x)

            # Nested directories should be created.
            self.assertTrue(os.path.exists(export_path))

    def test_context_manager_export_dict_format(self):
        """Test export with dict format (should convert to JSON)."""

        with tempfile.TemporaryDirectory() as tmpdir:
            export_path = os.path.join(tmpdir, "metrics.json")

            with torch_onnxmlir.explain_context(
                format="dict", export_to=export_path
            ) as ctx:
                x = torch.randn(2, 3)
                output = compiled_model(x)

            # File should exist and contain valid JSON.
            self.assertTrue(os.path.exists(export_path))
            with open(export_path, "r") as f:
                data = json.load(f)
            self.assertIn("summary", data)

    def test_context_manager_eager_fallbacks(self):
        """Test explain feature with eager fallbacks."""
        with torch_onnxmlir.explain_context(format="dict") as ctx:
            collector = metrics.global_metrics_collector

            # Simulate various fallback scenarios.
            collector.record_eager_fallback("export_failed", "model_fail_1")
            collector.record_eager_fallback("compilation_failed", "model_fail_2")
            collector.record_eager_fallback("no_inputs", "model_fail_3")
            collector.record_eager_fallback(
                "export_failed", "model_fail_4"
            )  # Same reason.

        # Get metrics.
        result = ctx.data

        # Verify.
        self.assertEqual(result["summary"]["total_eager_fallbacks"], 4)
        self.assertEqual(len(result["eager_fallbacks"]), 4)

        # Check table output includes fallbacks.
        with torch_onnxmlir.explain_context(format="table") as ctx:
            collector = metrics.global_metrics_collector
            collector.record_eager_fallback("export_failed", "test")

        table_output = str(ctx)
        self.assertIn("Eager Mode Fallbacks:", table_output)
        self.assertIn("export_failed", table_output)

    def test_explain_disabled_overhead(self):
        """Test that disabled explain has minimal overhead."""
        # Ensure explain is disabled.
        self.assertFalse(explain.is_enabled())

        collector = metrics.global_metrics_collector
        # Clear any metrics from previous tests.
        collector.clear()

        # These should be no-ops (just boolean checks).
        start = time.perf_counter()
        for _ in range(10000):
            collector.record_compilation("test", 1.0)
            collector.record_inference("test", 0.01, True)
            collector.record_eager_fallback("test")
        elapsed = time.perf_counter() - start

        # Should be very fast (< 10ms for 10k calls, ~1μs per call).
        self.assertLess(
            elapsed, 0.010, f"Disabled overhead too high: {elapsed*1000:.2f}ms"
        )

        # Verify no metrics collected.
        self.assertEqual(len(collector.get_metrics()), 0)
        self.assertEqual(len(collector.get_eager_fallbacks()), 0)


if __name__ == "__main__":
    unittest.main()
