# SPDX-License-Identifier: Apache-2.0

##################### test_explain_simple_model.py #############################
#
# Copyright 2026 The IBM Research Authors.
#
################################################################################
#
# Integration test for torch_onnxmlir.explain() with real model compilation.
#
################################################################################

import unittest
import logging
import torch
import torch.nn as nn
import torch_onnxmlir
from utils import TorchOMTestCase, COMPILER_IMAGE_NAME, COMPILER_PATH

logger = logging.basicConfig(level=logging.INFO)


class SimpleModel(nn.Module):
    """Simple model for testing explain feature."""

    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(10, 20)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(20, 5)

    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        return x


class TestExplainSimpleModel(TorchOMTestCase):

    def test_explain_with_real_model(self):
        """Test explain feature with real model compilation and inference."""
        # Create and compile model.
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

        # Use explain_context to automatically enable metrics collection.
        with torch_onnxmlir.explain_context(format="dict") as ctx:
            # Run inference multiple times.
            input_tensor = torch.randn(1, 10)

            with torch.no_grad():
                # First inference (triggers compilation or uses cached model).
                output1 = compiled_model(input_tensor)
                self.assertEqual(output1.shape, (1, 5))

                # Additional inferences (should use cache).
                for _ in range(5):
                    output = compiled_model(input_tensor)
                    self.assertEqual(output.shape, (1, 5))

        # After context exits, metrics are automatically captured.
        result = ctx.data

        # Check that metrics were collected.
        self.assertIsInstance(result, dict)
        self.assertIn("summary", result)
        self.assertIn("graphs", result)

        # Verify summary metrics.
        summary = result["summary"]
        self.assertEqual(summary["unique_graphs"], 1)
        self.assertEqual(summary["total_inference_calls"], 6)
        # Note: All inferences may be cache hits if model was already compiled.
        self.assertGreaterEqual(summary["total_cache_hits"], 5)

        # Verify graph metrics.
        self.assertEqual(len(result["graphs"]), 1)
        graph_key = list(result["graphs"].keys())[0]
        graph_metrics = result["graphs"][graph_key]

        self.assertEqual(graph_metrics["inference_count"], 6)
        self.assertGreaterEqual(graph_metrics["cache_hits"], 5)
        # Compilation time may be 0 if model was already cached from previous run.
        self.assertGreaterEqual(graph_metrics["compilation_time"], 0)
        self.assertGreater(graph_metrics["total_inference_time"], 0)
        self.assertGreater(graph_metrics["avg_inference_time"], 0)
        self.assertGreater(graph_metrics["cache_hit_rate"], 0.8)

        # Test table format output using a new context.
        with torch_onnxmlir.explain_context(format="table") as ctx:
            with torch.no_grad():
                compiled_model(input_tensor)

        table_output = str(ctx)
        self.assertIn("TORCH_ONNXMLIR PERFORMANCE METRICS", table_output)
        self.assertIn("Unique Graphs:", table_output)

        # Test JSON format output using a new context.
        import json

        with torch_onnxmlir.explain_context(format="json") as ctx:
            with torch.no_grad():
                compiled_model(input_tensor)

        json_output = ctx.data
        json_data = json.loads(json_output)
        self.assertEqual(json_data["summary"]["unique_graphs"], 1)

    def test_explain_sorting_with_real_model(self):
        """Test explain sorting options with real model."""
        # Create and compile model.
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

        # Run inference.
        input_tensor = torch.randn(1, 10)

        # Test different sorting options.
        with torch_onnxmlir.explain_context(format="dict", sort_by="time") as ctx:
            with torch.no_grad():
                for _ in range(3):
                    compiled_model(input_tensor)

        result_time = ctx.data
        self.assertIn("graphs", result_time)

        with torch_onnxmlir.explain_context(format="dict", sort_by="calls") as ctx:
            with torch.no_grad():
                for _ in range(3):
                    compiled_model(input_tensor)

        result_calls = ctx.data
        self.assertIn("graphs", result_calls)

        with torch_onnxmlir.explain_context(format="dict", sort_by="order") as ctx:
            with torch.no_grad():
                for _ in range(3):
                    compiled_model(input_tensor)

        result_order = ctx.data
        self.assertIn("graphs", result_order)


if __name__ == "__main__":
    unittest.main()
