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
        # Enable explain feature.
        torch_onnxmlir.config.enable_explain = True
        torch_onnxmlir.clear_metrics()
        
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
        
        # Verify explain output.
        result = torch_onnxmlir.explain(format="dict")
        
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
        
        # Test table format output.
        table_output = torch_onnxmlir.explain(format="table")
        self.assertIn("TORCH_ONNXMLIR PERFORMANCE METRICS", table_output)
        self.assertIn("Unique Graphs: 1", table_output)
        self.assertIn("Total Inference Calls: 6", table_output)
        
        # Test JSON format output.
        import json
        json_output = torch_onnxmlir.explain(format="json")
        json_data = json.loads(json_output)
        self.assertEqual(json_data["summary"]["unique_graphs"], 1)
        
        # Clear metrics and verify.
        torch_onnxmlir.clear_metrics()
        result_after_clear = torch_onnxmlir.explain()
        self.assertIn("no metrics", result_after_clear.lower())
    
    def test_explain_sorting_with_real_model(self):
        """Test explain sorting options with real model."""
        torch_onnxmlir.config.enable_explain = True
        torch_onnxmlir.clear_metrics()
        
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
        with torch.no_grad():
            for _ in range(3):
                compiled_model(input_tensor)
        
        # Test different sorting options.
        result_time = torch_onnxmlir.explain(format="dict", sort_by="time")
        self.assertIn("graphs", result_time)
        
        result_calls = torch_onnxmlir.explain(format="dict", sort_by="calls")
        self.assertIn("graphs", result_calls)
        
        result_order = torch_onnxmlir.explain(format="dict", sort_by="order")
        self.assertIn("graphs", result_order)


if __name__ == "__main__":
    unittest.main()
