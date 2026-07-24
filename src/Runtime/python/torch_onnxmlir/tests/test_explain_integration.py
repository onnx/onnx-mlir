# SPDX-License-Identifier: Apache-2.0

##################### test_explain_integration.py ##############################
#
# Copyright 2026 The IBM Research Authors.
#
################################################################################
#
# Integration tests for torch_onnxmlir.explain() feature with backend.
#
################################################################################

import unittest
import time
import torch
import torch.nn as nn
import torch_onnxmlir
from torch_onnxmlir import config, metrics
from utils import TorchOMTestCase


class SimpleModel(nn.Module):
    """Simple model for testing."""
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(10, 5)
    
    def forward(self, x):
        return self.linear(x)


class TestExplainIntegration(TorchOMTestCase):
    
    def test_explain_with_mock_backend(self):
        """Test explain feature with simulated backend behavior."""
        # Enable explain.
        config.enable_explain = True
        torch_onnxmlir.clear_metrics()
        
        collector = metrics.global_metrics_collector
        
        # Simulate what happens in the backend during compilation and inference.
        cache_key = "test_model_hash_123"
        
        # Simulate compilation (first run).
        compilation_start = time.perf_counter()
        time.sleep(0.001)  # Simulate compilation time.
        compilation_time = time.perf_counter() - compilation_start
        collector.record_compilation(cache_key, compilation_time)
        
        # Simulate first inference (cache miss).
        inference_start = time.perf_counter()
        time.sleep(0.0001)  # Simulate inference time.
        inference_time = time.perf_counter() - inference_start
        collector.record_inference(cache_key, inference_time, is_cache_hit=False)
        
        # Simulate subsequent inferences (cache hits).
        for _ in range(5):
            inference_start = time.perf_counter()
            time.sleep(0.0001)
            inference_time = time.perf_counter() - inference_start
            collector.record_inference(cache_key, inference_time, is_cache_hit=True)
        
        # Get metrics.
        result = torch_onnxmlir.explain(format="dict")
        
        # Verify.
        self.assertEqual(result["summary"]["unique_graphs"], 1)
        self.assertEqual(result["summary"]["total_inference_calls"], 6)
        self.assertEqual(result["summary"]["total_cache_hits"], 5)
        self.assertIn(cache_key, result["graphs"])
        
        graph_metrics = result["graphs"][cache_key]
        self.assertEqual(graph_metrics["inference_count"], 6)
        self.assertEqual(graph_metrics["cache_hits"], 5)
        self.assertGreater(graph_metrics["compilation_time"], 0)
        self.assertGreater(graph_metrics["cache_hit_rate"], 0.8)  # 5/6.
        
        # Test table output.
        table_output = torch_onnxmlir.explain(format="table")
        self.assertIn("TORCH_ONNXMLIR PERFORMANCE METRICS", table_output)
        self.assertIn("Unique Graphs: 1", table_output)
        self.assertIn("Total Inference Calls: 6", table_output)
        
        # Test JSON output.
        import json
        json_output = torch_onnxmlir.explain(format="json")
        json_data = json.loads(json_output)
        self.assertEqual(json_data["summary"]["unique_graphs"], 1)
    
    def test_explain_with_eager_fallbacks(self):
        """Test explain feature with eager fallbacks."""
        config.enable_explain = True
        torch_onnxmlir.clear_metrics()
        
        collector = metrics.global_metrics_collector
        
        # Simulate various fallback scenarios.
        collector.record_eager_fallback("export_failed", "model_fail_1")
        collector.record_eager_fallback("compilation_failed", "model_fail_2")
        collector.record_eager_fallback("no_inputs", "model_fail_3")
        collector.record_eager_fallback("export_failed", "model_fail_4")  # Same reason.
        
        # Get metrics.
        result = torch_onnxmlir.explain(format="dict")
        
        # Verify.
        self.assertEqual(result["summary"]["total_eager_fallbacks"], 4)
        self.assertEqual(len(result["eager_fallbacks"]), 4)
        
        # Check table output includes fallbacks.
        table_output = torch_onnxmlir.explain(format="table")
        self.assertIn("Eager Mode Fallbacks:", table_output)
        self.assertIn("export_failed", table_output)
        self.assertIn("compilation_failed", table_output)
    
    def test_explain_disabled_overhead(self):
        """Test that disabled explain has minimal overhead."""
        # Disable explain.
        config.enable_explain = False
        torch_onnxmlir.clear_metrics()
        
        collector = metrics.global_metrics_collector
        
        # These should be no-ops (just boolean checks).
        start = time.perf_counter()
        for _ in range(10000):
            collector.record_compilation("test", 1.0)
            collector.record_inference("test", 0.01, True)
            collector.record_eager_fallback("test")
        elapsed = time.perf_counter() - start
        
        # Should be very fast (< 10ms for 10k calls, ~1μs per call).
        self.assertLess(elapsed, 0.010, f"Disabled overhead too high: {elapsed*1000:.2f}ms")
        
        # Verify no metrics collected.
        self.assertEqual(len(collector.get_metrics()), 0)
        self.assertEqual(len(collector.get_eager_fallbacks()), 0)
    
    def test_multiple_graphs(self):
        """Test explain with multiple different graphs."""
        config.enable_explain = True
        torch_onnxmlir.clear_metrics()
        
        collector = metrics.global_metrics_collector
        
        # Simulate multiple different models.
        models = ["model_a", "model_b", "model_c"]
        
        for i, model_key in enumerate(models):
            # Different compilation times.
            collector.record_compilation(model_key, 1.0 + i * 0.5)
            
            # Different number of inferences.
            for j in range((i + 1) * 3):
                collector.record_inference(model_key, 0.01, j > 0)
        
        # Test sorting by time.
        result_time = torch_onnxmlir.explain(format="dict", sort_by="time")
        graphs_time = list(result_time["graphs"].keys())
        # model_c should have highest total time.
        self.assertEqual(graphs_time[0], "model_c")
        
        # Test sorting by calls.
        result_calls = torch_onnxmlir.explain(format="dict", sort_by="calls")
        graphs_calls = list(result_calls["graphs"].keys())
        # model_c should have most calls (9).
        self.assertEqual(graphs_calls[0], "model_c")
        
        # Test sorting by order.
        result_order = torch_onnxmlir.explain(format="dict", sort_by="order")
        graphs_order = list(result_order["graphs"].keys())
        self.assertEqual(graphs_order, ["model_a", "model_b", "model_c"])


if __name__ == "__main__":
    unittest.main()
