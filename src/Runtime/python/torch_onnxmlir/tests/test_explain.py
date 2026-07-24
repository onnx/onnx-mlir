# SPDX-License-Identifier: Apache-2.0

##################### test_explain.py ##########################################
#
# Copyright 2026 The IBM Research Authors.
#
################################################################################
#
# Unit tests for torch_onnxmlir.explain() feature.
#
################################################################################

import unittest
import torch_onnxmlir
from torch_onnxmlir import metrics
from utils import TorchOMTestCase


class TestExplain(TorchOMTestCase):

    def test_explain_disabled(self):
        """Test that explain returns message when disabled."""
        torch_onnxmlir.config.enable_explain = False
        result = torch_onnxmlir.explain()
        self.assertIn("not enabled", result.lower())

    def test_explain_no_metrics(self):
        """Test that explain returns message when no metrics collected."""
        torch_onnxmlir.config.enable_explain = True
        torch_onnxmlir.clear_metrics()
        result = torch_onnxmlir.explain()
        self.assertIn("no metrics", result.lower())

    def test_metrics_collection(self):
        """Test basic metrics collection."""
        torch_onnxmlir.config.enable_explain = True
        torch_onnxmlir.clear_metrics()

        collector = metrics.global_metrics_collector

        # Simulate compilation.
        collector.record_compilation("test_model_1", 1.5)

        # Simulate inference.
        collector.record_inference("test_model_1", 0.01, False)
        collector.record_inference("test_model_1", 0.005, True)
        collector.record_inference("test_model_1", 0.005, True)

        # Check metrics.
        all_metrics = collector.get_metrics()
        self.assertEqual(len(all_metrics), 1)
        self.assertIn("test_model_1", all_metrics)

        m = all_metrics["test_model_1"]
        self.assertEqual(m.compilation_time, 1.5)
        self.assertEqual(m.inference_count, 3)
        self.assertEqual(m.cache_hits, 2)
        self.assertAlmostEqual(m.total_inference_time, 0.02, places=3)
        self.assertAlmostEqual(m.avg_inference_time, 0.02 / 3, places=3)
        self.assertAlmostEqual(m.cache_hit_rate, 2 / 3, places=3)

    def test_explain_table_format(self):
        """Test explain with table format."""
        torch_onnxmlir.config.enable_explain = True
        torch_onnxmlir.clear_metrics()

        collector = metrics.global_metrics_collector

        # Add some metrics.
        collector.record_compilation("model_a", 2.0)
        collector.record_inference("model_a", 0.01, False)
        collector.record_inference("model_a", 0.005, True)

        collector.record_compilation("model_b", 1.0)
        collector.record_inference("model_b", 0.02, False)

        # Get table output.
        result = torch_onnxmlir.explain(format="table")

        self.assertIn("TORCH_ONNXMLIR PERFORMANCE METRICS", result)
        self.assertIn("Unique Graphs: 2", result)
        self.assertIn("Total Inference Calls: 3", result)
        self.assertIn("model_a", result)
        self.assertIn("model_b", result)

    def test_explain_dict_format(self):
        """Test explain with dict format."""
        torch_onnxmlir.config.enable_explain = True
        torch_onnxmlir.clear_metrics()

        collector = metrics.global_metrics_collector

        # Add metrics.
        collector.record_compilation("model_x", 1.5)
        collector.record_inference("model_x", 0.01, False)
        collector.record_inference("model_x", 0.005, True)

        # Get dict output.
        result = torch_onnxmlir.explain(format="dict")

        self.assertIsInstance(result, dict)
        self.assertIn("summary", result)
        self.assertIn("graphs", result)
        self.assertEqual(result["summary"]["unique_graphs"], 1)
        self.assertEqual(result["summary"]["total_inference_calls"], 2)
        self.assertIn("model_x", result["graphs"])
        self.assertEqual(result["graphs"]["model_x"]["inference_count"], 2)
        self.assertEqual(result["graphs"]["model_x"]["cache_hits"], 1)

    def test_explain_json_format(self):
        """Test explain with JSON format."""
        import json

        torch_onnxmlir.config.enable_explain = True
        torch_onnxmlir.clear_metrics()

        collector = metrics.global_metrics_collector

        # Add metrics.
        collector.record_compilation("model_y", 1.0)
        collector.record_inference("model_y", 0.01, False)

        # Get JSON output.
        result = torch_onnxmlir.explain(format="json")

        self.assertIsInstance(result, str)
        data = json.loads(result)
        self.assertIn("summary", data)
        self.assertIn("graphs", data)
        self.assertEqual(data["summary"]["unique_graphs"], 1)

    def test_sorting_options(self):
        """Test different sorting options."""
        torch_onnxmlir.config.enable_explain = True
        torch_onnxmlir.clear_metrics()

        collector = metrics.global_metrics_collector

        # Add metrics with different characteristics.
        collector.record_compilation("model_1", 3.0)  # Highest time.
        collector.record_inference("model_1", 0.01, False)

        collector.record_compilation("model_2", 1.0)  # Most calls.
        for i in range(10):
            collector.record_inference("model_2", 0.001, True)

        collector.record_compilation("model_3", 2.0)  # Middle.
        collector.record_inference("model_3", 0.005, False)

        # Test sort by time.
        result_time = torch_onnxmlir.explain(format="dict", sort_by="time")
        graphs_time = list(result_time["graphs"].keys())
        self.assertEqual(graphs_time[0], "model_1")  # Highest total time.

        # Test sort by calls.
        result_calls = torch_onnxmlir.explain(format="dict", sort_by="calls")
        graphs_calls = list(result_calls["graphs"].keys())
        self.assertEqual(graphs_calls[0], "model_2")  # Most calls.

        # Test sort by order.
        result_order = torch_onnxmlir.explain(format="dict", sort_by="order")
        graphs_order = list(result_order["graphs"].keys())
        self.assertEqual(graphs_order, ["model_1", "model_2", "model_3"])

    def test_eager_fallbacks(self):
        """Test eager fallback tracking."""
        torch_onnxmlir.config.enable_explain = True
        torch_onnxmlir.clear_metrics()

        collector = metrics.global_metrics_collector

        # Record fallbacks.
        collector.record_eager_fallback("export_failed", "model_fail_1")
        collector.record_eager_fallback("export_failed", "model_fail_2")
        collector.record_eager_fallback("compilation_failed", "model_fail_3")

        # Check fallbacks.
        fallbacks = collector.get_eager_fallbacks()
        self.assertEqual(len(fallbacks), 3)
        self.assertEqual(collector.total_eager_fallbacks, 3)

        # Get explain output.
        result = torch_onnxmlir.explain(format="dict")
        self.assertEqual(result["summary"]["total_eager_fallbacks"], 3)
        self.assertEqual(len(result["eager_fallbacks"]), 3)

    def test_clear_metrics(self):
        """Test clearing metrics."""
        torch_onnxmlir.config.enable_explain = True

        collector = metrics.global_metrics_collector

        # Add some metrics.
        collector.record_compilation("model_clear", 1.0)
        collector.record_inference("model_clear", 0.01, False)
        collector.record_eager_fallback("test_reason")

        # Verify metrics exist.
        self.assertGreater(len(collector.get_metrics()), 0)
        self.assertGreater(collector.total_eager_fallbacks, 0)

        # Clear metrics.
        torch_onnxmlir.clear_metrics()

        # Verify cleared.
        self.assertEqual(len(collector.get_metrics()), 0)
        self.assertEqual(len(collector.get_eager_fallbacks()), 0)
        self.assertEqual(collector.total_eager_fallbacks, 0)


if __name__ == "__main__":
    unittest.main()
