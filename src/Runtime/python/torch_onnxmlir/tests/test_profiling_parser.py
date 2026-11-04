# SPDX-License-Identifier: Apache-2.0

##################### test_profiling_parser.py #################################
#
# Copyright 2026 The IBM Research Authors.
#
################################################################################
#
# Unit tests for the profiling output parser (profiling.py).
# These tests use embedded sample data and do not require a real compiler.
#
################################################################################

import unittest
import tempfile
import os

from torch_onnxmlir.profiling import (
    OpTiming,
    ProfilingResult,
    parse_profiling_output,
    format_profiling_table,
)

SAMPLE_SINGLE_RUN = """\
==START-REPORT==
==PERF-REPORT==, onnx.Conv, /conv1, before, 0.000000, 1692654182.738500
==PERF-REPORT==, onnx.Conv, /conv1, after, 0.000045, 1692654182.738591
==PERF-REPORT==, onnx.Relu, /relu1, before, 0.000000, 1692654182.738592
==PERF-REPORT==, onnx.Relu, /relu1, after, 0.000012, 1692654182.738604
==PERF-REPORT==, onnx.Conv, /conv2, before, 0.000000, 1692654182.738605
==PERF-REPORT==, onnx.Conv, /conv2, after, 0.000038, 1692654182.738643
==PERF-REPORT==, onnx.Gemm, /fc1, before, 0.000000, 1692654182.738644
==PERF-REPORT==, onnx.Gemm, /fc1, after, 0.000020, 1692654182.738664
"""

SAMPLE_MULTI_RUN = """\
==START-REPORT==
==PERF-REPORT==, onnx.Conv, /conv1, before, 0.000000, 100.000000
==PERF-REPORT==, onnx.Conv, /conv1, after, 0.000100, 100.000100
==PERF-REPORT==, onnx.Relu, /relu1, before, 0.000000, 100.000101
==PERF-REPORT==, onnx.Relu, /relu1, after, 0.000010, 100.000111
==START-REPORT==
==PERF-REPORT==, onnx.Conv, /conv1, before, 0.000000, 200.000000
==PERF-REPORT==, onnx.Conv, /conv1, after, 0.000050, 200.000050
==PERF-REPORT==, onnx.Relu, /relu1, before, 0.000000, 200.000051
==PERF-REPORT==, onnx.Relu, /relu1, after, 0.000010, 200.000061
==START-REPORT==
==PERF-REPORT==, onnx.Conv, /conv1, before, 0.000000, 300.000000
==PERF-REPORT==, onnx.Conv, /conv1, after, 0.000060, 300.000060
==PERF-REPORT==, onnx.Relu, /relu1, before, 0.000000, 300.000061
==PERF-REPORT==, onnx.Relu, /relu1, after, 0.000010, 300.000071
==START-REPORT==
==PERF-REPORT==, onnx.Conv, /conv1, before, 0.000000, 400.000000
==PERF-REPORT==, onnx.Conv, /conv1, after, 0.000040, 400.000040
==PERF-REPORT==, onnx.Relu, /relu1, before, 0.000000, 400.000041
==PERF-REPORT==, onnx.Relu, /relu1, after, 0.000010, 400.000051
"""

SAMPLE_ZHIGH = """\
==START-REPORT==
==PERF-REPORT==, onnx.Reshape, /reshape1, before, 0.000000, 100.000000
==PERF-REPORT==, onnx.Reshape, /reshape1, after, 0.000005, 100.000005
==PERF-REPORT==, zhigh.Stick, /stick1, before, 0.000000, 100.000006
==PERF-REPORT==, zhigh.Stick, /stick1, after, 0.000020, 100.000026
==PERF-REPORT==, zhigh.MatMul, /matmul1, before, 0.000000, 100.000027
==PERF-REPORT==, zhigh.MatMul, /matmul1, after, 0.000050, 100.000077
==PERF-REPORT==, zhigh.Unstick, /unstick1, before, 0.000000, 100.000078
==PERF-REPORT==, zhigh.Unstick, /unstick1, after, 0.000015, 100.000093
"""


class TestParseSingleRun(unittest.TestCase):

    def test_basic_parsing(self):
        result = parse_profiling_output(text=SAMPLE_SINGLE_RUN)
        self.assertIsNotNone(result)
        self.assertIsInstance(result, ProfilingResult)
        self.assertEqual(result.num_measurements, 1)
        self.assertEqual(result.warmup_skipped, 0)

    def test_op_count(self):
        result = parse_profiling_output(text=SAMPLE_SINGLE_RUN)
        op_names = [op.op_name for op in result.op_timings]
        self.assertIn("onnx.Conv", op_names)
        self.assertIn("onnx.Relu", op_names)
        self.assertIn("onnx.Gemm", op_names)

    def test_conv_has_two_instances(self):
        result = parse_profiling_output(text=SAMPLE_SINGLE_RUN)
        conv = next(op for op in result.op_timings if op.op_name == "onnx.Conv")
        self.assertEqual(conv.count, 2)

    def test_total_time(self):
        result = parse_profiling_output(text=SAMPLE_SINGLE_RUN)
        # Total time = 0.000045 + 0.000012 + 0.000038 + 0.000020 = 0.000115
        self.assertAlmostEqual(result.total_time, 0.000115, places=6)

    def test_percentages_sum_to_100(self):
        result = parse_profiling_output(text=SAMPLE_SINGLE_RUN)
        total_pct = sum(op.percentage for op in result.op_timings)
        self.assertAlmostEqual(total_pct, 100.0, places=1)

    def test_sorted_by_total_time_desc(self):
        result = parse_profiling_output(text=SAMPLE_SINGLE_RUN)
        times = [op.total_time for op in result.op_timings]
        self.assertEqual(times, sorted(times, reverse=True))

    def test_before_entries_filtered(self):
        result = parse_profiling_output(text=SAMPLE_SINGLE_RUN)
        # "before" entries have time 0.0, so total must be > 0
        self.assertGreater(result.total_time, 0)


class TestParseMultiRun(unittest.TestCase):

    def test_quartile_reporting(self):
        result = parse_profiling_output(text=SAMPLE_MULTI_RUN, reporting="quartile")
        self.assertIsNotNone(result)
        # 4 runs, quartile discards 1 from each end -> 2 used
        self.assertEqual(result.num_measurements, 2)
        self.assertEqual(result.reporting_mode, "quartile")

    def test_all_reporting(self):
        result = parse_profiling_output(text=SAMPLE_MULTI_RUN, reporting="all")
        self.assertIsNotNone(result)
        self.assertEqual(result.num_measurements, 4)
        self.assertEqual(result.reporting_mode, "all")

    def test_sum_reporting(self):
        result = parse_profiling_output(text=SAMPLE_MULTI_RUN, reporting="sum")
        self.assertIsNotNone(result)
        self.assertEqual(result.reporting_mode, "sum")

    def test_warmup_skip(self):
        result = parse_profiling_output(
            text=SAMPLE_MULTI_RUN, warmup_num=1, reporting="all"
        )
        self.assertIsNotNone(result)
        # 4 runs - 1 warmup = 3 used
        self.assertEqual(result.num_measurements, 3)
        self.assertEqual(result.warmup_skipped, 1)


class TestParseEdgeCases(unittest.TestCase):

    def test_empty_text(self):
        result = parse_profiling_output(text="")
        self.assertIsNone(result)

    def test_none_inputs(self):
        result = parse_profiling_output()
        self.assertIsNone(result)

    def test_no_start_report(self):
        text = "==PERF-REPORT==, onnx.Conv, /conv1, after, 0.00005, 100.0\n"
        result = parse_profiling_output(text=text)
        self.assertIsNone(result)

    def test_only_before_entries(self):
        text = """\
==START-REPORT==
==PERF-REPORT==, onnx.Conv, /conv1, before, 0.000000, 100.000000
"""
        result = parse_profiling_output(text=text)
        self.assertIsNone(result)


class TestParseFromFile(unittest.TestCase):

    def test_file_parsing(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".prof", delete=False) as f:
            f.write(SAMPLE_SINGLE_RUN)
            f.flush()
            path = f.name
        try:
            result = parse_profiling_output(file_path=path)
            self.assertIsNotNone(result)
            self.assertEqual(len(result.op_timings), 3)
        finally:
            os.unlink(path)


class TestZHighBreakdown(unittest.TestCase):

    def test_cpu_accel_breakdown(self):
        result = parse_profiling_output(text=SAMPLE_ZHIGH)
        self.assertIsNotNone(result)
        self.assertIsNotNone(result.cpu_time)
        self.assertIsNotNone(result.accel_time)
        # CPU = onnx.Reshape(0.000005) + zhigh.Stick(0.000020) + zhigh.Unstick(0.000015)
        self.assertAlmostEqual(result.cpu_time, 0.000040, places=6)
        # Accel = zhigh.MatMul(0.000050)
        self.assertAlmostEqual(result.accel_time, 0.000050, places=6)

    def test_no_breakdown_without_zhigh(self):
        result = parse_profiling_output(text=SAMPLE_SINGLE_RUN)
        self.assertIsNone(result.cpu_time)
        self.assertIsNone(result.accel_time)


class TestFormatTable(unittest.TestCase):

    def test_format_produces_string(self):
        result = parse_profiling_output(text=SAMPLE_SINGLE_RUN)
        table = format_profiling_table(result)
        self.assertIsInstance(table, str)
        self.assertIn("onnx.Conv", table)
        self.assertIn("onnx.Relu", table)
        self.assertIn("Total model time", table)

    def test_format_time_units(self):
        result = parse_profiling_output(text=SAMPLE_SINGLE_RUN)
        for unit in ["s", "ms", "us"]:
            table = format_profiling_table(result, time_unit=unit)
            self.assertIn(unit, table)

    def test_format_zhigh_breakdown(self):
        result = parse_profiling_output(text=SAMPLE_ZHIGH)
        table = format_profiling_table(result)
        self.assertIn("CPU time", table)
        self.assertIn("Accelerator time", table)


class TestDataclasses(unittest.TestCase):

    def test_op_timing_fields(self):
        op = OpTiming(
            op_name="onnx.Conv",
            count=2,
            avg_time=0.001,
            total_time=0.002,
            percentage=50.0,
        )
        self.assertEqual(op.op_name, "onnx.Conv")
        self.assertEqual(op.count, 2)

    def test_profiling_result_fields(self):
        result = ProfilingResult(
            op_timings=[],
            total_time=0.1,
            num_measurements=5,
            warmup_skipped=1,
            reporting_mode="quartile",
        )
        self.assertEqual(result.total_time, 0.1)
        self.assertIsNone(result.cpu_time)
        self.assertIsNone(result.accel_time)


if __name__ == "__main__":
    unittest.main()
