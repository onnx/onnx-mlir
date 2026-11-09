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
    NodeTiming,
    ProfilingResult,
    parse_profiling_output,
    format_profiling_table,
    _parse_sig_lines,
    _get_timing_key,
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


SAMPLE_WITH_SIG = """\
==SIG-REPORT==, onnx.Conv, /conv1, sig, 1x3x32x32xf32, 16x3x3x3xf32
==SIG-REPORT==, onnx.Relu, /relu1, sig, 1x16x30x30xf32
==SIG-REPORT==, onnx.Conv, /conv2, sig, 1x16x30x30xf32, 32x16x3x3xf32
==START-REPORT==
==PERF-REPORT==, onnx.Conv, /conv1, before, 0.000000, 100.000000
==PERF-REPORT==, onnx.Conv, /conv1, after, 0.000045, 100.000045
==PERF-REPORT==, onnx.Relu, /relu1, before, 0.000000, 100.000046
==PERF-REPORT==, onnx.Relu, /relu1, after, 0.000012, 100.000058
==PERF-REPORT==, onnx.Conv, /conv2, before, 0.000000, 100.000059
==PERF-REPORT==, onnx.Conv, /conv2, after, 0.000038, 100.000097
"""

SAMPLE_NO_SIG = """\
==START-REPORT==
==PERF-REPORT==, onnx.Conv, /conv1, before, 0.000000, 100.000000
==PERF-REPORT==, onnx.Conv, /conv1, after, 0.000045, 100.000045
==PERF-REPORT==, onnx.Relu, /relu1, before, 0.000000, 100.000046
==PERF-REPORT==, onnx.Relu, /relu1, after, 0.000012, 100.000058
"""


class TestNodeTimings(unittest.TestCase):

    def test_node_timings_populated(self):
        result = parse_profiling_output(text=SAMPLE_SINGLE_RUN)
        self.assertIsNotNone(result)
        self.assertGreater(len(result.node_timings), 0)

    def test_node_timings_count_matches_perf_entries(self):
        result = parse_profiling_output(text=SAMPLE_SINGLE_RUN)
        # 4 distinct node entries: /conv1, /relu1, /conv2, /fc1
        self.assertEqual(len(result.node_timings), 4)

    def test_node_timing_fields(self):
        result = parse_profiling_output(text=SAMPLE_SINGLE_RUN)
        for nt in result.node_timings:
            self.assertIsInstance(nt, NodeTiming)
            self.assertTrue(len(nt.node_name) > 0)
            self.assertTrue(len(nt.op_name) > 0)
            self.assertGreater(nt.time, 0)
            self.assertGreaterEqual(nt.percentage, 0)

    def test_node_percentages_sum_to_100(self):
        result = parse_profiling_output(text=SAMPLE_SINGLE_RUN)
        total_pct = sum(nt.percentage for nt in result.node_timings)
        self.assertAlmostEqual(total_pct, 100.0, places=1)

    def test_node_timings_sorted_by_time_desc(self):
        result = parse_profiling_output(text=SAMPLE_SINGLE_RUN)
        times = [nt.time for nt in result.node_timings]
        self.assertEqual(times, sorted(times, reverse=True))

    def test_no_signature_without_sig_report(self):
        result = parse_profiling_output(text=SAMPLE_NO_SIG)
        for nt in result.node_timings:
            self.assertIsNone(nt.signature)

    def test_node_timing_dataclass(self):
        nt = NodeTiming(
            node_name="/conv1",
            op_name="onnx.Conv",
            time=0.001,
            percentage=50.0,
            signature="1x3x32x32xf32",
        )
        self.assertEqual(nt.node_name, "/conv1")
        self.assertEqual(nt.signature, "1x3x32x32xf32")

    def test_node_timing_default_signature(self):
        nt = NodeTiming(
            node_name="/conv1", op_name="onnx.Conv", time=0.001, percentage=50.0
        )
        self.assertIsNone(nt.signature)

    def test_profiling_result_default_node_timings(self):
        result = ProfilingResult(
            op_timings=[],
            total_time=0.1,
            num_measurements=1,
            warmup_skipped=0,
            reporting_mode="all",
        )
        self.assertEqual(result.node_timings, [])


class TestSigReportParsing(unittest.TestCase):

    def test_sig_lines_parsed(self):
        result = parse_profiling_output(text=SAMPLE_WITH_SIG)
        self.assertIsNotNone(result)
        sigs = {nt.node_name: nt.signature for nt in result.node_timings}
        self.assertIn("/conv1", sigs)
        self.assertIsNotNone(sigs["/conv1"])

    def test_sig_prefix_stripped(self):
        result = parse_profiling_output(text=SAMPLE_WITH_SIG)
        for nt in result.node_timings:
            if nt.signature is not None:
                self.assertFalse(
                    nt.signature.startswith("sig"),
                    f"Signature should not start with 'sig': {nt.signature}",
                )

    def test_sig_content_correct(self):
        result = parse_profiling_output(text=SAMPLE_WITH_SIG)
        conv1 = next(nt for nt in result.node_timings if nt.node_name == "/conv1")
        self.assertEqual(conv1.signature, "1x3x32x32xf32, 16x3x3x3xf32")

    def test_sig_correlated_to_correct_nodes(self):
        result = parse_profiling_output(text=SAMPLE_WITH_SIG)
        relu = next(nt for nt in result.node_timings if nt.node_name == "/relu1")
        self.assertEqual(relu.signature, "1x16x30x30xf32")

    def test_parse_sig_lines_directly(self):
        lines = [
            "==SIG-REPORT==, onnx.Conv, /conv1, sig, 1x3x32x32xf32\n",
            "==SIG-REPORT==, onnx.Relu, /relu1, sig, 1x16x30x30xf32\n",
        ]
        sig_dict = _parse_sig_lines(lines)
        key = _get_timing_key("onnx.Conv", "/conv1")
        self.assertIn(key, sig_dict)
        self.assertEqual(sig_dict[key], ["1x3x32x32xf32"])

    def test_parse_sig_lines_no_match(self):
        lines = ["Some random line\n", "Another line\n"]
        sig_dict = _parse_sig_lines(lines)
        self.assertEqual(len(sig_dict), 0)

    def test_mixed_perf_and_sig(self):
        result = parse_profiling_output(text=SAMPLE_WITH_SIG)
        self.assertEqual(len(result.op_timings), 2)  # onnx.Conv, onnx.Relu
        self.assertEqual(len(result.node_timings), 3)  # /conv1, /relu1, /conv2
        has_sig = sum(1 for nt in result.node_timings if nt.signature is not None)
        self.assertEqual(has_sig, 3)


SAMPLE_SAME_SIG = """\
==SIG-REPORT==, onnx.Conv, /conv1, sig, 1x3x32x32xf32, 16x3x3x3xf32
==SIG-REPORT==, onnx.Relu, /relu1, sig, 1x16x30x30xf32
==SIG-REPORT==, onnx.Conv, /conv2, sig, 1x3x32x32xf32, 16x3x3x3xf32
==SIG-REPORT==, onnx.Conv, /conv3, sig, 1x16x30x30xf32, 32x16x3x3xf32
==START-REPORT==
==PERF-REPORT==, onnx.Conv, /conv1, before, 0.000000, 100.000000
==PERF-REPORT==, onnx.Conv, /conv1, after, 0.000045, 100.000045
==PERF-REPORT==, onnx.Relu, /relu1, before, 0.000000, 100.000046
==PERF-REPORT==, onnx.Relu, /relu1, after, 0.000012, 100.000058
==PERF-REPORT==, onnx.Conv, /conv2, before, 0.000000, 100.000059
==PERF-REPORT==, onnx.Conv, /conv2, after, 0.000040, 100.000099
==PERF-REPORT==, onnx.Conv, /conv3, before, 0.000000, 100.000100
==PERF-REPORT==, onnx.Conv, /conv3, after, 0.000030, 100.000130
"""


class TestSignatureGrouping(unittest.TestCase):

    def _get_conv_result(self):
        return parse_profiling_output(text=SAMPLE_SAME_SIG)

    def test_same_sig_nodes_merged(self):
        result = self._get_conv_result()
        conv = next(op for op in result.op_timings if op.op_name == "onnx.Conv")
        self.assertEqual(conv.count, 3)
        node_timings_conv = [
            nt for nt in result.node_timings if nt.op_name == "onnx.Conv"
        ]
        sigs = set(nt.signature for nt in node_timings_conv)
        self.assertEqual(len(sigs), 2)

    def test_same_sig_count(self):
        result = self._get_conv_result()
        node_timings_conv = [
            nt for nt in result.node_timings if nt.op_name == "onnx.Conv"
        ]
        sig_groups: dict = {}
        for nt in node_timings_conv:
            sig_groups.setdefault(nt.signature, []).append(nt)
        same_sig = sig_groups["1x3x32x32xf32, 16x3x3x3xf32"]
        self.assertEqual(len(same_sig), 2)
        self.assertIn("/conv1", [nt.node_name for nt in same_sig])
        self.assertIn("/conv2", [nt.node_name for nt in same_sig])

    def test_same_sig_total_time(self):
        result = self._get_conv_result()
        node_timings_conv = [
            nt for nt in result.node_timings if nt.op_name == "onnx.Conv"
        ]
        sig_groups: dict = {}
        for nt in node_timings_conv:
            sig_groups.setdefault(nt.signature, []).append(nt)
        same_sig = sig_groups["1x3x32x32xf32, 16x3x3x3xf32"]
        total = sum(nt.time for nt in same_sig)
        self.assertAlmostEqual(total, 0.000085, places=6)

    def test_null_sig_group(self):
        result = parse_profiling_output(text=SAMPLE_NO_SIG)
        node_timings_conv = [
            nt for nt in result.node_timings if nt.op_name == "onnx.Conv"
        ]
        for nt in node_timings_conv:
            self.assertIsNone(nt.signature)


class TestTreeTableFormat(unittest.TestCase):

    def test_detail_0_no_tree(self):
        result = parse_profiling_output(text=SAMPLE_WITH_SIG)
        table = format_profiling_table(result, detail_level=0)
        self.assertNotIn("Per-node breakdown:", table)

    def test_detail_1_sig_only(self):
        result = parse_profiling_output(text=SAMPLE_SAME_SIG)
        table = format_profiling_table(result, detail_level=1)
        self.assertIn("Per-node breakdown:", table)
        self.assertIn("1x3x32x32xf32, 16x3x3x3xf32", table)
        self.assertNotIn("/conv1", table)

    def test_detail_2_with_nodes(self):
        result = parse_profiling_output(text=SAMPLE_SAME_SIG)
        table = format_profiling_table(result, detail_level=2)
        self.assertIn("├──", table)
        self.assertIn("└──", table)
        self.assertIn("/conv1", table)
        self.assertIn("/conv2", table)

    def test_no_tree_without_node_timings(self):
        result = ProfilingResult(
            op_timings=[
                OpTiming("onnx.Conv", 1, 0.001, 0.001, 100.0),
            ],
            total_time=0.001,
            num_measurements=1,
            warmup_skipped=0,
            reporting_mode="all",
        )
        table = format_profiling_table(result, detail_level=2)
        self.assertNotIn("Per-node breakdown:", table)

    def test_tree_local_percentages(self):
        result = parse_profiling_output(text=SAMPLE_SAME_SIG)
        table = format_profiling_table(result, detail_level=1)
        # conv total = 0.000115, same_sig total = 0.000085
        # same_sig local% of conv = 85/115 * 100 = 73.9%
        self.assertIn("73.9%", table)

    def test_null_sig_shows_no_signature(self):
        result = parse_profiling_output(text=SAMPLE_NO_SIG)
        table = format_profiling_table(result, detail_level=1)
        self.assertIn("(no signature)", table)


# Decoding scenario: same nodes, different signatures per run (seq len grows).
SAMPLE_SUM_DECODING = """\
==SIG-REPORT==, onnx.MatMul, /matmul1, sig, 1x1x128xf32, 1x128x128xf32
==SIG-REPORT==, onnx.Add, /add1, sig, 1x1x128xf32, 1x1x128xf32
==START-REPORT==
==PERF-REPORT==, onnx.MatMul, /matmul1, before, 0.000000, 100.000000
==PERF-REPORT==, onnx.MatMul, /matmul1, after, 0.000100, 100.000100
==PERF-REPORT==, onnx.Add, /add1, before, 0.000000, 100.000101
==PERF-REPORT==, onnx.Add, /add1, after, 0.000020, 100.000121
==SIG-REPORT==, onnx.MatMul, /matmul1, sig, 1x2x128xf32, 1x128x128xf32
==SIG-REPORT==, onnx.Add, /add1, sig, 1x2x128xf32, 1x2x128xf32
==START-REPORT==
==PERF-REPORT==, onnx.MatMul, /matmul1, before, 0.000000, 200.000000
==PERF-REPORT==, onnx.MatMul, /matmul1, after, 0.000120, 200.000120
==PERF-REPORT==, onnx.Add, /add1, before, 0.000000, 200.000121
==PERF-REPORT==, onnx.Add, /add1, after, 0.000025, 200.000146
==SIG-REPORT==, onnx.MatMul, /matmul1, sig, 1x3x128xf32, 1x128x128xf32
==SIG-REPORT==, onnx.Add, /add1, sig, 1x3x128xf32, 1x3x128xf32
==START-REPORT==
==PERF-REPORT==, onnx.MatMul, /matmul1, before, 0.000000, 300.000000
==PERF-REPORT==, onnx.MatMul, /matmul1, after, 0.000130, 300.000130
==PERF-REPORT==, onnx.Add, /add1, before, 0.000000, 300.000131
==PERF-REPORT==, onnx.Add, /add1, after, 0.000030, 300.000161
"""


class TestSumModeDecoding(unittest.TestCase):
    """Tests for sum mode with per-run signature expansion (decoding use case)."""

    def test_sum_count_is_total_invocations(self):
        result = parse_profiling_output(text=SAMPLE_SUM_DECODING, reporting="sum")
        matmul = next(op for op in result.op_timings if op.op_name == "onnx.MatMul")
        # 1 node x 3 runs = 3 invocations
        self.assertEqual(matmul.count, 3)

    def test_sum_cumulative_time(self):
        result = parse_profiling_output(text=SAMPLE_SUM_DECODING, reporting="sum")
        matmul = next(op for op in result.op_timings if op.op_name == "onnx.MatMul")
        self.assertAlmostEqual(matmul.total_time, 0.000350, places=6)

    def test_sum_node_timings_expanded(self):
        result = parse_profiling_output(text=SAMPLE_SUM_DECODING, reporting="sum")
        matmul_nts = [nt for nt in result.node_timings if nt.op_name == "onnx.MatMul"]
        # 1 node x 3 runs = 3 NodeTimings
        self.assertEqual(len(matmul_nts), 3)

    def test_sum_each_run_has_own_signature(self):
        result = parse_profiling_output(text=SAMPLE_SUM_DECODING, reporting="sum")
        matmul_nts = [nt for nt in result.node_timings if nt.op_name == "onnx.MatMul"]
        sigs = [nt.signature for nt in matmul_nts]
        self.assertIn("1x1x128xf32, 1x128x128xf32", sigs)
        self.assertIn("1x2x128xf32, 1x128x128xf32", sigs)
        self.assertIn("1x3x128xf32, 1x128x128xf32", sigs)

    def test_sum_all_different_signatures(self):
        result = parse_profiling_output(text=SAMPLE_SUM_DECODING, reporting="sum")
        matmul_nts = [nt for nt in result.node_timings if nt.op_name == "onnx.MatMul"]
        sigs = set(nt.signature for nt in matmul_nts)
        self.assertEqual(len(sigs), 3)

    def test_sum_individual_run_times(self):
        result = parse_profiling_output(text=SAMPLE_SUM_DECODING, reporting="sum")
        matmul_nts = sorted(
            [nt for nt in result.node_timings if nt.op_name == "onnx.MatMul"],
            key=lambda x: x.time,
        )
        self.assertAlmostEqual(matmul_nts[0].time, 0.000100, places=6)
        self.assertAlmostEqual(matmul_nts[1].time, 0.000120, places=6)
        self.assertAlmostEqual(matmul_nts[2].time, 0.000130, places=6)

    def test_all_mode_count_is_distinct_nodes(self):
        result = parse_profiling_output(text=SAMPLE_SUM_DECODING, reporting="all")
        matmul = next(op for op in result.op_timings if op.op_name == "onnx.MatMul")
        # all mode: count = distinct nodes, not invocations
        self.assertEqual(matmul.count, 1)

    def test_all_mode_uses_first_signature(self):
        result = parse_profiling_output(text=SAMPLE_SUM_DECODING, reporting="all")
        matmul_nts = [nt for nt in result.node_timings if nt.op_name == "onnx.MatMul"]
        self.assertEqual(len(matmul_nts), 1)
        self.assertEqual(matmul_nts[0].signature, "1x1x128xf32, 1x128x128xf32")

    def test_sum_with_warmup_skips_first_run(self):
        result = parse_profiling_output(
            text=SAMPLE_SUM_DECODING, reporting="sum", warmup_num=1
        )
        matmul = next(op for op in result.op_timings if op.op_name == "onnx.MatMul")
        # 1 node x 2 runs (3 - 1 warmup) = 2 invocations
        self.assertEqual(matmul.count, 2)
        matmul_nts = [nt for nt in result.node_timings if nt.op_name == "onnx.MatMul"]
        sigs = [nt.signature for nt in matmul_nts]
        # First run's sig (1x1x128xf32) should be skipped
        self.assertNotIn("1x1x128xf32, 1x128x128xf32", sigs)
        self.assertIn("1x2x128xf32, 1x128x128xf32", sigs)
        self.assertIn("1x3x128xf32, 1x128x128xf32", sigs)

    def test_sum_with_warmup_cumulative_time(self):
        result = parse_profiling_output(
            text=SAMPLE_SUM_DECODING, reporting="sum", warmup_num=1
        )
        matmul = next(op for op in result.op_timings if op.op_name == "onnx.MatMul")
        # Runs 2+3: 0.000120 + 0.000130 = 0.000250
        self.assertAlmostEqual(matmul.total_time, 0.000250, places=6)

    def test_sum_sig_grouping_for_tree(self):
        result = parse_profiling_output(text=SAMPLE_SUM_DECODING, reporting="sum")
        matmul_nts = [nt for nt in result.node_timings if nt.op_name == "onnx.MatMul"]
        sig_groups: dict = {}
        for nt in matmul_nts:
            sig_groups.setdefault(nt.signature, []).append(nt)
        # Each run has a unique sig, so 3 groups each with 1 entry
        self.assertEqual(len(sig_groups), 3)
        for nts in sig_groups.values():
            self.assertEqual(len(nts), 1)

    def test_sum_tree_table_shows_all_signatures(self):
        result = parse_profiling_output(text=SAMPLE_SUM_DECODING, reporting="sum")
        table = format_profiling_table(result, detail_level=1)
        self.assertIn("1x1x128xf32, 1x128x128xf32", table)
        self.assertIn("1x2x128xf32, 1x128x128xf32", table)
        self.assertIn("1x3x128xf32, 1x128x128xf32", table)

    def test_parse_sig_lines_collects_all_runs(self):
        lines = SAMPLE_SUM_DECODING.splitlines(keepends=True)
        sig_dict = _parse_sig_lines(lines)
        key = _get_timing_key("onnx.MatMul", "/matmul1")
        self.assertEqual(len(sig_dict[key]), 3)
        self.assertEqual(sig_dict[key][0], "1x1x128xf32, 1x128x128xf32")
        self.assertEqual(sig_dict[key][2], "1x3x128xf32, 1x128x128xf32")


if __name__ == "__main__":
    unittest.main()
