# SPDX-License-Identifier: Apache-2.0

##################### profiling.py ############################################
#
# Copyright 2026 The IBM Research Authors.
#
###############################################################################
#
# This module parses onnx-mlir profiling output (from --profile-ir-with-sig) and
# returns structured per-op timing data. It is a self-contained module
# usable independently of explain_context().
#
# The parsing logic is derived from utils/make-report.py but uses only
# the Python standard library (no numpy dependency).
#
###############################################################################

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


@dataclass
class OpTiming:
    """Timing data for a single ONNX op type."""

    op_name: str
    count: int
    avg_time: float
    total_time: float
    percentage: float


@dataclass
class NodeTiming:
    """Timing data for a single node instance."""

    node_name: str
    op_name: str
    time: float
    percentage: float
    signature: Optional[str] = None


@dataclass
class ProfilingResult:
    """Complete profiling result from one or more inference runs."""

    op_timings: List[OpTiming]
    total_time: float
    num_measurements: int
    warmup_skipped: int
    reporting_mode: str
    cpu_time: Optional[float] = None
    accel_time: Optional[float] = None
    node_timings: List[NodeTiming] = field(default_factory=list)


# Regex for ==PERF-REPORT==, <op>, <node>, <details...>
_PERF_REPORT_RE = re.compile(
    r"^==PERF-REPORT==,\s*([0-9a-zA-Z_.\-]+)\s*,\s*([^,]*),\s*(.*)"
)
_START_REPORT_RE = re.compile(r"==START-REPORT==")


def _get_timing_key(op: str, node_name: str) -> str:
    p = re.match(r"(.*)-(simd|par)", op)
    if p:
        op = p[1]
    return op + "_=_" + node_name


def _get_op_from_timing_key(timing_key: str) -> str:
    p = re.match(r"(.*)_=_(.*)", timing_key)
    assert p is not None
    return p[1]


def _get_node_name_from_timing_key(timing_key: str) -> str:
    p = re.match(r"(.*)_=_(.*)", timing_key)
    assert p is not None
    return p[2]


def _parse_perf_lines(
    lines, warmup_num: int = 0, reporting: str = "all"
) -> Tuple[Dict[str, float], float, Dict[str, List[float]]]:
    """Parse ==PERF-REPORT== lines.

    Returns (node_time_dict, total_time, per_run_times) where per_run_times
    holds individual run values after warmup (before averaging/discard).
    """
    line_iter = iter(lines)

    # Skip to first ==START-REPORT==.
    has_start = False
    for line in line_iter:
        if _START_REPORT_RE.search(line):
            has_start = True
            break
    if not has_start:
        return {}, 0.0, {}

    # Collect all measurement sets.
    time_stat_dict_all_meas: Dict[str, List[float]] = {}
    start_count = 0

    while has_start:
        time_stat_dict: Dict[str, List[float]] = {}
        has_start = False
        start_count += 1

        for line in line_iter:
            if _START_REPORT_RE.search(line):
                has_start = True
                break

            m = _PERF_REPORT_RE.match(line)
            if m is None:
                continue

            op = m[1]
            node_name = m[2].strip()
            details = m[3]

            # Keep only "after" entries.
            if "before" in details:
                continue

            detail_array = details.split(",")
            key = _get_timing_key(op, node_name)
            time_val = float(detail_array[1])

            if key not in time_stat_dict:
                time_stat_dict[key] = []
            time_stat_dict[key].append(time_val)

        # Average times for duplicate op+node pairs within a single run.
        for node in time_stat_dict:
            vals = time_stat_dict[node]
            avg = sum(vals) / len(vals)
            if node not in time_stat_dict_all_meas:
                time_stat_dict_all_meas[node] = []
            time_stat_dict_all_meas[node].append(avg)

    # Apply warmup and reporting mode.
    meas_num = start_count - warmup_num
    if meas_num <= 0:
        return {}, 0.0, {}

    discard_num = 0
    if reporting == "quartile":
        discard_num = int(meas_num / 4)
    elif reporting == "sum":
        pass  # no discard
    # "all": no discard

    per_run_times: Dict[str, List[float]] = {}
    node_time_dict: Dict[str, float] = {}
    for node in time_stat_dict_all_meas:
        if node == "tot_time":
            continue
        time_array = time_stat_dict_all_meas[node][warmup_num:]
        if len(time_array) != meas_num:
            continue
        per_run_times[node] = list(time_array)
        if discard_num > 0 and meas_num - 2 * discard_num > 0:
            time_array = sorted(time_array)
            time_array = time_array[discard_num : len(time_array) - discard_num]
        if reporting == "sum":
            node_time_dict[node] = sum(time_array)
        else:
            node_time_dict[node] = sum(time_array) / len(time_array)

    tot_time = sum(node_time_dict.values())
    return node_time_dict, tot_time, per_run_times


_SIG_REPORT_RE = re.compile(
    r"^==SIG-REPORT==,\s*([0-9a-zA-Z_.\-]+)\s*,\s*([^,]*),\s*(.*)"
)


def _parse_sig_lines(lines) -> Dict[str, List[str]]:
    """Parse ==SIG-REPORT== lines and return {timing_key: [sig, ...]}."""
    sig_dict: Dict[str, List[str]] = {}
    for line in lines:
        m = _SIG_REPORT_RE.match(line.strip() if isinstance(line, str) else line)
        if m is None:
            continue
        op = m[1]
        node_name = m[2].strip()
        details = m[3]
        parts = details.split(",", 1)
        if len(parts) > 1 and parts[0].strip() == "sig":
            sig_str = parts[1].strip()
        else:
            sig_str = details.strip()
        key = _get_timing_key(op, node_name)
        sig_dict.setdefault(key, []).append(sig_str)
    return sig_dict


def _build_op_summary(
    node_time_dict: Dict[str, float],
    tot_time: float,
    sig_dict: Optional[Dict[str, List[str]]] = None,
    per_run_times: Optional[Dict[str, List[float]]] = None,
    reporting: str = "all",
) -> Tuple[List[OpTiming], Optional[float], Optional[float], List[NodeTiming]]:
    """Aggregate per-node timings into per-op-type summaries.

    For "sum" reporting, each (node, run) pair becomes a separate NodeTiming
    with that run's time and signature, and OpTiming.count reflects total
    invocations across all runs.
    """
    if sig_dict is None:
        sig_dict = {}

    op_times: Dict[str, float] = {}
    op_counts: Dict[str, int] = {}
    node_timings: List[NodeTiming] = []

    for timing_key, time_val in node_time_dict.items():
        op = _get_op_from_timing_key(timing_key)
        node_name = _get_node_name_from_timing_key(timing_key)
        sigs = sig_dict.get(timing_key, [])

        if reporting == "sum" and per_run_times and timing_key in per_run_times:
            run_times = per_run_times[timing_key]
            sig_offset = max(0, len(sigs) - len(run_times))
            for ri, rt in enumerate(run_times):
                sig_idx = sig_offset + ri
                sig = sigs[sig_idx] if sig_idx < len(sigs) else None
                pct = (rt / tot_time * 100) if tot_time > 0 else 0.0
                node_timings.append(
                    NodeTiming(
                        node_name=node_name,
                        op_name=op,
                        time=rt,
                        percentage=pct,
                        signature=sig,
                    )
                )
            num_invocations = len(run_times)
        else:
            sig = sigs[0] if sigs else None
            pct = (time_val / tot_time * 100) if tot_time > 0 else 0.0
            node_timings.append(
                NodeTiming(
                    node_name=node_name,
                    op_name=op,
                    time=time_val,
                    percentage=pct,
                    signature=sig,
                )
            )
            num_invocations = 1

        if op not in op_times:
            op_times[op] = 0.0
            op_counts[op] = 0
        op_times[op] += time_val
        op_counts[op] += num_invocations

    node_timings.sort(key=lambda x: x.time, reverse=True)

    timings = []
    for op in op_times:
        count = op_counts[op]
        total = op_times[op]
        avg = total / count if count > 0 else 0.0
        pct = (total / tot_time * 100) if tot_time > 0 else 0.0
        timings.append(
            OpTiming(
                op_name=op,
                count=count,
                avg_time=avg,
                total_time=total,
                percentage=pct,
            )
        )

    timings.sort(key=lambda x: x.total_time, reverse=True)

    # CPU vs accelerator breakdown (when zhigh ops are present).
    cpu_time = None
    accel_time = None
    has_zhigh = any(re.match(r"zhigh\.", op) for op in op_times)
    if has_zhigh:
        cpu_time = 0.0
        accel_time = 0.0
        for op, t in op_times.items():
            if re.match(r"zhigh\.", op) and not re.search(r"Stick|Unstick", op):
                accel_time += t
            else:
                cpu_time += t

    return timings, cpu_time, accel_time, node_timings


def parse_profiling_output(
    file_path: str = None,
    text: str = None,
    warmup_num: int = 0,
    reporting: str = "all",
) -> Optional[ProfilingResult]:
    """Parse onnx-mlir profiling output and return structured results.

    Args:
        file_path: Path to a file containing profiling output.
        text: Raw profiling output text (alternative to file_path).
        warmup_num: Number of warmup measurement sets to skip.
        reporting: How per-node times are aggregated across multiple runs:
            "all" -- average time per node across all runs (default).
                Count = number of distinct node instances per op type.
            "quartile" -- sort runs, discard top/bottom 25%, average the rest.
                Count = number of distinct node instances per op type.
            "sum" -- cumulative time per node across all runs (not averaged).
                Count = total invocations (distinct nodes x runs).
                Each (node, run) becomes a separate entry with that run's
                signature, useful for decoding where tensor sizes change.

    Returns:
        ProfilingResult with per-op timing breakdown, or None if no data.
    """
    valid_modes = ("quartile", "all", "sum")
    if reporting not in valid_modes:
        raise ValueError(
            "reporting must be one of {}, got {!r}".format(valid_modes, reporting)
        )

    if file_path is not None:
        with open(file_path, "r") as f:
            lines = f.readlines()
    elif text is not None:
        lines = text.splitlines(keepends=True)
    else:
        return None

    if not lines:
        return None

    node_time_dict, tot_time, per_run_times = _parse_perf_lines(
        lines, warmup_num, reporting
    )
    if not node_time_dict and tot_time == 0.0:
        return None

    sig_dict = _parse_sig_lines(lines)
    timings, cpu_time, accel_time, node_timings = _build_op_summary(
        node_time_dict, tot_time, sig_dict, per_run_times, reporting
    )

    # Count measurement sets used.
    start_count = sum(1 for line in lines if _START_REPORT_RE.search(line))
    meas_num = start_count - warmup_num
    if reporting == "quartile":
        discard = int(meas_num / 4)
        used = max(0, meas_num - 2 * discard)
    else:
        used = meas_num

    return ProfilingResult(
        op_timings=timings,
        total_time=tot_time,
        num_measurements=used,
        warmup_skipped=warmup_num,
        reporting_mode=reporting,
        cpu_time=cpu_time,
        accel_time=accel_time,
        node_timings=node_timings,
    )


def _group_by_signature(
    nodes: List[NodeTiming],
) -> List[Tuple[Optional[str], List[NodeTiming], float]]:
    """Group NodeTiming objects by signature.

    Returns a list of (signature, nodes, total_time) tuples sorted by
    total_time descending.  Used by both table and dict formatters.
    """
    sig_groups: Dict[Optional[str], List[NodeTiming]] = {}
    for nt in nodes:
        sig_groups.setdefault(nt.signature, []).append(nt)
    items: List[Tuple[Optional[str], List[NodeTiming], float]] = []
    for sig_key, nts in sig_groups.items():
        total = sum(n.time for n in nts)
        items.append((sig_key, nts, total))
    items.sort(key=lambda x: x[2], reverse=True)
    return items


def format_profiling_table(
    result: ProfilingResult, time_unit: str = "ms", detail_level: int = 0
) -> str:
    """Format a ProfilingResult as a human-readable table.

    Args:
        result: The profiling result to format.
        time_unit: Time unit for display -- "s", "ms", or "us".
        detail_level: 0 = op summary only, 1 = + signature breakdown,
                      2 = + individual nodes.

    Returns:
        Formatted string.
    """
    if time_unit == "s":
        scale = 1.0
    elif time_unit == "us":
        scale = 1_000_000.0
    else:
        scale = 1000.0
        time_unit = "ms"

    mode_desc = {
        "all": "all -- average per-node time across all runs",
        "sum": "sum -- cumulative per-node time across all runs",
        "quartile": "quartile -- average after discarding top/bottom 25% of runs",
    }
    lines = []
    lines.append(
        "Reporting: {} ({} measurement(s), {} warmup skipped)".format(
            mode_desc.get(result.reporting_mode, result.reporting_mode),
            result.num_measurements,
            result.warmup_skipped,
        )
    )
    lines.append("")

    nodes_by_op: Dict[str, List[NodeTiming]] = {}
    for nt in result.node_timings:
        nodes_by_op.setdefault(nt.op_name, []).append(nt)

    def _tree_lines_for_op(op: OpTiming, dlevel: int) -> List[str]:
        op_nodes = nodes_by_op.get(op.op_name, [])
        if not op_nodes:
            return []
        sig_items = _group_by_signature(op_nodes)

        tree = []
        for si, (sig_key, nts, sig_total) in enumerate(sig_items):
            is_last_sig = si == len(sig_items) - 1
            prefix = "└── " if is_last_sig else "├── "
            child_prefix = "    " if is_last_sig else "│   "
            sig_label = sig_key if sig_key is not None else "(no signature)"
            op_total = op.total_time if op.total_time > 0 else 1.0
            local_pct = sig_total / op_total * 100
            tree.append(
                "{}{} ({}, {:.4f}, {:.1f}%)".format(
                    prefix, sig_label, len(nts), sig_total * scale, local_pct
                )
            )
            if dlevel >= 2:
                sorted_nts = sorted(nts, key=lambda x: x.time, reverse=True)
                for ni, nt in enumerate(sorted_nts):
                    is_last_node = ni == len(sorted_nts) - 1
                    node_prefix = child_prefix + ("└── " if is_last_node else "├── ")
                    node_pct = nt.time / sig_total * 100 if sig_total > 0 else 0.0
                    tree.append(
                        "{}{} ({:.4f}, {:.1f}%)".format(
                            node_prefix,
                            nt.node_name,
                            nt.time * scale,
                            node_pct,
                        )
                    )
        return tree

    try:
        from tabulate import tabulate

        rows = []
        for op in result.op_timings:
            rows.append(
                [
                    op.op_name,
                    str(op.count),
                    "{:.4f}".format(op.avg_time * scale),
                    "{:.4f}".format(op.total_time * scale),
                    "{:.1f}%".format(op.percentage),
                ]
            )
        headers = [
            "Op",
            "Count",
            "Avg ({})".format(time_unit),
            "Total ({})".format(time_unit),
            "%",
        ]
        lines.append(tabulate(rows, headers=headers, tablefmt="grid"))
    except ImportError:
        op_w = max(
            len("Op"),
            max((len(op.op_name) for op in result.op_timings), default=0),
        )
        fmt = "  {:<{op_w}s}  {:>6s}  {:>12s}  {:>12s}  {:>7s}"
        hdr = fmt.format(
            "Op",
            "Count",
            "Avg({})".format(time_unit),
            "Total({})".format(time_unit),
            "%",
            op_w=op_w,
        )
        lines.append(hdr)
        lines.append("  " + "=" * (len(hdr) - 2))
        for op in result.op_timings:
            lines.append(
                fmt.format(
                    op.op_name,
                    str(op.count),
                    "{:.4f}".format(op.avg_time * scale),
                    "{:.4f}".format(op.total_time * scale),
                    "{:.1f}%".format(op.percentage),
                    op_w=op_w,
                )
            )
            lines.append("  " + "-" * (len(hdr) - 2))

    if detail_level >= 1 and result.node_timings:
        lines.append("")
        lines.append("Per-node breakdown:")
        lines.append("")
        for op in result.op_timings:
            lines.append(
                "{} (count: {}, avg: {:.4f}, total: {:.4f}, {:.1f}%)".format(
                    op.op_name,
                    op.count,
                    op.avg_time * scale,
                    op.total_time * scale,
                    op.percentage,
                )
            )
            for tree_line in _tree_lines_for_op(op, detail_level):
                lines.append(tree_line)

    lines.append("")
    lines.append(
        "Total model time: {:.4f} {}".format(result.total_time * scale, time_unit)
    )

    if result.cpu_time is not None and result.accel_time is not None:
        tot = result.total_time if result.total_time > 0 else 1.0
        lines.append(
            "CPU time: {:.4f} {} ({:.1f}%), Accelerator time: {:.4f} {} ({:.1f}%)".format(
                result.cpu_time * scale,
                time_unit,
                result.cpu_time / tot * 100,
                result.accel_time * scale,
                time_unit,
                result.accel_time / tot * 100,
            )
        )

    return "\n".join(lines)
