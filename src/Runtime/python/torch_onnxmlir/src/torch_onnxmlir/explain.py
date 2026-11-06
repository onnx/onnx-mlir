# SPDX-License-Identifier: Apache-2.0

##################### explain.py *******########################################
#
# Copyright 2026 The IBM Research Authors.
#
################################################################################
#
# This file implements the explain() function for torch_onnxmlir package,
# providing insights into compilation and inference performance metrics.
#
################################################################################

import json
import os
from typing import Optional, Dict, List, Literal
from . import metrics, config

try:
    from tabulate import tabulate

    HAS_TABULATE = True
except ImportError:
    HAS_TABULATE = False


# Private flag for explain metrics collection.
# This is controlled internally by ExplainContext and should not be accessed directly.
_enable_explain = False


def is_enabled() -> bool:
    """
    Check if explain metrics collection is currently enabled.

    This function is used by backend.py and metrics.py to check if metrics
    should be collected. The flag is controlled automatically by explain_context().

    Returns:
        bool: True if explain metrics collection is enabled, False otherwise.
    """
    return _enable_explain


def _set_enable_explain(value: bool):
    """
    Internal setter for enable_explain flag.

    This function is for internal use only by ExplainContext.
    Do not call this directly. Use explain_context() instead.

    Args:
        value: Boolean value to set for enable_explain.
    """
    global _enable_explain
    _enable_explain = value


def _explain(
    format: str = "table",
    detailed: bool = False,
    sort_by: Literal["time", "order", "calls"] = "time",
    profiling_detail: int = 0,
) -> Optional[str]:
    """
    Internal function to format and return performance metrics.

    This is used internally by ExplainContext. Do not call directly.
    Use explain_context() instead.

    Args:
        format: "table", "json", or "dict"
        detailed: Not used (reserved for future)
        sort_by: "time" (default), "order", or "calls"
        profiling_detail: 0 = op summary, 1 = + signatures, 2 = + nodes.

    Returns:
        Formatted string or dict
    """
    if not _enable_explain:
        return (
            "Explain feature is not enabled. Use explain_context() to collect metrics."
        )

    all_metrics = metrics.global_metrics_collector.get_metrics()
    eager_fallbacks = metrics.global_metrics_collector.get_eager_fallbacks()
    total_eager_fallbacks = metrics.global_metrics_collector.total_eager_fallbacks

    if not all_metrics and not eager_fallbacks:
        return "No metrics collected yet. Run some models first."

    if format == "dict":
        return _format_as_dict(
            all_metrics,
            eager_fallbacks,
            total_eager_fallbacks,
            sort_by,
            profiling_detail,
        )
    elif format == "json":
        return json.dumps(
            _format_as_dict(
                all_metrics,
                eager_fallbacks,
                total_eager_fallbacks,
                sort_by,
                profiling_detail,
            ),
            indent=2,
        )
    else:
        return _format_as_table(
            all_metrics,
            eager_fallbacks,
            total_eager_fallbacks,
            sort_by,
            profiling_detail,
        )


def _sort_models(all_metrics: Dict, sort_by: str) -> List[tuple]:
    """Sort models based on criteria."""
    if sort_by == "order":
        return sorted(all_metrics.items(), key=lambda x: x[1].call_order)
    elif sort_by == "calls":
        return sorted(
            all_metrics.items(), key=lambda x: x[1].inference_count, reverse=True
        )
    else:
        return sorted(all_metrics.items(), key=lambda x: x[1].total_time, reverse=True)


def _format_table_simple(headers: List[str], rows: List[List[str]]) -> str:
    """Simple table formatting fallback when tabulate is not available."""
    col_widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            col_widths[i] = max(col_widths[i], len(str(cell)))

    sep = "+" + "+".join("-" * (w + 2) for w in col_widths) + "+"
    hdr = "|" + "|".join(f" {h:<{col_widths[i]}} " for i, h in enumerate(headers)) + "|"

    lines = [sep, hdr, sep]
    for row in rows:
        lines.append(
            "|"
            + "|".join(f" {str(cell):<{col_widths[i]}} " for i, cell in enumerate(row))
            + "|"
        )
    lines.append(sep)
    return "\n".join(lines)


def _format_as_table(
    all_metrics: Dict,
    eager_fallbacks: Dict,
    total_eager_fallbacks: int,
    sort_by: str,
    profiling_detail: int = 0,
) -> str:
    """Format metrics as a table."""
    from . import sessioncache

    out = ["=" * 80, "TORCH_ONNXMLIR PERFORMANCE METRICS", "=" * 80, ""]

    # Summary statistics
    total_inferences = sum(m.inference_count for m in all_metrics.values())
    total_cache_hits = sum(m.cache_hits for m in all_metrics.values())

    out.append(f"Cache Directory: {sessioncache.cache_dir()}")
    out.append(f"Unique Graphs: {len(all_metrics)}")
    out.append(f"Total Inference Calls: {total_inferences}")
    if total_inferences > 0:
        out.append(
            f"Cache Hit Rate: {total_cache_hits}/{total_inferences} ({100*total_cache_hits/total_inferences:.1f}%)"
        )
    out.append(f"Eager Fallbacks: {total_eager_fallbacks}")
    out.append("")

    # Eager fallback summary
    if eager_fallbacks:
        out.append("Eager Mode Fallbacks:")
        out.append("-" * 80)

        reason_counts = {}
        for info in eager_fallbacks.values():
            reason_counts[info.reason] = reason_counts.get(info.reason, 0) + info.count

        fb_table = []
        for reason, count in sorted(
            reason_counts.items(), key=lambda x: x[1], reverse=True
        ):
            pct = (
                100 * count / total_eager_fallbacks if total_eager_fallbacks > 0 else 0
            )
            fb_table.append([reason, str(count), f"{pct:.1f}%"])

        if HAS_TABULATE:
            out.append(
                tabulate(fb_table, headers=["Reason", "Count", "%"], tablefmt="grid")
            )
        else:
            out.append(_format_table_simple(["Reason", "Count", "%"], fb_table))
        out.append("")

    # Per-graph statistics
    if all_metrics:
        labels = {
            "time": "by Total Time",
            "order": "by Call Order",
            "calls": "by Call Count",
        }
        out.append(
            f"Per-Graph Statistics (sorted {labels.get(sort_by, 'by Total Time')}):"
        )
        out.append("-" * 80)

        if sort_by == "order":
            hdrs = [
                "Order",
                "Graph ID",
                "Calls",
                "Hits",
                "Compile(s)",
                "Avg Inference(ms)",
                "Total(s)",
                "%",
            ]
        elif sort_by == "calls":
            hdrs = [
                "Graph ID",
                "Calls",
                "Hits",
                "Hit%",
                "Compile(s)",
                "Avg Inference(ms)",
                "Total(s)",
                "%",
            ]
        else:
            hdrs = [
                "Graph ID",
                "Calls",
                "Hits",
                "Compile(s)",
                "Avg Inference(ms)",
                "Total(s)",
                "%",
            ]

        total_time_all = sum(m.total_time for m in all_metrics.values())
        sorted_models = _sort_models(all_metrics, sort_by)

        tbl = []
        for key, m in sorted_models:
            mid = key[:16] + "..." if len(key) > 16 else key
            pct = 100 * m.total_time / total_time_all if total_time_all > 0 else 0

            if sort_by == "order":
                row = [
                    str(m.call_order),
                    mid,
                    str(m.inference_count),
                    str(m.cache_hits),
                    f"{m.compilation_time:.3f}",
                    f"{m.avg_inference_time * 1000:.2f}",
                    f"{m.total_time:.3f}",
                    f"{pct:.1f}%",
                ]
            elif sort_by == "calls":
                row = [
                    mid,
                    str(m.inference_count),
                    str(m.cache_hits),
                    f"{100 * m.cache_hit_rate:.1f}%",
                    f"{m.compilation_time:.3f}",
                    f"{m.avg_inference_time * 1000:.2f}",
                    f"{m.total_time:.3f}",
                    f"{pct:.1f}%",
                ]
            else:
                row = [
                    mid,
                    str(m.inference_count),
                    str(m.cache_hits),
                    f"{m.compilation_time:.3f}",
                    f"{m.avg_inference_time * 1000:.2f}",
                    f"{m.total_time:.3f}",
                    f"{pct:.1f}%",
                ]
            tbl.append(row)

        if HAS_TABULATE:
            out.append(tabulate(tbl, headers=hdrs, tablefmt="grid"))
        else:
            out.append(_format_table_simple(hdrs, tbl))
        out.append("")

        # Optimization hint
        if sorted_models:
            if sort_by == "time":
                top = sorted_models[0]
                out.append(
                    f"💡 Top time: '{top[0][:16]}...' ({100 * top[1].total_time / total_time_all:.1f}%)"
                )
            elif sort_by == "calls":
                top = sorted_models[0]
                out.append(
                    f"💡 Most called: '{top[0][:16]}...' ({top[1].inference_count} calls)"
                )
            out.append("")

    # Profiling data section.
    profiling_data = metrics.global_metrics_collector.get_profiling_data()
    if profiling_data:
        from . import profiling as profiling_mod

        out.append("=" * 80)
        out.append("PER-OP PROFILING DATA")
        out.append("=" * 80)
        for cache_key, prof_result in profiling_data.items():
            mid = cache_key[:16] + "..." if len(cache_key) > 16 else cache_key
            out.append("")
            out.append(f"Graph: {mid}")
            out.append(
                profiling_mod.format_profiling_table(
                    prof_result, detail_level=profiling_detail
                )
            )

    if not HAS_TABULATE:
        out.append("Tip: pip install tabulate")

    return "\n".join(out)


def _format_as_dict(
    all_metrics: Dict,
    eager_fallbacks: Dict,
    total_eager_fallbacks: int,
    sort_by: str,
    profiling_detail: int = 0,
) -> dict:
    """Format metrics as a dictionary."""
    result = {
        "summary": {
            "unique_graphs": len(all_metrics),
            "total_inference_calls": sum(
                m.inference_count for m in all_metrics.values()
            ),
            "total_cache_hits": sum(m.cache_hits for m in all_metrics.values()),
            "total_eager_fallbacks": total_eager_fallbacks,
        },
        "graphs": {},
        "eager_fallbacks": {},
        "sort_by": sort_by,
    }

    from . import sessioncache

    for key, m in _sort_models(all_metrics, sort_by):
        graph_entry = {
            "call_order": m.call_order,
            "compilation_time": m.compilation_time,
            "inference_count": m.inference_count,
            "cache_hits": m.cache_hits,
            "cache_hit_rate": m.cache_hit_rate,
            "avg_inference_time": m.avg_inference_time,
            "total_inference_time": m.total_inference_time,
            "total_time": m.total_time,
        }
        config_path = os.path.join(
            sessioncache.cache_dir(), key, sessioncache.OM_BACKEND_CONFIG_FILE
        )
        try:
            with open(config_path, "r") as f:
                disk_cfg = json.load(f)
            for cfg_key in ("compilation_info", "input_signature", "output_signature"):
                raw = disk_cfg.get(cfg_key, "")
                try:
                    graph_entry[cfg_key] = json.loads(raw) if raw else raw
                except (json.JSONDecodeError, TypeError):
                    graph_entry[cfg_key] = raw
        except (OSError, json.JSONDecodeError):
            pass
        result["graphs"][key] = graph_entry

    sorted_fb = sorted(
        eager_fallbacks.items(),
        key=lambda x: x[1].call_order if sort_by == "order" else x[1].count,
        reverse=(sort_by != "order"),
    )
    for key, info in sorted_fb:
        result["eager_fallbacks"][key] = {
            "call_order": info.call_order,
            "reason": info.reason,
            "count": info.count,
        }
        if info.cache_key:
            result["eager_fallbacks"][key]["cache_key"] = info.cache_key

    # Profiling data.
    profiling_data = metrics.global_metrics_collector.get_profiling_data()
    if profiling_data:
        result["profiling"] = {}
        for cache_key, prof_result in profiling_data.items():
            nodes_by_op: dict = {}
            for nt in prof_result.node_timings:
                nodes_by_op.setdefault(nt.op_name, []).append(nt)

            ops_list = []
            for op in prof_result.op_timings:
                op_entry = {
                    "op_name": op.op_name,
                    "count": op.count,
                    "avg_time": op.avg_time,
                    "total_time": op.total_time,
                    "percentage": op.percentage,
                }
                op_nodes = nodes_by_op.get(op.op_name, [])
                if profiling_detail >= 1 and op_nodes:
                    sig_groups: dict = {}
                    for nt in op_nodes:
                        sig_groups.setdefault(nt.signature, []).append(nt)
                    op_total = op.total_time if op.total_time > 0 else 1.0
                    sig_list = []
                    for sig_key, nts in sig_groups.items():
                        sig_total = sum(n.time for n in nts)
                        count = len(nts)
                        sig_pct = sig_total / op_total * 100
                        sig_entry = {
                            "signature": sig_key,
                            "count": count,
                            "avg_time": sig_total / count if count else 0.0,
                            "total_time": sig_total,
                            "percentage": sig_pct,
                        }
                        if profiling_detail >= 2:
                            node_list = []
                            for nt in sorted(nts, key=lambda x: x.time, reverse=True):
                                node_pct = (
                                    nt.time / sig_total * 100 if sig_total > 0 else 0.0
                                )
                                node_list.append(
                                    {
                                        "node_name": nt.node_name,
                                        "time": nt.time,
                                        "percentage": node_pct,
                                    }
                                )
                            sig_entry["nodes"] = node_list
                        sig_list.append(sig_entry)
                    sig_list.sort(key=lambda x: x["total_time"], reverse=True)
                    op_entry["signatures"] = sig_list
                ops_list.append(op_entry)

            entry = {
                "total_time": prof_result.total_time,
                "num_measurements": prof_result.num_measurements,
                "reporting_mode": prof_result.reporting_mode,
                "ops": ops_list,
            }
            if prof_result.cpu_time is not None:
                entry["cpu_time"] = prof_result.cpu_time
            if prof_result.accel_time is not None:
                entry["accel_time"] = prof_result.accel_time
            result["profiling"][cache_key] = entry

    return result


class ExplainContext:
    """
    Context manager for automatic metrics collection.

    Automatically enables metrics collection on entry and captures
    metrics on exit. Provides access to metrics during and after
    the context.

    Attributes:
        format: Output format ('table', 'dict', 'json').
        sort_by: Sort order ('time', 'calls', 'order').
        export_to: Optional file path to automatically export metrics on exit.
        metrics: Captured metrics (available after __exit__).

    Example:
        with torch_onnxmlir.explain_context() as ctx:
            output = model(input)
            print(ctx)  # Print formatted metrics.

        # Access metrics after context.
        data = ctx.data
    """

    def __init__(
        self,
        format="table",
        sort_by="time",
        export_to=None,
        profiling=False,
    ):
        """
        Initialize context manager.

        Args:
            format: Output format ('table', 'dict', 'json'). Default: 'table'.
            sort_by: Sort order ('time', 'calls', 'order'). Default: 'time'.
            export_to: Optional file path to automatically export metrics on exit.
            profiling: Enable per-op profiling. Accepts:
                - False/None: disabled (default).
                - True: enabled with defaults from config.
                - dict: enabled with overrides, keys:
                    "detail": 0 (op summary), 1 (+ signatures), 2 (+ nodes).
                    "warmup": number of initial runs to skip before collecting.
                    "reporting": how per-node times aggregate across runs:
                        "all" -- average across all runs (default).
                        "quartile" -- discard top/bottom 25%, average the rest.
                        "sum" -- cumulative time, not averaged.
        """
        self.format = format
        self.sort_by = sort_by
        self.export_to = export_to

        if isinstance(profiling, dict):
            self.profiling = True
            self.profiling_detail = profiling.get("detail", config.profiling_detail)
            self.profiling_warmup = profiling.get("warmup", config.profiling_warmup)
            self.profiling_reporting = profiling.get(
                "reporting", config.profiling_reporting
            )
        elif profiling:
            self.profiling = True
            self.profiling_detail = config.profiling_detail
            self.profiling_warmup = config.profiling_warmup
            self.profiling_reporting = config.profiling_reporting
        else:
            self.profiling = False
            self.profiling_detail = config.profiling_detail
            self.profiling_warmup = config.profiling_warmup
            self.profiling_reporting = config.profiling_reporting

        self.metrics = None
        self._previous_enable_state = None
        self._previous_profiling_state = None
        self._previous_instrument_file = None

    def __enter__(self):
        """Enable metrics collection and clear existing metrics."""
        # Save previous state.
        self._previous_enable_state = _enable_explain
        self._previous_profiling_state = config._profiling_enabled

        # Enable metrics collection using internal setter.
        _set_enable_explain(True)

        # Clear any existing metrics.
        _clear_metrics()

        # Set up profiling if requested.
        if self.profiling:
            import torch._dynamo

            config._profiling_enabled = True
            self._previous_instrument_file = os.environ.get("ONNX_MLIR_INSTRUMENT_FILE")
            torch._dynamo.reset()

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Capture metrics and restore previous state."""
        # Parse profiling output if profiling was active.
        if self.profiling:
            if self._previous_instrument_file is not None:
                os.environ["ONNX_MLIR_INSTRUMENT_FILE"] = self._previous_instrument_file
            else:
                os.environ.pop("ONNX_MLIR_INSTRUMENT_FILE", None)
            self._collect_profiling_data()

        # Capture metrics before disabling.
        self.metrics = _explain(
            format=self.format,
            sort_by=self.sort_by,
            profiling_detail=self.profiling_detail,
        )

        # Export to file if specified.
        if self.export_to is not None:
            self._export_metrics()

        # Restore previous state using internal setter.
        _set_enable_explain(self._previous_enable_state)
        config._profiling_enabled = self._previous_profiling_state

        if self.profiling:
            import torch._dynamo

            torch._dynamo.reset()

        # Don't suppress exceptions.
        return False

    def _collect_profiling_data(self):
        """Parse per-graph profiling files from cache dir and record results."""
        import glob

        from . import profiling as profiling_mod
        from . import sessioncache

        pattern = os.path.join(sessioncache.cache_dir(), "*", "profile.prof")
        for path in glob.glob(pattern):
            if os.path.getsize(path) == 0:
                continue
            cache_key = os.path.basename(os.path.dirname(path))
            prof_result = profiling_mod.parse_profiling_output(
                file_path=path,
                warmup_num=self.profiling_warmup,
                reporting=self.profiling_reporting,
            )
            if prof_result is not None:
                metrics.global_metrics_collector.record_profiling(
                    cache_key, prof_result
                )
            try:
                os.unlink(path)
            except OSError:
                pass

    def _export_metrics(self):
        """Export metrics to file."""
        import os

        if self.metrics is None:
            return

        # Create directory if it doesn't exist.
        export_dir = os.path.dirname(os.path.abspath(self.export_to))
        if export_dir:
            os.makedirs(export_dir, exist_ok=True)

        with open(self.export_to, "w") as f:
            if isinstance(self.metrics, str):
                # For table or json format, write as-is.
                f.write(self.metrics)
            else:
                # For dict format, convert to JSON.
                json.dump(self.metrics, f, indent=2)

    def __str__(self):
        """Return formatted metrics as string."""
        if self.metrics is None:
            return "ExplainContext: No metrics collected yet"

        if isinstance(self.metrics, str):
            return self.metrics

        # For dict/json, format nicely.
        if isinstance(self.metrics, dict):
            return json.dumps(self.metrics, indent=2)

        return str(self.metrics)

    def __repr__(self):
        """Return representation of context manager."""
        return (
            f"ExplainContext(format={self.format!r}, sort_by={self.sort_by!r}"
            f", profiling={self.profiling!r})"
        )

    @property
    def data(self):
        """
        Access raw metrics data.

        Returns:
            Metrics in the specified format (dict, str, or json str).
        """
        return self.metrics


def explain_context(format="table", sort_by="time", export_to=None, profiling=False):
    """
    Create a context manager for automatic metrics collection.

    Args:
        format: Output format - 'table', 'dict', or 'json' (default: 'table').
        sort_by: Sort order - 'time', 'calls', or 'order' (default: 'time').
        export_to: Optional file path to automatically export metrics on exit.
        profiling: Enable per-op profiling. Accepts:
            - False/None: disabled (default).
            - True: enabled with defaults from config.
            - dict: enabled with overrides, keys:
                "detail": 0 (op summary), 1 (+ signatures), 2 (+ nodes).
                "warmup": number of initial runs to skip before collecting data.
                "reporting": how per-node times are aggregated across runs:
                    "all" -- average time per node across all runs (default).
                    "quartile" -- discard top/bottom 25% of runs, average the rest.
                    "sum" -- cumulative time per node across all runs.

    Returns:
        ExplainContext: Context manager instance.

    Example:
        Basic usage:
        >>> with torch_onnxmlir.explain_context() as ctx:
        ...     output = compiled_model(input)
        ...     print(ctx)

        With profiling (defaults):
        >>> with torch_onnxmlir.explain_context(profiling=True) as ctx:
        ...     output = compiled_model(input)

        With profiling config dict:
        >>> with torch_onnxmlir.explain_context(
        ...     profiling={"detail": 2, "warmup": 1, "reporting": "all"},
        ...     format="dict",
        ... ) as ctx:
        ...     for _ in range(5):
        ...         output = compiled_model(input)
        ...     print(ctx.data["profiling"])
    """
    return ExplainContext(
        format=format, sort_by=sort_by, export_to=export_to, profiling=profiling
    )


def _clear_metrics():
    """
    Internal function to clear all collected metrics.

    This is used internally by ExplainContext. Do not call directly.
    Use explain_context() instead which handles metrics lifecycle automatically.
    """
    metrics.global_metrics_collector.clear()
