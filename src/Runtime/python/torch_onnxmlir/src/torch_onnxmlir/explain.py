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
) -> Optional[str]:
    """
    Internal function to format and return performance metrics.

    This is used internally by ExplainContext. Do not call directly.
    Use explain_context() instead.

    Args:
        format: "table", "json", or "dict"
        detailed: Not used (reserved for future)
        sort_by: "time" (default), "order", or "calls"

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
            all_metrics, eager_fallbacks, total_eager_fallbacks, sort_by
        )
    elif format == "json":
        return json.dumps(
            _format_as_dict(
                all_metrics, eager_fallbacks, total_eager_fallbacks, sort_by
            ),
            indent=2,
        )
    else:
        return _format_as_table(
            all_metrics, eager_fallbacks, total_eager_fallbacks, sort_by
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
    all_metrics: Dict, eager_fallbacks: Dict, total_eager_fallbacks: int, sort_by: str
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

    if not HAS_TABULATE:
        out.append("Tip: pip install tabulate")

    return "\n".join(out)


def _format_as_dict(
    all_metrics: Dict, eager_fallbacks: Dict, total_eager_fallbacks: int, sort_by: str
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

    for key, m in _sort_models(all_metrics, sort_by):
        result["graphs"][key] = {
            "call_order": m.call_order,
            "compilation_time": m.compilation_time,
            "inference_count": m.inference_count,
            "cache_hits": m.cache_hits,
            "cache_hit_rate": m.cache_hit_rate,
            "avg_inference_time": m.avg_inference_time,
            "total_inference_time": m.total_inference_time,
            "total_time": m.total_time,
        }

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

    def __init__(self, format="table", sort_by="time", export_to=None):
        """
        Initialize context manager.

        Args:
            format: Output format ('table', 'dict', 'json'). Default: 'table'.
            sort_by: Sort order ('time', 'calls', 'order'). Default: 'time'.
            export_to: Optional file path to automatically export metrics on exit.
        """
        self.format = format
        self.sort_by = sort_by
        self.export_to = export_to
        self.metrics = None
        self._previous_enable_state = None

    def __enter__(self):
        """Enable metrics collection and clear existing metrics."""
        # Save previous state.
        self._previous_enable_state = _enable_explain

        # Enable metrics collection using internal setter.
        _set_enable_explain(True)

        # Clear any existing metrics.
        _clear_metrics()

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Capture metrics and restore previous state."""
        # Capture metrics before disabling.
        self.metrics = _explain(format=self.format, sort_by=self.sort_by)

        # Export to file if specified.
        if self.export_to is not None:
            self._export_metrics()

        # Restore previous state using internal setter.
        _set_enable_explain(self._previous_enable_state)

        # Don't suppress exceptions.
        return False

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
        return f"ExplainContext(format={self.format!r}, sort_by={self.sort_by!r})"

    @property
    def data(self):
        """
        Access raw metrics data.

        Returns:
            Metrics in the specified format (dict, str, or json str).
        """
        return self.metrics


def explain_context(format="table", sort_by="time", export_to=None):
    """
    Create a context manager for automatic metrics collection.

    This context manager automatically enables metrics collection on entry,
    captures metrics on exit, and provides access to the metrics.

    Args:
        format: Output format - 'table', 'dict', or 'json' (default: 'table').
        sort_by: Sort order - 'time', 'calls', or 'order' (default: 'time').
        export_to: Optional file path to automatically export metrics on exit (default: None).

    Returns:
        ExplainContext: Context manager instance.

    Example:
        Basic usage:
        >>> with torch_onnxmlir.explain_context() as metrics:
        ...     output = compiled_model(input)
        ...     print(metrics)

        Access metrics data:
        >>> with torch_onnxmlir.explain_context(format="dict") as metrics:
        ...     output = compiled_model(input)
        ...     cache_hits = metrics.data['summary']['total_cache_hits']

        Automatic export:
        >>> with torch_onnxmlir.explain_context(export_to="metrics.json") as metrics:
        ...     output = compiled_model(input)
        ...     # Metrics automatically saved to metrics.json on exit.
    """
    return ExplainContext(format=format, sort_by=sort_by, export_to=export_to)


def _clear_metrics():
    """
    Internal function to clear all collected metrics.

    This is used internally by ExplainContext. Do not call directly.
    Use explain_context() instead which handles metrics lifecycle automatically.
    """
    metrics.global_metrics_collector.clear()
