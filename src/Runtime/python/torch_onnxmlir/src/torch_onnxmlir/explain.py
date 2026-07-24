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


def explain(
    format: str = "table",
    detailed: bool = False,
    sort_by: Literal["time", "order", "calls"] = "time",
) -> Optional[str]:
    """
    Display performance metrics for compiled models.

    Args:
        format: "table", "json", or "dict"
        detailed: Not used (reserved for future)
        sort_by: "time" (default), "order", or "calls"

    Returns:
        Formatted string or dict

    Example:
        >>> import torch_onnxmlir
        >>> torch_onnxmlir.config.enable_explain = True
        >>> # ... compile and run models ...
        >>> print(torch_onnxmlir.explain())
        >>> print(torch_onnxmlir.explain(sort_by="calls"))
    """
    if not config.enable_explain:
        return "Explain feature is not enabled. Set torch_onnxmlir.config.enable_explain = True"

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
    out = ["=" * 80, "TORCH_ONNXMLIR PERFORMANCE METRICS", "=" * 80, ""]

    # Summary statistics
    total_inferences = sum(m.inference_count for m in all_metrics.values())
    total_cache_hits = sum(m.cache_hits for m in all_metrics.values())

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
                "Compile(s)",
                "Calls",
                "Hits",
                "Avg(ms)",
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
                "Avg(ms)",
                "Total(s)",
                "%",
            ]
        else:
            hdrs = [
                "Graph ID",
                "Compile(s)",
                "Calls",
                "Hits",
                "Avg(ms)",
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
                    f"{m.compilation_time:.3f}",
                    str(m.inference_count),
                    str(m.cache_hits),
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
                    f"{m.compilation_time:.3f}",
                    str(m.inference_count),
                    str(m.cache_hits),
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


def clear_metrics():
    """Clear all collected metrics."""
    metrics.global_metrics_collector.clear()
