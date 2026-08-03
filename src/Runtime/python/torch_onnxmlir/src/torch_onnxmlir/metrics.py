# SPDX-License-Identifier: Apache-2.0

##################### metrics.py *******########################################
#
# Copyright 2026 The IBM Research Authors.
#
################################################################################
#
# This file implements metrics collection for the torch_onnxmlir.explain()
# feature.
#
################################################################################

from dataclasses import dataclass
from typing import Dict, Optional
from . import config, explain


@dataclass
class ModelMetrics:
    """Metrics for a single compiled model (unique graph)."""

    cache_key: str
    compilation_time: float = 0.0
    total_inference_time: float = 0.0
    inference_count: int = 0
    cache_hits: int = 0
    call_order: int = 0

    @property
    def avg_inference_time(self) -> float:
        """Average inference time per call."""
        return (
            self.total_inference_time / self.inference_count
            if self.inference_count > 0
            else 0.0
        )

    @property
    def total_time(self) -> float:
        """Total time spent (compilation + all inferences)."""
        return self.compilation_time + self.total_inference_time

    @property
    def cache_hit_rate(self) -> float:
        """Cache hit rate for this graph."""
        return (
            self.cache_hits / self.inference_count if self.inference_count > 0 else 0.0
        )


@dataclass
class EagerFallbackInfo:
    """Information about a fallback to eager mode."""

    reason: str
    cache_key: Optional[str] = None
    count: int = 1
    call_order: int = 0


class MetricsCollector:
    """
    Metrics collector for debugging and performance analysis.

    When disabled: Single boolean check (minimal overhead).
    When enabled: Direct dict updates for fast metric recording.
    """

    def __init__(self):
        self.models: Dict[str, ModelMetrics] = {}
        self.eager_fallbacks: Dict[str, EagerFallbackInfo] = {}
        self.total_eager_fallbacks: int = 0
        self._call_counter: int = 0

    def record_compilation(self, cache_key: str, compilation_time: float):
        """Record compilation metrics."""
        if not explain.is_enabled():
            return

        if cache_key not in self.models:
            self._call_counter += 1
            self.models[cache_key] = ModelMetrics(
                cache_key=cache_key, call_order=self._call_counter
            )
        self.models[cache_key].compilation_time = compilation_time

    def record_inference(
        self, cache_key: str, inference_time: float, is_cache_hit: bool
    ):
        """Record inference metrics."""
        if not explain.is_enabled():
            return

        if cache_key not in self.models:
            self._call_counter += 1
            self.models[cache_key] = ModelMetrics(
                cache_key=cache_key, call_order=self._call_counter
            )

        m = self.models[cache_key]
        m.total_inference_time += inference_time
        m.inference_count += 1
        if is_cache_hit:
            m.cache_hits += 1

    def record_eager_fallback(self, reason: str, cache_key: Optional[str] = None):
        """Record eager mode fallback."""
        if not explain.is_enabled():
            return

        self.total_eager_fallbacks += 1
        key = cache_key if cache_key else f"reason_{reason}"

        if key in self.eager_fallbacks:
            self.eager_fallbacks[key].count += 1
        else:
            self._call_counter += 1
            self.eager_fallbacks[key] = EagerFallbackInfo(
                reason=reason,
                cache_key=cache_key,
                count=1,
                call_order=self._call_counter,
            )

    def get_metrics(self) -> Dict[str, ModelMetrics]:
        """Get all collected metrics."""
        return dict(self.models)

    def get_eager_fallbacks(self) -> Dict[str, EagerFallbackInfo]:
        """Get all eager fallback information."""
        return dict(self.eager_fallbacks)

    def clear(self):
        """Clear all metrics."""
        self.models.clear()
        self.eager_fallbacks.clear()
        self.total_eager_fallbacks = 0
        self._call_counter = 0


# Global metrics collector instance
global_metrics_collector = MetricsCollector()
