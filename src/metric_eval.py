# metric_eval.py
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from time import perf_counter_ns
from typing import Dict, List, Tuple
from metrics.base_metric import BaseMetric
import math
import time

class MetricEval:
    def __init__(self, metrics: List[BaseMetric], weights: Dict[str, float]):
        self.metrics = metrics
        self.weights = weights

    def evaluate_all_timed(self, repo_cxt) -> Tuple[Dict[str, float], Dict[str, int]]:
        """
        Threaded evaluation. Returns (scores, latencies_ms).
        Exceptions or invalid values are normalized to 0.0.
        """
        scores: Dict[str, float] = {}
        lats: Dict[str, int] = {}

        def timed(metric: BaseMetric):
            t0 = time.perf_counter_ns()
            try:
                val = metric.evaluate(repo_cxt)
            except Exception:
                val = 0.0
            dur_ms = max(1, int((time.perf_counter_ns() - t0) // 1_000_000))
            # clamp val to [0,1] and handle NaNs/inf/negatives
            try:
                v = float(val)
            except Exception:
                v = 0.0
            if not math.isfinite(v) or v < 0.0:
                v = 0.0
            elif v > 1.0:
                v = 1.0
            return metric.name, v, dur_ms

        with ThreadPoolExecutor() as ex:
            futs = [ex.submit(timed, m) for m in self.metrics]
            for fut in as_completed(futs):
                name, v, ms = fut.result()
                scores[name] = v
                lats[name] = ms

        return scores, lats

    # Backward-compatible wrapper used by some tests
    def evaluateAll(self, repo_cxt) -> Dict[str, float]:
        scores, _ = self.evaluate_all_timed(repo_cxt)
        return scores

    def aggregateScores(self, scores: Dict[str, float]) -> float:
        total_weight = sum(self.weights.get(name, 0.0) for name in scores)
        if total_weight <= 0.0:
            return 0.0
        weighted_sum = sum(scores[name] * self.weights.get(name, 0.0) for name in scores)
        return max(0.0, min(1.0, weighted_sum / total_weight))

def init_metrics() -> List[BaseMetric]:
    """Load only concrete metric classes, skipping any that fail to import."""
    specs = [
        ("BusFactor", "metrics.bus_factor_metric", "BusFactorMetric"),
        ("CodeQuality", "metrics.code_quality_metric", "CodeQualityMetric"),
        ("CommunityRating", "metrics.community_rating_metric", "CommunityRatingMetric"),
        ("DatasetAvailability", "metrics.dataset_availability_metric", "DatasetAvailabilityMetric"),
        ("DatasetQuality", "metrics.dataset_quality_metric", "DatasetQualityMetric"),
        ("License", "metrics.license_metric", "LicenseMetric"),
        ("PerformanceClaims", "metrics.performance_claims_metric", "PerformanceClaimsMetric"),
        ("RampUpTime", "metrics.ramp_up_time_metric", "RampUpTimeMetric"),
        ("Size", "metrics.size_metric", "SizeMetric"),
    ]

    out: List[BaseMetric] = []
    for expected_name, mod_path, cls_name in specs:
        try:
            mod = __import__(mod_path, fromlist=[cls_name])
            cls = getattr(mod, cls_name)
            m: BaseMetric = cls()
            if getattr(m, "name", None) != expected_name:
                raise ValueError(
                    f"Metric '{cls_name}' name='{getattr(m, 'name', None)}' "
                    f"does not match expected '{expected_name}'"
                )
            out.append(m)
        except Exception:
            pass
    return out


def init_weights() -> Dict[str, float]:
    # Keys MUST match metric.name exactly.
    return {
        "BusFactor": 0.15,
        "CodeQuality": 0.15,
        "CommunityRating": 0.15,
        "DatasetAvailability": 0.10,
        "DatasetQuality": 0.10,
        "License": 0.10,
        "PerformanceClaims": 0.10,
        "RampUpTime": 0.10,
        "Size": 0.05,
    }