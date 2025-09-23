# metric_eval.py
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List

from metrics.base_metric import BaseMetric


class MetricEval:
    def __init__(self, metrics: List[BaseMetric], weights: Dict[str, float]):
        self.metrics = metrics
        self.weights = weights

    def evaluateAll(self, repo_cxt) -> Dict[str, float]:
        def safe_eval(metric):
            try:
                return (metric.name, metric.evaluate(repo_cxt))
            except Exception as e:
                print(f"Error evaluating {metric.name}: {e}")
                return (metric.name, -1)

        with ThreadPoolExecutor() as executor:
            return dict(executor.map(safe_eval, self.metrics))

    def aggregateScores(self, scores: Dict[str, float]) -> float:
        total_weight = sum(self.weights.get(name, 0.0) for name in scores)
        if total_weight <= 0.0:
            return 0.0
        weighted_sum = sum(
            scores[name] * self.weights.get(name, 0.0) for name in scores
        )
        return max(0.0, min(1.0, weighted_sum / total_weight))


def init_metrics() -> List[BaseMetric]:
    """Load only concrete metric classes, skipping any that fail to import."""
    specs = [
        ("BusFactor", "metrics.bus_factor_metric", "BusFactorMetric"),
        ("CodeQuality", "metrics.code_quality_metric", "CodeQualityMetric"),
        (
            "CommunityRating",
            "metrics.community_rating_metric",
            "CommunityRatingMetric",
        ),
        (
            "DatasetAvailability",
            "metrics.dataset_availability_metric",
            "DatasetAvailabilityMetric",
        ),
        (
            "DatasetQuality",
            "metrics.dataset_quality_metric",
            "DatasetQualityMetric",
        ),
        ("License", "metrics.license_metric", "LicenseMetric"),
        (
            "PerformanceClaims",
            "metrics.performance_claims_metric",
            "PerformanceClaimsMetric",
        ),
        ("RampUpTime", "metrics.ramp_up_time_metric", "RampUpTimeMetric"),
        ("Size", "metrics.size_metric", "SizeMetric"),
    ]

    out: List[BaseMetric] = []
    for expected_name, mod_path, cls_name in specs:
        try:
            mod = __import__(mod_path, fromlist=[cls_name])
            cls = getattr(mod, cls_name)
            m: BaseMetric = cls()  # type: ignore
            if getattr(m, "name", None) != expected_name:
                raise ValueError(
                    f"Metric '{cls_name}' name='{getattr(m, 'name', None)}' "
                    f"does not match expected '{expected_name}'"
                )
            out.append(m)
        except Exception as e:
            # Don't crash the app; just skip the broken metric
            print(f"[WARN] skipping metric {expected_name}: {e}")
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
