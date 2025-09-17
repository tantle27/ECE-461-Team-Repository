from concurrent.futures import ThreadPoolExecutor
from typing import List, Dict
from .base_metric import BaseMetric

from metrics import base_metric, bus_factor_metric, code_quality_metric, community_rating_metric, dataset_availability_metric, dataset_quality_metric, license_metric, performance_claims_metric, ramp_up_time_metric, size_metric


class MetricEval:
    def __init__(self, metrics: List[BaseMetric], weights: Dict[str, float]):
        """
        :param metrics: list of BaseMetric subclasses
        :param weights: dict mapping metric name -> weight
        """
        self.metrics = metrics
        self.weights = weights

    def evaluateAll(self, repo_cxt) -> Dict[str, float]:
        """
        Run all metrics in parallel and return raw scores.
        Each metric is evaluated safely with exception handling.
        """

        def safe_eval(metric):
            try:
                return (metric.name, metric.evaluate(repo_cxt))
            except Exception as e:
                print(f"Error evaluating {metric.name}: {e}")
                return (metric.name, -1)  # failing score

        with ThreadPoolExecutor() as executor:
            results = dict(
                executor.map(lambda m: safe_eval(m, repo_cxt), self.metrics)
            )

        return results

    def aggregateScores(self, scores: Dict[str, float]) -> float:
        """
        Aggregate individual metric scores into a final score using weights.

        Args:
            scores (Dict[str, float]): Dictionary of metric name to score.

        Returns:
            float: Final aggregated score between 0.0 and 1.0.
        """
        total_weight = sum(self.weights.get(name, 0) for name in scores.keys())
        if total_weight == 0:
            return 0.0

        weighted_sum = sum(
            scores[name] * self.weights.get(name, 0)
            for name in scores.keys() if name in self.weights
        )

        final_score = weighted_sum / total_weight
        return max(0.0, min(1.0, final_score))


def init_metrics() -> List[BaseMetric]:
    metric_classes = [
        base_metric.BaseMetric,
        bus_factor_metric.BusFactorMetric,
        code_quality_metric.CodeQualityMetric,
        community_rating_metric.CommunityRatingMetric,
        dataset_availability_metric.DatasetAvailabilityMetric,
        dataset_quality_metric.DatasetQualityMetric,
        license_metric.LicenseMetric,
        performance_claims_metric.PerformanceClaimsMetric,
        ramp_up_time_metric.RampUpTimeMetric,
        size_metric.SizeMetric,
    ]
    
    return [cls() for cls in metric_classes]
    

def init_weights() -> Dict[str, float]:
    return {
        "Bus Factor": 0.15,
        "Code Quality": 0.15,
        "Community Rating": 0.15,
        "Dataset Availability": 0.10,
        "Dataset Quality": 0.10,
        "License": 0.10,
        "Performance Claims": 0.10,
        "Ramp Up Time": 0.10,
        "Size": 0.05,
    }
