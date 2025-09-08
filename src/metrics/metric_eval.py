from concurrent.futures import ThreadPoolExecutor
from typing import List, Dict
from .base_metric import BaseMetric


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
