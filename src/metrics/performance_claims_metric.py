"""
Performance Claims Metric for evaluating verification of performance claims.
"""

from .base_metric import BaseMetric


class PerformanceClaimsMetric(BaseMetric):
    """
    Metric to evaluate the verification of performance claims and benchmarks.

    This metric checks if the model's performance claims are backed by
    verifiable benchmarks and proper documentation.
    """

    def __init__(self, weight: float = 0.15):
        super().__init__(name="PerformanceClaims", weight=weight)

    def evaluate(self, repo_context: dict) -> float:
        """
        Evaluate the performance claims verification for a given repository.

        Args:
            repo_context (dict): Dictionary containing repository information
                               including benchmark and performance data.

        Returns:
            float: Score between 0.0 and 1.0, where 1.0 indicates fully
                  verified performance claims and 0.0 indicates no
                  verification.
        """
        has_benchmarks = repo_context.get('has_benchmarks', False)
        benchmark_results = repo_context.get('benchmark_results', [])
        has_performance_docs = repo_context.get('has_performance_docs', False)
        claims_verified = repo_context.get('claims_verified', False)

        score = 0.0

        if has_benchmarks:
            score += 0.3

        if benchmark_results:
            score += min(0.3, len(benchmark_results) * 0.1)

        if has_performance_docs:
            score += 0.2

        if claims_verified:
            score += 0.2

        return min(1.0, score)

    def get_description(self) -> str:
        """Get description of the metric."""
        return "Evaluates verification of performance claims and benchmarks"
