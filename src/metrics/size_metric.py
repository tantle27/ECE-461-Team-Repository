"""
Size Metric for evaluating model size impact on usability.
"""

from .base_metric import BaseMetric


class SizeMetric(BaseMetric):
    """
    Metric to evaluate the size impact of AI/ML models on usability.

    Smaller models are generally more usable for deployment,
    especially in resource-constrained environments.
    """

    def __init__(self, weight: float = 0.1):
        super().__init__(name="Size", weight=weight)

    def evaluate(self, repo_context: dict) -> float:
        """
        Evaluate the size impact for a given model repository.

        Args:
            repo_context (dict): Dictionary containing repository information
                               including size data.

        Returns:
            float: Score between 0.0 and 1.0, where 1.0 indicates optimal size
                  (≤2GB) and 0.0 indicates very large size (>16GB).
        """
        size_gb = repo_context.get('size', 0) / (1024**3)

        if size_gb <= 2:
            return 1.0
        elif size_gb <= 16:
            return max(0, 1 - (size_gb - 2) / 14)
        else:
            return 0.0

    def get_description(self) -> str:
        """Get description of the metric."""
        return "Evaluates model size impact on usability"
