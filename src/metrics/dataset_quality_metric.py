"""
Dataset Quality Metric for evaluating dataset quality and diversity.
"""

from .base_metric import BaseMetric


class DatasetQualityMetric(BaseMetric):
    """
    Metric to evaluate the quality and diversity of datasets used in
    training or fine-tuning AI/ML models.

    This metric considers data validation, diversity, and completeness.
    """

    def __init__(self, weight: float = 0.15):
        super().__init__(name="DatasetQuality", weight=weight)

    def evaluate(self, repo_context: dict) -> float:
        """
        Evaluate the dataset quality for a given model repository.

        Args:
            repo_context (dict): Dictionary containing repository information
                               including dataset quality metrics.

        Returns:
            float: Score between 0.0 and 1.0, where 1.0 indicates excellent
                  dataset quality and 0.0 indicates poor quality.
        """
        has_validation = repo_context.get('has_data_validation', False)
        data_diversity = repo_context.get('data_diversity_score', 0.0)
        data_completeness = repo_context.get('data_completeness', 0.0)

        score = 0.0

        if has_validation:
            score += 0.4

        score += data_diversity * 0.3
        score += data_completeness * 0.3

        return min(1.0, score)

    def get_description(self) -> str:
        """Get description of the metric."""
        return "Evaluates quality and diversity of training datasets"
