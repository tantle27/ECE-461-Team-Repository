"""
Dataset Availability Metric for evaluating dataset accessibility.
"""

from .base_metric import BaseMetric


class DatasetAvailabilityMetric(BaseMetric):
    """
    Metric to evaluate the availability of datasets used in training or
    fine-tuning AI/ML models.

    This metric checks if the datasets referenced in the model's documentation
    or repository are publicly accessible and properly cited.
    """

    def __init__(self, weight: float = 0.1):
        super().__init__(name="DatasetAvailability", weight=weight)

    def evaluate(self, repo_context: dict) -> float:
        """
        Evaluate the dataset availability for a given model repository.

        Args:
            repo_context (dict): Dictionary containing repository information
                               including dataset information.

        Returns:
            float: Score between 0.0 and 1.0, where 1.0 indicates all datasets
                  are publicly available and properly cited, and 0.0 indicates
                  none are available.
        """
        has_dataset = repo_context.get('has_dataset', False)
        dataset_accessible = repo_context.get('dataset_accessible', False)
        dataset_size = repo_context.get('dataset_size', 0)

        if not has_dataset:
            return 0.0

        score = 0.5  # Base score for having dataset

        if dataset_accessible:
            score += 0.3

        if dataset_size > 1000:  # Substantial dataset
            score += 0.2

        return min(1.0, score)

    def get_description(self) -> str:
        """Get description of the metric."""
        return "Evaluates availability and accessibility of training datasets"
