"""
Ramp-up Time Metric for evaluating ease of getting started.
"""

from .base_metric import BaseMetric


class RampUpTimeMetric(BaseMetric):
    """
    Metric to evaluate the ramp-up time for developers to get started
    with an AI/ML model.

    This metric considers documentation quality, examples availability,
    and overall ease of understanding the model.
    """

    def __init__(self, weight: float = 0.15):
        super().__init__(name="RampUpTime", weight=weight)

    def evaluate(self, repo_context: dict) -> float:
        """
        Evaluate the ramp-up time for a given model repository.

        Args:
            repo_context (dict): Dictionary containing repository information
                               including documentation and example data.

        Returns:
            float: Score between 0.0 and 1.0, where 1.0 indicates excellent
                  documentation and examples, and 0.0 indicates poor ramp-up.
        """
        has_readme = repo_context.get('has_readme', False)
        has_examples = repo_context.get('has_examples', False)
        has_docs = repo_context.get('has_documentation', False)

        score = 0.0
        if has_readme:
            score += 0.4
        if has_examples:
            score += 0.4
        if has_docs:
            score += 0.2

        return min(1.0, score)

    def get_description(self) -> str:
        """Get description of the metric."""
        return "Evaluates ease of getting started with the model"
