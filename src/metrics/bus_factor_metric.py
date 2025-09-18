"""
Bus Factor Metric for evaluating project sustainability.
"""

from .base_metric import BaseMetric


class BusFactorMetric(BaseMetric):
    """
    Metric to evaluate the bus factor of a project.

    The bus factor is defined as the minimum number of key developers
    that need to be incapacitated (e.g., hit by a bus) before the project
    is in serious trouble. A higher bus factor indicates better project
    sustainability and lower risk.
    """

    def __init__(self, weight: float = 0.15):
        super().__init__(name="BusFactor", weight=weight)

    def evaluate(self, repo_context: dict) -> float:
        """
        Evaluate the bus factor for a given model repository.

        Args:
            repo_context (dict): Dictionary containing repository information
                               including contributor data.

        Returns:
            float: Score between 0.0 and 1.0, where 1.0 indicates a
                  high bus factor and 0.0 indicates a low bus factor.
        """
        contributors = repo_context.get('contributors', [])
        if not contributors:
            return 0.0

        total_contributions = sum(
            c.get('contributions', 0) for c in contributors
        )
        if total_contributions == 0:
            return 0.0

        # Calculate contribution distribution
        max_contribution = max(c.get('contributions', 0) for c in contributors)
        concentration = max_contribution / total_contributions

        # Higher concentration = lower bus factor = lower score
        return max(0.0, 1.0 - concentration)

    def get_description(self) -> str:
        """Get description of the metric."""
        return ("Evaluates project sustainability based on "
                "contributor diversity")
