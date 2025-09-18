"""
Code Quality Metric for evaluating code quality through testing and analysis.
"""

from .base_metric import BaseMetric


class CodeQualityMetric(BaseMetric):
    """
    Metric to evaluate the code quality of AI/ML model repositories.

    This metric considers testing coverage, linting, and code complexity
    to assess overall code quality and maintainability.
    """

    def __init__(self, weight: float = 0.2):
        super().__init__(name="CodeQuality", weight=weight)

    def evaluate(self, repo_context: dict) -> float:
        """
        Evaluate the code quality for a given model repository.

        Args:
            repo_context (dict): Dictionary containing repository information
                               including testing and code quality metrics.

        Returns:
            float: Score between 0.0 and 1.0, where 1.0 indicates excellent
                  code quality and 0.0 indicates poor quality.
        """
        has_tests = repo_context.get('has_tests', False)
        test_coverage = repo_context.get('test_coverage', 0.0)
        has_linting = repo_context.get('has_linting', False)
        code_complexity = repo_context.get('code_complexity', 10.0)

        score = 0.0

        if has_tests:
            score += 0.3

        score += (test_coverage / 100.0) * 0.3

        if has_linting:
            score += 0.2

        # Lower complexity is better (invert and normalize)
        complexity_score = max(0, 1 - (code_complexity / 20.0))
        score += complexity_score * 0.2

        return min(1.0, score)

    def get_description(self) -> str:
        """Get description of the metric."""
        return "Evaluates code quality through testing and static analysis"
