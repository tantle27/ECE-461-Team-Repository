"""
Code Quality Metric for evaluating code quality using LLM analysis.
"""

from .base_metric import BaseMetric
import os
import sys

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


try:
    from utils.llm_analyzer import LLMAnalyzer
except ImportError:
    LLMAnalyzer = None


class CodeQualityMetric(BaseMetric):
    """
    Metric to evaluate code quality using LLM analysis.

    Uses LLM to analyze code structure, maintainability, documentation,
    and best practices to assess overall code quality.
    """

    def __init__(self, weight: float = 0.2):
        super().__init__(name="CodeQuality", weight=weight)
        self.llm_analyzer = LLMAnalyzer() if LLMAnalyzer else None

    def evaluate(self, repo_context: dict) -> float:
        """
        Evaluate code quality using LLM analysis of code and documentation.

        Args:
            repo_context (dict): Dictionary containing repository information
                               including code content and README.

        Returns:
            float: Score between 0.0 and 1.0 from LLM analysis of code
                  quality, maintainability, and best practices.
        """
        code_content = repo_context.get('code_content', '')
        readme_content = repo_context.get('readme_content', '')

        if self.llm_analyzer and code_content:
            return self.llm_analyzer.analyze_code_quality(
                code_content, readme_content)

        # Fallback to basic analysis if LLM unavailable
        return self._fallback_analysis(repo_context)

    def _fallback_analysis(self, repo_context: dict) -> float:
        """Fallback analysis when LLM is unavailable."""
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
        return "Evaluates code quality using LLM analysis"
