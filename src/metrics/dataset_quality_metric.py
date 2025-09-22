"""
Dataset Quality Metric for evaluating dataset quality using LLM analysis.
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


class DatasetQualityMetric(BaseMetric):
    """
    Metric to evaluate dataset quality using LLM analysis.

    Uses LLM to analyze documentation and metadata to assess dataset quality,
    diversity, completeness, and overall suitability for AI/ML training.
    """

    def __init__(self, weight: float = 0.15):
        super().__init__(name="DatasetQuality", weight=weight)
        self.llm_analyzer = LLMAnalyzer() if LLMAnalyzer else None

    def evaluate(self, repo_context: dict) -> float:
        """
        Evaluate dataset quality using LLM analysis of documentation.

        Args:
            repo_context (dict): Dictionary containing repository information
                               including README content and metadata.

        Returns:
            float: Score between 0.0 and 1.0 from LLM analysis of dataset
                  quality indicators including documentation, validation,
                  diversity, and completeness.
        """
        readme_content = repo_context.get('readme_text', '')
        
        # Construct metadata from available sources
        metadata = {}
        
        # Add card_data if available
        if repo_context.get('card_data'):
            metadata.update(repo_context.get('card_data', {}))
        
        # Add model_index if available
        if repo_context.get('model_index'):
            metadata.update(repo_context.get('model_index', {}))

        if self.llm_analyzer and readme_content:
            return self.llm_analyzer.analyze_dataset_quality(
                readme_content, metadata)

        # Fallback to basic analysis if LLM unavailable
        return self._fallback_analysis(repo_context)

    def _fallback_analysis(self, repo_context: dict) -> float:
        """Fallback analysis when LLM is unavailable."""
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
        return "Evaluates dataset quality using LLM analysis"
