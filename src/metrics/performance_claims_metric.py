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
        Evaluate performance claims by analyzing README for benchmarks.

        Args:
            repo_context (dict): Dictionary containing repository information
                               including README content and benchmark data.

        Returns:
            float: Score between 0.0 and 1.0:
                  - 0.0: No evaluation section
                  - 0.0-1.0: Based on benchmark performance scores
                            (averages scores and divides by 100)
        """
        readme_content = repo_context.get('readme_text', '').lower()
        
        # Extract benchmark scores from metadata or model card
        benchmark_scores = []
        
        # Check card_data for benchmark information
        card_data = repo_context.get('card_data', {})
        if card_data and isinstance(card_data, dict):
            # Extract benchmark scores from card_data if available
            if 'benchmarks' in card_data:
                benchmarks = card_data.get('benchmarks', [])
                if isinstance(benchmarks, list):
                    for benchmark in benchmarks:
                        if (isinstance(benchmark, dict) and
                                'score' in benchmark):
                            benchmark_scores.append(benchmark.get('score', 0))
        
        # Check for evaluation/benchmark sections in README
        eval_indicators = ['evaluation', 'benchmark', 'performance',
                           'results', 'metrics', 'accuracy', 'f1', 'bleu']
        has_eval_section = any(indicator in readme_content
                               for indicator in eval_indicators)
        
        if not has_eval_section:
            return 0.0
        
        # If specific benchmark scores are provided, use them
        if benchmark_scores:
            # Average the scores and normalize to 0-1 scale
            avg_score = sum(benchmark_scores) / len(benchmark_scores)
            return min(1.0, avg_score / 100.0)
        
        # Fallback: estimate based on keywords in README
        performance_keywords = ['state-of-the-art', 'sota', 'best',
                                'high accuracy', 'excellent', 'superior']
        keyword_count = sum(1 for kw in performance_keywords
                            if kw in readme_content)
        
        # Base score for having evaluation section
        score = 0.2
        
        # Additional score based on performance claims
        score += min(0.8, keyword_count * 0.2)
        
        return min(1.0, score)

    def get_description(self) -> str:
        """Get description of the metric."""
        return "Evaluates verification of performance claims and benchmarks"
