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
        Evaluate the ramp-up time using LLM analysis of documentation.

        Args:
            repo_context (dict): Dictionary containing repository information
                               including README content and documentation.

        Returns:
            float: Score based on documentation quality:
                  - 1.0: External docs + long how-to + examples
                  - 0.75: Documentation + some how-to + examples
                  - 0.5: Documentation + some how-to OR examples
                  - 0.25: Basic documentation present
                  - 0.0: No documentation
        """
        readme_content = repo_context.get('readme_content', '').lower()
        has_external_docs = repo_context.get('has_documentation', False)
        
        if not readme_content and not has_external_docs:
            return 0.0
        
        # Analyze README content for documentation quality
        score = 0.0
        
        # Base documentation presence
        if readme_content or has_external_docs:
            score = 0.25
        
        # Check for how-to sections
        how_to_indicators = ['how to', 'tutorial', 'guide', 'getting started',
                             'quickstart', 'setup', 'installation']
        has_how_to = any(indicator in readme_content
                         for indicator in how_to_indicators)
        
        # Check for examples
        example_indicators = ['example', 'demo', 'sample', 'usage',
                              'code example', '```']
        has_examples = any(indicator in readme_content
                           for indicator in example_indicators)
        
        # Check for extensive documentation
        extensive_indicators = ['detailed', 'comprehensive', 'documentation',
                                'reference', 'api', 'troubleshooting']
        has_extensive = any(indicator in readme_content
                            for indicator in extensive_indicators)
        
        # Calculate score based on presence
        if has_external_docs and has_extensive and has_how_to and has_examples:
            score = 1.0  # Full documentation
        elif ((has_external_docs or has_extensive) and
              has_how_to and has_examples):
            score = 0.75  # Good documentation
        elif has_how_to or has_examples:
            score = 0.5  # Some documentation
        elif readme_content or has_external_docs:
            score = 0.25  # Basic documentation
        
        return score

    def get_description(self) -> str:
        """Get description of the metric."""
        return "Evaluates ease of getting started with the model"
