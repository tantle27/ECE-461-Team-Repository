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
                               including files data for size calculation.

        Returns:
            float: Score between 0.0 and 1.0 based on device compatibility:
                  - <2GB: 1.0 (fits all devices)
                  - 2-16GB: 1.0 to 0.5 (PC compatible)
                  - 16-512GB: 0.5 to 0.0 (cloud only)
                  - >512GB: 0.0 (impractical)
        """
        # Calculate size from files or use total_weight_bytes if available
        if repo_context.get('total_weight_bytes') is not None:
            size_bytes = repo_context.get('total_weight_bytes', 0)
        else:
            # Sum up file sizes from files list
            files = repo_context.get('files', [])
            size_bytes = 0
            if files:
                size_bytes = sum(f.get('size_bytes', 0) for f in files)
        
        size_gb = size_bytes / (1024**3)

        if size_gb < 2:
            return 1.0
        elif size_gb <= 16:
            # Formula: 1 - 0.5((s-2)/14)
            return 1.0 - 0.5 * ((size_gb - 2) / 14)
        elif size_gb <= 512:
            # Formula: 0.5 - 0.5((s-16)/496)
            return 0.5 - 0.5 * ((size_gb - 16) / 496)
        else:
            return 0.0

    def get_description(self) -> str:
        """Get description of the metric."""
        return "Evaluates model size impact on usability"
