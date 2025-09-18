"""
License Metric for evaluating license compatibility.
"""

from .base_metric import BaseMetric


class LicenseMetric(BaseMetric):
    """
    Metric to evaluate the license compatibility for commercial use.

    This metric checks if the model's license is compatible with
    commercial applications and deployment.
    """

    COMPATIBLE_LICENSES = ['mit', 'apache-2.0', 'bsd-3-clause', 'unlicense']

    def __init__(self, weight: float = 0.1):
        super().__init__(name="License", weight=weight)

    def evaluate(self, repo_context: dict) -> float:
        """
        Evaluate the license compatibility for a given model repository.

        Args:
            repo_context (dict): Dictionary containing repository information
                               including license data.

        Returns:
            float: Score between 0.0 and 1.0, where 1.0 indicates a fully
                  compatible license and 0.0 indicates incompatible license.
        """
        license_name = repo_context.get('license', '').lower()
        return 1.0 if license_name in self.COMPATIBLE_LICENSES else 0.0

    def get_description(self) -> str:
        """Get description of the metric."""
        return "Checks license compatibility for commercial use"
