"""
License Metric for evaluating license compatibility.
"""

from .base_metric import BaseMetric


class LicenseMetric(BaseMetric):
    """
    Metric to evaluate license compatibility with ACME's LGPLv2.1.

    This metric checks license compatibility and assigns scores based on
    permissiveness levels for compatible licenses.
    """

    # Compatible licenses with permissiveness scores
    LICENSE_SCORES = {
        'public domain': 1.0,
        'unlicense': 1.0,
        'mit': 1.0,
        'bsd-3-clause': 0.8,
        'bsd-2-clause': 0.8,
        'apache-2.0': 0.6,
        'apache': 0.6,
        'mpl-2.0': 0.4,
        'mozilla': 0.4,
        'lgpl-2.1': 0.2,
        'lgplv2.1': 0.2
    }

    # Incompatible licenses (score 0.0)
    INCOMPATIBLE_LICENSES = [
        'gpl', 'gpl-2.0', 'gpl-3.0', 'gplv2', 'gplv3',
        'agpl', 'agpl-3.0', 'agplv3',
        'lgpl-3.0', 'lgplv3',
        'copyleft'
    ]

    def __init__(self, weight: float = 0.1):
        super().__init__(name="License", weight=weight)

    def evaluate(self, repo_context: dict) -> float:
        """
        Evaluate license compatibility with ACME's LGPLv2.1.

        Args:
            repo_context (dict): Dictionary containing repository information
                               including license data.

        Returns:
            float: Score based on license compatibility:
                  - 1.0: Public domain, MIT
                  - 0.8: BSD-3-Clause
                  - 0.6: Apache-2.0
                  - 0.4: MPL-2.0
                  - 0.2: LGPLv2.1
                  - 0.0: Incompatible (GPL, AGPL, etc.)
        """
        license_name = repo_context.get('license', '').lower().strip()

        if not license_name:
            return 0.0  # No license specified

        # Check for direct matches in compatible licenses
        if license_name in self.LICENSE_SCORES:
            return self.LICENSE_SCORES[license_name]

        # Check for incompatible licenses
        for incompatible in self.INCOMPATIBLE_LICENSES:
            if incompatible in license_name:
                return 0.0

        # Check for partial matches in compatible licenses
        for license_key, score in self.LICENSE_SCORES.items():
            if license_key in license_name:
                return score

        # Default to incompatible if unknown
        return 0.0

    def get_description(self) -> str:
        """Get description of the metric."""
        return "Evaluates license compatibility with ACME's LGPLv2.1"
