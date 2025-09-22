"""
Dataset Availability Metric for evaluating dataset accessibility.
"""

from .base_metric import BaseMetric


class DatasetAvailabilityMetric(BaseMetric):
    """
    Metric to evaluate the availability of datasets used in training or
    fine-tuning AI/ML models.

    This metric checks if the datasets referenced in the model's documentation
    or repository are publicly accessible and properly cited.
    """

    def __init__(self, weight: float = 0.1):
        super().__init__(name="DatasetAvailability", weight=weight)

    def evaluate(self, repo_context: dict) -> float:
        """
        Evaluate dataset availability and code documentation (ADACS).

        Args:
            repo_context (dict): Dictionary containing repository information
                               including dataset and training documentation.

        Returns:
            float: Score based on availability and documentation:
                  - 0.0: No dataset available
                  - 0.33: Dataset available
                  - 0.67: Dataset available + well documented OR
                          training documented
                  - 1.0: Dataset available + well documented AND
                         training documented
        """
        has_dataset = repo_context.get('has_dataset', False)
        dataset_documented = repo_context.get('dataset_documented', False)
        training_documented = repo_context.get('training_documented', False)
        
        # Check README for dataset/training documentation
        readme_content = repo_context.get('readme_content', '').lower()
        
        # Analyze README for dataset documentation
        dataset_indicators = ['dataset', 'data', 'training data', 'corpus']
        has_dataset_in_readme = any(indicator in readme_content
                                    for indicator in dataset_indicators)
        
        # Analyze README for training documentation
        training_indicators = ['training', 'fine-tuning', 'model training',
                               'train', 'training procedure', 'training setup']
        has_training_in_readme = any(indicator in readme_content
                                     for indicator in training_indicators)
        
        # Combine explicit flags with README analysis
        dataset_well_documented = (dataset_documented or
                                   has_dataset_in_readme)
        training_well_documented = (training_documented or
                                    has_training_in_readme)
        
        if not has_dataset:
            return 0.0
        
        if dataset_well_documented and training_well_documented:
            return 1.0  # Both documented
        elif dataset_well_documented or training_well_documented:
            return 0.67  # One documented
        else:
            return 0.33  # Available but not documented

    def get_description(self) -> str:
        """Get description of the metric."""
        return "Evaluates availability and accessibility of training datasets"
