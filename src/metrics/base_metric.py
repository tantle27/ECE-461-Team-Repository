from abc import ABC, abstractmethod


class BaseMetric(ABC):
    """
    Abstract base class for all metrics when evaluating AI/ML models.

    This class defines the interface that all specific metrics must implement
    to evaluate different aspects of models from Hugging Face or other sources.
    """

    def __init__(self, name: str, weight: float = 0.0):
        """
        Initialize the base metric.

        Args:
            name (str): Name of the metric
            weight (float): Weight of this metric (default: 0.0)
        """
        self.name = name
        self.weight = weight

    @abstractmethod
    def evaluate(self, model_info: dict) -> float:
        """
        Evaluate the metric for a given model.

        Args:
            model_info (dict): Dictionary containing model info including
                             URL, repository data, documentation, etc.

        Returns:
            float: Score between 0.0 and 1.0, where 1.0 is the best score
        """
        pass

    def __str__(self) -> str:
        """String representation of the metric."""
        return f"{self.name} (weight: {self.weight})"
