"""
Unit tests for the BaseMetric abstract class.
"""

import pytest
import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.metrics.base_metric import BaseMetric


class ConcreteMetric(BaseMetric):
    """Concrete implementation of BaseMetric for testing."""

    def __init__(self, name: str = "test_metric", weight: float = 0.5):
        super().__init__(name, weight)

    def evaluate(self, repo_context: dict) -> float:
        """Simple evaluation that returns a fixed score for testing."""
        return 0.8

    def get_description(self) -> str:
        """Return test description."""
        return "Test metric for unit testing"


class TestBaseMetric:
    """Test suite for BaseMetric class."""

    def test_base_metric_cannot_be_instantiated(self):
        """Test that BaseMetric cannot be instantiated directly."""
        with pytest.raises(TypeError):
            BaseMetric("test", 0.5)

    def test_concrete_metric_initialization(self):
        """Test that concrete metric can be initialized properly."""
        metric = ConcreteMetric("test_metric", 0.7)
        assert metric.name == "test_metric"
        assert metric.weight == 0.7

    def test_default_weight(self):
        """Test that default weight is set correctly."""
        metric = ConcreteMetric("test_metric")
        assert metric.weight == 0.5  # Default from ConcreteMetric

    def test_zero_weight(self):
        """Test metric with zero weight."""
        metric = ConcreteMetric("test_metric", 0.0)
        assert metric.weight == 0.0

    def test_negative_weight(self):
        """Test metric with negative weight."""
        metric = ConcreteMetric("test_metric", -0.1)
        assert metric.weight == -0.1

    def test_weight_greater_than_one(self):
        """Test metric with weight greater than 1."""
        metric = ConcreteMetric("test_metric", 1.5)
        assert metric.weight == 1.5

    def test_evaluate_method(self):
        """Test that evaluate method works correctly."""
        metric = ConcreteMetric()
        test_data = {"test": "data"}
        result = metric.evaluate(test_data)
        assert result == 0.8

    def test_get_description_method(self):
        """Test that get_description method works correctly."""
        metric = ConcreteMetric()
        description = metric.get_description()
        assert description == "Test metric for unit testing"

    def test_str_representation(self):
        """Test string representation of metric."""
        metric = ConcreteMetric("my_metric", 0.3)
        expected = "my_metric (weight: 0.3)"
        assert str(metric) == expected

    def test_name_attribute(self):
        """Test that name attribute is accessible."""
        metric = ConcreteMetric("custom_name", 0.5)
        assert metric.name == "custom_name"

    def test_weight_attribute(self):
        """Test that weight attribute is accessible."""
        metric = ConcreteMetric("test", 0.75)
        assert metric.weight == 0.75


if __name__ == "__main__":
    pytest.main([__file__])
