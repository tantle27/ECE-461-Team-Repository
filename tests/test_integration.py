"""
Integration tests for the complete evaluation system.
"""

import pytest
import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.metrics.base_metric import BaseMetric
from src.metric_eval import MetricEval
from src.metrics.community_rating_metric import CommunityRatingMetric


class SimpleTestMetric(BaseMetric):
    """Simple metric for integration testing."""

    def __init__(self, name: str, weight: float = 0.5):
        super().__init__(name, weight)

    def evaluate(self, repo_context: dict) -> float:
        """Return a score based on model name length."""
        name_length = len(repo_context.get("model_name", ""))
        return min(1.0, name_length / 20.0)  # Normalize by 20 chars

    def get_description(self) -> str:
        return "Test metric based on model name length"


class TestIntegrationMetricsSystem:
    """Integration tests for the complete metrics system."""

    def test_full_evaluation_pipeline(self):
        """Test complete evaluation pipeline with multiple metrics."""
        # Create metrics
        community_metric = CommunityRatingMetric(weight=0.4)
        simple_metric = SimpleTestMetric("name_length", weight=0.6)

        # Create evaluator
        metrics = [community_metric, simple_metric]
        weights = {
            "CommunityRating": 0.4,
            "name_length": 0.6
        }
        evaluator = MetricEval(metrics, weights)

        # Create test data
        repo_context = {
            "model_name": "bert-base-uncased",
            "hf_likes": 250,
            "hf_downloads": 15000
        }

        # Run evaluation
        scores = evaluator.evaluateAll(repo_context)
        final_score = evaluator.aggregateScores(scores)

        # Verify results
        assert len(scores) == 2
        assert "CommunityRating" in scores
        assert "name_length" in scores

        # Verify scores are in valid range
        for metric_name, score in scores.items():
            assert 0.0 <= score <= 1.0, (
                f"{metric_name} score {score} out of range"
            )

        # Verify final score
        assert 0.0 <= final_score <= 1.0

        # Manual verification of calculation
        community_score = scores["CommunityRating"]
        name_score = scores["name_length"]
        expected_final = (community_score * 0.4 + name_score * 0.6) / 1.0
        assert abs(final_score - expected_final) < 0.001

    def test_multiple_models_evaluation(self):
        """Test evaluating multiple models."""
        # Create evaluator
        metrics = [
            CommunityRatingMetric(weight=0.5),
            SimpleTestMetric("test_metric", weight=0.5)
        ]
        weights = {"CommunityRating": 0.5, "test_metric": 0.5}
        evaluator = MetricEval(metrics, weights)

        # Test data for multiple models
        models = [
            {
                "model_name": "bert-base",
                "hf_likes": 50,
                "hf_downloads": 2000
            },
            {
                "model_name": "gpt-2-large",
                "hf_likes": 150,
                "hf_downloads": 5000
            },
            {
                "model_name": "t5",
                "hf_likes": 25,
                "hf_downloads": 500
            }
        ]

        # Evaluate all models
        results = []
        for model_data in models:
            scores = evaluator.evaluateAll(model_data)
            final_score = evaluator.aggregateScores(scores)
            results.append({
                "model": model_data["model_name"],
                "scores": scores,
                "final_score": final_score
            })

        # Verify all evaluations completed
        assert len(results) == 3

        # Verify all scores are valid
        for result in results:
            assert 0.0 <= result["final_score"] <= 1.0
            for score in result["scores"].values():
                assert 0.0 <= score <= 1.0

        # Popular model (gpt-2-large) should score highest on community rating
        gpt2_result = next(r for r in results if r["model"] == "gpt-2-large")
        bert_result = next(r for r in results if r["model"] == "bert-base")

        gpt2_rating = gpt2_result["scores"]["CommunityRating"]
        bert_rating = bert_result["scores"]["CommunityRating"]
        assert gpt2_rating > bert_rating

    def test_error_handling_in_pipeline(self):
        """Test that the pipeline handles errors gracefully."""

        class FailingMetric(BaseMetric):
            """Metric that always fails."""

            def __init__(self):
                super().__init__("failing_metric", 0.3)

            def evaluate(self, repo_context: dict) -> float:
                raise RuntimeError("Simulated failure")

            def get_description(self) -> str:
                return "Metric that fails for testing"

        # Create evaluator with one good and one failing metric
        metrics = [
            CommunityRatingMetric(weight=0.7),
            FailingMetric()
        ]
        weights = {"CommunityRating": 0.7, "failing_metric": 0.3}
        evaluator = MetricEval(metrics, weights)

        repo_context = {
            "hf_likes": 100,
            "hf_downloads": 5000
        }

        # Run evaluation (should not crash)
        scores = evaluator.evaluateAll(repo_context)
        final_score = evaluator.aggregateScores(scores)

        # Verify partial results
        assert len(scores) == 2
        assert scores["CommunityRating"] > 0  # Should work
        assert scores["failing_metric"] == -1  # Should fail gracefully

        # Final score should still be calculated from working metrics
        assert final_score > 0

    def test_empty_model_data(self):
        """Test evaluation with empty model data."""
        metrics = [CommunityRatingMetric()]
        weights = {"CommunityRating": 1.0}
        evaluator = MetricEval(metrics, weights)

        # Empty data
        scores = evaluator.evaluateAll({})
        final_score = evaluator.aggregateScores(scores)

        assert scores["CommunityRating"] == 0.0
        assert final_score == 0.0

    def test_metric_weight_impact(self):
        """Test that metric weights properly impact final scores."""
        # Create two identical metrics with different weights
        metric1 = SimpleTestMetric("metric1", weight=0.8)
        metric2 = SimpleTestMetric("metric2", weight=0.2)

        metrics = [metric1, metric2]
        weights = {"metric1": 0.8, "metric2": 0.2}
        evaluator = MetricEval(metrics, weights)

        repo_context = {"model_name": "test-model"}

        scores = evaluator.evaluateAll(repo_context)
        final_score = evaluator.aggregateScores(scores)

        # Both metrics should return the same score
        assert scores["metric1"] == scores["metric2"]

        # Final score should equal individual scores since weights sum to 1.0
        expected = scores["metric1"]
        assert abs(final_score - expected) < 0.001


class TestSystemPerformance:
    """Performance and scalability tests."""

    @pytest.mark.slow
    def test_large_number_of_metrics(self):
        """Test system with many metrics."""
        # Create many simple metrics
        metrics = []
        weights = {}
        num_metrics = 20

        for i in range(num_metrics):
            metric_name = f"metric_{i}"
            weight = 1.0 / num_metrics
            metrics.append(SimpleTestMetric(metric_name, weight=weight))
            weights[metric_name] = 1.0 / num_metrics

        evaluator = MetricEval(metrics, weights)
        repo_context = {"model_name": "performance-test-model"}

        # This should complete without timing out
        scores = evaluator.evaluateAll(repo_context)
        final_score = evaluator.aggregateScores(scores)

        assert len(scores) == num_metrics
        assert 0.0 <= final_score <= 1.0


if __name__ == "__main__":
    pytest.main([__file__])
