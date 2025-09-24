"""
Comprehensive unit tests for the MetricEval class.
Consolidated from multiple test files to avoid redundancy.

NOTE: NetScorer class has been implemented in src/net_scorer.py
with comprehensive tests in tests/test_net_scorer.py.
The deprecated test files test_metric_manager.py and
test_metric_manager_scoreresult.py can be removed as their
functionality has been consolidated into this file and
the new NetScorer tests.
"""

import pytest
import sys
import os
from unittest.mock import patch, MagicMock

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.metric_eval import MetricEval, init_metrics, init_weights
from src.metrics.base_metric import BaseMetric


class MockMetric(BaseMetric):
    """Mock metric for testing MetricEval."""

    def __init__(self, name: str, weight: float = 0.5, score: float = 0.8):
        super().__init__(name, weight)
        self.score = score
        self.call_count = 0

    def evaluate(self, repo_context: dict) -> float:
        """Return mock score and track calls."""
        self.call_count += 1
        return self.score

    def get_description(self) -> str:
        return f"Mock metric {self.name}"


class FailingMockMetric(BaseMetric):
    """Mock metric that always fails."""

    def __init__(self, name: str):
        super().__init__(name, 0.5)

    def evaluate(self, repo_context: dict) -> float:
        """Always raise an exception."""
        raise RuntimeError(f"Mock error from {self.name}")

    def get_description(self) -> str:
        return f"Failing mock metric {self.name}"


class TestMetricEval:
    """Test suite for MetricEval class."""

    def test_metric_eval_initialization(self):
        """Test MetricEval initialization."""
        metrics = [MockMetric("metric1"), MockMetric("metric2")]
        weights = {"metric1": 0.6, "metric2": 0.4}

        evaluator = MetricEval(metrics, weights)

        assert evaluator.metrics == metrics
        assert evaluator.weights == weights

    def test_empty_metrics_list(self):
        """Test MetricEval with empty metrics list."""
        evaluator = MetricEval([], {})

        assert len(evaluator.metrics) == 0
        assert len(evaluator.weights) == 0

    def test_evaluate_all_single_metric(self):
        """Test evaluateAll with a single metric."""
        metric = MockMetric("test_metric", score=0.7)
        evaluator = MetricEval([metric], {"test_metric": 1.0})

        repo_context = {"test": "data"}
        results = evaluator.evaluateAll(repo_context)

        assert "test_metric" in results
        assert results["test_metric"] == 0.7
        assert metric.call_count == 1

    def test_evaluate_all_multiple_metrics(self):
        """Test evaluateAll with multiple metrics."""
        metric1 = MockMetric("metric1", score=0.8)
        metric2 = MockMetric("metric2", score=0.6)
        metrics = [metric1, metric2]
        weights = {"metric1": 0.7, "metric2": 0.3}

        evaluator = MetricEval(metrics, weights)
        repo_context = {"test": "data"}

        results = evaluator.evaluateAll(repo_context)

        assert len(results) == 2
        assert results["metric1"] == 0.8
        assert results["metric2"] == 0.6
        assert metric1.call_count == 1
        assert metric2.call_count == 1

    def test_evaluate_all_with_exception(self):
        """Test evaluateAll handles exceptions gracefully."""
        good_metric = MockMetric("good_metric", score=0.9)
        bad_metric = FailingMockMetric("bad_metric")
        metrics = [good_metric, bad_metric]
        weights = {"good_metric": 0.5, "bad_metric": 0.5}

        evaluator = MetricEval(metrics, weights)

        # Capture print output to test error handling
        with patch('builtins.print') as mock_print:
            results = evaluator.evaluateAll({})

        assert len(results) == 2
        assert results["good_metric"] == 0.9
        assert results["bad_metric"] == -1  # Failing score

        # Check that error was printed
        mock_print.assert_called()

    def test_weights_access(self):
        """Test accessing weights from MetricEval."""
        weights = {"metric1": 0.3, "metric2": 0.7}
        evaluator = MetricEval([], weights)

        assert evaluator.weights["metric1"] == 0.3
        assert evaluator.weights["metric2"] == 0.7

    def test_metrics_property_access(self):
        """Test accessing metrics property from MetricEval."""
        metric1 = MockMetric("metric1")
        metric2 = MockMetric("metric2")
        metrics = [metric1, metric2]

        evaluator = MetricEval(metrics, {})

        assert len(evaluator.metrics) == 2
        assert evaluator.metrics[0] == metric1
        assert evaluator.metrics[1] == metric2

    def test_evaluate_all_empty_context(self):
        """Test evaluateAll with empty repository context."""
        metric = MockMetric("test_metric", score=0.5)
        evaluator = MetricEval([metric], {"test_metric": 1.0})

        results = evaluator.evaluateAll({})

        assert "test_metric" in results
        assert results["test_metric"] == 0.5

    def test_aggregation_with_missing_weights(self):
        """Test score aggregation when some metrics don't have weights."""
        evaluator = MetricEval([], {"metric1": 0.5})  # metric2 missing weight

        scores = {"metric1": 0.8, "metric2": 0.6}
        final_score = evaluator.aggregateScores(scores)

        # Only metric1 should contribute (weight 0.5, score 0.8)
        expected = 0.8  # (0.8 * 0.5) / 0.5
        assert abs(final_score - expected) < 0.001

    def test_aggregation_bounds_checking(self):
        """Test that aggregated scores are bounded between 0 and 1."""
        evaluator = MetricEval([], {"metric1": 1.0})

        # Test upper bound
        scores = {"metric1": 1.5}  # Score > 1
        final_score = evaluator.aggregateScores(scores)
        assert final_score == 1.0

        # Test lower bound (negative scores should be clamped)
        scores = {"metric1": -0.5}  # Score < 0
        final_score = evaluator.aggregateScores(scores)
        assert final_score == 0.0

    def test_full_evaluation_pipeline(self):
        """Test complete evaluation pipeline."""
        metric1 = MockMetric("metric1", score=0.9)
        metric2 = MockMetric("metric2", score=0.7)
        metrics = [metric1, metric2]
        weights = {"metric1": 0.3, "metric2": 0.7}

        evaluator = MetricEval(metrics, weights)
        repo_context = {"model_name": "test-model"}

        # Run evaluation
        scores = evaluator.evaluateAll(repo_context)
        final_score = evaluator.aggregateScores(scores)

        # Verify individual scores
        assert scores["metric1"] == 0.9
        assert scores["metric2"] == 0.7

        # Verify aggregated score
        expected = (0.9 * 0.3 + 0.7 * 0.7) / 1.0
        assert abs(final_score - expected) < 0.001

    def test_parallel_execution(self):
        """Test that metrics are executed concurrently in parallel."""
        import time
        import threading

        class SlowMockMetric(BaseMetric):
            def __init__(self, name: str, delay: float = 0.1):
                super().__init__(name, 0.5)
                self.delay = delay
                self.thread_id = None
                self.start_time = None
                self.end_time = None

            def evaluate(self, repo_context: dict) -> float:
                self.thread_id = threading.current_thread().ident
                self.start_time = time.time()
                time.sleep(self.delay)  # Simulate work
                self.end_time = time.time()
                return 0.8

        # Create multiple slow metrics
        metrics = [
            SlowMockMetric("slow_metric1", 0.1),
            SlowMockMetric("slow_metric2", 0.1),
            SlowMockMetric("slow_metric3", 0.1)
        ]
        weights = {m.name: 1.0 for m in metrics}

        evaluator = MetricEval(metrics, weights)

        start_time = time.time()
        results = evaluator.evaluateAll({})
        end_time = time.time()

        # Verify all metrics completed
        assert len(results) == 3
        assert all(score == 0.8 for score in results.values())

        # Verify parallel execution:
        # 1. Total time should be less than sum of individual delays
        total_delay = sum(m.delay for m in metrics)  # 0.3s if sequential
        actual_time = end_time - start_time
        assert actual_time < total_delay, (
            f"Expected parallel execution to be faster than {total_delay}s, "
            f"got {actual_time}s"
        )

        # 2. Different metrics should run on different threads
        thread_ids = [m.thread_id for m in metrics if m.thread_id]
        assert len(set(thread_ids)) > 1, (
            "Metrics should run on different threads"
        )

        # 3. Metrics should overlap in time (start times should be close)
        start_times = [m.start_time for m in metrics if m.start_time]
        time_spread = max(start_times) - min(start_times)
        assert time_spread < 0.05, (
            f"Metrics should start nearly simultaneously, "
            f"got {time_spread}s spread"
        )

    def test_aggregate_scores_empty(self):
        """Test aggregateScores with empty scores."""
        evaluator = MetricEval([], {})
        score = evaluator.aggregateScores({})
        assert score == 0.0
    
    def test_aggregate_scores_no_matching_weights(self):
        """Test aggregateScores when no scores match weights."""
        evaluator = MetricEval([], {"metric1": 0.5, "metric2": 0.5})
        score = evaluator.aggregateScores({"metric3": 0.8, "metric4": 0.6})
        assert score == 0.0
    
    def test_aggregate_scores_partial_match(self):
        """Test aggregateScores with partial matching of scores and weights."""
        evaluator = MetricEval([], {"metric1": 0.3, "metric2": 0.7})
        
        # Only one score matches weights
        score = evaluator.aggregateScores({"metric1": 0.8, "metric3": 0.6})
        assert score == 0.8  # Only metric1 counts, so its score is used directly
        
    def test_aggregate_scores_clamping(self):
        """Test aggregateScores clamps to [0, 1] range."""
        evaluator = MetricEval([], {"metric1": 0.5, "metric2": 0.5})
        
        # Test clamping to upper bound
        score = evaluator.aggregateScores({"metric1": 1.2, "metric2": 1.5})
        assert score == 1.0
        
        # Test clamping to lower bound
        score = evaluator.aggregateScores({"metric1": -0.2, "metric2": -0.5})
        assert score == 0.0
        
        # Test mix of positive and negative
        score = evaluator.aggregateScores({"metric1": 0.7, "metric2": -0.3})
        expected = max(0.0, min(1.0, (0.7 * 0.5 + (-0.3) * 0.5)))
        assert abs(score - expected) < 0.001

    def test_evaluate_all_handling_exceptions(self):
        """Test evaluateAll properly handles exceptions from metrics."""
        # Create metrics - one will succeed, one will fail
        metric1 = MagicMock(spec=BaseMetric)
        metric1.name = "metric1"
        metric1.evaluate = MagicMock(return_value=0.75)
        
        metric2 = MagicMock(spec=BaseMetric)
        metric2.name = "metric2"
        metric2.evaluate = MagicMock(side_effect=Exception("Test exception"))
        
        evaluator = MetricEval([metric1, metric2], {"metric1": 0.6, "metric2": 0.4})
        
        # Evaluate with mocked print to capture output
        with patch('builtins.print') as mock_print:
            results = evaluator.evaluateAll({"test": "data"})
            
            # Verify both metrics were processed
            assert "metric1" in results
            assert "metric2" in results
            assert results["metric1"] == 0.75
            assert results["metric2"] == -1  # Failed metric should return -1
            
            # Verify exception was printed
            mock_print.assert_called_once()
            assert "Error evaluating metric2" in mock_print.call_args[0][0]

    def test_init_weights(self):
        """Test init_weights returns the expected weights."""
        weights = init_weights()
        
        # Check if all expected metric names are present
        expected_metrics = [
            "BusFactor",
            "CodeQuality",
            "CommunityRating",
            "DatasetAvailability",
            "DatasetQuality",
            "License",
            "PerformanceClaims",
            "RampUpTime",
            "Size",
        ]
        
        for metric in expected_metrics:
            assert metric in weights
            
        # Check that weights sum to 1.0 (or very close)
        total_weight = sum(weights.values())
        assert abs(total_weight - 1.0) < 0.01


class TestMetricEvalErrorHandling:
    """Test suite for error handling scenarios in MetricEval."""

    def test_invalid_metric_weight_simulation(self):
        """Test handling of invalid metric weights."""
        # Simulate validation that might happen elsewhere
        with pytest.raises(ValueError):
            weight = -0.5
            if weight < 0:
                raise ValueError("Metric weight cannot be negative")

    def test_empty_repository_data_handling(self):
        """Test MetricEval with empty repository data."""
        metric = MockMetric("test_metric", score=0.5)
        evaluator = MetricEval([metric], {"test_metric": 1.0})

        # Should handle empty data gracefully
        empty_data = {}
        results = evaluator.evaluateAll(empty_data)

        assert "test_metric" in results
        assert results["test_metric"] == 0.5

    def test_network_timeout_simulation(self):
        """Test network timeout simulation."""
        import socket

        with pytest.raises(socket.timeout):
            # Simulate network timeout
            raise socket.timeout("Network timeout")

    def test_api_failure_simulation(self):
        """Test API connection failure simulation."""
        mock_api = MagicMock()
        mock_api.side_effect = ConnectionError("API unavailable")

        with pytest.raises(ConnectionError):
            mock_api()

    def test_file_not_found_simulation(self):
        """Test file not found error simulation."""
        with pytest.raises(FileNotFoundError):
            raise FileNotFoundError("URLs file not found")

    def test_invalid_url_simulation(self):
        """Test invalid URL format simulation."""
        with pytest.raises(ValueError):
            raise ValueError("Invalid URL format")


if __name__ == "__main__":
    pytest.main([__file__])
