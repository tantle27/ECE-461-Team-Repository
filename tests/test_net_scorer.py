"""
Unit tests for NetScorer class.

Tests cover weighted average computation, NDJSON output formatting,
and edge cases including zero weights and empty scores.
"""

import json

from src.net_scorer import NetScorer


class TestNetScorer:
    """Test suite for NetScorer class."""

    def setup_method(self):
        """Setup test fixtures."""
        self.sample_scores = {
            'CommunityRating': 0.8,
            'SizeMetric': 0.9,
            'LicenseMetric': 1.0
        }
        self.sample_weights = {
            'CommunityRating': 0.4,
            'SizeMetric': 0.3,
            'LicenseMetric': 0.3
        }
        self.sample_latencies = {
            'CommunityRating': 0.05,
            'SizeMetric': 0.03,
            'LicenseMetric': 0.01,
            'NetScore': 0.1
        }

    def test_net_scorer_initialization(self):
        """Test NetScorer can be instantiated."""
        net_scorer = NetScorer(self.sample_scores, self.sample_weights)
        assert net_scorer is not None
        assert net_scorer.scores == self.sample_scores
        assert net_scorer.weights == self.sample_weights
        assert net_scorer.url == ""
        assert net_scorer.latencies == {}

    def test_net_scorer_initialization_with_optional_params(self):
        """Test NetScorer initialization with URL and latencies."""
        url = "https://github.com/example/repo"
        net_scorer = NetScorer(
            self.sample_scores,
            self.sample_weights,
            url=url,
            latencies=self.sample_latencies
        )
        assert net_scorer.url == url
        assert net_scorer.latencies == self.sample_latencies

    def test_compute_net_score(self):
        """Computes weighted average correctly."""
        net_scorer = NetScorer(self.sample_scores, self.sample_weights)

        net_score = net_scorer.compute_net_score()

        # Expected: (0.8*0.4 + 0.9*0.3 + 1.0*0.3) = 0.32 + 0.27 + 0.3 = 0.89
        expected = 0.89
        assert abs(net_score - expected) < 0.01

    def test_compute_net_score_single_metric(self):
        """Test compute_net_score with single metric."""
        single_scores = {'OnlyMetric': 0.75}
        single_weights = {'OnlyMetric': 1.0}

        net_scorer = NetScorer(single_scores, single_weights)
        net_score = net_scorer.compute_net_score()

        assert net_score == 0.75

    def test_compute_net_score_zero_weights(self):
        """Handles zero weights gracefully."""
        zero_weights = {
            'CommunityRating': 0.0,
            'SizeMetric': 0.0,
            'LicenseMetric': 0.0
        }

        net_scorer = NetScorer(self.sample_scores, zero_weights)
        net_score = net_scorer.compute_net_score()

        assert net_score == 0.0

    def test_compute_net_score_empty_scores(self):
        """Test compute_net_score with empty scores."""
        empty_scores = {}
        empty_weights = {}

        net_scorer = NetScorer(empty_scores, empty_weights)
        net_score = net_scorer.compute_net_score()

        assert net_score == 0.0

    def test_compute_net_score_mismatched_scores_weights(self):
        """Test handling when scores and weights don't match."""
        mismatched_weights = {
            'CommunityRating': 0.5,
            'NonExistentMetric': 0.5
        }

        net_scorer = NetScorer(self.sample_scores, mismatched_weights)

        # Should handle gracefully, only use CommunityRating
        net_score = net_scorer.compute_net_score()
        # Expected: 0.8 * 0.5 / 0.5 = 0.8
        assert net_score == 0.8

    def test_to_ndjson_basic(self):
        """Test basic NDJSON output format."""
        url = "https://github.com/example/repo"
        net_scorer = NetScorer(
            self.sample_scores,
            self.sample_weights,
            url=url,
            latencies=self.sample_latencies
        )

        ndjson_output = net_scorer.to_ndjson()

        # Check required fields
        assert 'URL' in ndjson_output
        assert 'NetScore' in ndjson_output
        assert 'NetScore_Latency' in ndjson_output

        # Check individual metrics
        assert 'CommunityRating' in ndjson_output
        assert 'SizeMetric' in ndjson_output
        assert 'LicenseMetric' in ndjson_output

        # Check latencies
        assert 'CommunityRating_Latency' in ndjson_output
        assert 'SizeMetric_Latency' in ndjson_output
        assert 'LicenseMetric_Latency' in ndjson_output

        # Check values
        assert ndjson_output['URL'] == url
        assert ndjson_output['NetScore'] == 0.89
        assert isinstance(ndjson_output['NetScore'], float)
        assert isinstance(ndjson_output['CommunityRating'], float)

        # EXTRA TESTS BELOW

    def test_to_ndjson_no_latencies(self):
        """Test NDJSON output without latencies."""
        net_scorer = NetScorer(self.sample_scores, self.sample_weights)

        ndjson_output = net_scorer.to_ndjson()

        # All latencies should be 0.0
        assert ndjson_output['NetScore_Latency'] == 0.0
        assert ndjson_output['CommunityRating_Latency'] == 0.0
        assert ndjson_output['SizeMetric_Latency'] == 0.0
        assert ndjson_output['LicenseMetric_Latency'] == 0.0

    def test_to_ndjson_rounding(self):
        """Test NDJSON output rounds scores correctly."""
        scores_with_precision = {
            'TestMetric': 0.123456789
        }
        weights = {'TestMetric': 1.0}

        net_scorer = NetScorer(scores_with_precision, weights)
        ndjson_output = net_scorer.to_ndjson()

        assert ndjson_output['TestMetric'] == 0.12
        assert ndjson_output['NetScore'] == 0.12

    def test_to_ndjson_string(self):
        """Test NDJSON string output."""
        net_scorer = NetScorer(self.sample_scores, self.sample_weights)

        ndjson_string = net_scorer.to_ndjson_string()

        # Should be valid JSON
        parsed = json.loads(ndjson_string)
        assert isinstance(parsed, dict)
        assert 'NetScore' in parsed
        assert 'CommunityRating' in parsed

    def test_string_representation(self):
        """Test NetScorer string representation."""
        net_scorer = NetScorer(self.sample_scores, self.sample_weights)

        str_repr = str(net_scorer)

        assert isinstance(str_repr, str)
        assert 'NetScorer' in str_repr
        assert '0.89' in str_repr
        assert '3' in str_repr  # number of metrics

    def test_repr_representation(self):
        """Test NetScorer detailed representation."""
        url = "https://example.com"
        net_scorer = NetScorer(
            self.sample_scores,
            self.sample_weights,
            url=url,
            latencies=self.sample_latencies
        )

        repr_str = repr(net_scorer)

        assert isinstance(repr_str, str)
        assert 'NetScorer' in repr_str
        assert url in repr_str
        assert 'scores=' in repr_str
        assert 'weights=' in repr_str
        assert 'latencies=' in repr_str

    def test_none_values_handling(self):
        """Test NetScorer with None values for optional parameters."""
        net_scorer = NetScorer(
            self.sample_scores,
            self.sample_weights,
            url=None,
            latencies=None
        )

        assert net_scorer.url == ""
        assert net_scorer.latencies == {}

        # Should still work normally
        net_score = net_scorer.compute_net_score()
        assert abs(net_score - 0.89) < 0.01

    def test_edge_case_negative_scores(self):
        """Test handling of negative scores (should still compute)."""
        negative_scores = {
            'MetricA': -0.5,
            'MetricB': 0.5
        }
        weights = {
            'MetricA': 0.5,
            'MetricB': 0.5
        }

        net_scorer = NetScorer(negative_scores, weights)
        net_score = net_scorer.compute_net_score()

        # Expected: (-0.5 * 0.5 + 0.5 * 0.5) / 1.0 = 0.0
        assert net_score == 0.0

    def test_edge_case_scores_above_one(self):
        """Test handling of scores above 1.0 (should still compute)."""
        high_scores = {
            'MetricA': 1.5,
            'MetricB': 2.0
        }
        weights = {
            'MetricA': 0.4,
            'MetricB': 0.6
        }

        net_scorer = NetScorer(high_scores, weights)
        net_score = net_scorer.compute_net_score()

        # Expected: (1.5 * 0.4 + 2.0 * 0.6) / 1.0 = 0.6 + 1.2 = 1.8
        assert net_score == 1.8

    def test_partial_weight_coverage(self):
        """Test when weights don't cover all scores."""
        partial_weights = {
            'CommunityRating': 0.7
            # Missing weights for SizeMetric and LicenseMetric
        }

        net_scorer = NetScorer(self.sample_scores, partial_weights)
        net_score = net_scorer.compute_net_score()

        # Only CommunityRating should be counted
        # Expected: 0.8 * 0.7 / 0.7 = 0.8
        assert abs(net_score - 0.8) < 0.01

    def test_ndjson_consistency(self):
        """Test that NDJSON output is consistent across multiple calls."""
        net_scorer = NetScorer(
            self.sample_scores,
            self.sample_weights,
            url="https://test.com",
            latencies=self.sample_latencies
        )

        output1 = net_scorer.to_ndjson()
        output2 = net_scorer.to_ndjson()

        assert output1 == output2
