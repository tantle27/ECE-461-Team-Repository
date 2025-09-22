"""
Unit tests for CommunityRatingMetric class.
"""

import pytest
import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from metrics.community_rating_metric import CommunityRatingMetric


class TestCommunityRatingMetric:
    """Test suite for CommunityRatingMetric class."""

    def setup_method(self):
        """Setup test fixtures."""
        self.metric = CommunityRatingMetric()

    def test_initialization(self):
        """Test CommunityRatingMetric initialization."""
        assert self.metric.name == "CommunityRating"
        assert self.metric.weight == 0.15

    def test_initialization_custom_weight(self):
        """Test CommunityRatingMetric with custom weight."""
        metric = CommunityRatingMetric(weight=0.25)
        assert metric.weight == 0.25
        assert metric.name == "CommunityRating"

    def test_no_engagement(self):
        """Test with no community engagement."""
        repo_context = {'likes': 0, 'downloads_all_time': 0}
        score = self.metric.evaluate(repo_context)
        assert score == 0.0

    def test_basic_scoring(self):
        """Test basic scoring functionality."""
        # Test low likes
        repo_context = {'likes': 3, 'downloads_all_time': 0}
        score = self.metric.evaluate(repo_context)
        assert score == 0.1

        # Test medium likes
        repo_context = {'likes': 25, 'downloads_all_time': 0}
        score = self.metric.evaluate(repo_context)
        assert score == 0.3

        # Test high likes
        repo_context = {'likes': 150, 'downloads_all_time': 0}
        score = self.metric.evaluate(repo_context)
        assert score == 0.5

    def test_downloads_scoring(self):
        """Test downloads scoring."""
        # Test low downloads
        repo_context = {'likes': 0, 'downloads_all_time': 3000}
        score = self.metric.evaluate(repo_context)
        assert score == 0.1

        # Test high downloads
        repo_context = {'likes': 0, 'downloads_all_time': 150000}
        score = self.metric.evaluate(repo_context)
        assert score == 0.5

    def test_combined_scoring(self):
        """Test combined likes and downloads scoring."""
        repo_context = {'likes': 200, 'downloads_all_time': 200000}
        score = self.metric.evaluate(repo_context)
        assert score == 1.0  # 0.5 + 0.5 = 1.0

    def test_missing_data(self):
        """Test handling of missing data fields."""
        repo_context = {}
        score = self.metric.evaluate(repo_context)
        assert score == 0.0

    def test_none_context(self):
        """Test with None context."""
        score = self.metric.evaluate(None)
        assert score == 0.0

    def test_negative_values_error(self):
        """Test that negative values raise ValueError."""
        with pytest.raises(ValueError):
            self.metric.evaluate({'likes': -5, 'downloads_all_time': 100})

    def test_get_description(self):
        """Test the metric description."""
        description = self.metric.get_description()
        assert isinstance(description, str)
        assert len(description) > 0


if __name__ == "__main__":
    pytest.main([__file__])
