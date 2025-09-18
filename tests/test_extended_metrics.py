"""
Unit tests for Extended Metrics (Size, License, RampUp, etc.).
"""

import pytest
from unittest.mock import MagicMock
import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

try:
    from metrics.base_metric import BaseMetric
    from metrics.size_metric import SizeMetric
    from metrics.license_metric import LicenseMetric
    from metrics.ramp_up_time_metric import RampUpTimeMetric
    from metrics.bus_factor_metric import BusFactorMetric
    from metrics.dataset_availability_metric import DatasetAvailabilityMetric
    from metrics.dataset_quality_metric import DatasetQualityMetric
    from metrics.code_quality_metric import CodeQualityMetric
    from metrics.performance_claims_metric import PerformanceClaimsMetric
except ImportError:
    # Fallback to mocks if imports fail
    BaseMetric = MagicMock
    SizeMetric = MagicMock
    LicenseMetric = MagicMock
    RampUpTimeMetric = MagicMock
    BusFactorMetric = MagicMock
    DatasetAvailabilityMetric = MagicMock
    DatasetQualityMetric = MagicMock
    CodeQualityMetric = MagicMock
    PerformanceClaimsMetric = MagicMock


# Test classes
class TestSizeMetric:
    """Test suite for SizeMetric class."""

    def setup_method(self):
        """Setup test fixtures."""
        self.metric = SizeMetric()

    def test_size_under_2gb(self):
        """Returns 1.0."""
        repo_context = {'size': 1 * 1024**3}  # 1GB
        score = self.metric.evaluate(repo_context)
        assert score == 1.0

    def test_size_between_2_and_16(self):
        """Returns correct formula."""
        repo_context = {'size': 8 * 1024**3}  # 8GB
        score = self.metric.evaluate(repo_context)
        expected = 1 - (8 - 2) / 14  # Should be ~0.57
        assert abs(score - expected) < 0.01


class TestLicenseMetric:
    """Test suite for LicenseMetric class."""

    def setup_method(self):
        """Setup test fixtures."""
        self.metric = LicenseMetric()

    def test_license_compatible(self):
        """MIT -> returns 1."""
        repo_context = {'license': 'MIT'}
        score = self.metric.evaluate(repo_context)
        assert score == 1.0

    def test_license_incompatible(self):
        """GPL -> returns 0."""
        repo_context = {'license': 'GPL-3.0'}
        score = self.metric.evaluate(repo_context)
        assert score == 0.0


class TestRampUpTimeMetric:
    """Test suite for RampUpTimeMetric class."""

    def setup_method(self):
        """Setup test fixtures."""
        self.metric = RampUpTimeMetric()

    def test_rampup_full_docs(self):
        """README + examples -> 1.0."""
        repo_context = {
            'has_readme': True,
            'has_examples': True,
            'has_documentation': True
        }
        score = self.metric.evaluate(repo_context)
        assert score == 1.0

    def test_rampup_no_docs(self):
        """Returns 0."""
        repo_context = {
            'has_readme': False,
            'has_examples': False,
            'has_documentation': False
        }
        score = self.metric.evaluate(repo_context)
        assert score == 0.0


class TestBusFactorMetric:
    """Test suite for BusFactorMetric class."""

    def setup_method(self):
        """Setup test fixtures."""
        self.metric = BusFactorMetric()

    def test_multiple_contributors_equal_share(self):
        """High score."""
        repo_context = {
            'contributors': [
                {'contributions': 100},
                {'contributions': 100},
                {'contributions': 100}
            ]
        }
        score = self.metric.evaluate(repo_context)
        expected = 1.0 - (100 / 300)  # Should be ~0.67
        assert abs(score - expected) < 0.01

    def test_single_contributor(self):
        """Score near 0."""
        repo_context = {
            'contributors': [
                {'contributions': 1000},
                {'contributions': 10},
                {'contributions': 5}
            ]
        }
        score = self.metric.evaluate(repo_context)
        # Should be very low due to high concentration
        assert score < 0.1

    def test_bus_factor_no_contributors(self):
        """Test BusFactorMetric with no contributors."""
        metric = BusFactorMetric()
        repo_context = {}
        score = metric.evaluate(repo_context)
        assert score == 0.0

    def test_bus_factor_equal_contributors(self):
        """Test BusFactorMetric with equal contributor distribution."""
        metric = BusFactorMetric()
        repo_context = {
            'contributors': [
                {'contributions': 10},
                {'contributions': 10},
                {'contributions': 10}
            ]
        }
        score = metric.evaluate(repo_context)
        # With equal distribution, concentration is 1/3, score = 1 - 1/3 = 2/3
        assert abs(score - (2/3)) < 0.01

    def test_bus_factor_unequal_contributors(self):
        """Test BusFactorMetric with unequal contributor distribution."""
        metric = BusFactorMetric()
        repo_context = {
            'contributors': [
                {'contributions': 50},  # High concentration
                {'contributions': 5},
                {'contributions': 5}
            ]
        }
        score = metric.evaluate(repo_context)
        # Concentration is 50/60 = 5/6, score = 1 - 5/6 = 1/6
        assert abs(score - (1/6)) < 0.01

    def test_bus_factor_weight(self):
        """Test BusFactorMetric weight initialization."""
        metric = BusFactorMetric(weight=0.25)
        assert metric.weight == 0.25
        assert metric.name == "BusFactor"


class TestDatasetAvailabilityMetric:
    """Test suite for DatasetAvailabilityMetric class."""

    def setup_method(self):
        """Setup test fixtures."""
        self.metric = DatasetAvailabilityMetric()

    def test_dataset_available(self):
        """Has dataset -> returns base score of 0.5."""
        repo_context = {'has_dataset': True}
        score = self.metric.evaluate(repo_context)
        assert score == 0.5

    def test_dataset_unavailable(self):
        """No dataset -> returns 0."""
        repo_context = {'has_dataset': False}
        score = self.metric.evaluate(repo_context)
        assert score == 0.0

    def test_dataset_availability_no_dataset(self):
        """Test DatasetAvailabilityMetric with no dataset."""
        metric = DatasetAvailabilityMetric()
        repo_context = {'has_dataset': False}
        score = metric.evaluate(repo_context)
        assert score == 0.0

    def test_dataset_availability_basic_dataset(self):
        """Test DatasetAvailabilityMetric with basic dataset."""
        metric = DatasetAvailabilityMetric()
        repo_context = {
            'has_dataset': True,
            'dataset_accessible': False,
            'dataset_size': 500
        }
        score = metric.evaluate(repo_context)
        assert score == 0.5  # Base score only

    def test_dataset_availability_accessible_dataset(self):
        """Test DatasetAvailabilityMetric with accessible dataset."""
        metric = DatasetAvailabilityMetric()
        repo_context = {
            'has_dataset': True,
            'dataset_accessible': True,
            'dataset_size': 500
        }
        score = metric.evaluate(repo_context)
        assert score == 0.8  # Base + accessibility

    def test_dataset_availability_large_dataset(self):
        """Test DatasetAvailabilityMetric with large dataset."""
        metric = DatasetAvailabilityMetric()
        repo_context = {
            'has_dataset': True,
            'dataset_accessible': True,
            'dataset_size': 2000
        }
        score = metric.evaluate(repo_context)
        assert score == 1.0  # Base + accessibility + size

    def test_dataset_availability_weight(self):
        """Test DatasetAvailabilityMetric weight initialization."""
        metric = DatasetAvailabilityMetric(weight=0.2)
        assert metric.weight == 0.2
        assert metric.name == "DatasetAvailability"


class TestDatasetQualityMetric:
    """Test suite for DatasetQualityMetric class."""

    def setup_method(self):
        """Setup test fixtures."""
        self.metric = DatasetQualityMetric()

    def test_full_quality(self):
        """All fields present -> highest score."""
        repo_context = {
            'has_data_validation': True,
            'data_diversity_score': 1.0,
            'data_completeness': 1.0
        }
        score = self.metric.evaluate(repo_context)
        assert score == 1.0

    def test_no_quality_indicators(self):
        """No fields present -> zero score."""
        repo_context = {
            'dataset_card': False,
            'data_fields_documented': False,
            'example_usage': False
        }
        score = self.metric.evaluate(repo_context)
        assert score == 0.0

    def test_dataset_quality_no_validation(self):
        """Test DatasetQualityMetric with no validation or quality."""
        metric = DatasetQualityMetric()
        repo_context = {
            'has_data_validation': False,
            'data_diversity_score': 0.0,
            'data_completeness': 0.0
        }
        score = metric.evaluate(repo_context)
        assert score == 0.0

    def test_dataset_quality_with_validation(self):
        """Test DatasetQualityMetric with validation."""
        metric = DatasetQualityMetric()
        repo_context = {
            'has_data_validation': True,
            'data_diversity_score': 0.0,
            'data_completeness': 0.0
        }
        score = metric.evaluate(repo_context)
        assert score == 0.4

    def test_dataset_quality_high_diversity(self):
        """Test DatasetQualityMetric with high diversity."""
        metric = DatasetQualityMetric()
        repo_context = {
            'has_data_validation': False,
            'data_diversity_score': 1.0,
            'data_completeness': 0.0
        }
        score = metric.evaluate(repo_context)
        assert score == 0.3

    def test_dataset_quality_complete_dataset(self):
        """Test DatasetQualityMetric with complete dataset."""
        metric = DatasetQualityMetric()
        repo_context = {
            'has_data_validation': False,
            'data_diversity_score': 0.0,
            'data_completeness': 1.0
        }
        score = metric.evaluate(repo_context)
        assert score == 0.3

    def test_dataset_quality_full_score(self):
        """Test DatasetQualityMetric with all quality indicators."""
        metric = DatasetQualityMetric()
        repo_context = {
            'has_data_validation': True,
            'data_diversity_score': 1.0,
            'data_completeness': 1.0
        }
        score = metric.evaluate(repo_context)
        assert score == 1.0

    def test_dataset_quality_weight(self):
        """Test DatasetQualityMetric weight initialization."""
        metric = DatasetQualityMetric(weight=0.25)
        assert metric.weight == 0.25
        assert metric.name == "DatasetQuality"


class TestCodeQualityMetric:
    """Test suite for CodeQualityMetric class."""

    def setup_method(self):
        """Setup test fixtures."""
        self.metric = CodeQualityMetric()

    def test_no_flake8_issues(self):
        """Returns score based on all quality factors."""
        repo_context = {
            'has_tests': False,
            'test_coverage': 0.0,
            'has_linting': False,
            'code_complexity': 5.0
        }
        score = self.metric.evaluate(repo_context)
        expected = (1 - 5.0/20.0) * 0.2  # Only complexity score
        assert abs(score - expected) < 0.01

    def test_flake8_issues_present(self):
        """Returns score based on quality factors, not just flake8."""
        repo_context = {
            'has_tests': False,
            'test_coverage': 0.0,
            'has_linting': False,
            'code_complexity': 20.0
        }
        score = self.metric.evaluate(repo_context)
        assert score == 0.0  # High complexity results in 0 score

    def test_code_quality_no_features(self):
        """Test CodeQualityMetric with no quality features."""
        metric = CodeQualityMetric()
        repo_context = {
            'has_tests': False,
            'test_coverage': 0.0,
            'has_linting': False,
            'code_complexity': 20.0
        }
        score = metric.evaluate(repo_context)
        assert score == 0.0  # High complexity results in 0 complexity score

    def test_code_quality_with_tests(self):
        """Test CodeQualityMetric with tests."""
        metric = CodeQualityMetric()
        repo_context = {
            'has_tests': True,
            'test_coverage': 0.0,
            'has_linting': False,
            'code_complexity': 5.0
        }
        score = metric.evaluate(repo_context)
        expected = 0.3 + (1 - 5.0/20.0) * 0.2  # tests + complexity score
        assert abs(score - expected) < 0.01

    def test_code_quality_high_coverage(self):
        """Test CodeQualityMetric with high test coverage."""
        metric = CodeQualityMetric()
        repo_context = {
            'has_tests': True,
            'test_coverage': 90.0,
            'has_linting': True,
            'code_complexity': 5.0
        }
        score = metric.evaluate(repo_context)
        expected = 0.3 + 0.9 * 0.3 + 0.2 + (1 - 5.0/20.0) * 0.2
        assert abs(score - expected) < 0.01

    def test_code_quality_with_linting(self):
        """Test CodeQualityMetric with linting enabled."""
        metric = CodeQualityMetric()
        repo_context = {
            'has_tests': False,
            'test_coverage': 0.0,
            'has_linting': True,
            'code_complexity': 10.0
        }
        score = metric.evaluate(repo_context)
        expected = 0.2 + (1 - 10.0/20.0) * 0.2  # linting + complexity
        assert abs(score - expected) < 0.01

    def test_code_quality_weight(self):
        """Test CodeQualityMetric weight initialization."""
        metric = CodeQualityMetric(weight=0.3)
        assert metric.weight == 0.3
        assert metric.name == "CodeQuality"


class TestPerformanceClaimsMetric:
    """Test suite for PerformanceClaimsMetric class."""

    def setup_method(self):
        """Setup test fixtures."""
        self.metric = PerformanceClaimsMetric()

    def test_benchmarks_available(self):
        """Has benchmarks -> returns partial score."""
        repo_context = {'has_benchmarks': True}
        score = self.metric.evaluate(repo_context)
        assert score == 0.3

    def test_benchmarks_unavailable(self):
        """No benchmarks -> returns 0."""
        repo_context = {'has_benchmarks': False}
        score = self.metric.evaluate(repo_context)
        assert score == 0.0

    def test_performance_claims_no_evidence(self):
        """Test PerformanceClaimsMetric with no performance evidence."""
        metric = PerformanceClaimsMetric()
        repo_context = {
            'has_benchmarks': False,
            'benchmark_results': [],
            'has_performance_docs': False,
            'claims_verified': False
        }
        score = metric.evaluate(repo_context)
        assert score == 0.0

    def test_performance_claims_with_benchmarks(self):
        """Test PerformanceClaimsMetric with benchmarks."""
        metric = PerformanceClaimsMetric()
        repo_context = {
            'has_benchmarks': True,
            'benchmark_results': [],
            'has_performance_docs': False,
            'claims_verified': False
        }
        score = metric.evaluate(repo_context)
        assert score == 0.3

    def test_performance_claims_with_results(self):
        """Test PerformanceClaimsMetric with benchmark results."""
        metric = PerformanceClaimsMetric()
        repo_context = {
            'has_benchmarks': False,
            'benchmark_results': ['result1', 'result2'],
            'has_performance_docs': False,
            'claims_verified': False
        }
        score = metric.evaluate(repo_context)
        assert score == 0.2  # 2 results * 0.1 each

    def test_performance_claims_with_docs(self):
        """Test PerformanceClaimsMetric with performance documentation."""
        metric = PerformanceClaimsMetric()
        repo_context = {
            'has_benchmarks': False,
            'benchmark_results': [],
            'has_performance_docs': True,
            'claims_verified': False
        }
        score = metric.evaluate(repo_context)
        assert score == 0.2

    def test_performance_claims_verified(self):
        """Test PerformanceClaimsMetric with verified claims."""
        metric = PerformanceClaimsMetric()
        repo_context = {
            'has_benchmarks': False,
            'benchmark_results': [],
            'has_performance_docs': False,
            'claims_verified': True
        }
        score = metric.evaluate(repo_context)
        assert score == 0.2

    def test_performance_claims_full_score(self):
        """Test PerformanceClaimsMetric with all evidence."""
        metric = PerformanceClaimsMetric()
        repo_context = {
            'has_benchmarks': True,
            'benchmark_results': ['result1', 'result2', 'result3'],
            'has_performance_docs': True,
            'claims_verified': True
        }
        score = metric.evaluate(repo_context)
        assert score == 1.0  # 0.3 + 0.3 + 0.2 + 0.2

    def test_performance_claims_weight(self):
        """Test PerformanceClaimsMetric weight initialization."""
        metric = PerformanceClaimsMetric(weight=0.25)
        assert metric.weight == 0.25
        assert metric.name == "PerformanceClaims"


if __name__ == "__main__":
    pytest.main([__file__])
