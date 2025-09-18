"""
Unit tests for RunTest class.
"""

import pytest
from unittest.mock import patch, MagicMock
import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

try:
    from app import RunTest
except ImportError:
    # Create mock if class doesn't exist yet
    RunTest = MagicMock


class TestRunTest:
    """Test suite for RunTest class."""

    def setup_method(self):
        """Setup test fixtures."""
        try:
            from app import RunTest
            self.run_test = RunTest()
        except ImportError:
            self.run_test = MagicMock()

    @patch('pytest.main')
    def test_run_pytest_success(self, mock_pytest):
        """Mock pytest main returns 0"""
        mock_pytest.return_value = 0

        result = self.run_test.execute()

        mock_pytest.assert_called_once()
        assert result == 0

    @patch('pytest.main')
    @patch('sys.exit')
    def test_run_pytest_failure(self, mock_exit, mock_pytest):
        """Pytest main returns nonzero -> assert error handling"""
        mock_pytest.return_value = 1

        with patch('builtins.print') as mock_print:
            self.run_test.execute()

        mock_pytest.assert_called_once()
        mock_exit.assert_called_with(1)
        mock_print.assert_called_with("Tests failed", file=sys.stderr)

    def test_format_coverage_report(self):
        """Input raw coverage, output matches NDJSON schema"""
        raw_coverage = {
            "files": {
                "src/app.py": {"summary": {"percent_covered": 85.5}},
                "src/metrics/base_metric.py": {
                    "summary": {"percent_covered": 92.1}
                }
            }
        }

        expected_ndjson = {
            "coverage": 88.8,  # Average coverage
            "files": [
                {"file": "src/app.py", "coverage": 85.5},
                {"file": "src/metrics/base_metric.py", "coverage": 92.1}
            ]
        }

        result = self.run_test.format_coverage_report(raw_coverage)
        assert result == expected_ndjson

    # --- EXTRA TESTS ---
    def test_run_test_initialization(self):
        """Test RunTest class can be instantiated."""
        tester = RunTest()
        assert tester is not None

    @patch('pytest.main')
    def test_run_pytest_with_coverage(self, mock_pytest):
        """Test running pytest with coverage options."""
        mock_pytest.return_value = 0

        result = self.run_test.execute_with_coverage()

        mock_pytest.assert_called_once()
        assert result == 0

    @patch('pytest.main')
    def test_run_specific_test_file(self, mock_pytest):
        """Test running a specific test file."""
        mock_pytest.return_value = 0
        test_file = "tests/test_specific.py"

        result = self.run_test.execute_file(test_file)

        mock_pytest.assert_called_once()
        assert result == 0

    def test_format_coverage_report_empty_input(self):
        """Test coverage report formatting with empty input."""
        raw_coverage = {"files": {}}

        expected_ndjson = {
            "coverage": 0.0,
            "files": []
        }

        result = self.run_test.format_coverage_report(raw_coverage)
        assert result == expected_ndjson

    def test_format_coverage_report_single_file(self):
        """Test coverage report formatting with single file."""
        raw_coverage = {
            "files": {
                "src/single.py": {"summary": {"percent_covered": 75.0}}
            }
        }

        expected_ndjson = {
            "coverage": 75.0,
            "files": [
                {"file": "src/single.py", "coverage": 75.0}
            ]
        }

        result = self.run_test.format_coverage_report(raw_coverage)
        assert result == expected_ndjson

    @patch('pytest.main')
    def test_run_pytest_with_custom_args(self, mock_pytest):
        """Test running pytest with custom arguments."""
        mock_pytest.return_value = 0
        custom_args = ["-v", "--tb=short"]

        result = self.run_test.execute_with_args(custom_args)

        mock_pytest.assert_called_once()
        assert result == 0


if __name__ == "__main__":
    pytest.main([__file__])
