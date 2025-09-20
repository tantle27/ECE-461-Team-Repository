"""
Unit tests for RunTest class with pytest integration and coverage reporting.
"""

import pytest
from unittest.mock import patch, MagicMock, mock_open
import sys
import os
import json
import subprocess

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from src.run_test import RunTest, TestExecutionResult


class TestRunTestClass:
    """Test suite for RunTest class functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        self.runner = RunTest()

    def test_runner_initialization(self):
        """Test RunTest class initialization."""
        runner = RunTest()
        assert runner.project_root is not None
        assert runner.src_dir.name == "src"
        assert runner.tests_dir.name == "tests"

    def test_runner_initialization_with_custom_path(self):
        """Test RunTest initialization with custom project root."""
        custom_path = "/custom/project/path"
        runner = RunTest(custom_path)
        # Convert to string and normalize path separators for Windows
        assert str(runner.project_root).replace('\\', '/') == custom_path

    @patch('subprocess.run')
    @patch('src.run_test.RunTest._parse_coverage_report')
    @patch('src.run_test.RunTest._parse_test_output')
    def test_run_tests_with_coverage_success(
            self, mock_parse_output, mock_parse_coverage, mock_subprocess):
        """Test successful test execution with coverage."""
        # Mock subprocess result
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = "169 passed in 0.53s"
        mock_subprocess.return_value = mock_result

        # Mock coverage data
        mock_parse_coverage.return_value = {
            'totals': {'percent_covered': 85.5},
            'files': {'src/app.py': {'summary': {'percent_covered': 90.0}}}
        }

        # Mock test output parsing
        mock_parse_output.return_value = {
            'total': 169, 'passed': 169, 'failed': 0, 'duration': 0.53
        }

        result = self.runner.run_tests_with_coverage()

        assert result.exit_code == 0
        assert result.tests_run == 169
        assert result.tests_passed == 169
        assert result.tests_failed == 0
        assert result.coverage_percentage == 85.5
        assert result.execution_time == 0.53

    @patch('subprocess.run')
    def test_run_tests_with_coverage_failure(self, mock_subprocess):
        """Test test execution failure handling."""
        # Mock subprocess failure
        mock_result = MagicMock()
        mock_result.returncode = 1
        mock_result.stdout = "5 failed, 164 passed in 0.65s"
        mock_subprocess.return_value = mock_result

        with patch.object(self.runner, '_parse_coverage_report') as mock_cov:
            mock_cov.return_value = {'totals': {'percent_covered': 75.0}}
            
            with patch.object(self.runner, '_parse_test_output') as mock_out:
                mock_out.return_value = {
                    'total': 169, 'passed': 164, 'failed': 5, 'duration': 0.65
                }

                result = self.runner.run_tests_with_coverage()

                assert result.exit_code == 1
                assert result.tests_failed == 5
                assert result.coverage_percentage == 75.0

    @patch('subprocess.run')
    def test_run_tests_timeout(self, mock_subprocess):
        """Test test execution timeout handling."""
        mock_subprocess.side_effect = subprocess.TimeoutExpired(['pytest'], 300)

        result = self.runner.run_tests_with_coverage()

        assert result.exit_code == 124  # Timeout exit code
        assert result.tests_run == 0
        assert "timeout" in result.ndjson_output.lower()

    def test_parse_test_output_success(self):
        """Test parsing of successful pytest output."""
        output = """
        ===== test session starts =====
        tests/test_cli_layer.py::TestAppCLI::test_read_urls_success PASSED
        tests/test_cli_layer.py::TestAppCLI::test_read_urls_file_not_found PASSED
        ===== 169 passed, 1 warning in 0.53s =====
        """
        stats = self.runner._parse_test_output(output)
        assert stats['passed'] == 169
        assert stats['failed'] == 0
        assert stats['total'] == 169
        assert stats['duration'] == 0.53
        assert stats['total'] == 169
        assert stats['failed'] == 0
        assert stats['duration'] == 0.53

    def test_parse_test_output_with_failures(self):
        """Test parsing of pytest output with failures."""
        output = """
        ===== test session starts =====
        tests/test_cli_layer.py::TestAppCLI::test_example FAILED
        ===== 5 failed, 164 passed in 0.65s =====
        """

        stats = self.runner._parse_test_output(output)

        assert stats['passed'] == 164
        assert stats['failed'] == 5
        assert stats['total'] == 169
        assert stats['duration'] == 0.65

    def test_parse_coverage_report_success(self):
        """Test parsing of coverage JSON report."""
        coverage_data = {
            'summary': {
                'percent_covered': 85.5,
                'covered_lines': 450,
                'num_statements': 527
            },
            'files': {
                'src/app.py': {
                    'summary': {'percent_covered': 90.0}
                },
                'src/url_router.py': {
                    'summary': {'percent_covered': 95.0}
                }
            }
        }

        with patch('builtins.open', mock_open(read_data=json.dumps(coverage_data))):
            with patch('pathlib.Path.exists', return_value=True):
                result = self.runner._parse_coverage_report()

                assert result['summary']['percent_covered'] == 85.5
                assert len(result['files']) == 2

    def test_parse_coverage_report_file_not_found(self):
        """Test coverage report parsing when file doesn't exist."""
        with patch('pathlib.Path.exists', return_value=False):
            result = self.runner._parse_coverage_report()
            assert result == {}

    def test_extract_file_coverage(self):
        """Test extraction of per-file coverage data."""
        coverage_data = {
            'files': {
                str(self.runner.src_dir / 'app.py'): {
                    'summary': {'percent_covered': 90.0}
                },
                str(self.runner.src_dir / 'url_router.py'): {
                    'summary': {'percent_covered': 95.0}
                }
            }
        }

        file_coverage = self.runner._extract_file_coverage(coverage_data)

        assert 'app.py' in file_coverage
        assert 'url_router.py' in file_coverage
        assert file_coverage['app.py'] == 90.0
        assert file_coverage['url_router.py'] == 95.0

    def test_generate_ndjson_output_success(self):
        """Test NDJSON output generation for successful tests."""
        test_stats = {'total': 169, 'passed': 169, 'failed': 0, 'duration': 0.53}
        coverage_data = {
            'totals': {
                'percent_covered': 85.5,
                'covered_lines': 450,
                'num_statements': 527
            },
            'files': {'src/app.py': {}}
        }

        ndjson_output = self.runner._generate_ndjson_output(test_stats, coverage_data, 0)
        result = json.loads(ndjson_output)

        assert result['test_execution']['status'] == 'success'
        assert result['test_execution']['tests_run'] == 169
        assert result['test_execution']['tests_passed'] == 169
        assert result['coverage']['overall_percentage'] == 85.5
        assert result['coverage']['lines_covered'] == 450

    def test_generate_ndjson_output_failure(self):
        """Test NDJSON output generation for failed tests."""
        test_stats = {'total': 169, 'passed': 164, 'failed': 5, 'duration': 0.65}
        coverage_data = {'totals': {'percent_covered': 75.0}}

        ndjson_output = self.runner._generate_ndjson_output(test_stats, coverage_data, 1)
        result = json.loads(ndjson_output)

        assert result['test_execution']['status'] == 'failure'
        assert result['test_execution']['exit_code'] == 1
        assert result['test_execution']['tests_failed'] == 5

    def test_generate_error_ndjson(self):
        """Test NDJSON generation for error conditions."""
        error_message = "Test execution timeout"
        ndjson_output = self.runner._generate_error_ndjson(error_message)
        result = json.loads(ndjson_output)

        assert result['test_execution']['status'] == 'error'
        assert result['test_execution']['error_message'] == error_message
        assert result['test_execution']['exit_code'] == 1
        assert result['coverage']['overall_percentage'] == 0.0

    def test_test_result_dataclass(self):
        """Test TestResult dataclass functionality."""
        result = TestExecutionResult(
            exit_code=0,
            tests_run=169,
            tests_passed=169,
            tests_failed=0,
            coverage_percentage=85.5,
            coverage_report={'app.py': 90.0},
            execution_time=0.53,
            ndjson_output='{"status": "success"}'
        )

        assert result.exit_code == 0
        assert result.tests_run == 169
        assert result.coverage_percentage == 85.5
        assert 'app.py' in result.coverage_report


class TestPytestIntegration:
    """Integration tests with actual pytest execution."""

    def test_pytest_available(self):
        """Test that pytest is available and working."""
        import pytest
        
        assert pytest is not None
        assert hasattr(pytest, 'main')

    def test_pytest_cov_available(self):
        """Test that pytest-cov is available."""
        try:
            import pytest_cov
            assert pytest_cov is not None
        except ImportError:
            pytest.fail("pytest-cov not available")

    def test_coverage_data_format(self):
        """Test coverage data format concepts."""
        # Test NDJSON format validation
        test_ndjson = {
            "test_execution": {
                "status": "success",
                "exit_code": 0,
                "tests_run": 169,
                "tests_passed": 169,
                "tests_failed": 0,
                "execution_time": 0.53
            },
            "coverage": {
                "overall_percentage": 85.5,
                "files_covered": 15,
                "lines_covered": 450,
                "lines_total": 527
            }
        }

        # Validate JSON serialization
        json_str = json.dumps(test_ndjson, separators=(',', ':'))
        parsed = json.loads(json_str)
        
        assert parsed['test_execution']['status'] == 'success'
        assert parsed['coverage']['overall_percentage'] == 85.5


class TestRunTestCLI:
    """Test suite for RunTest command-line interface."""

    @patch('src.run_test.sys.exit')  # Mock sys.exit in the run_test module
    @patch('argparse.ArgumentParser.parse_args')
    @patch('src.run_test.RunTest')
    def test_main_function_basic(self, mock_run_test, mock_parse_args,
                                 mock_exit):
        """Test main CLI function with basic arguments."""
        from src.run_test import main
        
        # Mock arguments
        mock_args = MagicMock()
        mock_args.pattern = None
        mock_args.min_coverage = 80.0
        mock_args.html = False
        mock_args.markers = None
        mock_parse_args.return_value = mock_args
        
        # Mock runner
        mock_runner = MagicMock()
        mock_result = MagicMock()
        mock_result.exit_code = 0
        mock_result.tests_run = 100
        mock_result.tests_passed = 100
        mock_result.tests_failed = 0
        mock_result.coverage_percentage = 85.0
        mock_result.execution_time = 1.5
        mock_result.ndjson_output = '{"test": "data"}'
        mock_runner.run_tests_with_coverage.return_value = mock_result
        mock_run_test.return_value = mock_runner
        
        # Test main function
        with patch('builtins.print'):  # Suppress output
            main()
        
        mock_exit.assert_called_once_with(0)
        mock_runner.run_tests_with_coverage.assert_called_once_with(None, 80.0)

    def test_main_function_with_pattern_simple(self):
        """Test main CLI function basics."""
        from src.run_test import main
        # Just test that main function exists and is callable
        assert callable(main)

    def test_generate_error_ndjson(self):
        """Test error NDJSON generation."""
        runner = RunTest()
        error_msg = "Test execution failed"
        
        ndjson_output = runner._generate_error_ndjson(error_msg)
        result = json.loads(ndjson_output)
        
        assert result['test_execution']['status'] == 'error'
        assert result['test_execution']['error_message'] == error_msg
        assert result['test_execution']['exit_code'] == 1
        assert result['coverage']['overall_percentage'] == 0.0

    def test_extract_file_coverage_with_absolute_paths(self):
        """Test file coverage extraction with absolute paths."""
        runner = RunTest()
        
        # Create test data with absolute paths
        abs_path = str(runner.src_dir / "test_file.py")
        coverage_data = {
            'files': {
                abs_path: {
                    'summary': {'percent_covered': 85.0}
                },
                '/other/path/file.py': {
                    'summary': {'percent_covered': 75.0}
                }
            }
        }
        
        file_coverage = runner._extract_file_coverage(coverage_data)
        
        # Should convert absolute path to relative
        assert 'test_file.py' in file_coverage
        assert file_coverage['test_file.py'] == 85.0
        
        # Should keep non-src paths as is
        assert '/other/path/file.py' in file_coverage
        assert file_coverage['/other/path/file.py'] == 75.0

    def test_parse_test_output_edge_cases(self):
        """Test test output parsing with edge cases."""
        runner = RunTest()
        
        # Test with no matching lines
        output = "No relevant output"
        stats = runner._parse_test_output(output)
        
        assert stats['total'] == 0
        assert stats['passed'] == 0
        assert stats['failed'] == 0
        assert stats['duration'] == 0.0
        
        # Test with empty output
        output = ""
        stats = runner._parse_test_output(output)
        
        assert stats['total'] == 0
        assert stats['passed'] == 0
        assert stats['failed'] == 0
        assert stats['duration'] == 0.0


if __name__ == "__main__":
    pytest.main([__file__])
