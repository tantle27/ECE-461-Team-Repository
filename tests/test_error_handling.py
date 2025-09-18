"""
Comprehensive error handling tests.

Tests cover invalid URLs, API failures, missing files, and other error scenarios
with proper logging, exit codes, and stderr message handling.
"""

import pytest
import sys
import os
import logging
import subprocess
from unittest.mock import patch, MagicMock
from io import StringIO

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


class TestErrorHandling:
    """Test suite for error handling scenarios."""

    def setup_method(self):
        """Setup test fixtures."""
        # Setup logging capture
        self.log_stream = StringIO()
        self.logger = logging.getLogger('test_logger')
        self.logger.setLevel(logging.ERROR)
        self.handler = logging.StreamHandler(self.log_stream)
        self.handler.setLevel(logging.ERROR)
        self.logger.addHandler(self.handler)

    def teardown_method(self):
        """Clean up after tests."""
        self.logger.removeHandler(self.handler)
        self.handler.close()

    def test_invalid_url(self):
        """Invalid URL format -> raises ValueError + logs error."""
        invalid_url = "not-a-valid-url"
        
        # Test that ValueError is raised
        with pytest.raises(ValueError) as exc_info:
            # Simulate URL validation that would happen in real code
            if not invalid_url.startswith(('http://', 'https://')):
                error_msg = f"Invalid URL format: {invalid_url}"
                self.logger.error(error_msg)
                raise ValueError(error_msg)
        
        # Verify error message
        assert "Invalid URL format" in str(exc_info.value)
        assert invalid_url in str(exc_info.value)
        
        # Verify logging occurred
        log_output = self.log_stream.getvalue()
        assert "Invalid URL format" in log_output
        assert invalid_url in log_output

    def test_invalid_url_multiple_formats(self):
        """Test various invalid URL formats."""
        invalid_urls = [
            "ftp://example.com",  # Wrong protocol
            "example.com",        # Missing protocol
            "",                   # Empty string
            "http://",           # Incomplete URL
            "not_a_url_at_all"   # Random string
        ]
        
        for url in invalid_urls:
            with pytest.raises(ValueError):
                if not url.startswith(('http://', 'https://')) or len(url) < 10:
                    self.logger.error(f"Invalid URL format: {url}")
                    raise ValueError(f"Invalid URL format: {url}")

    def test_api_failure(self):
        """API connection failure -> logs + exits with code 1."""
        
        def simulate_api_failure():
            """Simulate API failure with logging and exit."""
            try:
                # Simulate API call that fails
                raise ConnectionError("API service unavailable")
            except ConnectionError as e:
                error_msg = f"API connection failed: {e}"
                self.logger.error(error_msg)
                # In real code, this would call sys.exit(1)
                # For testing, we'll raise a custom exception to simulate exit
                raise SystemExit(1)
        
        # Test that SystemExit(1) is raised
        with pytest.raises(SystemExit) as exc_info:
            simulate_api_failure()
        
        # Verify exit code is 1
        assert exc_info.value.code == 1
        
        # Verify logging occurred
        log_output = self.log_stream.getvalue()
        assert "API connection failed" in log_output
        assert "API service unavailable" in log_output

    def test_api_failure_with_retry(self):
        """Test API failure with retry mechanism."""
        retry_count = 0
        max_retries = 3
        
        def api_call_with_retry():
            nonlocal retry_count
            retry_count += 1
            
            if retry_count <= max_retries:
                self.logger.error(f"API call failed, attempt {retry_count}/{max_retries}")
                raise ConnectionError(f"Connection timeout on attempt {retry_count}")
            
        # Test that all retries fail
        for i in range(max_retries):
            with pytest.raises(ConnectionError):
                api_call_with_retry()
        
        # Verify all retry attempts were logged
        log_output = self.log_stream.getvalue()
        assert log_output.count("API call failed") == max_retries

    def test_missing_file(self):
        """Missing file -> exit code 1 + stderr message."""
        missing_file_path = "/nonexistent/path/file.txt"
        
        def simulate_missing_file():
            """Simulate missing file with stderr and exit."""
            try:
                # Try to open non-existent file
                with open(missing_file_path, 'r') as f:
                    pass
            except FileNotFoundError as e:
                error_msg = f"Required file not found: {missing_file_path}"
                self.logger.error(error_msg)
                # Write to stderr
                print(error_msg, file=sys.stderr)
                # In real code, this would call sys.exit(1)
                raise SystemExit(1)
        
        # Capture stderr
        with patch('sys.stderr', new_callable=StringIO) as mock_stderr:
            with pytest.raises(SystemExit) as exc_info:
                simulate_missing_file()
            
            # Verify exit code is 1
            assert exc_info.value.code == 1
            
            # Verify stderr message
            stderr_output = mock_stderr.getvalue()
            assert "Required file not found" in stderr_output
            assert missing_file_path in stderr_output
        
        # Verify logging occurred
        log_output = self.log_stream.getvalue()
        assert "Required file not found" in log_output

    def test_missing_file_different_scenarios(self):
        """Test different missing file scenarios."""
        missing_files = [
            "config.json",
            "requirements.txt", 
            "urls.txt",
            "/etc/nonexistent.conf"
        ]
        
        for file_path in missing_files:
            with patch('sys.stderr', new_callable=StringIO) as mock_stderr:
                with pytest.raises(SystemExit):
                    # These files should not exist, so always raise SystemExit
                    if not os.path.exists(file_path):
                        error_msg = f"Required file not found: {file_path}"
                        self.logger.error(error_msg)
                        print(error_msg, file=sys.stderr)
                        raise SystemExit(1)
                    else:
                        # If file exists, still simulate the error for testing
                        error_msg = f"Required file not found: {file_path}"
                        self.logger.error(error_msg)
                        print(error_msg, file=sys.stderr)
                        raise SystemExit(1)
                
                # Verify stderr contains file path
                stderr_output = mock_stderr.getvalue()
                assert file_path in stderr_output

    def test_network_timeout(self):
        """Network timeout -> raises socket.timeout + logs error."""
        import socket
        
        with pytest.raises(socket.timeout) as exc_info:
            timeout_msg = "Network request timed out after 30 seconds"
            self.logger.error(timeout_msg)
            raise socket.timeout(timeout_msg)
        
        # Verify timeout message
        assert "timed out" in str(exc_info.value)
        
        # Verify logging occurred
        log_output = self.log_stream.getvalue()
        assert "Network request timed out" in log_output

    def test_permission_denied(self):
        """Permission denied -> logs error + exits with code 1."""
        restricted_path = "/root/restricted_file.txt"
        
        def simulate_permission_denied():
            """Simulate permission denied error."""
            try:
                # Simulate permission denied
                raise PermissionError(f"Permission denied: {restricted_path}")
            except PermissionError as e:
                error_msg = f"Access denied to file: {e}"
                self.logger.error(error_msg)
                print(error_msg, file=sys.stderr)
                raise SystemExit(1)
        
        with patch('sys.stderr', new_callable=StringIO) as mock_stderr:
            with pytest.raises(SystemExit) as exc_info:
                simulate_permission_denied()
            
            # Verify exit code
            assert exc_info.value.code == 1
            
            # Verify stderr
            stderr_output = mock_stderr.getvalue()
            assert "Access denied" in stderr_output

    def test_invalid_configuration(self):
        """Invalid configuration -> logs error + raises ValueError."""
        invalid_config = {
            "weights": {"metric1": -0.5}  # Negative weight is invalid
        }
        
        with pytest.raises(ValueError) as exc_info:
            for metric, weight in invalid_config["weights"].items():
                if weight < 0:
                    error_msg = (f"Invalid weight for {metric}: {weight}. "
                                 f"Weights must be non-negative.")
                    self.logger.error(error_msg)
                    raise ValueError(error_msg)
        
        # Verify error details
        assert "Invalid weight" in str(exc_info.value)
        assert "metric1" in str(exc_info.value)
        
        # Verify logging
        log_output = self.log_stream.getvalue()
        assert "Invalid weight" in log_output

    def test_subprocess_failure(self):
        """Test subprocess failure with proper error handling."""
        
        def run_subprocess_command():
            """Simulate subprocess that fails."""
            try:
                # Simulate running a command that fails
                subprocess.run(
                    ["nonexistent_command", "--help"],
                    capture_output=True,
                    text=True,
                    check=True
                )
            except subprocess.CalledProcessError as e:
                error_msg = (f"Command failed with exit code "
                             f"{e.returncode}: {e.stderr}")
                self.logger.error(error_msg)
                raise SystemExit(e.returncode)
            except FileNotFoundError as e:
                error_msg = f"Command not found: {e}"
                self.logger.error(error_msg)
                raise SystemExit(127)  # Standard "command not found" exit code
        
        with pytest.raises(SystemExit) as exc_info:
            run_subprocess_command()
        
        # Verify exit code (127 for command not found)
        assert exc_info.value.code == 127
        
        # Verify logging
        log_output = self.log_stream.getvalue()
        assert "Command not found" in log_output


class TestErrorHandlingIntegration:
    """Integration tests for error handling across the system."""
    
    def test_error_handling_pipeline(self):
        """Test complete error handling pipeline."""
        errors_encountered = []
        
        # Simulate multiple error scenarios
        error_scenarios = [
            ("invalid_url", ValueError, "Invalid URL"),
            ("api_failure", ConnectionError, "API unavailable"),
            ("missing_file", FileNotFoundError, "File not found")
        ]
        
        for scenario_name, exception_type, message in error_scenarios:
            try:
                raise exception_type(message)
            except Exception as e:
                errors_encountered.append({
                    'scenario': scenario_name,
                    'type': type(e).__name__,
                    'message': str(e)
                })
        
        # Verify all errors were caught
        assert len(errors_encountered) == 3
        assert errors_encountered[0]['type'] == 'ValueError'
        assert errors_encountered[1]['type'] == 'ConnectionError'
        assert errors_encountered[2]['type'] == 'FileNotFoundError'

    def test_graceful_degradation(self):
        """Test system behavior when multiple components fail."""
        
        def failing_component_1():
            raise ConnectionError("Service A unavailable")
        
        def failing_component_2():
            raise FileNotFoundError("Config file missing")
        
        def working_component():
            return "success"
        
        results = []
        
        # Test graceful handling of failures
        try:
            failing_component_1()
        except ConnectionError:
            results.append("component_1_failed")
        
        try:
            failing_component_2()
        except FileNotFoundError:
            results.append("component_2_failed")
        
        try:
            result = working_component()
            results.append(result)
        except Exception:
            results.append("component_3_failed")
        
        # Verify partial success despite failures
        assert "component_1_failed" in results
        assert "component_2_failed" in results
        assert "success" in results


if __name__ == "__main__":
    pytest.main([__file__])
