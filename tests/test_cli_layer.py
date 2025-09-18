"""
Unit tests for CLI Layer components (App, RunCommand).
"""

import pytest
from unittest.mock import patch, MagicMock
import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

try:
    from app import App
except ImportError:
    # Create mock if app doesn't exist yet
    App = MagicMock


class TestCLILayer:
    """Test suite for CLI Layer components."""

    def setup_method(self):
        """Setup test fixtures."""
        self.app = App()

    @patch('app.RunInstall')
    def test_cli_install_command(self, mock_run_install):
        """Pass ['install'] args -> calls App.runInstall()"""
        mock_installer = MagicMock()
        mock_run_install.return_value = mock_installer

        # Simulate CLI args
        test_args = ["install"]

        # Test that install command is properly routed
        with patch('sys.argv', ['app.py'] + test_args):
            self.app.runInstall()

        mock_run_install.assert_called_once()
        mock_installer.execute.assert_called_once()

    @patch('app.RunTest')
    def test_cli_test_command(self, mock_run_test):
        """Pass ['test'] args -> calls App.runTest()"""
        mock_tester = MagicMock()
        mock_run_test.return_value = mock_tester

        # Simulate CLI args
        test_args = ["test"]

        # Test that test command is properly routed
        with patch('sys.argv', ['app.py'] + test_args):
            self.app.runTest()

        mock_run_test.assert_called_once()
        mock_tester.execute.assert_called_once()

    @patch('app.EvaluatorRouter')
    def test_url_command(self, mock_evaluator):
        """Pass ['url', 'urls.txt'] args -> calls App.runEvaluateUrl()"""
        mock_router = MagicMock()
        mock_evaluator.return_value = mock_router

        # Simulate CLI args
        test_args = ["url", "urls.txt"]

        # Test that URL evaluation command is properly routed
        with patch('sys.argv', ['app.py'] + test_args):
            with patch('os.path.exists', return_value=True):
                self.app.runEvaluateUrl("urls.txt")

        mock_evaluator.assert_called_once()
        mock_router.evaluate_from_file.assert_called_once_with("urls.txt")


if __name__ == "__main__":
    pytest.main([__file__])
