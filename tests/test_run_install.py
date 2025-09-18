"""
Unit tests for RunInstall class.
"""

import pytest
from unittest.mock import patch, MagicMock
import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

try:
    from app import RunInstall
except ImportError:
    # Create mock if class doesn't exist yet
    RunInstall = MagicMock


class TestRunInstall:
    """Test suite for RunInstall class."""

    def setup_method(self):
        """Setup test fixtures."""
        try:
            from app import RunInstall
            self.run_install = RunInstall()
        except ImportError:
            # Create mock if class doesn't exist yet
            self.run_install = MagicMock()

    @patch('subprocess.run')
    def test_install_success(self, mock_subprocess):
        """Simulate subprocess.run pip install returns success."""
        # Mock successful pip install
        mock_subprocess.return_value.returncode = 0
        mock_subprocess.return_value.stdout = "Successfully installed packages"

        result = self.run_install.execute()

        mock_subprocess.assert_called_with(
            ["pip", "install", "-r", "requirements.txt"],
            capture_output=True,
            text=True
        )
        assert result == 0

    @patch('subprocess.run')
    @patch('sys.exit')
    def test_install_failure(self, mock_exit, mock_subprocess):
        """Simulate subprocess failure -> assert exit code = 1 + error log."""
        # Mock failed pip install
        mock_subprocess.return_value.returncode = 1
        mock_subprocess.return_value.stderr = "Package installation failed"

        with patch('builtins.print') as mock_print:
            self.run_install.execute()

        mock_subprocess.assert_called_once()
        mock_exit.assert_called_with(1)
        mock_print.assert_called_with(
            "Error: Package installation failed", file=sys.stderr
        )

    @patch('sys.version_info')
    def test_verify_environment_python_version(self, mock_version):
        """Confirm correct version check."""
        # Mock Python 3.8+ version
        mock_version.major = 3
        mock_version.minor = 9

        result = self.run_install.verify_python_version()
        assert result is True

        # Mock unsupported version
        mock_version.minor = 6
        result = self.run_install.verify_python_version()
        assert result is False

    @patch('subprocess.run')
    def test_verify_environment_linter_installed(self, mock_subprocess):
        """Simulate flake8 presence."""
        # Mock flake8 available
        mock_subprocess.return_value.returncode = 0

        result = self.run_install.verify_linter_installed()

        mock_subprocess.assert_called_with(
            ["flake8", "--version"],
            capture_output=True,
            text=True
        )
        assert result is True

    # --- EXTRA TESTS ---
    @patch('subprocess.run')
    def test_verify_environment_linter_missing(self, mock_subprocess):
        """Test behavior when flake8 is not installed."""
        # Mock flake8 not available
        mock_subprocess.return_value.returncode = 1

        result = self.run_install.verify_linter_installed()
        assert result is False

    def test_run_install_initialization(self):
        """Test RunInstall class can be instantiated."""
        installer = RunInstall()
        assert installer is not None

    @patch('subprocess.run')
    def test_install_requirements_file_missing(self, mock_subprocess):
        """Test behavior when requirements.txt is missing."""
        mock_subprocess.side_effect = FileNotFoundError(
            "requirements.txt not found"
        )

        with pytest.raises(FileNotFoundError):
            self.run_install.execute()

    @patch('sys.version_info')
    def test_python_version_boundary_conditions(self, mock_version):
        """Test Python version boundary conditions."""
        # Test exactly Python 3.8 (should pass)
        mock_version.major = 3
        mock_version.minor = 8
        result = self.run_install.verify_python_version()
        assert result is True

        # Test Python 3.7 (should fail)
        mock_version.minor = 7
        result = self.run_install.verify_python_version()
        assert result is False

        # Test Python 2.x (should fail)
        mock_version.major = 2
        mock_version.minor = 7
        result = self.run_install.verify_python_version()
        assert result is False


if __name__ == "__main__":
    pytest.main([__file__])
