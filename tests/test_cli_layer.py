"""
Unit tests for CLI functionality in app.py.
"""

import pytest
from unittest.mock import patch, MagicMock, mock_open
import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from app import main, read_urls


class TestAppCLI:
    """Test suite for app.py CLI functionality."""

    def test_read_urls_success(self):
        """Test successful reading of URLs from file."""
        test_content = (
            "https://huggingface.co/bert-base-uncased\n"
            "https://github.com/pytorch/pytorch\n"
        )

        with patch('builtins.open',
                   mock_open(read_data=test_content)) as mock_file:
            urls = read_urls("test_urls.txt")

            assert len(urls) == 2
            assert "https://huggingface.co/bert-base-uncased" in urls
            assert "https://github.com/pytorch/pytorch" in urls
            mock_file.assert_called_once_with(
                "test_urls.txt", "r", encoding="ascii")

    def test_read_urls_file_not_found(self):
        """Test FileNotFoundError handling in read_urls."""
        with patch('builtins.open', side_effect=FileNotFoundError):
            with pytest.raises(SystemExit) as exc_info:
                read_urls("nonexistent.txt")

            assert exc_info.value.code == 1

    def test_read_urls_general_exception(self):
        """Test general exception handling in read_urls."""
        error = PermissionError("Permission denied")
        with patch('builtins.open', side_effect=error):
            with pytest.raises(SystemExit) as exc_info:
                read_urls("protected.txt")

            assert exc_info.value.code == 1

    def test_read_urls_empty_lines_filtered(self):
        """Test that empty lines are filtered out."""
        test_content = (
            "https://huggingface.co/bert-base-uncased\n"
            "\n\n"
            "https://github.com/pytorch/pytorch\n\n"
        )

        with patch('builtins.open',
                   mock_open(read_data=test_content)):
            urls = read_urls("test_urls.txt")

            assert len(urls) == 2
            assert "" not in urls

    @patch('api.hf_client.HFClient')
    @patch('url_router.UrlRouter')
    @patch('app.read_urls')
    def test_main_function_success(self, mock_read_urls,
                                   mock_url_router, mock_hf_client):
        """Test successful execution of main function."""
        # Setup mocks
        test_url = "https://huggingface.co/bert-base-uncased"
        mock_read_urls.return_value = [test_url]

        mock_parsed = MagicMock()
        mock_parsed.hf_id = "bert-base-uncased"
        mock_router_instance = MagicMock()
        mock_router_instance.parse.return_value = mock_parsed
        mock_url_router.return_value = mock_router_instance

        mock_client_instance = MagicMock()
        mock_hf_client.return_value = mock_client_instance

        # Test with mock sys.argv
        with patch('sys.argv', ['app.py', 'test_urls.txt']):
            result = main()

        # Verify calls
        mock_read_urls.assert_called_once_with('test_urls.txt')
        mock_router_instance.parse.assert_called_once_with(test_url)
        mock_client_instance.get_model_info.assert_called_once_with(
            "bert-base-uncased")
        assert result == 0

    @patch('api.hf_client.HFClient')
    @patch('url_router.UrlRouter')
    @patch('app.read_urls')
    def test_main_function_multiple_urls(self, mock_read_urls,
                                         mock_url_router, mock_hf_client):
        """Test main function with multiple URLs."""
        # Setup mocks
        test_urls = [
            "https://huggingface.co/bert-base-uncased",
            "https://huggingface.co/gpt2"
        ]
        mock_read_urls.return_value = test_urls

        mock_parsed1 = MagicMock()
        mock_parsed1.hf_id = "bert-base-uncased"
        mock_parsed2 = MagicMock()
        mock_parsed2.hf_id = "gpt2"

        mock_router_instance = MagicMock()
        mock_router_instance.parse.side_effect = [mock_parsed1, mock_parsed2]
        mock_url_router.return_value = mock_router_instance

        mock_client_instance = MagicMock()
        mock_hf_client.return_value = mock_client_instance

        # Test with mock sys.argv
        with patch('sys.argv', ['app.py', 'test_urls.txt']):
            result = main()

        # Verify calls
        assert mock_router_instance.parse.call_count == 2
        assert mock_client_instance.get_model_info.call_count == 2
        mock_client_instance.get_model_info.assert_any_call(
            "bert-base-uncased")
        mock_client_instance.get_model_info.assert_any_call("gpt2")
        assert result == 0

    def test_main_function_missing_argument(self):
        """Test main function with missing command line argument."""
        with patch('sys.argv', ['app.py']):  # Missing URL file argument
            with pytest.raises(IndexError):
                main()


if __name__ == "__main__":
    pytest.main([__file__])
