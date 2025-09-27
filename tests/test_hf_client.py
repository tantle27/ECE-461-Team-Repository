"""
Comprehensive unit tests for hf_client.py aligned with the refactored client
(no token handling, public endpoints only).
"""

import pytest
import sys
import os
from unittest.mock import patch, Mock

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from src.api.hf_client import (  # noqa: E402
    HFClient,
    _create_session,
    _TimeoutHTTPAdapter,
    _normalize_card_data,
    _retry,
    GitHubMatcher
)


class TestHFClientHelpers:
    """Test helper functions in HF Client module."""

    def test_normalize_card_data_none_input(self):
        """Test _normalize_card_data with None input."""
        result = _normalize_card_data(None)
        assert result == {}

    def test_normalize_card_data_to_dict_method(self):
        """Test _normalize_card_data with object having to_dict method."""
        mock_obj = Mock()
        mock_obj.to_dict.return_value = {"key": "value"}

        result = _normalize_card_data(mock_obj)
        assert result == {"key": "value"}

    def test_normalize_card_data_data_attribute(self):
        """Test _normalize_card_data with object having data attribute."""
        mock_obj = Mock()
        # Remove to_dict to test fallback
        del mock_obj.to_dict
        mock_obj.data = {"license": "MIT"}

        result = _normalize_card_data(mock_obj)
        assert result == {"license": "MIT"}

    def test_normalize_card_data_dict_input(self):
        """Test _normalize_card_data with dict input."""
        data = {"license": "MIT", "tags": ["nlp"]}
        result = _normalize_card_data(data)
        assert result == data

    def test_normalize_card_data_fallback_str(self):
        """Test _normalize_card_data fallback for unknown objects."""
        class MockObj:
            def __str__(self):
                return '{"license": "MIT"}'

        mock_obj = MockObj()
        result = _normalize_card_data(mock_obj)
        # Should return empty dict when object can't be converted via known APIs
        assert result == {}

    def test_create_session_function(self):
        """Test _create_session function creates session with timeout adapter."""
        with patch('src.api.hf_client.requests.Session') as mock_session_class:
            mock_session = Mock()
            mock_session_class.return_value = mock_session

            session = _create_session()  # <- no token

            # Should have mounted timeout adapter
            mock_session.mount.assert_called()
            assert session == mock_session

    def test_timeout_http_adapter_initialization(self):
        """Test _TimeoutHTTPAdapter initialization."""
        adapter = _TimeoutHTTPAdapter(timeout=30)
        assert adapter._timeout == 30

    def test_timeout_adapter_send_method(self):
        """Test TimeoutHTTPAdapter send method with timeout override."""
        adapter = _TimeoutHTTPAdapter(timeout=10)

        # Mock parent send method
        with patch('requests.adapters.HTTPAdapter.send') as mock_send:
            mock_response = Mock()
            mock_send.return_value = mock_response

            mock_request = Mock()
            result = adapter.send(mock_request)

            # Should use adapter's timeout when none provided in kwargs
            mock_send.assert_called_once_with(mock_request, timeout=10)
            assert result == mock_response

    def test_retry_decorator_success(self):
        """Test _retry function with successful call."""
        call_count = 0

        def test_function():
            nonlocal call_count
            call_count += 1
            return "success"

        result = _retry(test_function)
        assert result == "success"
        assert call_count == 1

    def test_retry_decorator_with_retries(self):
        """Test _retry function with retries."""
        call_count = 0

        def test_function():
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise Exception("retry needed")
            return "success"

        result = _retry(test_function, attempts=3)
        assert result == "success"
        assert call_count == 2

    def test_retry_decorator_final_failure(self):
        """Test _retry function when all attempts fail."""
        call_count = 0

        def failing_function():
            nonlocal call_count
            call_count += 1
            raise ValueError("persistent error")

        with pytest.raises(ValueError, match="persistent error"):
            _retry(failing_function, attempts=2)
        assert call_count == 2

    def test_normalize_card_data_with_different_objects(self):
        """Test _normalize_card_data function with various input types."""
        # Test with dict-like object
        class DictLike:
            def __init__(self, data):
                self.data = data

        dict_obj = DictLike({"test": "value"})
        result = _normalize_card_data(dict_obj)
        assert result == {"test": "value"}

        # Test with object that has model_dump method
        class ModelDumpObject:
            def model_dump(self):
                return {"model": "data"}

        model_obj = ModelDumpObject()
        result = _normalize_card_data(model_obj)
        assert result == {"model": "data"}

        # Test with object that has dict method
        class DictMethodObject:
            def dict(self):
                return {"dict": "method"}

        dict_method_obj = DictMethodObject()
        result = _normalize_card_data(dict_method_obj)
        assert result == {"dict": "method"}

        # Test with object that has json method
        class JsonObject:
            def json(self):
                return '{"json": "data"}'

        json_obj = JsonObject()
        result = _normalize_card_data(json_obj)
        assert result == {"json": "data"}

        # Test with object that raises exception
        class ExceptionObject:
            def model_dump(self):
                raise Exception("test error")

        exc_obj = ExceptionObject()
        result = _normalize_card_data(exc_obj)
        assert result == {}

        # Test with None
        result = _normalize_card_data(None)
        assert result == {}


class TestGitHubMatcher:
    """Test GitHubMatcher functionality."""

    def test_github_matcher_normalize_method(self):
        """Test GitHubMatcher._normalize static method."""
        assert GitHubMatcher._normalize("hf-transformers") == "transformers"
        assert GitHubMatcher._normalize("HUGGINGFACE-models") == "models"
        assert "test" in GitHubMatcher._normalize("test-project-dev")

    def test_github_matcher_tokenize_method(self):
        """Test GitHubMatcher._tokens static method."""
        tokens = GitHubMatcher._tokens("test-model-123")
        assert isinstance(tokens, set)
        assert "test" in tokens
        assert "model" in tokens


class TestHFClientInitialization:
    """Test HFClient initialization and configuration (no token path)."""

    def test_hf_client_initializes_without_token(self):
        """HFClient should initialize a session and an HfApi instance."""
        with patch('src.api.hf_client._create_session') as mock_create:
            with patch('src.api.hf_client.HfApi') as mock_hfapi:
                mock_session = Mock()
                mock_create.return_value = mock_session

                client = HFClient()

                mock_create.assert_called_once_with()
                mock_hfapi.assert_called_once_with()
                assert getattr(client, "_session", None) is mock_session
                assert getattr(client, "api", None) is not None


class TestHFClientMethods:
    """Test HFClient methods that require coverage."""

    def setup_method(self):
        """Setup HFClient for testing."""
        with patch('src.api.hf_client._create_session'):
            with patch('src.api.hf_client.HfApi'):
                self.client = HFClient()

    def test_list_files_with_attribute_error_handling(self):
        """Test HFClient.list_files with AttributeError handling."""
        # Mock api to raise AttributeError for get_paths_info
        mock_api = Mock()
        mock_api.list_repo_files.return_value = ['file1.txt', 'file2.py']
        mock_api.get_paths_info.side_effect = AttributeError("Missing attribute")

        with patch.object(self.client, 'api', mock_api):
            result = self.client.list_files("test-model")
            assert len(result) == 2
            assert all(f.size is None for f in result)

    def test_get_readme_returns_none_for_not_found(self):
        """Test get_readme returns None when README files are not found."""
        # Patch _text_get to always return None, simulating file not found
        with patch("src.api.hf_client._text_get", return_value=None):
            result = self.client.get_readme("test-model")
            assert result is None

    def test_get_dataset_info_basic(self):
        """Test get_dataset_info method via HfApi normalization path."""
        mock_api = Mock()
        mock_info = Mock()
        mock_info.modelId = "test-dataset"
        mock_info.tags = ["dataset"]
        mock_info.likes = 50
        mock_info.downloads = 1000
        mock_info.cardData = {"license": "apache-2.0"}
        mock_info.createdAt = "2023-01-01"
        mock_info.lastModified = "2023-06-01"
        mock_info.gated = False
        mock_info.private = False

        mock_info.datasetId = "test-dataset"  # Ensure fallback logic sets correct hf_id
        mock_api.dataset_info.return_value = mock_info

        # Patch _api_dataset_json to return None so the fallback path is used
        with patch.object(self.client, 'api', mock_api):
            with patch.object(self.client, '_api_dataset_json', return_value=None):
                result = self.client.get_dataset_info("test-dataset")
                mock_api.dataset_info.assert_called_once_with("test-dataset")
                assert getattr(result, 'hf_id', None) == "test-dataset"
                assert getattr(result, 'card_data', None) == {"license": "apache-2.0"}
                assert getattr(result, 'tags', None) == ["dataset"]
                assert getattr(result, 'likes', None) == 50
                assert getattr(result, 'downloads_30d', None) == 1000
                assert getattr(result, 'private', None) is False

    def test_get_model_readme_method(self):
        """Test get_model_readme convenience method."""
        with patch.object(self.client, 'get_readme',
                          return_value="Model README") as mock_get_readme:
            result = self.client.get_model_readme("test-model")

            mock_get_readme.assert_called_once_with("test-model",
                                                    revision="main",
                                                    repo_type="model")
            assert result == "Model README"

    def test_get_dataset_readme_method(self):
        """Test get_dataset_readme convenience method."""
        with patch.object(self.client, 'get_readme',
                          return_value="Dataset README") as mock_get_readme:
            result = self.client.get_dataset_readme("test-dataset")

            mock_get_readme.assert_called_once_with("test-dataset",
                                                    revision="main",
                                                    repo_type="dataset")
            assert result == "Dataset README"

    def test_get_github_urls_with_content(self):
        """Test get_github_urls extracting URLs from README."""
        readme_content = """
        # Test Model
        Code: https://github.com/user/repo
        Also: https://github.com/another/project
        """

        with patch.object(self.client, 'get_readme', return_value=readme_content):
            urls = self.client.get_github_urls("test-model")

            assert len(urls) >= 1  # At least one URL should be found
            assert "https://github.com/user/repo" in urls


class TestHFClientEdgeCases:
    """Test edge cases and error conditions."""

    def setup_method(self):
        """Setup HFClient for testing."""
        with patch('src.api.hf_client._create_session'):
            with patch('src.api.hf_client.HfApi'):
                self.client = HFClient()

    def test_normalize_card_data_with_exception(self):
        """Test _normalize_card_data when to_dict raises exception."""
        mock_obj = Mock()
        mock_obj.to_dict.side_effect = Exception("Conversion failed")

        result = _normalize_card_data(mock_obj)
        assert result == {}

    def test_client_session_creation_error(self):
        """Test HFClient when session creation fails."""
        with patch('src.api.hf_client._create_session',
                   side_effect=Exception("Session creation failed")):
            with patch('src.api.hf_client.HfApi'):
                with pytest.raises(Exception, match="Session creation failed"):
                    HFClient()


if __name__ == "__main__":
    pytest.main([__file__])
