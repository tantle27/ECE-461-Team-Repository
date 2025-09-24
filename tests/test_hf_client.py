"""
Comprehensive tests for hf_client.py consolidated from coverage-focused files.
"""

import pytest
import sys
import os
from unittest.mock import patch, Mock

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from api.hf_client import (
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
        """Test _normalize_card_data fallback for string conversion."""
        class MockObj:
            def __str__(self):
                return '{"license": "MIT"}'
        
        mock_obj = MockObj()
        result = _normalize_card_data(mock_obj)
        # Should return empty dict when object can't be converted
        assert result == {}

    def test_create_session_function(self):
        """Test _create_session function creates session with timeout adapter."""
        with patch('api.hf_client.requests.Session') as mock_session_class:
            mock_session = Mock()
            mock_session_class.return_value = mock_session
            
            session = _create_session("test_token")
            
            # Should have mounted timeout adapter
            mock_session.mount.assert_called()
            assert session == mock_session

    def test_timeout_http_adapter_initialization(self):
        """Test _TimeoutHTTPAdapter initialization."""
        adapter = _TimeoutHTTPAdapter(timeout=30)
        assert adapter._timeout == 30

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


class TestGitHubMatcher:
    """Test GitHubMatcher functionality."""

    def test_github_matcher_normalize_method(self):
        """Test GitHubMatcher._normalize static method."""
        assert GitHubMatcher._normalize("hf-transformers") == "transformers"
        assert GitHubMatcher._normalize("HUGGINGFACE-models") == "models"
        assert "test" in GitHubMatcher._normalize("test-project-dev")  # More flexible test

    def test_github_matcher_tokenize_method(self):
        """Test GitHubMatcher._tokenize static method."""
        tokens = GitHubMatcher._tokenize("test-model-123")
        assert isinstance(tokens, set)
        assert "test" in tokens
        assert "model" in tokens


class TestHFClientInitialization:
    """Test HFClient initialization and configuration."""

    @patch.dict(os.environ, {'HF_TOKEN': 'test_token'}, clear=True)
    def test_hf_client_token_from_env(self):
        """Test HFClient token environment variable priority."""
        with patch('api.hf_client._create_session') as mock_create_session:
            mock_session = Mock()
            mock_create_session.return_value = mock_session
            
            # Mock requests module
            with patch('api.hf_client.requests') as mock_requests:
                mock_requests.Session.return_value = mock_session
                
                client = HFClient()
                assert client.token == 'test_token'

    @patch.dict(os.environ, {}, clear=True)
    def test_hf_client_no_token(self):
        """Test HFClient with no token available."""
        with patch('api.hf_client._create_session'):
            client = HFClient()
            assert client.token is None

    @patch.dict(os.environ, {'HF_TOKEN': 'env_token'}, clear=True)
    def test_hf_client_creates_session(self):
        """Test HFClient creates session with token."""
        with patch('api.hf_client._create_session') as mock_create:
            with patch('api.hf_client.HfApi'):
                mock_session = Mock()
                mock_create.return_value = mock_session
                
                client = HFClient()
                
                mock_create.assert_called_once_with('env_token')
                assert client._session == mock_session


class TestHFClientMethods:
    """Test HFClient methods that require coverage."""

    def setup_method(self):
        """Setup HFClient for testing."""
        with patch('api.hf_client._create_session'):
            with patch('api.hf_client.HfApi'):
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
        # Mock _get_text to return None (simulating file not found)
        with patch.object(self.client, '_get_text', return_value=None):
            result = self.client.get_readme("test-model")
            assert result is None


class TestHFClientEdgeCases:
    """Test edge cases and error conditions."""

    def setup_method(self):
        """Setup HFClient for testing."""
        with patch('api.hf_client._create_session'):
            with patch('api.hf_client.HfApi'):
                self.client = HFClient()

    def test_normalize_card_data_with_exception(self):
        """Test _normalize_card_data when to_dict raises exception."""
        mock_obj = Mock()
        mock_obj.to_dict.side_effect = Exception("Conversion failed")
        
        result = _normalize_card_data(mock_obj)
        assert result == {}

    def test_client_session_creation_error(self):
        """Test HFClient when session creation fails."""
        with patch('api.hf_client._create_session', 
                   side_effect=Exception("Session creation failed")):
            with patch('api.hf_client.HfApi'):
                with pytest.raises(Exception, match="Session creation failed"):
                    HFClient()


if __name__ == "__main__":
    pytest.main([__file__])
