"""
Targeted tests for hf_client.py missing lines to increase coverage.
"""

import pytest
import sys
import os
from unittest.mock import patch, Mock, MagicMock
import json
from collections.abc import Mapping

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


class TestHFClientMissingLines:
    """Test missing lines in HF Client for coverage."""

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

    def test_normalize_card_data_mapping_object(self):
        """Test _normalize_card_data with mapping object."""
        data = {"model": "test", "task": "classification"}
        result = _normalize_card_data(data)
        assert result == data

    def test_normalize_card_data_model_dump_method(self):
        """Test _normalize_card_data with model_dump method."""
        mock_obj = Mock()
        del mock_obj.to_dict
        mock_obj.data = "not_a_mapping"  # Non-mapping data
        mock_obj.model_dump.return_value = {"dumped": True}
        
        result = _normalize_card_data(mock_obj)
        assert result == {"dumped": True}

    def test_normalize_card_data_dict_method(self):
        """Test _normalize_card_data with dict method."""
        mock_obj = Mock()
        del mock_obj.to_dict
        mock_obj.data = "not_a_mapping"
        del mock_obj.model_dump
        mock_obj.dict.return_value = {"dict_data": True}
        
        result = _normalize_card_data(mock_obj)
        assert result == {"dict_data": True}

    def test_normalize_card_data_json_method(self):
        """Test _normalize_card_data with json method."""
        mock_obj = Mock()
        del mock_obj.to_dict
        mock_obj.data = "not_a_mapping"
        del mock_obj.model_dump
        del mock_obj.dict
        mock_obj.json.return_value = '{"json_data": true}'
        
        result = _normalize_card_data(mock_obj)
        assert result == {"json_data": True}

    def test_normalize_card_data_exception_handling(self):
        """Test _normalize_card_data exception handling."""
        mock_obj = Mock()
        mock_obj.to_dict.side_effect = Exception("Conversion failed")
        
        result = _normalize_card_data(mock_obj)
        assert result == {}

    def test_normalize_card_data_no_methods(self):
        """Test _normalize_card_data with object having no convertible methods."""
        # Basic object with no special methods
        obj = object()
        result = _normalize_card_data(obj)
        assert result == {}

    def test_timeout_http_adapter_send_with_timeout(self):
        """Test _TimeoutHTTPAdapter send method sets timeout."""
        adapter = _TimeoutHTTPAdapter(timeout=45)
        
        mock_request = Mock()
        mock_super_send = Mock(return_value="response")
        
        with patch('requests.adapters.HTTPAdapter.send', mock_super_send):
            result = adapter.send(mock_request)
            
            # Should have called super().send with timeout
            mock_super_send.assert_called_once_with(mock_request, timeout=45)
            assert result == "response"

    def test_timeout_http_adapter_send_preserve_existing_timeout(self):
        """Test _TimeoutHTTPAdapter preserves existing timeout in kwargs."""
        adapter = _TimeoutHTTPAdapter(timeout=30)
        
        mock_request = Mock()
        mock_super_send = Mock(return_value="response")
        
        with patch('requests.adapters.HTTPAdapter.send', mock_super_send):
            result = adapter.send(mock_request, timeout=60)  # Explicit timeout
            
            # Should use the explicit timeout, not adapter's default
            mock_super_send.assert_called_once_with(mock_request, timeout=60)
            assert result == "response"

    @patch('api.hf_client.requests.Session')
    def test_create_session_with_token(self, mock_session_class):
        """Test _create_session with token."""
        mock_session = Mock()
        mock_session_class.return_value = mock_session
        
        session = _create_session("test-token")
        
        # Should have updated headers with auth
        mock_session.headers.update.assert_called_once()
        headers_arg = mock_session.headers.update.call_args[0][0]
        assert "Authorization" in headers_arg
        assert headers_arg["Authorization"] == "Bearer test-token"

    @patch('api.hf_client.requests.Session')
    def test_create_session_without_token(self, mock_session_class):
        """Test _create_session without token."""
        mock_session = Mock()
        mock_session_class.return_value = mock_session
        
        session = _create_session(None)
        
        # Should have updated headers without auth
        mock_session.headers.update.assert_called_once()
        headers_arg = mock_session.headers.update.call_args[0][0]
        assert "Authorization" not in headers_arg

    def test_retry_success_first_attempt(self):
        """Test _retry with success on first attempt."""
        mock_operation = Mock(return_value="success")
        
        result = _retry(mock_operation, attempts=3)
        
        assert result == "success"
        mock_operation.assert_called_once()

    def test_retry_success_after_failures(self):
        """Test _retry with success after some failures."""
        mock_operation = Mock()
        # Fail twice, succeed on third attempt
        mock_operation.side_effect = [Exception("fail1"), Exception("fail2"), "success"]
        
        result = _retry(mock_operation, attempts=4)
        
        assert result == "success"
        assert mock_operation.call_count == 3

    def test_retry_exhaust_attempts(self):
        """Test _retry exhausting all attempts."""
        mock_operation = Mock()
        mock_operation.side_effect = Exception("persistent failure")
        
        with pytest.raises(Exception, match="persistent failure"):
            _retry(mock_operation, attempts=3)
        
        assert mock_operation.call_count == 3

    @patch('api.hf_client.os.getenv')
    @patch('api.hf_client.HfApi')
    def test_hf_client_token_priority(self, mock_hf_api, mock_getenv):
        """Test HFClient token environment variable priority."""
        # Mock different token sources with priority
        def getenv_side_effect(key):
            if key == "HUGGINGFACE_HUB_TOKEN":
                return "primary-token"
            elif key == "HUGGINGFACEHUB_API_TOKEN":
                return "fallback-token"
            elif key == "HF_TOKEN":
                return "third-token"
            return None
        
        mock_getenv.side_effect = getenv_side_effect
        
        client = HFClient()
        
        # Should use the first available token (primary)
        assert client.token == "primary-token"

    @patch('api.hf_client.os.getenv')
    @patch('api.hf_client.HfApi')
    def test_hf_client_no_token(self, mock_hf_api, mock_getenv):
        """Test HFClient with no token available."""
        mock_getenv.return_value = None
        
        client = HFClient()
        
        # Should have no token
        assert client.token is None

    @patch('api.hf_client._create_session')
    @patch('api.hf_client.os.getenv')  
    @patch('api.hf_client.HfApi')
    def test_hf_client_session_creation(self, mock_hf_api, mock_getenv, mock_create_session):
        """Test HFClient creates session with token."""
        mock_getenv.return_value = "test-token"
        mock_session = Mock()
        mock_create_session.return_value = mock_session
        
        client = HFClient()
        
        # Should have created session with token
        mock_create_session.assert_called_once_with("test-token")
        assert client._session == mock_session


class TestGitHubMatcher:
    """Test GitHubMatcher functionality."""

    def test_normalize_method(self):
        """Test GitHubMatcher._normalize method."""
        assert GitHubMatcher._normalize("Test-Model_Name") == "test-model-name"
        assert GitHubMatcher._normalize("BERT-base-uncased") == "bert-base-uncased"
        assert GitHubMatcher._normalize("model_v2.1") == "model-v2-1"

    def test_tokenize_method(self):
        """Test GitHubMatcher._tokenize method."""
        tokens = GitHubMatcher._tokenize("test-model-v2")
        assert "test" in tokens
        assert "model" in tokens
        # "v2" is too short (< 3 chars), so it won't be included

    def test_get_aliases_method(self):
        """Test GitHubMatcher._get_aliases method."""
        hf_id = "user/bert-model"
        card_data = {"model_name": "BERT Model"}
        
        aliases = GitHubMatcher._get_aliases(hf_id, card_data)
        
        assert isinstance(aliases, set)
        assert len(aliases) > 0

    def test_jaccard_similarity_method(self):
        """Test GitHubMatcher._jaccard_similarity method."""
        set1 = {"test", "model"}
        set2 = {"test", "project"}
        
        similarity = GitHubMatcher._jaccard_similarity(set1, set2)
        
        # Should be 1/3 = 0.333... (1 common element, 3 total unique)
        assert 0.3 < similarity < 0.4

    def test_jaccard_similarity_identical_sets(self):
        """Test Jaccard similarity with identical sets."""
        set1 = {"test", "model"}
        set2 = {"test", "model"}
        
        similarity = GitHubMatcher._jaccard_similarity(set1, set2)
        
        assert similarity == 1.0

    def test_jaccard_similarity_empty_sets(self):
        """Test Jaccard similarity with empty sets."""
        set1 = set()
        set2 = set()
        
        similarity = GitHubMatcher._jaccard_similarity(set1, set2)
        
        assert similarity == 0.0

    def test_version_bonus_method(self):
        """Test GitHubMatcher._version_bonus method."""
        # Test with version match
        bonus = GitHubMatcher._version_bonus("model-v1", "model-v1-repo")
        assert bonus > 0
        
        # Test without version match
        bonus_no_match = GitHubMatcher._version_bonus("model-v1", "different-repo")
        assert bonus_no_match == 0


class TestHFClientMethods:
    """Test HFClient methods that are missing coverage."""

    def setup_method(self):
        """Setup HFClient for testing."""
        with patch('api.hf_client.os.getenv') as mock_getenv:
            with patch('api.hf_client.HfApi'):
                with patch('api.hf_client._create_session') as mock_create_session:
                    mock_getenv.return_value = None
                    self.mock_session = Mock()
                    mock_create_session.return_value = self.mock_session
                    self.client = HFClient()

    def test_get_text_404_response(self):
        """Test _get_text with 404 response."""
        mock_response = Mock()
        mock_response.status_code = 404
        self.mock_session.get.return_value = mock_response
        
        result = self.client._get_text("https://example.com/not-found")
        
        assert result is None

    def test_get_text_401_response(self):
        """Test _get_text with 401 response."""
        from huggingface_hub.utils import GatedRepoError
        
        mock_response = Mock()
        mock_response.status_code = 401
        self.mock_session.get.return_value = mock_response
        
        with pytest.raises(GatedRepoError):
            self.client._get_text("https://example.com/gated")

    def test_get_text_success(self):
        """Test _get_text with successful response."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.text = "Success content"
        mock_response.raise_for_status.return_value = None
        self.mock_session.get.return_value = mock_response
        
        result = self.client._get_text("https://example.com/file.txt")
        
        assert result == "Success content"

    def test_get_json_404_response(self):
        """Test _get_json with 404 response."""
        mock_response = Mock()
        mock_response.status_code = 404
        self.mock_session.get.return_value = mock_response
        
        result = self.client._get_json("https://example.com/not-found.json")
        
        assert result is None

    def test_get_json_401_response(self):
        """Test _get_json with 401 response."""
        from huggingface_hub.utils import GatedRepoError
        
        mock_response = Mock()
        mock_response.status_code = 401
        self.mock_session.get.return_value = mock_response
        
        with pytest.raises(GatedRepoError):
            self.client._get_json("https://example.com/gated.json")

    def test_get_json_success(self):
        """Test _get_json with successful response."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"key": "value"}
        mock_response.raise_for_status.return_value = None
        self.mock_session.get.return_value = mock_response
        
        result = self.client._get_json("https://example.com/data.json")
        
        assert result == {"key": "value"}
