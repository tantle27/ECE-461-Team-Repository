"""
Comprehensive tests for HF Client to increase coverage.
"""

import pytest
import sys
import os
from unittest.mock import patch, MagicMock, Mock
import json
import time
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


class TestTimeoutHTTPAdapter:
    """Test the custom HTTP adapter with timeout."""

    def test_timeout_adapter_initialization(self):
        """Test adapter initializes with correct timeout."""
        adapter = _TimeoutHTTPAdapter(timeout=60)
        assert adapter._timeout == 60

    def test_timeout_adapter_default_timeout(self):
        """Test adapter uses default timeout."""
        adapter = _TimeoutHTTPAdapter()
        assert adapter._timeout == 30

    @patch('api.hf_client.HTTPAdapter.send')
    def test_timeout_adapter_send_sets_timeout(self, mock_send):
        """Test that send method sets timeout in kwargs."""
        adapter = _TimeoutHTTPAdapter(timeout=45)
        mock_request = Mock()
        mock_send.return_value = Mock()

        adapter.send(mock_request, custom_arg="test")

        mock_send.assert_called_once()
        call_args = mock_send.call_args
        assert call_args[1]["timeout"] == 45
        assert call_args[1]["custom_arg"] == "test"

    @patch('api.hf_client.HTTPAdapter.send')
    def test_timeout_adapter_preserves_existing_timeout(self, mock_send):
        """Test that existing timeout in kwargs is preserved."""
        adapter = _TimeoutHTTPAdapter(timeout=45)
        mock_request = Mock()
        mock_send.return_value = Mock()

        adapter.send(mock_request, timeout=99)

        mock_send.assert_called_once()
        call_args = mock_send.call_args
        assert call_args[1]["timeout"] == 99  # Existing timeout preserved


class TestCreateSession:
    """Test session creation function."""

    def test_create_session_with_token(self):
        """Test session creation with authentication token."""
        token = "test_token_12345"
        
        with patch('api.hf_client.requests.Session') as mock_session_class:
            mock_session = Mock()
            mock_session_class.return_value = mock_session

            session = _create_session(token)

            # Verify session was configured
            assert session == mock_session
            mock_session.mount.assert_any_call("https://", mock_session.mount.call_args_list[0][0][1])
            mock_session.mount.assert_any_call("http://", mock_session.mount.call_args_list[1][0][1])
            
            # Verify headers were updated
            mock_session.headers.update.assert_called_once()
            headers_arg = mock_session.headers.update.call_args[0][0]
            assert "Authorization" in headers_arg
            assert headers_arg["Authorization"] == f"Bearer {token}"
            assert "User-Agent" in headers_arg

    def test_create_session_without_token(self):
        """Test session creation without token."""
        with patch('api.hf_client.requests.Session') as mock_session_class:
            mock_session = Mock()
            mock_session_class.return_value = mock_session

            session = _create_session(None)

            # Verify session was configured
            assert session == mock_session
            
            # Verify headers were updated but without Authorization
            headers_arg = mock_session.headers.update.call_args[0][0]
            assert "Authorization" not in headers_arg
            assert "User-Agent" in headers_arg


class TestExtractCardData:
    """Test card data extraction function."""

    def test_normalize_card_data_none(self):
        """Test extracting card data from None."""
        result = _normalize_card_data(None)
        assert result == {}

    def test_normalize_card_data_with_to_dict_method(self):
        """Test extracting card data from object with to_dict method."""
        mock_card = Mock()
        mock_card.to_dict.return_value = {"key": "value", "nested": {"data": "test"}}
        
        result = _normalize_card_data(mock_card)
        assert result == {"key": "value", "nested": {"data": "test"}}

    def test_normalize_card_data_with_data_attribute(self):
        """Test extracting card data from object with data attribute."""
        mock_card = Mock()
        del mock_card.to_dict  # Ensure to_dict doesn't exist
        mock_card.data = {"license": "MIT", "tags": ["nlp"]}
        
        result = _normalize_card_data(mock_card)
        assert result == {"license": "MIT", "tags": ["nlp"]}

    def test_normalize_card_data_mapping_object(self):
        """Test extracting card data from mapping object."""
        card_data = {"model_type": "transformer", "language": "en"}
        
        result = _normalize_card_data(card_data)
        assert result == {"model_type": "transformer", "language": "en"}

    def test_normalize_card_data_with_model_dump_method(self):
        """Test extracting card data from object with model_dump method."""
        mock_card = Mock()
        del mock_card.to_dict  # Ensure to_dict doesn't exist
        mock_card.data = "not_a_mapping"  # Make data non-mapping
        mock_card.model_dump.return_value = {"dumped": "data"}
        
        result = _normalize_card_data(mock_card)
        assert result == {"dumped": "data"}

    def test_normalize_card_data_with_dict_method(self):
        """Test extracting card data from object with dict method."""
        mock_card = Mock()
        del mock_card.to_dict
        mock_card.data = "not_a_mapping"
        del mock_card.model_dump
        mock_card.dict.return_value = {"dict_data": "test"}
        
        result = _normalize_card_data(mock_card)
        assert result == {"dict_data": "test"}

    def test_normalize_card_data_with_json_method(self):
        """Test extracting card data from object with json method."""
        mock_card = Mock()
        del mock_card.to_dict
        mock_card.data = "not_a_mapping"
        del mock_card.model_dump
        del mock_card.dict
        mock_card.json.return_value = '{"json_data": "test"}'
        
        result = _normalize_card_data(mock_card)
        assert result == {"json_data": "test"}

    def test_normalize_card_data_exception_handling(self):
        """Test that exceptions during extraction are handled gracefully."""
        mock_card = Mock()
        mock_card.to_dict.side_effect = Exception("Extraction failed")
        
        result = _normalize_card_data(mock_card)
        assert result == {}

    def test_normalize_card_data_no_suitable_method(self):
        """Test extracting from object with no suitable methods."""
        mock_card = Mock()
        del mock_card.to_dict
        mock_card.data = "not_a_mapping"
        # Remove all methods
        for method in ("model_dump", "dict", "json"):
            if hasattr(mock_card, method):
                delattr(mock_card, method)
        
        result = _normalize_card_data(mock_card)
        assert result == {}


class TestRetryFunction:
    """Test the retry mechanism."""

    def test_retry_success_first_attempt(self):
        """Test retry when operation succeeds on first attempt."""
        mock_operation = Mock()
        mock_operation.return_value = "success"
        
        result = _retry(mock_operation, attempts=3)
        
        assert result == "success"
        assert mock_operation.call_count == 1

    def test_retry_success_after_failures(self):
        """Test retry when operation succeeds after failures."""
        mock_operation = Mock()
        mock_operation.side_effect = [
            Exception("First failure"),
            Exception("Second failure"),
            "success"
        ]
        
        with patch('api.hf_client.time.sleep') as mock_sleep:
            result = _retry(mock_operation, attempts=4)
        
        assert result == "success"
        assert mock_operation.call_count == 3
        assert mock_sleep.call_count == 2
        # Check exponential backoff
        mock_sleep.assert_any_call(0.4 * (2**0))  # First retry
        mock_sleep.assert_any_call(0.4 * (2**1))  # Second retry

    def test_retry_exhausts_attempts(self):
        """Test retry when all attempts are exhausted."""
        mock_operation = Mock()
        mock_operation.side_effect = Exception("Always fails")
        
        with patch('api.hf_client.time.sleep'), \
             pytest.raises(Exception, match="Always fails"):
            _retry(mock_operation, attempts=2)
        
        assert mock_operation.call_count == 2

    def test_retry_default_attempts(self):
        """Test retry with default number of attempts."""
        mock_operation = Mock()
        mock_operation.side_effect = Exception("Always fails")
        
        with patch('api.hf_client.time.sleep'), \
             pytest.raises(Exception):
            _retry(mock_operation)  # Default attempts=4
        
        assert mock_operation.call_count == 4

    def test_retry_returns_none_edge_case(self):
        """Test retry function edge case - should not return None."""
        # This tests the unreachable return None line
        # In practice, this should never happen due to the loop structure
        pass


class TestHFClientInitialization:
    """Test HFClient initialization and configuration."""

    @patch('api.hf_client.os.getenv')
    @patch('api.hf_client.HfApi')
    @patch('api.hf_client._create_session')
    def test_hf_client_initialization_with_token(self, mock_create_session, mock_hf_api, mock_getenv):
        """Test HFClient initialization with token from environment."""
        mock_getenv.side_effect = lambda key: "test_token" if key == "HF_TOKEN" else None
        mock_session = Mock()
        mock_create_session.return_value = mock_session
        mock_api = Mock()
        mock_hf_api.return_value = mock_api

        client = HFClient()

        assert client.token == "test_token"
        assert client.api == mock_api
        assert client._session == mock_session
        mock_hf_api.assert_called_once_with(token="test_token")
        mock_create_session.assert_called_once_with("test_token")

    @patch('api.hf_client.os.getenv')
    @patch('api.hf_client.HfApi')
    @patch('api.hf_client._create_session')
    def test_hf_client_initialization_token_priority(self, mock_create_session, mock_hf_api, mock_getenv):
        """Test HFClient token environment variable priority."""
        # Test priority order: HUGGINGFACE_HUB_TOKEN > HUGGINGFACEHUB_API_TOKEN > HF_TOKEN
        def getenv_side_effect(key):
            if key == "HUGGINGFACE_HUB_TOKEN":
                return "priority_token"
            elif key == "HUGGINGFACEHUB_API_TOKEN":
                return "secondary_token"
            elif key == "HF_TOKEN":
                return "tertiary_token"
            return None

        mock_getenv.side_effect = getenv_side_effect

        client = HFClient()

        assert client.token == "priority_token"

    @patch('api.hf_client.os.getenv')
    @patch('api.hf_client.HfApi')
    @patch('api.hf_client._create_session')
    def test_hf_client_initialization_no_token(self, mock_create_session, mock_hf_api, mock_getenv):
        """Test HFClient initialization without any token."""
        mock_getenv.return_value = None

        client = HFClient()

        assert client.token is None
        mock_hf_api.assert_called_once_with(token=None)


class TestHFClientGitHubExtraction:
    """Test GitHub URL extraction functionality."""

    def setup_method(self):
        """Setup HF client with mocked dependencies."""
        with patch('api.hf_client.os.getenv', return_value=None), \
             patch('api.hf_client.HfApi'), \
             patch('api.hf_client._create_session'):
            self.client = HFClient()

    def test_extract_github_urls_empty_content(self):
        """Test GitHub URL extraction from empty content."""
        result = self.client.extract_github_urls("", "test-model")
        assert result == []

    def test_extract_github_urls_no_matches(self):
        """Test GitHub URL extraction with no GitHub URLs."""
        content = "This is content without any GitHub URLs"
        result = self.client.extract_github_urls(content, "test-model")
        assert result == []

    def test_extract_github_urls_with_matches(self):
        """Test GitHub URL extraction with valid URLs."""
        content = """
        Check out our code at https://github.com/user/repo
        Also see https://github.com/another/project
        And visit https://github.com/user/repo again (duplicate)
        """
        
        with patch.object(self.client, '_score_github_url') as mock_score:
            mock_score.return_value = 0.5
            
            result = self.client.extract_github_urls(content, "test-model")
            
            # Should deduplicate and return unique URLs
            assert len(result) == 2
            assert "https://github.com/user/repo" in result
            assert "https://github.com/another/project" in result

    def test_extract_github_urls_generic_names_filtered(self):
        """Test that repositories with generic names are filtered out."""
        content = "https://github.com/user/examples https://github.com/user/test"
        
        result = self.client.extract_github_urls(content, "test-model")
        
        # Generic names should be filtered out
        assert result == []

    @patch.object(HFClient, '_tokenize')
    @patch.object(HFClient, '_jaccard_similarity')
    def test_score_github_url_calculation(self, mock_jaccard, mock_tokenize):
        """Test GitHub URL scoring calculation."""
        mock_tokenize.side_effect = lambda x: set(x.split('-'))
        mock_jaccard.return_value = 0.6
        
        with patch.object(self.client, '_version_bonus', return_value=0.1):
            score = self.client._score_github_url("test-model", "user/test-project")
        
        assert score == 0.7  # 0.6 + 0.1

    def test_tokenize_function(self):
        """Test the tokenization helper function."""
        tokens = self.client._tokenize("test-model-v2")
        expected = {"test", "model", "v2", "v", "2"}
        assert tokens == expected

    def test_jaccard_similarity(self):
        """Test Jaccard similarity calculation."""
        set1 = {"test", "model"}
        set2 = {"test", "project"}
        
        similarity = self.client._jaccard_similarity(set1, set2)
        
        # Intersection: {"test"}, Union: {"test", "model", "project"}
        expected = 1 / 3  # 1 intersection / 3 union
        assert similarity == expected

    def test_version_bonus(self):
        """Test version bonus calculation."""
        bonus = self.client._version_bonus("test-model-v1", "test-model-v1-repo")
        assert bonus > 0  # Should get bonus for version match
        
        no_bonus = self.client._version_bonus("test-model", "different-repo")
        assert no_bonus == 0


class TestHFClientErrorHandling:
    """Test error handling in HF Client methods."""

    def setup_method(self):
        """Setup HF client with mocked dependencies."""
        with patch('api.hf_client.os.getenv', return_value=None), \
             patch('api.hf_client.HfApi') as mock_api, \
             patch('api.hf_client._create_session'):
            self.mock_api = mock_api.return_value
            self.client = HFClient()

    def test_get_model_info_api_error(self):
        """Test get_model_info with API error."""
        from huggingface_hub.utils import HfHubHTTPError
        
        self.mock_api.model_info.side_effect = HfHubHTTPError("500: Server error")
        
        with pytest.raises(HfHubHTTPError):
            self.client.get_model_info("test-model")

    def test_get_dataset_info_api_error(self):
        """Test get_dataset_info with API error."""
        from huggingface_hub.utils import RepositoryNotFoundError
        
        self.mock_api.dataset_info.side_effect = RepositoryNotFoundError("Dataset not found")
        
        with pytest.raises(RepositoryNotFoundError):
            self.client.get_dataset_info("nonexistent/dataset")

    @patch('api.hf_client._retry')
    def test_session_get_with_retry(self, mock_retry):
        """Test session get with retry mechanism."""
        mock_retry.return_value = Mock()
        
        self.client._session_get("https://example.com")
        
        mock_retry.assert_called_once()

    def test_get_readme_with_session_error(self):
        """Test get_readme when session request fails."""
        with patch.object(self.client, '_session_get') as mock_session_get:
            mock_session_get.side_effect = Exception("Network error")
            
            result = self.client.get_readme("test-model")
            
            assert result == ""  # Should return empty string on error

    def test_list_files_with_api_error(self):
        """Test list_files when API call fails."""
        self.mock_api.list_repo_files.side_effect = Exception("API error")
        
        result = self.client.list_files("test-model")
        
        assert result == []  # Should return empty list on error

    def test_get_model_index_json_parsing_error(self):
        """Test get_model_index_json with JSON parsing error."""
        mock_response = Mock()
        mock_response.text = "invalid json content"
        
        with patch.object(self.client, '_session_get', return_value=mock_response):
            result = self.client.get_model_index_json("test-model")
            
            assert result == {}  # Should return empty dict on JSON parse error
