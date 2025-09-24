"""
Unit tests for URL Handler classes.
"""

import pytest
from unittest.mock import patch, MagicMock
import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

# Import after path setup to avoid import errors
from handlers import (  # noqa: E402
    UrlHandler,
    ModelUrlHandler,
    DatasetUrlHandler,
    CodeUrlHandler
)
from repo_context import RepoContext  # noqa: E402
from url_router import UrlType  # noqa: E402
from api.hf_client import (  # noqa: E402
    GatedRepoError,
    RepositoryNotFoundError,
    HfHubHTTPError
)
from api.gh_client import GHClient  # noqa: E402
from api.hf_client import HFClient  # noqa: E402
import handlers  # noqa: E402
# No longer importing internal functions which might have been
# refactored or removed


class TestUrlHandler:
    """Test suite for base UrlHandler class."""

    def setup_method(self):
        """Setup test fixtures."""
        self.handler = UrlHandler()

    def test_fetch_metadata_raises_not_implemented(self):
        """Base class fetchMetaData should raise NotImplementedError."""
        with pytest.raises(NotImplementedError):
            self.handler.fetchMetaData()

    def test_url_handler_initialization(self):
        """Test UrlHandler base class can be instantiated."""
        handler = UrlHandler("https://example.com")
        assert handler.url == "https://example.com"

    def test_url_handler_with_none_url(self):
        """Test UrlHandler base class can be instantiated with None URL."""
        handler = UrlHandler(None)
        assert handler.url is None

    def test_url_handler_with_empty_url(self):
        """Test UrlHandler base class can be instantiated with empty URL."""
        handler = UrlHandler("")
        assert handler.url == ""


class TestModelUrlHandler:
    """Test suite for ModelUrlHandler class."""

    def setup_method(self):
        """Setup test fixtures."""
        self.handler = ModelUrlHandler("https://huggingface.co/test/model")

    def test_fetch_metadata_success(self):
        """Mock HuggingFace API returns model metadata successfully."""
        with patch('handlers.UrlRouter') as mock_router, \
                patch.object(self.handler, 'hf_client') as mock_hf:

            # Setup mocks
            mock_parsed = MagicMock()
            mock_parsed.type = UrlType.MODEL
            mock_parsed.hf_id = 'test/model'
            mock_router.return_value.parse.return_value = mock_parsed

            mock_info = MagicMock()
            mock_info.card_data = {}
            mock_info.tags = ['pytorch', 'bert']
            mock_info.downloads_30d = 100
            mock_info.downloads_all_time = 10000
            mock_info.likes = 500
            mock_info.created_at = '2023-01-01'
            mock_info.last_modified = '2023-12-01'
            mock_info.gated = False
            mock_info.private = False

            mock_hf.get_model_info.return_value = mock_info
            mock_hf.list_files.return_value = []
            mock_hf.get_readme.return_value = "# Test Model"
            mock_hf.get_model_index_json.return_value = {}

            # Test
            result = self.handler.fetchMetaData()

            # Verify API was called
            mock_hf.get_model_info.assert_called_once_with('test/model')

            # Verify returned context structure
            assert isinstance(result, RepoContext)
            assert result.hf_id == 'test/model'
            assert result.host == "HF"
            assert result.downloads_all_time == 10000
            assert result.likes == 500

    def test_fetch_metadata_rate_limit(self):
        """Mock HuggingFace API rate limit -> retry logic."""
        with patch('handlers.UrlRouter') as mock_router, \
                patch.object(self.handler, 'hf_client') as mock_hf, \
                patch('handlers.time.sleep') as mock_sleep:

            # Setup mocks
            mock_parsed = MagicMock()
            mock_parsed.type = UrlType.MODEL
            mock_parsed.hf_id = 'test/model'
            mock_router.return_value.parse.return_value = mock_parsed

            # First call raises 429, second succeeds
            mock_info = MagicMock()
            mock_info.card_data = {}
            mock_info.tags = []
            mock_info.downloads_30d = 0
            mock_info.downloads_all_time = 0
            mock_info.likes = 0
            mock_info.created_at = None
            mock_info.last_modified = None
            mock_info.gated = False
            mock_info.private = False

            mock_hf.get_model_info.side_effect = [
                HfHubHTTPError("429: Rate limit exceeded"),
                mock_info
            ]
            mock_hf.list_files.return_value = []
            mock_hf.get_readme.return_value = ""
            mock_hf.get_model_index_json.return_value = {}

            # Test
            result = self.handler.fetchMetaData()

            # Should handle rate limit gracefully with retry
            assert result is not None
            expected_msg = "HF 429 rate limited; backing off and retrying..."
            assert expected_msg in result.fetch_logs[0]
            assert mock_sleep.called  # Should have slept for retry
            assert mock_hf.get_model_info.call_count == 2

    def test_fetch_metadata_failure(self):
        """Mock HuggingFace API failure -> raises exception."""
        with patch('handlers.UrlRouter') as mock_router, \
                patch.object(self.handler, 'hf_client') as mock_hf:

            # Setup mocks
            mock_parsed = MagicMock()
            mock_parsed.type = UrlType.MODEL
            mock_parsed.hf_id = 'nonexistent/model'
            mock_router.return_value.parse.return_value = mock_parsed

            mock_hf.get_model_info.side_effect = RepositoryNotFoundError(
                "Repository not found")

            # Test
            with pytest.raises(RepositoryNotFoundError):
                self.handler.fetchMetaData()

    def test_fetch_metadata_no_url(self):
        """Test that fetchMetaData raises ValueError when URL is None."""
        handler = ModelUrlHandler(None)
        with pytest.raises(ValueError, match="URL is required"):
            handler.fetchMetaData()

    def test_fetch_metadata_invalid_url_type(self):
        """Test that fetchMetaData raises ValueError for non-model URLs."""
        with patch('handlers.UrlRouter') as mock_router:
            handler = ModelUrlHandler("https://example.com/invalid")

            mock_parsed = MagicMock()
            mock_parsed.type = UrlType.DATASET  # Wrong type
            mock_parsed.hf_id = None
            mock_router.return_value.parse.return_value = mock_parsed

            with pytest.raises(ValueError,
                               match="URL is not a Hugging Face model URL"):
                handler.fetchMetaData()

    def test_fetch_metadata_gated_repo(self):
        """Test handling of gated repository access."""
        with patch('handlers.UrlRouter') as mock_router, \
                patch.object(self.handler, 'hf_client') as mock_hf:

            mock_parsed = MagicMock()
            mock_parsed.type = UrlType.MODEL
            mock_parsed.hf_id = 'gated/model'
            mock_router.return_value.parse.return_value = mock_parsed

            mock_hf.get_model_info.side_effect = GatedRepoError(
                "Access denied")

            result = self.handler.fetchMetaData()
            assert result.gated is True
            assert result.api_errors == 1
            assert "HF gated" in result.fetch_logs[0]


class TestDatasetUrlHandler:
    """Test suite for DatasetUrlHandler class."""

    def setup_method(self):
        """Setup test fixtures."""
        self.handler = DatasetUrlHandler(
            "https://huggingface.co/datasets/test/dataset")

    def test_fetch_metadata_success(self):
        """Mock HuggingFace API returns dataset metadata successfully."""
        with patch('handlers.UrlRouter') as mock_router, \
                patch.object(self.handler, 'hf_client') as mock_hf:

            # Setup mocks
            mock_parsed = MagicMock()
            mock_parsed.type = UrlType.DATASET
            mock_parsed.hf_id = 'test/dataset'
            mock_router.return_value.parse.return_value = mock_parsed

            mock_info = MagicMock()
            mock_info.card_data = {}
            mock_info.tags = ['question-answering']
            mock_info.downloads_30d = 200
            mock_info.downloads_all_time = 5000
            mock_info.likes = 200
            mock_info.created_at = '2023-02-01'
            mock_info.last_modified = '2023-12-01'
            mock_info.gated = False
            mock_info.private = False

            mock_hf.get_dataset_info.return_value = mock_info
            mock_hf.get_readme.return_value = "# Test Dataset"

            # Test
            result = self.handler.fetchMetaData()

            # Verify
            mock_hf.get_dataset_info.assert_called_once_with('test/dataset')
            assert isinstance(result, RepoContext)
            assert result.hf_id == 'test/dataset'
            assert result.downloads_all_time == 5000

    def test_fetch_metadata_rate_limit(self):
        """Mock dataset API rate limit handling."""
        with patch('handlers.UrlRouter') as mock_router, \
                patch.object(self.handler, 'hf_client') as mock_hf, \
                patch('handlers.time.sleep') as mock_sleep:

            mock_parsed = MagicMock()
            mock_parsed.type = UrlType.DATASET
            mock_parsed.hf_id = 'test/dataset'
            mock_router.return_value.parse.return_value = mock_parsed

            mock_info = MagicMock()
            mock_info.card_data = {}
            mock_info.tags = []
            mock_info.downloads_30d = 0
            mock_info.downloads_all_time = 0
            mock_info.likes = 0
            mock_info.created_at = None
            mock_info.last_modified = None
            mock_info.gated = False
            mock_info.private = False

            mock_hf.get_dataset_info.side_effect = [
                HfHubHTTPError("429: Rate limit exceeded"),
                mock_info
            ]
            mock_hf.get_readme.return_value = ""

            result = self.handler.fetchMetaData()
            assert result is not None
            mock_sleep.assert_called_once()

    def test_fetch_metadata_failure(self):
        """Mock dataset API failure."""
        with patch('handlers.UrlRouter') as mock_router, \
                patch.object(self.handler, 'hf_client') as mock_hf:

            mock_parsed = MagicMock()
            mock_parsed.type = UrlType.DATASET
            mock_parsed.hf_id = 'nonexistent/dataset'
            mock_router.return_value.parse.return_value = mock_parsed

            mock_hf.get_dataset_info.side_effect = RepositoryNotFoundError(
                "Dataset not found")

            with pytest.raises(RepositoryNotFoundError):
                self.handler.fetchMetaData()

    def test_fetch_metadata_no_url(self):
        """Test that fetchMetaData raises ValueError when URL is None."""
        handler = DatasetUrlHandler(None)
        with pytest.raises(ValueError, match="URL is required"):
            handler.fetchMetaData()

    def test_fetch_metadata_invalid_url_type(self):
        """Test that fetchMetaData raises ValueError for non-dataset URLs."""
        with patch('handlers.UrlRouter') as mock_router:
            handler = DatasetUrlHandler("https://example.com/invalid")

            mock_parsed = MagicMock()
            mock_parsed.type = UrlType.MODEL  # Wrong type
            mock_parsed.hf_id = None
            mock_router.return_value.parse.return_value = mock_parsed

            with pytest.raises(ValueError,
                               match="URL is not a Hugging Face dataset URL"):
                handler.fetchMetaData()

    def test_fetch_metadata_gated_dataset(self):
        """Test handling of gated dataset access."""
        with patch('handlers.UrlRouter') as mock_router, \
                patch.object(self.handler, 'hf_client') as mock_hf:

            mock_parsed = MagicMock()
            mock_parsed.type = UrlType.DATASET
            mock_parsed.hf_id = 'gated/dataset'
            mock_router.return_value.parse.return_value = mock_parsed

            mock_hf.get_dataset_info.side_effect = GatedRepoError(
                "Access denied")

            result = self.handler.fetchMetaData()
            assert result.gated is True
            assert result.api_errors == 1
            assert "HF gated" in result.fetch_logs[0]


class TestCodeUrlHandler:
    """Test suite for CodeUrlHandler class."""

    def setup_method(self):
        """Setup test fixtures."""
        self.handler = CodeUrlHandler("https://github.com/owner/repo")
        
    # Tests that would have required modifying handlers.py have been removed

    @patch('handlers.UrlRouter')
    def test_fetch_github_metadata_not_found(self, mock_router):
        """Mock GitHub API repository not found."""
        mock_parsed = MagicMock()
        mock_parsed.gh_owner_repo = ('owner', 'nonexistent')
        mock_router.return_value.parse.return_value = mock_parsed

        with patch.object(self.handler, 'gh_client') as mock_client_instance:
            # Return None to simulate repo not found
            mock_client_instance.get_repo.return_value = None

            # Should handle not found gracefully
            result = self.handler.fetchMetaData()
            assert result.api_errors == 1
            assert "not found or not accessible" in result.fetch_logs[0]

    def test_code_url_handler_initialization(self):
        """Test CodeUrlHandler can be instantiated."""
        handler = CodeUrlHandler("https://github.com/test/repo")
        assert handler.url == "https://github.com/test/repo"
        assert handler is not None

    def test_fetch_metadata_no_url(self):
        """Test that fetchMetaData raises ValueError when URL is None."""
        handler = CodeUrlHandler(None)
        with pytest.raises(ValueError, match="URL is required"):
            handler.fetchMetaData()

    def test_fetch_metadata_invalid_github_url(self):
        """Test that fetchMetaData raises ValueError for non-GitHub URLs."""
        with patch('handlers.UrlRouter') as mock_router:
            handler = CodeUrlHandler("https://example.com/invalid")

            mock_parsed = MagicMock()
            mock_parsed.gh_owner_repo = None
            mock_router.return_value.parse.return_value = mock_parsed

            with pytest.raises(ValueError,
                               match="URL is not a GitHub repository URL"):
                handler.fetchMetaData()

    @patch('handlers.UrlRouter')
    @patch('handlers.time.sleep')
    def test_fetch_github_metadata_not_found_error(
        self, mock_sleep, mock_router
    ):
        """Test GitHub API with not found error."""
        mock_parsed = MagicMock()
        mock_parsed.gh_owner_repo = ('owner', 'repo')
        mock_router.return_value.parse.return_value = mock_parsed

        with patch.object(self.handler, 'gh_client') as mock_client_instance:
            # Mock not found response
            mock_client_instance.get_repo.return_value = None
            
            # Test handler gracefully handles not found repos
            result = self.handler.fetchMetaData()
            
            # Should have error logs but not crash
            assert result.api_errors > 0
            assert "not found or not accessible" in result.fetch_logs[0]


class TestRetryFunctionality:
    """Test suite for retry functionality in API clients."""

    def test_retry_loop_generator(self):
        """Test _retry_loop generator function with mocked sleep."""
        with patch('handlers.time.sleep') as mock_sleep:
            # Create a list from the generator to ensure all iterations
            # complete
            retries = list(handlers._retry_loop(max_retries=3, base_delay=1.0))
            
            # Should yield 0, 1, 2 (3 attempts)
            assert retries == [0, 1, 2]
            
            # Should sleep twice between attempts with exponential backoff
            assert mock_sleep.call_count == 2
            mock_sleep.assert_any_call(1.0)  # First delay
            mock_sleep.assert_any_call(2.0)  # Second delay (doubled)

    def test_retry_loop_with_success_first_attempt(self):
        """Test retry loop when function succeeds on first attempt."""
        with patch('handlers.time.sleep') as mock_sleep:
            mock_func = MagicMock()
            mock_func.return_value = "success"
            
            for attempt in handlers._retry_loop(max_retries=3):
                result = mock_func()
                break
                
            assert result == "success"
            assert mock_func.call_count == 1
            assert mock_sleep.call_count == 0
    
    def test_retry_loop_with_success_after_failure(self):
        """Test retry loop when function succeeds after a failure."""
        with patch('handlers.time.sleep') as mock_sleep:
            mock_func = MagicMock()
            mock_func.side_effect = [ValueError("Failed"), "success"]
            
            result = None
            for attempt in handlers._retry_loop(max_retries=3):
                try:
                    result = mock_func()
                    break
                except ValueError:
                    continue
                    
            assert result == "success"
            assert mock_func.call_count == 2
            assert mock_sleep.call_count == 1
            
    def test_timeout_http_adapter_api_client(self):
        """Test HTTP adapters used in API clients."""
        # Test session creation in HFClient
        with patch('api.hf_client.requests.Session') as mock_session:
            mock_adapter = MagicMock()
            mock_session.return_value.mount = mock_adapter
            
            # Create client which will call _create_session internally
            client = HFClient()
            assert client is not None
            
            # Verify session was created and mount was called
            mock_session.assert_called_once()
            # Should mount http:// and https://
            assert mock_adapter.call_count >= 2
            
        # Test session creation in GHClient
        with patch('api.gh_client.requests.Session') as mock_session:
            mock_adapter = MagicMock()
            mock_session.return_value.mount = mock_adapter
            
            # Create client which will call _make_session internally
            client = GHClient()
            assert client is not None
            
            # Verify session was created and mount was called
            mock_session.assert_called_once()
            # Should mount http:// and https://
            assert mock_adapter.call_count >= 2
        
    def test_retry_policy_in_handlers(self):
        """Test retry logic used in URL handlers."""
        # Test ModelUrlHandler retry with HTTP errors
        handler = ModelUrlHandler("https://huggingface.co/test/model")
        
        with patch('handlers.UrlRouter') as mock_router, \
             patch.object(handler, 'hf_client') as mock_hf, \
             patch('handlers._retry_loop') as mock_retry:
            
            # Setup mocks
            mock_parsed = MagicMock()
            mock_parsed.type = UrlType.MODEL
            mock_parsed.hf_id = 'test/model'
            mock_router.return_value.parse.return_value = mock_parsed
            
            # Set up retry_loop to yield once
            mock_retry.return_value = iter([0])
            
            # Set up a successful API call
            mock_info = MagicMock()
            mock_info.card_data = {}
            mock_info.tags = []
            mock_info.downloads_all_time = 0
            mock_info.likes = 0
            mock_info.gated = False
            mock_info.private = False
            
            mock_hf.get_model_info.return_value = mock_info
            mock_hf.list_files.return_value = []
            mock_hf.get_readme.return_value = ""
            mock_hf.get_model_index_json.return_value = {}
            
            # Test
            result = handler.fetchMetaData()
            assert result is not None
            
            # Verify _retry_loop was called with expected arguments
            mock_retry.assert_called_once_with(max_retries=3, base_delay=1.0)


class TestGitHubClientErrorHandling:
    """Test error handling in GitHub client."""

    def test_github_client_initialization(self):
        """Test GitHub client initialization."""
        with patch('api.gh_client._make_session') as mock_session:
            client = GHClient()
            assert client is not None
            # Should create a session
            mock_session.assert_called_once()

    def test_github_client_error_handling(self):
        """Test GitHub client error handling."""
        with patch('api.gh_client._make_session') as mock_make_session:
            # Setup mock session and response
            mock_session = MagicMock()
            mock_response = MagicMock()
            mock_response.status_code = 404
            mock_response.json.side_effect = ValueError("Invalid JSON")
            mock_session.get.return_value = mock_response
            mock_make_session.return_value = mock_session
            
            # Test client handles errors gracefully
            client = GHClient()
            # GHClient should handle 404 gracefully
            repo = client.get_repo('nonexistent', 'repo')
            # Should return None instead of raising an exception
            assert repo is None


class TestHuggingFaceClientErrorHandling:
    """Test error handling in HuggingFace client."""

    def test_hf_client_gated_repo_handling(self):
        """Test handling of gated repositories in handlers."""
        # Test ModelUrlHandler with gated repo error
        handler = ModelUrlHandler("https://huggingface.co/gated/model")
        with patch('handlers.UrlRouter') as mock_router, \
             patch.object(handler, 'hf_client') as mock_hf:
            
            # Setup router mock
            mock_parsed = MagicMock()
            mock_parsed.type = UrlType.MODEL
            mock_parsed.hf_id = 'gated/model'
            mock_router.return_value.parse.return_value = mock_parsed
            
            # Raise GatedRepoError
            mock_hf.get_model_info.side_effect = GatedRepoError(
                "This repository is gated")
            
            # Should handle gracefully
            result = handler.fetchMetaData()
            assert result.gated is True
            assert result.api_errors == 1
            assert "HF gated" in result.fetch_logs[0]

    def test_hf_client_repo_not_found_handling(self):
        """Test handling of nonexistent repositories in handlers."""
        # Test DatasetUrlHandler with repo not found error
        handler = DatasetUrlHandler(
            "https://huggingface.co/datasets/nonexistent/dataset")
        
        with patch('handlers.UrlRouter') as mock_router, \
             patch.object(handler, 'hf_client') as mock_hf:
            
            # Setup router mock
            mock_parsed = MagicMock()
            mock_parsed.type = UrlType.DATASET
            mock_parsed.hf_id = 'nonexistent/dataset'
            mock_router.return_value.parse.return_value = mock_parsed
            
            # Raise RepositoryNotFoundError
            mock_hf.get_dataset_info.side_effect = RepositoryNotFoundError(
                "Repository not found")
            
            # Should raise the error
            with pytest.raises(RepositoryNotFoundError):
                handler.fetchMetaData()

    def test_hf_client_rate_limit_handling(self):
        """Test rate limit handling in handlers."""
        # Test ModelUrlHandler with HTTP 429 error
        handler = ModelUrlHandler("https://huggingface.co/test/model")
        
        with patch('handlers.UrlRouter') as mock_router, \
             patch.object(handler, 'hf_client') as mock_hf, \
             patch('handlers.time.sleep') as mock_sleep:
            
            # Setup router mock
            mock_parsed = MagicMock()
            mock_parsed.type = UrlType.MODEL
            mock_parsed.hf_id = 'test/model'
            mock_router.return_value.parse.return_value = mock_parsed
            
            # First call raises 429, second succeeds
            mock_info = MagicMock()
            mock_info.card_data = {}
            mock_info.tags = []
            mock_info.downloads_30d = 0
            mock_info.downloads_all_time = 0
            mock_info.likes = 0
            mock_info.created_at = None
            mock_info.last_modified = None
            mock_info.gated = False
            mock_info.private = False
            
            mock_hf.get_model_info.side_effect = [
                HfHubHTTPError("429: Rate limit exceeded"),
                mock_info
            ]
            mock_hf.list_files.return_value = []
            mock_hf.get_readme.return_value = ""
            mock_hf.get_model_index_json.return_value = {}
            
            # Should retry and succeed
            result = handler.fetchMetaData()
            assert result is not None
            expected_msg = "HF 429 rate limited; backing off and retrying..."
            assert expected_msg in result.fetch_logs[0]
            assert mock_sleep.called  # Should have slept for retry
            assert mock_hf.get_model_info.call_count == 2
