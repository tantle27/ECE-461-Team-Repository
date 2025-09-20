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
from api.gh_client import (  # noqa: E402
    _retry, _TimeoutHTTPAdapter, _retry_policy
)


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
            assert "Rate limited, retrying" in result.fetch_logs[0]
            mock_sleep.assert_called_once_with(1)  # First retry delay
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

    @patch('handlers.UrlRouter')
    @patch('handlers.requests.get')
    def test_fetch_github_metadata(self, mock_requests, mock_router):
        """Mock GitHub API returns repository metadata successfully."""
        # Setup mocks
        mock_parsed = MagicMock()
        mock_parsed.gh_owner_repo = ('owner', 'repo')
        mock_router.return_value.parse.return_value = mock_parsed

        # Mock successful GitHub API response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            'name': 'repo',
            'full_name': 'owner/repo',
            'private': False,
            'created_at': '2020-01-01T00:00:00Z',
            'updated_at': '2023-12-01T00:00:00Z'
        }
        mock_requests.return_value = mock_response

        # Test
        result = self.handler.fetchMetaData()

        # Verify API call
        mock_requests.assert_called_once_with(
            "https://api.github.com/repos/owner/repo")

        # Verify returned context
        assert isinstance(result, RepoContext)
        assert result.gh_url == "https://github.com/owner/repo"
        assert result.host == "GitHub"
        assert result.private is False

    @patch('handlers.UrlRouter')
    @patch('handlers.requests.get')
    @patch('handlers.time.sleep')
    def test_fetch_github_metadata_rate_limit(self, mock_sleep, mock_requests,
                                              mock_router):
        """Mock GitHub API rate limit handling."""
        mock_parsed = MagicMock()
        mock_parsed.gh_owner_repo = ('owner', 'repo')
        mock_router.return_value.parse.return_value = mock_parsed

        # First call returns 403 (rate limit), then 200
        mock_response_403 = MagicMock()
        mock_response_403.status_code = 403

        mock_response_200 = MagicMock()
        mock_response_200.status_code = 200
        mock_response_200.json.return_value = {
            'name': 'repo',
            'private': False,
            'created_at': '2020-01-01T00:00:00Z',
            'updated_at': '2023-12-01T00:00:00Z'
        }

        mock_requests.side_effect = [mock_response_403, mock_response_200]

        result = self.handler.fetchMetaData()
        assert result is not None
        mock_sleep.assert_called_once_with(1)  # First retry delay

    @patch('handlers.UrlRouter')
    @patch('handlers.requests.get')
    def test_fetch_github_metadata_not_found(self, mock_requests, mock_router):
        """Mock GitHub API repository not found."""
        mock_parsed = MagicMock()
        mock_parsed.gh_owner_repo = ('owner', 'nonexistent')
        mock_router.return_value.parse.return_value = mock_parsed

        mock_response = MagicMock()
        mock_response.status_code = 404
        mock_requests.return_value = mock_response

        with pytest.raises(Exception, match="Repository not found"):
            self.handler.fetchMetaData()

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

    def test_fetch_github_metadata_max_retries_exhausted(self):
        """Test GitHub API with max retries exhausted on rate limit."""
        with patch('handlers.UrlRouter') as mock_router, \
                patch('handlers.requests.get') as mock_requests, \
                patch('handlers.time.sleep') as mock_sleep:

            mock_parsed = MagicMock()
            mock_parsed.gh_owner_repo = ('owner', 'repo')
            mock_router.return_value.parse.return_value = mock_parsed

            # All attempts return 403 (rate limit)
            mock_response = MagicMock()
            mock_response.status_code = 403
            mock_requests.return_value = mock_response

            handler = CodeUrlHandler("https://github.com/owner/repo")
            result = handler.fetchMetaData()

            # Should return context with rate limit error
            assert result.api_errors == 1
            assert "GitHub API rate limited" in result.fetch_logs[0]
            assert mock_sleep.call_count == 2  # 3 attempts = 2 sleeps


class TestRetryFunctionality:
    """Test suite for retry functionality in API clients."""

    def test_retry_success_on_first_attempt(self):
        """Test _retry function when function succeeds on first attempt."""
        def success_fn():
            return "success"

        result = _retry(success_fn)
        assert result == "success"

    def test_retry_success_after_failures(self):
        """Test _retry function when function succeeds after failures."""
        call_count = 0

        def fail_then_succeed():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ValueError("Temporary failure")
            return "success"

        with patch('time.sleep'):  # Mock sleep to speed up test
            result = _retry(fail_then_succeed, attempts=4)
            assert result == "success"
            assert call_count == 3

    def test_retry_exhausts_attempts(self):
        """Test _retry function when all attempts are exhausted."""
        def always_fail():
            raise ValueError("Always fails")

        with patch('time.sleep'):  # Mock sleep to speed up test
            with pytest.raises(ValueError, match="Always fails"):
                _retry(always_fail, attempts=3)

    def test_timeout_http_adapter_initialization(self):
        """Test TimeoutHTTPAdapter initialization."""
        adapter = _TimeoutHTTPAdapter(timeout=45)
        assert adapter._timeout == 45

    def test_timeout_http_adapter_default_timeout(self):
        """Test TimeoutHTTPAdapter with default timeout."""
        adapter = _TimeoutHTTPAdapter()
        assert adapter._timeout == 30  # DEFAULT_TIMEOUT

    def test_retry_policy_configuration(self):
        """Test retry policy configuration."""
        policy = _retry_policy()
        assert policy.total == 5
        assert policy.connect == 5
        assert policy.read == 5
        assert policy.backoff_factor == 0.4


class TestGitHubClientErrorHandling:
    """Test error handling in GitHub client."""

    @patch('api.gh_client.GHClient')
    def test_github_client_initialization_error(self, mock_client):
        """Test GitHub client initialization with errors."""
        from api.gh_client import GHClient

        # Test with invalid token
        mock_client.side_effect = Exception("Invalid token")

        with pytest.raises(Exception, match="Invalid token"):
            GHClient()

    @patch('requests.Session.get')
    def test_github_client_request_timeout(self, mock_get):
        """Test GitHub client request timeout handling."""
        from api.gh_client import GHClient

        mock_get.side_effect = Exception("Request timeout")

        client = GHClient()

        with pytest.raises(Exception, match="Request timeout"):
            client.get_repo("owner", "repo")


class TestHuggingFaceClientErrorHandling:
    """Test error handling in HuggingFace client."""

    @patch('api.hf_client.HFClient')
    def test_hf_client_gated_repo_error(self, mock_client):
        """Test HuggingFace client gated repository error."""
        mock_client_instance = MagicMock()
        mock_client_instance.get_model_info.side_effect = GatedRepoError(
            "Gated repo")

        with pytest.raises(GatedRepoError, match="Gated repo"):
            mock_client_instance.get_model_info("gated/model")

    @patch('api.hf_client.HFClient')
    def test_hf_client_repo_not_found_error(self, mock_client):
        """Test HuggingFace client repository not found error."""
        mock_client_instance = MagicMock()
        error = RepositoryNotFoundError("Not found")
        mock_client_instance.get_model_info.side_effect = error

        with pytest.raises(RepositoryNotFoundError, match="Not found"):
            mock_client_instance.get_model_info("nonexistent/model")

    @patch('api.hf_client.HFClient')
    def test_hf_client_http_error(self, mock_client):
        """Test HuggingFace client HTTP error."""
        mock_client_instance = MagicMock()
        mock_client_instance.get_model_info.side_effect = HfHubHTTPError(
            "HTTP error")

        with pytest.raises(HfHubHTTPError, match="HTTP error"):
            mock_client_instance.get_model_info("some/model")
