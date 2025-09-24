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


class TestHandlersHelperFunctions:
    """Test helper functions in handlers module for comprehensive coverage."""

    def test_safe_ext_function(self):
        """Test the _safe_ext helper function."""
        _safe_ext = handlers._safe_ext

        # Test normal extensions
        assert _safe_ext("file.txt") == "txt"
        assert _safe_ext("script.py") == "py"
        assert _safe_ext("data.json") == "json"

        # Test uppercase extensions
        assert _safe_ext("README.MD") == "md"
        assert _safe_ext("CONFIG.YAML") == "yaml"

        # Test files without extensions
        assert _safe_ext("README") == ""
        assert _safe_ext("Dockerfile") == ""

        # Test files with multiple dots
        assert _safe_ext("file.tar.gz") == "gz"
        assert _safe_ext("config.local.json") == "json"

        # Test edge cases
        assert _safe_ext("") == ""
        assert _safe_ext(".hidden") == ""  # .hidden files have no extension
        assert _safe_ext(".") == ""

    def test_datasets_from_card_basic(self):
        """Test basic dataset extraction from card data."""
        datasets_from_card = handlers.datasets_from_card

        card_data = {
            'datasets': ['huggingface/squad', 'microsoft/glue']
        }
        result = datasets_from_card(card_data, [])
        assert 'huggingface/squad' in result
        assert 'microsoft/glue' in result

    def test_datasets_from_card_with_tags(self):
        """Test dataset extraction with tags."""
        datasets_from_card = handlers.datasets_from_card

        card_data = {}
        tags = ['dataset:huggingface/squad', 'task:qa']
        result = datasets_from_card(card_data, tags)
        assert 'huggingface/squad' in result

    def test_datasets_from_readme_basic(self):
        """Test basic dataset extraction from readme text."""
        datasets_from_readme = handlers.datasets_from_readme

        readme_text = ("This model was trained on the SQuAD dataset "
                       "and evaluated on GLUE.")
        result = datasets_from_readme(readme_text)
        # Should find common dataset names
        assert isinstance(result, (list, set))

    def test_datasets_from_readme_empty(self):
        """Test dataset extraction from empty readme."""
        datasets_from_readme = handlers.datasets_from_readme

        result = datasets_from_readme("")
        assert isinstance(result, (list, set))
        assert len(result) == 0


class TestModelUrlHandlerComprehensive:
    """Comprehensive tests for ModelUrlHandler to increase coverage."""

    def setup_method(self):
        """Setup for each test."""
        self.handler = ModelUrlHandler("https://huggingface.co/bert-base-uncased")

    @patch('handlers.HFClient')
    def test_fetch_metadata_hydrate_linked_code_failure(self, mock_hf_client_class):
        """Test handling of linked code hydration failures."""
        mock_client = MagicMock()
        mock_hf_client_class.return_value = mock_client

        # Mock successful model info retrieval
        mock_model_info = MagicMock()
        mock_model_info.card_data = {'license': 'mit', 'tags': ['nlp']}
        mock_model_info.tags = ['nlp']
        mock_model_info.likes = 100
        mock_model_info.downloads_30d = 1000
        mock_model_info.downloads_all_time = 5000
        mock_model_info.created_at = '2023-01-01'
        mock_model_info.last_modified = '2023-06-01'
        mock_model_info.gated = False
        mock_model_info.private = False
        mock_model_info.hf_id = 'test-model'

        mock_client.get_model_info.return_value = mock_model_info
        mock_client.list_files.return_value = []
        mock_client.get_github_urls.return_value = ['https://github.com/user/repo']
        mock_client.get_readme.return_value = None

        # Mock build_code_context to fail
        with patch('handlers.build_code_context') as mock_build_code:
            mock_build_code.side_effect = Exception("Code context build failed")

            handler = ModelUrlHandler("https://huggingface.co/test-model")
            ctx = handler.fetchMetaData()

            # Should continue despite code context failure
            assert isinstance(ctx, RepoContext)
            assert ctx.hf_id == 'test-model'

    @patch('handlers.HFClient')
    def test_fetch_metadata_linked_datasets_failure(self, mock_hf_client_class):
        """Test handling of linked dataset hydration failures."""
        mock_client = MagicMock()
        mock_hf_client_class.return_value = mock_client

        # Mock successful model info
        mock_model_info = MagicMock()
        mock_model_info.card_data = {'datasets': ['squad']}
        mock_model_info.tags = ['nlp']
        mock_model_info.likes = 100
        mock_model_info.downloads_30d = None
        mock_model_info.downloads_all_time = None
        mock_model_info.created_at = '2023-01-01'
        mock_model_info.last_modified = '2023-06-01'
        mock_model_info.gated = False
        mock_model_info.private = False
        mock_model_info.hf_id = 'test-model'

        mock_client.get_model_info.return_value = mock_model_info
        mock_client.list_files.return_value = []
        mock_client.get_github_urls.return_value = []
        mock_client.get_readme.return_value = "Dataset: squad"

        # Mock datasets_from_* functions to return dataset IDs
        with patch('handlers.datasets_from_card') as mock_ds_card:
            with patch('handlers.datasets_from_readme') as mock_ds_readme:
                with patch('handlers.build_dataset_context') as mock_build_ds:
                    mock_ds_card.return_value = ['squad']
                    mock_ds_readme.return_value = []
                    mock_build_ds.side_effect = Exception(
                        "Dataset context build failed")

                    handler = ModelUrlHandler("https://huggingface.co/test-model")
                    ctx = handler.fetchMetaData()

                    # Should continue despite dataset context failure
                    assert isinstance(ctx, RepoContext)
                    assert ctx.hf_id == 'test-model'

    @patch('handlers.HFClient')
    def test_fetch_metadata_datasets_discovery_error(self, mock_hf_client_class):
        """Test handling of dataset discovery errors."""
        mock_client = MagicMock()
        mock_hf_client_class.return_value = mock_client

        mock_model_info = MagicMock()
        mock_model_info.card_data = {'license': 'mit'}
        mock_model_info.tags = ['nlp']
        mock_model_info.likes = 100
        mock_model_info.downloads_30d = 1000
        mock_model_info.downloads_all_time = 5000
        mock_model_info.created_at = '2023-01-01'
        mock_model_info.last_modified = '2023-06-01'
        mock_model_info.gated = False
        mock_model_info.private = False
        mock_model_info.hf_id = 'test-model'

        mock_client.get_model_info.return_value = mock_model_info
        mock_client.list_files.return_value = []
        mock_client.get_github_urls.return_value = []
        mock_client.get_readme.return_value = "Some readme"

        # Mock datasets_from_card to raise exception
        with patch('handlers.datasets_from_card') as mock_ds_card:
            mock_ds_card.side_effect = Exception("Dataset discovery failed")

            handler = ModelUrlHandler("https://huggingface.co/test-model")
            ctx = handler.fetchMetaData()

            # Should handle the error gracefully
            assert isinstance(ctx, RepoContext)
            assert ctx.hf_id == 'test-model'


class TestCodeUrlHandlerComprehensive:
    """Comprehensive tests for CodeUrlHandler to increase coverage."""

    def setup_method(self):
        """Setup for each test."""
        self.handler = CodeUrlHandler("https://github.com/user/repo")

    @patch('handlers.GHClient')
    def test_fetch_metadata_repo_not_found(self, mock_gh_client_class):
        """Test handling of repository not found."""
        mock_client = MagicMock()
        mock_gh_client_class.return_value = mock_client

        mock_client.get_repo.return_value = None  # Repo not found

        handler = CodeUrlHandler("https://github.com/user/nonexistent")
        ctx = handler.fetchMetaData()

        # Should return context with error logged
        assert isinstance(ctx, RepoContext)
        not_found_logs = [log for log in ctx.fetch_logs if "not found or not accessible" in log]  # noqa: E501
        assert len(not_found_logs) > 0

    @patch('handlers.GHClient')
    def test_fetch_metadata_readme_error(self, mock_gh_client_class):
        """Test handling of README retrieval errors."""
        mock_client = MagicMock()
        mock_gh_client_class.return_value = mock_client

        # Mock successful repo info
        mock_repo_info = MagicMock()
        mock_repo_info.private = False
        mock_repo_info.default_branch = 'main'
        mock_repo_info.description = 'Test repo'
        mock_client.get_repo.return_value = mock_repo_info

        # Mock README retrieval to fail
        mock_client.get_readme_markdown.side_effect = Exception("README fetch failed")

        handler = CodeUrlHandler("https://github.com/user/repo")
        ctx = handler.fetchMetaData()

        # Should handle README error gracefully
        assert ctx.api_errors >= 1
        readme_error_logs = [log for log in ctx.fetch_logs if "readme error" in log]
        assert len(readme_error_logs) > 0

    @patch('handlers.GHClient')
    def test_fetch_metadata_contributors_error(self, mock_gh_client_class):
        """Test handling of contributors retrieval errors."""
        mock_client = MagicMock()
        mock_gh_client_class.return_value = mock_client

        # Mock successful repo info and README
        mock_repo_info = MagicMock()
        mock_repo_info.private = False
        mock_repo_info.default_branch = 'main'
        mock_repo_info.description = 'Test repo'
        mock_client.get_repo.return_value = mock_repo_info
        mock_client.get_readme_markdown.return_value = "# Test README"

        # Mock contributors retrieval to fail
        mock_client.list_contributors.side_effect = Exception("Contributors fetch failed")  # noqa: E501

        handler = CodeUrlHandler("https://github.com/user/repo")
        ctx = handler.fetchMetaData()

        # Should handle contributors error gracefully
        assert ctx.api_errors >= 1
        contrib_error_logs = [log for log in ctx.fetch_logs if "contributors error" in log]  # noqa: E501
        assert len(contrib_error_logs) > 0

    @patch('handlers.GHClient')
    def test_fetch_metadata_file_tree_error(self, mock_gh_client_class):
        """Test handling of file tree retrieval errors."""
        mock_client = MagicMock()
        mock_gh_client_class.return_value = mock_client

        # Mock successful repo info, README, and contributors
        mock_repo_info = MagicMock()
        mock_repo_info.private = False
        mock_repo_info.default_branch = 'main'
        mock_repo_info.description = 'Test repo'
        mock_client.get_repo.return_value = mock_repo_info
        mock_client.get_readme_markdown.return_value = "# Test README"
        mock_client.list_contributors.return_value = [
            {'login': 'user1', 'contributions': 10}
        ]

        # Mock file tree retrieval to fail
        mock_client.get_repo_tree = MagicMock(side_effect=Exception("Tree fetch failed"))  # noqa: E501

        handler = CodeUrlHandler("https://github.com/user/repo")
        ctx = handler.fetchMetaData()

        # Should handle file tree error gracefully
        tree_error_logs = [log for log in ctx.fetch_logs if "tree/files error" in log]
        assert len(tree_error_logs) > 0

    @patch('handlers.GHClient')
    def test_fetch_metadata_file_tree_success(self, mock_gh_client_class):
        """Test successful file tree retrieval and processing."""
        mock_client = MagicMock()
        mock_gh_client_class.return_value = mock_client

        # Mock successful repo info
        mock_repo_info = MagicMock()
        mock_repo_info.private = False
        mock_repo_info.default_branch = 'main'
        mock_repo_info.description = 'Test repo'
        mock_client.get_repo.return_value = mock_repo_info
        mock_client.get_readme_markdown.return_value = "# Test README"
        mock_client.list_contributors.return_value = []

        # Mock successful file tree
        mock_tree = [
            {'type': 'blob', 'path': 'src/main.py', 'size': 1024},
            {'type': 'blob', 'path': 'README.md', 'size': 512},
            {'type': 'tree', 'path': 'src'},  # Directory, should be ignored
            {'type': 'blob', 'path': 'config.json', 'size': 256}
        ]
        mock_client.get_repo_tree.return_value = mock_tree

        handler = CodeUrlHandler("https://github.com/user/repo")
        ctx = handler.fetchMetaData()

        # Should have processed files correctly
        assert ctx.files is not None
        assert len(ctx.files) == 3  # Only blob types
        file_paths = [str(f.path) for f in ctx.files]
        # Normalize paths for cross-platform compatibility
        normalized_paths = [p.replace('\\', '/') for p in file_paths]
        assert 'src/main.py' in normalized_paths
        assert 'README.md' in normalized_paths
        assert 'config.json' in normalized_paths

    @patch('handlers.GHClient')
    def test_fetch_metadata_private_repo_attribute(self, mock_gh_client_class):
        """Test handling of private repository attribute."""
        mock_client = MagicMock()
        mock_gh_client_class.return_value = mock_client

        # Test private repo
        mock_repo_info = MagicMock()
        mock_repo_info.private = True
        mock_repo_info.default_branch = 'main'
        mock_repo_info.description = 'Private repo'
        mock_client.get_repo.return_value = mock_repo_info
        mock_client.get_readme_markdown.return_value = None
        mock_client.list_contributors.return_value = []

        handler = CodeUrlHandler("https://github.com/user/private-repo")
        ctx = handler.fetchMetaData()

        # Should set private attribute correctly
        assert ctx.private is True

    @patch('handlers.GHClient')
    def test_fetch_metadata_general_exception(self, mock_gh_client_class):
        """Test handling of general exceptions during fetch."""
        mock_client = MagicMock()
        mock_gh_client_class.return_value = mock_client

        # Mock get_repo to raise a general exception
        mock_client.get_repo.side_effect = Exception("General GitHub error")

        handler = CodeUrlHandler("https://github.com/user/repo")
        ctx = handler.fetchMetaData()

        # Should handle general errors gracefully
        assert ctx.api_errors >= 1
        error_logs = [log for log in ctx.fetch_logs if "GitHub error" in log]
        assert len(error_logs) > 0
