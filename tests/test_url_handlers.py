"""
Unit tests for URL Handler classes (updated to match thin handlers).
"""

import pytest
from unittest.mock import patch, MagicMock
import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from handlers import (  # noqa: E402
    UrlHandler,
    ModelUrlHandler,
    DatasetUrlHandler,
    CodeUrlHandler
)
from repo_context import RepoContext  # noqa: E402
from url_router import UrlType  # noqa: E402
import handlers  # noqa: E402


# ---------------- Base ----------------

class TestUrlHandler:
    def setup_method(self):
        self.handler = UrlHandler()

    def test_fetch_metadata_raises_not_implemented(self):
        with pytest.raises(NotImplementedError):
            self.handler.fetchMetaData()

    def test_url_handler_initialization(self):
        handler = UrlHandler("https://example.com")
        assert handler.url == "https://example.com"

    def test_url_handler_with_none_or_empty_url(self):
        assert UrlHandler(None).url is None
        assert UrlHandler("").url == ""


# ---------------- Model (HF) ----------------

class TestModelUrlHandler:
    def setup_method(self):
        self.handler = ModelUrlHandler("https://huggingface.co/test/model")

    def test_fetch_metadata_success(self):
        with patch('handlers.UrlRouter') as mock_router, \
             patch.object(self.handler, 'hf_client') as mock_hf:

            parsed = MagicMock(type=UrlType.MODEL, hf_id='test/model')
            mock_router.return_value.parse.return_value = parsed

            mock_info = MagicMock(
                card_data={},
                tags=['pytorch', 'bert'],
                downloads_30d=100,
                downloads_all_time=10000,
                likes=500,
                created_at='2023-01-01',
                last_modified='2023-12-01',
                gated=False,
                private=False,
            )
            mock_hf.get_model_info.return_value = mock_info
            mock_hf.list_files.return_value = []
            mock_hf.get_readme.return_value = "# Test Model"
            mock_hf.get_model_index_json.return_value = {}

            ctx = self.handler.fetchMetaData()

            mock_hf.get_model_info.assert_called_once_with('test/model')
            assert isinstance(ctx, RepoContext)
            assert ctx.hf_id == 'test/model'
            assert ctx.host == 'HF'
            assert ctx.downloads_all_time == 10000
            assert ctx.likes == 500
            assert ctx.gated is False
            assert ctx.private is False

    def test_fetch_metadata_retry_on_error(self):
        with patch('handlers.UrlRouter') as mock_router, \
             patch.object(self.handler, 'hf_client') as mock_hf, \
             patch('handlers.time.sleep') as mock_sleep:

            parsed = MagicMock(type=UrlType.MODEL, hf_id='test/model')
            mock_router.return_value.parse.return_value = parsed

            mock_info = MagicMock(
                card_data={}, tags=[], downloads_30d=0, downloads_all_time=0,
                likes=0, created_at=None, last_modified=None, gated=False, private=False
            )
            # first attempt raises generic Exception, second succeeds
            mock_hf.get_model_info.side_effect = [Exception("429"), mock_info]
            mock_hf.list_files.return_value = []
            mock_hf.get_readme.return_value = ""
            mock_hf.get_model_index_json.return_value = {}

            ctx = self.handler.fetchMetaData()
            assert isinstance(ctx, RepoContext)
            # one backoff sleep after first attempt
            assert mock_sleep.called
            assert mock_hf.get_model_info.call_count == 2
            # error logs recorded
            assert any("HF error:" in log for log in ctx.fetch_logs)

    def test_fetch_metadata_failure_raises_on_filenotfound(self):
        with patch('handlers.UrlRouter') as mock_router, \
             patch.object(self.handler, 'hf_client') as mock_hf:

            parsed = MagicMock(type=UrlType.MODEL, hf_id='nonexistent/model')
            mock_router.return_value.parse.return_value = parsed

            mock_hf.get_model_info.side_effect = FileNotFoundError("not found")

            with pytest.raises(FileNotFoundError):
                self.handler.fetchMetaData()

    def test_fetch_metadata_no_url(self):
        handler = ModelUrlHandler(None)
        with pytest.raises(ValueError, match="URL is required"):
            handler.fetchMetaData()

    def test_fetch_metadata_invalid_url_type(self):
        with patch('handlers.UrlRouter') as mock_router:
            handler = ModelUrlHandler("https://example.com/invalid")

            parsed = MagicMock(type=UrlType.DATASET, hf_id=None)  # wrong type
            mock_router.return_value.parse.return_value = parsed

            with pytest.raises(ValueError, match="Not an HF model URL"):
                handler.fetchMetaData()

    def test_fetch_metadata_gated_repo_via_info_flag(self):
        with patch('handlers.UrlRouter') as mock_router, \
             patch.object(self.handler, 'hf_client') as mock_hf:

            parsed = MagicMock(type=UrlType.MODEL, hf_id='gated/model')
            mock_router.return_value.parse.return_value = parsed

            mock_info = MagicMock(
                card_data={}, tags=[], downloads_30d=None, downloads_all_time=None,
                likes=None, created_at=None, last_modified=None, gated=True, private=None
            )
            mock_hf.get_model_info.return_value = mock_info
            mock_hf.list_files.return_value = []
            mock_hf.get_readme.return_value = ""
            mock_hf.get_model_index_json.return_value = None

            ctx = self.handler.fetchMetaData()
            assert ctx.gated is True
            assert ctx.api_errors == 0  # not an error path
            # still a normal completion
            assert isinstance(ctx, RepoContext)


# ---------------- Dataset (HF) ----------------

class TestDatasetUrlHandler:
    def setup_method(self):
        self.handler = DatasetUrlHandler("https://huggingface.co/datasets/test/ds")

    def test_fetch_metadata_success(self):
        with patch('handlers.UrlRouter') as mock_router, \
             patch.object(self.handler, 'hf_client') as mock_hf:

            parsed = MagicMock(type=UrlType.DATASET, hf_id='test/ds')
            mock_router.return_value.parse.return_value = parsed

            mock_info = MagicMock(
                card_data={}, tags=['qa'], downloads_30d=200, downloads_all_time=5000,
                likes=200, created_at='2023-02-01', last_modified='2023-12-01',
                gated=False, private=False
            )
            mock_hf.get_dataset_info.return_value = mock_info
            mock_hf.get_readme.return_value = "# Test Dataset"

            ctx = self.handler.fetchMetaData()
            mock_hf.get_dataset_info.assert_called_once_with('test/ds')
            assert isinstance(ctx, RepoContext)
            assert ctx.hf_id == 'test/ds'
            assert ctx.downloads_all_time == 5000
            assert ctx.gated is False

    def test_fetch_metadata_retry_on_error(self):
        with patch('handlers.UrlRouter') as mock_router, \
             patch.object(self.handler, 'hf_client') as mock_hf, \
             patch('handlers.time.sleep') as mock_sleep:

            parsed = MagicMock(type=UrlType.DATASET, hf_id='test/ds')
            mock_router.return_value.parse.return_value = parsed

            mock_info = MagicMock(
                card_data={}, tags=[], downloads_30d=0, downloads_all_time=0,
                likes=0, created_at=None, last_modified=None, gated=False, private=False
            )
            mock_hf.get_dataset_info.side_effect = [Exception("429"), mock_info]
            mock_hf.get_readme.return_value = ""

            ctx = self.handler.fetchMetaData()
            assert isinstance(ctx, RepoContext)
            assert mock_sleep.called
            assert mock_hf.get_dataset_info.call_count == 2
            assert any("HF error:" in log for log in ctx.fetch_logs)

    def test_fetch_metadata_failure_raises_on_filenotfound(self):
        with patch('handlers.UrlRouter') as mock_router, \
             patch.object(self.handler, 'hf_client') as mock_hf:

            parsed = MagicMock(type=UrlType.DATASET, hf_id='missing/ds')
            mock_router.return_value.parse.return_value = parsed

            mock_hf.get_dataset_info.side_effect = FileNotFoundError("missing")
            with pytest.raises(FileNotFoundError):
                self.handler.fetchMetaData()

    def test_fetch_metadata_no_url(self):
        handler = DatasetUrlHandler(None)
        with pytest.raises(ValueError, match="URL is required"):
            handler.fetchMetaData()

    def test_fetch_metadata_invalid_url_type(self):
        with patch('handlers.UrlRouter') as mock_router:
            handler = DatasetUrlHandler("https://example.com/invalid")
            parsed = MagicMock(type=UrlType.MODEL, hf_id=None)
            mock_router.return_value.parse.return_value = parsed
            with pytest.raises(ValueError, match="Not an HF dataset URL"):
                handler.fetchMetaData()

    def test_fetch_metadata_gated_dataset_via_info_flag(self):
        with patch('handlers.UrlRouter') as mock_router, \
             patch.object(self.handler, 'hf_client') as mock_hf:

            parsed = MagicMock(type=UrlType.DATASET, hf_id='org/ds')
            mock_router.return_value.parse.return_value = parsed

            mock_info = MagicMock(
                card_data={}, tags=[], downloads_30d=None, downloads_all_time=None,
                likes=None, created_at=None, last_modified=None, gated=True, private=False
            )
            mock_hf.get_dataset_info.return_value = mock_info
            mock_hf.get_readme.return_value = ""

            ctx = self.handler.fetchMetaData()
            assert isinstance(ctx, RepoContext)
            assert ctx.gated is True


# ---------------- Code (GitHub) ----------------

class TestCodeUrlHandler:
    def setup_method(self):
        self.handler = CodeUrlHandler("https://github.com/owner/repo")

    @patch('handlers.UrlRouter')
    def test_fetch_github_metadata_not_found(self, mock_router):
        parsed = MagicMock()
        parsed.gh_owner_repo = ('owner', 'nonexistent')
        mock_router.return_value.parse.return_value = parsed

        with patch.object(self.handler, 'gh_client') as mock_gh:
            mock_gh.get_repo.return_value = None  # not found

            ctx = self.handler.fetchMetaData()
            assert ctx.api_errors == 1
            assert any("GitHub repo owner/nonexistent not found" in log for log in ctx.fetch_logs)

    def test_code_url_handler_initialization(self):
        handler = CodeUrlHandler("https://github.com/test/repo")
        assert handler.url == "https://github.com/test/repo"

    def test_fetch_metadata_no_url(self):
        handler = CodeUrlHandler(None)
        with pytest.raises(ValueError, match="URL is required"):
            handler.fetchMetaData()

    def test_fetch_metadata_invalid_github_url(self):
        with patch('handlers.UrlRouter') as mock_router:
            handler = CodeUrlHandler("https://example.com/invalid")
            parsed = MagicMock()
            parsed.gh_owner_repo = None
            mock_router.return_value.parse.return_value = parsed
            with pytest.raises(ValueError, match="Not a GitHub repo URL"):
                handler.fetchMetaData()

    @patch('handlers.UrlRouter')
    def test_fetch_github_metadata_general_error(self, mock_router):
        parsed = MagicMock()
        parsed.gh_owner_repo = ('owner', 'repo')
        mock_router.return_value.parse.return_value = parsed

        with patch.object(self.handler, 'gh_client') as mock_gh:
            mock_gh.get_repo.side_effect = Exception("boom")

            ctx = self.handler.fetchMetaData()
            assert ctx.api_errors >= 1
            assert any("GitHub error:" in log for log in ctx.fetch_logs)

    @patch('handlers.GHClient')
    def test_fetch_metadata_readme_error(self, mock_gh_class):
        mock_gh = MagicMock()
        mock_gh_class.return_value = mock_gh

        repo = MagicMock(private=False, default_branch='main', description='x')
        mock_gh.get_repo.return_value = repo
        mock_gh.get_readme_markdown.side_effect = Exception("README fetch failed")
        mock_gh.list_contributors.return_value = []

        handler = CodeUrlHandler("https://github.com/user/repo")
        ctx = handler.fetchMetaData()
        assert ctx.api_errors >= 1
        assert any("GitHub readme error:" in log for log in ctx.fetch_logs)

    @patch('handlers.GHClient')
    def test_fetch_metadata_contributors_error(self, mock_gh_class):
        mock_gh = MagicMock()
        mock_gh_class.return_value = mock_gh

        repo = MagicMock(private=False, default_branch='main', description='x')
        mock_gh.get_repo.return_value = repo
        mock_gh.get_readme_markdown.return_value = "# Readme"
        mock_gh.list_contributors.side_effect = Exception("Contrib fail")

        handler = CodeUrlHandler("https://github.com/user/repo")
        ctx = handler.fetchMetaData()
        assert ctx.api_errors >= 1
        assert any("GitHub contributors error:" in log for log in ctx.fetch_logs)

    @patch('handlers.GHClient')
    def test_fetch_metadata_file_tree_error(self, mock_gh_class):
        mock_gh = MagicMock()
        mock_gh_class.return_value = mock_gh
        repo = MagicMock(private=False, default_branch='main', description='x')
        mock_gh.get_repo.return_value = repo
        mock_gh.get_readme_markdown.return_value = "# Readme"
        mock_gh.list_contributors.return_value = [{'login': 'u', 'contributions': 1}]
        mock_gh.get_repo_tree.side_effect = Exception("Tree fetch failed")

        handler = CodeUrlHandler("https://github.com/user/repo")
        ctx = handler.fetchMetaData()
        assert any("GitHub tree error:" in log for log in ctx.fetch_logs)

    @patch('handlers.GHClient')
    def test_fetch_metadata_file_tree_success(self, mock_gh_class):
        mock_gh = MagicMock()
        mock_gh_class.return_value = mock_gh

        repo = MagicMock(private=False, default_branch='main', description='Test repo')
        mock_gh.get_repo.return_value = repo
        mock_gh.get_readme_markdown.return_value = "# README"
        mock_gh.list_contributors.return_value = []

        mock_tree = [
            {'type': 'blob', 'path': 'src/main.py', 'size': 1024},
            {'type': 'blob', 'path': 'README.md', 'size': 512},
            {'type': 'tree', 'path': 'src'},
            {'type': 'blob', 'path': 'config.json', 'size': 256},
        ]
        mock_gh.get_repo_tree.return_value = mock_tree

        handler = CodeUrlHandler("https://github.com/user/repo")
        ctx = handler.fetchMetaData()
        assert ctx.files is not None
        paths = [str(f.path).replace('\\', '/') for f in ctx.files]
        assert paths == ['src/main.py', 'README.md', 'config.json']


# ---------------- Helpers ----------------

class TestHandlersHelperFunctions:
    def test_safe_ext_function(self):
        _safe_ext = handlers._safe_ext
        assert _safe_ext("file.txt") == "txt"
        assert _safe_ext("script.py") == "py"
        assert _safe_ext("data.json") == "json"
        assert _safe_ext("README.MD") == "md"
        assert _safe_ext("CONFIG.YAML") == "yaml"
        assert _safe_ext("README") == ""
        assert _safe_ext("Dockerfile") == ""
        assert _safe_ext("file.tar.gz") == "gz"
        assert _safe_ext("config.local.json") == "json"
        assert _safe_ext("") == ""
        assert _safe_ext(".hidden") == ""
        assert _safe_ext(".") == ""