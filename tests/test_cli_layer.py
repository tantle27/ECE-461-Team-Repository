"""
Unit tests for CLI functionality in app.py.
"""

import pytest
from unittest.mock import patch, MagicMock, mock_open
import sys
import os
import tempfile
import stat
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from app import (main, read_urls, _build_context_for_url, persist_context,  # noqa: E402
                 _canon_for, _resolve_db_path, _ensure_path_secure,
                 _find_project_root, _user_cache_base, _evaluate_and_persist)
from repo_context import RepoContext  # noqa: E402
from url_router import UrlType  # noqa: E402


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

    @patch('src.handlers.build_model_context')
    @patch('app.read_urls')
    def test_main_function_success(
        self, mock_read_urls, mock_build_model_context
    ):
        """Test successful execution of main function."""
        # Setup mocks
        test_url = "https://huggingface.co/bert-base-uncased"
        mock_read_urls.return_value = [test_url]

        # Mock the model context builder and its return value
        mock_context = MagicMock()
        mock_build_model_context.return_value = mock_context

        # Mock other dependencies
        with patch('app._build_context_for_url') as mock_build_context, \
             patch('app.persist_context') as mock_persist, \
             patch('app._evaluate_and_persist') as mock_evaluate, \
             patch('app._resolve_db_path') as mock_db_path, \
             patch('app._ensure_path_secure'):

            mock_db_path.return_value = "mock_db_path"
            mock_build_context.return_value = ("MODEL", mock_context)
            mock_persist.return_value = 1  # repo ID

            # Test with mock sys.argv
            with patch('sys.argv', ['app.py', 'test_urls.txt']):
                result = main()

            # Verify calls
            mock_read_urls.assert_called_once_with('test_urls.txt')
            mock_build_context.assert_called_once_with(test_url)
            mock_persist.assert_called_once()
            mock_evaluate.assert_called_once()
            assert result == 0

    @patch('src.handlers.build_model_context')
    @patch('app.read_urls')
    def test_main_function_multiple_urls(
        self, mock_read_urls, mock_build_model_context
    ):
        """Test main function with multiple URLs."""
        # Setup mocks
        test_urls = [
            "https://huggingface.co/bert-base-uncased",
            "https://huggingface.co/gpt2"
        ]
        mock_read_urls.return_value = test_urls

        # Mock the model context builder and its return values
        mock_context1 = MagicMock()
        mock_context2 = MagicMock()
        mock_build_model_context.side_effect = [mock_context1, mock_context2]

        # Mock other dependencies
        with patch('app._build_context_for_url') as mock_build_context, \
             patch('app.persist_context') as mock_persist, \
             patch('app._evaluate_and_persist') as mock_evaluate, \
             patch('app._resolve_db_path') as mock_db_path, \
             patch('app._ensure_path_secure'):

            mock_db_path.return_value = "mock_db_path"
            # Set up return values for each URL
            mock_build_context.side_effect = [
                ("MODEL", mock_context1),
                ("MODEL", mock_context2)
            ]
            mock_persist.side_effect = [1, 2]  # repo IDs

            # Test with mock sys.argv
            with patch('sys.argv', ['app.py', 'test_urls.txt']):
                result = main()

            # Verify calls
            mock_read_urls.assert_called_once_with('test_urls.txt')
            assert mock_build_context.call_count == 2
            assert mock_persist.call_count == 2
            assert mock_evaluate.call_count == 2
            assert result == 0

    def test_main_function_missing_argument(self):
        """Test main function with missing command line argument."""
        with patch('sys.argv', ['app.py']):  # Missing URL file argument
            result = main()
            assert result == 1  # Should return error code 1


class TestAppCLIFunctions:
    """Test suite for app.py CLI helper functions."""

    def test_build_context_for_url_model(self):
        """Test building context for a model URL."""
        with patch('app.UrlRouter') as mock_router, \
             patch('app.build_model_context') as mock_build_model:

            # Setup mocks
            mock_parsed = MagicMock()
            mock_parsed.type = UrlType.MODEL
            mock_router.return_value.parse.return_value = mock_parsed

            mock_context = MagicMock(spec=RepoContext)
            mock_build_model.return_value = mock_context

            # Test
            category, context = _build_context_for_url(
                "https://huggingface.co/bert-base-uncased")

            # Verify
            assert category == "MODEL"
            assert context == mock_context
            mock_build_model.assert_called_once_with(
                "https://huggingface.co/bert-base-uncased")

    def test_build_context_for_url_dataset(self):
        """Test building context for a dataset URL."""
        with patch('app.UrlRouter') as mock_router, \
             patch('app.build_dataset_context') as mock_build_dataset:

            # Setup mocks
            mock_parsed = MagicMock()
            mock_parsed.type = UrlType.DATASET
            mock_router.return_value.parse.return_value = mock_parsed

            mock_context = MagicMock(spec=RepoContext)
            mock_build_dataset.return_value = mock_context

            # Test
            category, context = _build_context_for_url(
                "https://huggingface.co/datasets/squad")

            # Verify
            assert category == "DATASET"
            assert context == mock_context
            mock_build_dataset.assert_called_once_with(
                "https://huggingface.co/datasets/squad")

    def test_build_context_for_url_code(self):
        """Test building context for a code URL."""
        with patch('app.UrlRouter') as mock_router, \
             patch('app.build_code_context') as mock_build_code:

            # Setup mocks
            mock_parsed = MagicMock()
            mock_parsed.type = UrlType.CODE
            mock_router.return_value.parse.return_value = mock_parsed

            mock_context = MagicMock(spec=RepoContext)
            mock_build_code.return_value = mock_context

            # Test
            category, context = _build_context_for_url(
                "https://github.com/pytorch/pytorch")  # noqa: E501

            # Verify
            assert category == "CODE"
            assert context == mock_context
            mock_build_code.assert_called_once_with(
                "https://github.com/pytorch/pytorch")

    def test_build_context_for_url_unsupported(self):
        """Test building context for an unsupported URL type."""
        with patch('app.UrlRouter') as mock_router:
            # Setup mock to return unsupported type
            mock_parsed = MagicMock()
            mock_parsed.type = None  # Unsupported type
            mock_router.return_value.parse.return_value = mock_parsed

            # Test - should raise ValueError
            with pytest.raises(ValueError, match="Unsupported URL type"):
                _build_context_for_url("https://invalid-url.com")

    def test_canon_for_model(self):
        """Test canonical key generation for models."""
        mock_ctx = MagicMock(spec=RepoContext)
        mock_ctx.hf_id = "bert-base-uncased"
        mock_ctx.url = "https://huggingface.co/bert-base-uncased"

        result = _canon_for(mock_ctx, "MODEL")
        assert result == "bert-base-uncased"

    def test_canon_for_dataset(self):
        """Test canonical key generation for datasets."""
        mock_ctx = MagicMock(spec=RepoContext)
        mock_ctx.hf_id = "datasets/squad"
        mock_ctx.url = "https://huggingface.co/datasets/squad"

        result = _canon_for(mock_ctx, "DATASET")
        assert result == "squad"  # removes "datasets/" prefix

    def test_canon_for_code(self):
        """Test canonical key generation for code repositories."""
        mock_ctx = MagicMock(spec=RepoContext)
        mock_ctx.hf_id = None
        mock_ctx.gh_url = "https://github.com/pytorch/pytorch"
        mock_ctx.url = "https://github.com/pytorch/pytorch"

        result = _canon_for(mock_ctx, "CODE")
        assert result == "https://github.com/pytorch/pytorch"

    def test_canon_for_no_identifiers(self):
        """Test canonical key generation when no identifiers are available."""
        mock_ctx = MagicMock(spec=RepoContext)
        mock_ctx.hf_id = None
        mock_ctx.gh_url = None
        mock_ctx.url = None

        result = _canon_for(mock_ctx, "MODEL")
        assert result == ""  # empty string when no identifiers

    def test_find_project_root_with_git(self):
        """Test finding project root with .git directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)

            # Create a .git directory
            git_dir = tmp_path / ".git"
            git_dir.mkdir()

            # Create a subdirectory
            sub_dir = tmp_path / "subdir" / "deep"
            sub_dir.mkdir(parents=True)

            # Test from subdirectory
            result = _find_project_root(sub_dir)
            assert result == tmp_path

    def test_find_project_root_no_git(self):
        """Test finding project root when no .git directory exists."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            sub_dir = tmp_path / "subdir"
            sub_dir.mkdir()

            # Test - should return None when no .git found
            result = _find_project_root(sub_dir)
            assert result is None

    @patch('app.platform.system')
    def test_user_cache_base_env_override(self, mock_system, monkeypatch):
        monkeypatch.setenv("ACME_CACHE_DIR", "/custom/cache")
        result = _user_cache_base()
        assert result == Path("/custom/cache")

    @patch('app.platform.system')
    def test_user_cache_base_windows(self, mock_system, monkeypatch):
        mock_system.return_value = "Windows"
        monkeypatch.setenv("APPDATA", "C:/Users/test/AppData/Roaming")
        assert _user_cache_base() == Path("C:/Users/test/AppData/Roaming/acme-cli")

    @patch('app.platform.system')
    @patch('pathlib.Path.home', return_value=Path("/Users/test"))
    def test_user_cache_base_darwin(self, mock_home, mock_system, monkeypatch):
        mock_system.return_value = "Darwin"
        monkeypatch.delenv("ACME_CACHE_DIR", raising=False)
        monkeypatch.delenv("XDG_CACHE_HOME", raising=False)
        monkeypatch.delenv("HOME", raising=False)  # forces Path.home() path
        assert _user_cache_base() == Path("/Users/test/Library/Application Support/acme-cli")

    def test_resolve_db_path_with_project_root(self):
        """Test resolving DB path when project root exists."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            git_dir = tmp_path / ".git"
            git_dir.mkdir()

            with patch('app._find_project_root') as mock_find_root:
                mock_find_root.return_value = tmp_path

                result = _resolve_db_path()
                expected = tmp_path / ".acme" / "state.sqlite"
                assert result == expected

    def test_resolve_db_path_no_project_root(self):
        """Test resolving DB path when no project root exists."""
        with patch('app._find_project_root') as mock_find_root, \
             patch('app._user_cache_base') as mock_cache_base:

            mock_find_root.return_value = None
            mock_cache_base.return_value = Path('/home/user/.cache/acme-cli')

            result = _resolve_db_path()
            expected = Path('/home/user/.cache/acme-cli/state.sqlite')
            assert result == expected

    def test_ensure_path_secure_creates_directory(self):
        """Test that _ensure_path_secure creates parent directories."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            db_path = tmp_path / "cache" / "metrics.db"

            # Ensure directory doesn't exist yet
            assert not db_path.parent.exists()

            # Test
            _ensure_path_secure(db_path)

            # Verify directory was created
            assert db_path.parent.exists()
            assert db_path.parent.is_dir()

    def test_ensure_path_secure_sets_permissions(self):
        """Test that _ensure_path_secure sets correct permissions."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            db_path = tmp_path / "cache" / "state.sqlite"

            # Mock path.chmod instead of os.chmod
            with patch.object(Path, 'chmod') as mock_chmod:
                # Test
                _ensure_path_secure(db_path)

                # Verify chmod was called with correct permissions for the file
                expected_mode = stat.S_IRUSR | stat.S_IWUSR  # 0o600
                mock_chmod.assert_called_with(expected_mode)

    def test_persist_context_success(self):
        """Test successful context persistence."""
        mock_ctx = MagicMock(spec=RepoContext)
        # Set all required attributes
        mock_ctx.url = "https://huggingface.co/bert-base-uncased"
        mock_ctx.hf_id = "bert-base-uncased"
        mock_ctx.gh_url = None
        mock_ctx.host = "HF"
        mock_ctx.repo_path = None
        mock_ctx.readme_text = "# BERT Base Uncased"
        mock_ctx.card_data = {}
        mock_ctx.config_json = {}
        mock_ctx.model_index = {}
        mock_ctx.tags = ["pytorch", "bert"]
        mock_ctx.downloads_30d = 1000
        mock_ctx.downloads_all_time = 10000
        mock_ctx.likes = 500
        mock_ctx.created_at = "2023-01-01"
        mock_ctx.last_modified = "2023-12-01"
        mock_ctx.gated = False
        mock_ctx.private = False
        mock_ctx.contributors = []
        mock_ctx.commit_history = []
        mock_ctx.fetch_logs = []
        mock_ctx.cache_hits = 0
        mock_ctx.api_errors = 0
        mock_ctx.files = []
        mock_ctx.linked_datasets = []
        mock_ctx.linked_code = []

        with patch('app._canon_for') as mock_canon, \
             patch('app.db.open_db') as mock_open_db, \
             patch('app.db.upsert_resource') as mock_upsert:

            mock_canon.return_value = "bert-base-uncased"
            mock_conn = MagicMock()
            mock_open_db.return_value = mock_conn
            mock_upsert.return_value = 123  # Mock resource ID

            # Test
            result = persist_context(Path("test.db"), mock_ctx, "MODEL")

            # Verify
            assert result == 123
            mock_canon.assert_called_once_with(mock_ctx, "MODEL")
            mock_open_db.assert_called_once_with(Path("test.db"))
            mock_upsert.assert_called_once()

    def test_persist_context_no_canonical_key(self):
        """Test context persistence when canonical key cannot be derived."""
        mock_ctx = MagicMock(spec=RepoContext)

        with patch('app._canon_for') as mock_canon:
            mock_canon.return_value = None  # No canonical key

            # Test - should raise ValueError
            with pytest.raises(ValueError, match="Cannot derive canonical key"):
                persist_context(Path("test.db"), mock_ctx, "MODEL")

    def test_main_with_exception_handling(self):
        """Test main function handles exceptions properly."""
        with patch('app.read_urls') as mock_read_urls, \
             patch('app._resolve_db_path') as mock_db_path, \
             patch('app._ensure_path_secure'), \
             patch('app._build_context_for_url') as mock_build_context, \
             patch('sys.argv', ['app.py', 'test_urls.txt']):

            mock_read_urls.return_value = ["https://invalid-url.com"]
            mock_db_path.return_value = Path("test.db")
            mock_build_context.side_effect = Exception("Build failed")

            # Capture stderr
            with patch('sys.stderr') as mock_stderr:
                result = main()

                # Should return 1 (failure) and print error
                assert result == 1
                mock_stderr.write.assert_called()

    def test_main_partial_success(self):
        """Test main function with partial success (some URLs fail)."""
        with patch('app.read_urls') as mock_read_urls, \
             patch('app._resolve_db_path') as mock_db_path, \
             patch('app._ensure_path_secure'), \
             patch('app._build_context_for_url') as mock_build_context, \
             patch('app.persist_context') as mock_persist, \
             patch('app._evaluate_and_persist'), \
             patch('sys.argv', ['app.py', 'test_urls.txt']):

            mock_read_urls.return_value = [
                "https://huggingface.co/bert-base-uncased",
                "https://invalid-url.com"
            ]
            mock_db_path.return_value = Path("test.db")

            # First URL succeeds, second fails
            mock_ctx = MagicMock(spec=RepoContext)
            mock_build_context.side_effect = [
                ("MODEL", mock_ctx),
                Exception("Build failed")
            ]
            mock_persist.return_value = 1

            result = main()

            # Should return 1 because not all URLs succeeded
            assert result == 1

    def test_main_all_success(self):
        """Test main function with all URLs succeeding."""
        with patch('app.read_urls') as mock_read_urls, \
             patch('app._resolve_db_path') as mock_db_path, \
             patch('app._ensure_path_secure'), \
             patch('app._build_context_for_url') as mock_build_context, \
             patch('app.persist_context') as mock_persist, \
             patch('app._evaluate_and_persist') as mock_evaluate, \
             patch('sys.argv', ['app.py', 'test_urls.txt']):

            mock_read_urls.return_value = [
                "https://huggingface.co/bert-base-uncased",
                "https://huggingface.co/gpt2"
            ]
            mock_db_path.return_value = Path("test.db")

            mock_ctx1 = MagicMock(spec=RepoContext)
            mock_ctx1.files = []
            mock_ctx1.tags = []
            mock_ctx1.contributors = []
            mock_ctx2 = MagicMock(spec=RepoContext)
            mock_ctx2.files = []
            mock_ctx2.tags = []
            mock_ctx2.contributors = []
            mock_build_context.side_effect = [
                ("MODEL", mock_ctx1),
                ("MODEL", mock_ctx2)
            ]
            mock_persist.side_effect = [1, 2]

            result = main()

            assert result == 0
            assert mock_build_context.call_count == 2
            assert mock_persist.call_count == 2
            assert mock_evaluate.call_count == 2


class TestAppMainFunction:
    """Comprehensive tests for the main() function in app.py."""

    @patch('app.sys.argv', ['app.py'])
    def test_main_insufficient_args(self):
        """Test main function with insufficient arguments."""
        result = main()
        assert result == 1

    @patch('app.sys.argv', ['app.py', 'file1', 'file2', 'extra'])
    def test_main_too_many_args(self):
        """Test main function with too many arguments."""
        result = main()
        assert result == 1

    @patch('app.sys.argv', ['app.py', 'test_urls.txt'])
    @patch('app.read_urls')
    @patch('app._resolve_db_path')
    @patch('app._ensure_path_secure')
    @patch('app._build_context_for_url')
    @patch('app.persist_context')
    @patch('app._evaluate_and_persist')
    def test_main_success_all_urls(
        self, mock_evaluate, mock_persist, mock_build_context,
        mock_ensure_secure, mock_resolve_db, mock_read_urls
    ):
        """Test main function with successful processing of all URLs."""
        # Setup
        mock_read_urls.return_value = [
            "https://huggingface.co/bert-base-uncased",
            "https://github.com/pytorch/pytorch"
        ]
        mock_resolve_db.return_value = Path("/tmp/test.db")
        mock_ctx1 = MagicMock(spec=RepoContext)
        mock_ctx1.files = []
        mock_ctx1.tags = []
        mock_ctx1.contributors = []
        mock_ctx2 = MagicMock(spec=RepoContext)
        mock_ctx2.files = []
        mock_ctx2.tags = []
        mock_ctx2.contributors = []
        mock_build_context.side_effect = [
            ("MODEL", mock_ctx1),
            ("CODE", mock_ctx2)
        ]
        mock_persist.side_effect = [1, 2]

        # Execute
        result = main()

        # Verify
        assert result == 0
        assert mock_read_urls.call_count == 1
        assert mock_build_context.call_count == 2
        assert mock_persist.call_count == 2
        assert mock_evaluate.call_count == 1

    @patch('app.sys.argv', ['app.py', 'test_urls.txt'])
    @patch('app.read_urls')
    @patch('app._resolve_db_path')
    @patch('app._ensure_path_secure')
    @patch('app._build_context_for_url')
    @patch('app.persist_context')
    @patch('app._evaluate_and_persist')
    def test_main_partial_failure(
        self, mock_evaluate, mock_persist, mock_build_context,
        mock_ensure_secure, mock_resolve_db, mock_read_urls
    ):
        """Test main function with some URLs failing."""
        # Setup
        mock_read_urls.return_value = [
            "https://huggingface.co/bert-base-uncased",
            "https://invalid-url.com/not-found",
            "https://github.com/pytorch/pytorch"
        ]
        mock_resolve_db.return_value = Path("/tmp/test.db")
        mock_ctx1 = MagicMock(spec=RepoContext)
        mock_ctx1.files = []
        mock_ctx1.tags = []
        mock_ctx1.contributors = []
        mock_ctx3 = MagicMock(spec=RepoContext)
        mock_ctx3.files = []
        mock_ctx3.tags = []
        mock_ctx3.contributors = []
        mock_build_context.side_effect = [
            ("MODEL", mock_ctx1),
            ValueError("Unsupported URL type"),
            ("CODE", mock_ctx3)
        ]
        mock_persist.side_effect = [1, 3]

        # Execute
        result = main()

        # Verify - should return 1 since not all URLs succeeded
        assert result == 1
        assert mock_build_context.call_count == 3
        assert mock_persist.call_count == 2
        assert mock_evaluate.call_count == 1


class TestBuildContextForUrl:
    """Test suite for _build_context_for_url function."""

    @patch('app.UrlRouter')
    @patch('app.build_model_context')
    def test_build_context_model_url(self, mock_build_model, mock_url_router):
        """Test building context for model URL."""
        # Setup
        mock_parsed = MagicMock()
        mock_parsed.type = UrlType.MODEL
        mock_router_instance = MagicMock()
        mock_router_instance.parse.return_value = mock_parsed
        mock_url_router.return_value = mock_router_instance

        mock_context = MagicMock(spec=RepoContext)
        mock_build_model.return_value = mock_context

        # Execute
        url = "https://huggingface.co/bert-base-uncased"
        category, ctx = _build_context_for_url(url)

        # Verify
        assert category == "MODEL"
        assert ctx == mock_context
        mock_build_model.assert_called_once_with(url)

    @patch('app.UrlRouter')
    @patch('app.build_dataset_context')
    def test_build_context_dataset_url(self, mock_build_dataset, mock_url_router):
        """Test building context for dataset URL."""
        # Setup
        mock_parsed = MagicMock()
        mock_parsed.type = UrlType.DATASET
        mock_router_instance = MagicMock()
        mock_router_instance.parse.return_value = mock_parsed
        mock_url_router.return_value = mock_router_instance

        mock_context = MagicMock(spec=RepoContext)
        mock_build_dataset.return_value = mock_context

        # Execute
        url = "https://huggingface.co/datasets/squad"
        category, ctx = _build_context_for_url(url)

        # Verify
        assert category == "DATASET"
        assert ctx == mock_context
        mock_build_dataset.assert_called_once_with(url)

    @patch('app.UrlRouter')
    @patch('app.build_code_context')
    def test_build_context_code_url(self, mock_build_code, mock_url_router):
        """Test building context for code URL."""
        # Setup
        mock_parsed = MagicMock()
        mock_parsed.type = UrlType.CODE
        mock_router_instance = MagicMock()
        mock_router_instance.parse.return_value = mock_parsed
        mock_url_router.return_value = mock_router_instance

        mock_context = MagicMock(spec=RepoContext)
        mock_build_code.return_value = mock_context

        # Execute
        url = "https://github.com/pytorch/pytorch"
        category, ctx = _build_context_for_url(url)

        # Verify
        assert category == "CODE"
        assert ctx == mock_context
        mock_build_code.assert_called_once_with(url)

    @patch('app.UrlRouter')
    def test_build_context_unsupported_url(self, mock_url_router):
        """Test building context for unsupported URL type."""
        # Setup
        mock_parsed = MagicMock()
        mock_parsed.type = "UNSUPPORTED"  # Invalid type
        mock_router_instance = MagicMock()
        mock_router_instance.parse.return_value = mock_parsed
        mock_url_router.return_value = mock_router_instance

        # Execute & Verify
        url = "https://example.com/unsupported"
        with pytest.raises(ValueError, match="Unsupported URL type"):
            _build_context_for_url(url)


class TestAppEvaluationAndPersistenceFixed:
    """Test suite for _evaluate_and_persist function with proper mocking."""

    @patch('app.init_metrics')
    @patch('app.init_weights')
    @patch('app.MetricEval')
    @patch('app.db.open_db')
    @patch('app.NetScorer')
    def test_evaluate_and_persist_basic_flow(
        self, mock_net_scorer, mock_open_db, mock_metric_eval_class,
        mock_init_weights, mock_init_metrics
    ):
        """Test basic evaluation and persistence flow."""
        # Setup mocks
        mock_metrics = [MagicMock(), MagicMock()]
        mock_metrics[0].name = "BusFactor"
        mock_metrics[1].name = "License"
        mock_init_metrics.return_value = mock_metrics
        mock_init_weights.return_value = {"BusFactor": 0.3, "License": 0.4}

        mock_evaluator = MagicMock()
        mock_evaluator.evaluateAll.return_value = {"BusFactor": 0.8, "License": 0.9}
        mock_evaluator.aggregateScores.return_value = 0.85
        mock_metric_eval_class.return_value = mock_evaluator

        mock_conn = MagicMock()
        mock_open_db.return_value = mock_conn

        # Mock NetScorer
        mock_ns = MagicMock()
        mock_ns.to_ndjson_string.return_value = '{"URL": "test", "NetScore": 0.85}'
        mock_ns.__str__.return_value = "NetScorer(net_score=0.85, metrics=2)"
        mock_net_scorer.return_value = mock_ns

        # Setup context
        mock_ctx = MagicMock(spec=RepoContext)
        mock_ctx.hf_id = "test-model"
        mock_ctx.gh_url = None
        mock_ctx.url = "https://huggingface.co/test-model"

        # Execute
        _evaluate_and_persist(Path("test.db"), 123, "MODEL", mock_ctx)

        # Verify
        mock_evaluator.evaluateAll.assert_called_once()
        mock_evaluator.aggregateScores.assert_called_once()
        mock_conn.commit.assert_called_once()
        mock_conn.close.assert_called_once()

    @patch('app.init_metrics')
    @patch('app.init_weights')
    @patch('app.MetricEval')
    @patch('app.db.open_db')
    def test_evaluate_and_persist_with_metric_errors(
        self, mock_open_db, mock_metric_eval_class, mock_init_weights, mock_init_metrics
    ):
        """Test evaluation with metric errors in context."""
        # Setup mocks
        mock_metrics = [MagicMock()]
        mock_metrics[0].name = "BusFactor"
        mock_init_metrics.return_value = mock_metrics
        mock_init_weights.return_value = {"BusFactor": 1.0}

        mock_evaluator = MagicMock()
        mock_evaluator.evaluateAll.return_value = {"BusFactor": -1.0}  # Error value
        mock_evaluator.aggregateScores.return_value = 0.0
        mock_metric_eval_class.return_value = mock_evaluator

        mock_conn = MagicMock()
        mock_open_db.return_value = mock_conn

        # Setup context with metric errors
        mock_ctx = MagicMock(spec=RepoContext)
        mock_ctx.hf_id = "test-model"
        mock_ctx.gh_url = None
        mock_ctx.url = "https://huggingface.co/test-model"

        with patch('app.NetScorer'):
            # Execute
            _evaluate_and_persist(Path("test.db"), 456, "MODEL", mock_ctx)

            # Verify error handling
            assert mock_evaluator.evaluateAll.called
            assert mock_conn.commit.called


class TestPersistContext:
    """Test suite for persist_context function."""

    @patch('app.db.open_db')
    @patch('app.db.upsert_resource')
    def test_persist_context_model_basic(self, mock_upsert, mock_open_db):
        """Test basic model context persistence."""
        # Setup mocks
        mock_conn = MagicMock()
        mock_open_db.return_value = mock_conn
        mock_upsert.return_value = 1

        # Setup context with all required attributes
        mock_ctx = MagicMock(spec=RepoContext)
        mock_ctx.hf_id = "test-model"
        mock_ctx.url = "https://huggingface.co/test-model"
        mock_ctx.gh_url = None
        mock_ctx.host = "HF"
        mock_ctx.repo_path = None
        mock_ctx.files = []
        mock_ctx.linked_datasets = []
        mock_ctx.linked_code = []
        # Add all required attributes for RepoContext
        mock_ctx.readme_text = ""
        mock_ctx.card_data = {}
        mock_ctx.config_json = {}
        mock_ctx.model_index = {}
        mock_ctx.tags = []
        mock_ctx.downloads_30d = 0
        mock_ctx.downloads_all_time = 0
        mock_ctx.likes = 0
        mock_ctx.created_at = None
        mock_ctx.last_modified = None
        mock_ctx.gated = False
        mock_ctx.private = False
        mock_ctx.contributors = []
        mock_ctx.commit_history = []
        mock_ctx.fetch_logs = []
        mock_ctx.cache_hits = 0
        mock_ctx.api_errors = 0

        # Execute
        result = persist_context(Path("test.db"), mock_ctx, "MODEL")

        # Verify
        assert result == 1
        mock_upsert.assert_called_once()
        mock_conn.commit.assert_called_once()
        mock_conn.close.assert_called_once()

    @patch('app._canon_for')
    def test_persist_context_no_canonical_key(self, mock_canon_for):
        """Test persist_context when canonical key cannot be derived."""
        mock_canon_for.return_value = None
        mock_ctx = MagicMock(spec=RepoContext)

        with pytest.raises(ValueError, match="Cannot derive canonical key"):
            persist_context(Path("test.db"), mock_ctx, "MODEL")


class TestCanonFor:
    """Test suite for _canon_for helper function."""

    def test_canon_for_model(self):
        """Test canonical key generation for models."""
        mock_ctx = MagicMock(spec=RepoContext)
        mock_ctx.hf_id = "Bert-Base-Uncased"
        mock_ctx.url = "https://huggingface.co/bert-base-uncased"

        result = _canon_for(mock_ctx, "MODEL")
        assert result == "bert-base-uncased"

    def test_canon_for_model_no_hf_id(self):
        """Test canonical key generation for models without HF ID."""
        mock_ctx = MagicMock(spec=RepoContext)
        mock_ctx.hf_id = None
        mock_ctx.url = "https://Example.Com/Model"

        result = _canon_for(mock_ctx, "MODEL")
        assert result == "https://example.com/model"

    @patch('app.RepoContext._canon_dataset_key')
    def test_canon_for_dataset(self, mock_canon_dataset):
        """Test canonical key generation for datasets."""
        mock_canon_dataset.return_value = "dataset-key"
        mock_ctx = MagicMock(spec=RepoContext)

        result = _canon_for(mock_ctx, "DATASET")
        assert result == "dataset-key"
        mock_canon_dataset.assert_called_once_with(mock_ctx)

    @patch('app.RepoContext._canon_code_key')
    def test_canon_for_code(self, mock_canon_code):
        """Test canonical key generation for code repos."""
        mock_canon_code.return_value = "code-key"
        mock_ctx = MagicMock(spec=RepoContext)

        result = _canon_for(mock_ctx, "CODE")
        assert result == "code-key"
        mock_canon_code.assert_called_once_with(mock_ctx)


class TestPathHelpers:
    """Test suite for path helper functions."""

    def test_find_project_root_with_git(self):
        """Test finding project root when .git exists."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Create a nested structure with .git at the top
            git_dir = temp_path / ".git"
            git_dir.mkdir()

            nested_dir = temp_path / "nested" / "deeply"
            nested_dir.mkdir(parents=True)

            result = _find_project_root(nested_dir)
            assert result == temp_path

    def test_find_project_root_no_git(self):
        """Test finding project root when .git doesn't exist."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            nested_dir = temp_path / "nested" / "deeply"
            nested_dir.mkdir(parents=True)

            result = _find_project_root(nested_dir)
            assert result is None

    def test_user_cache_base_windows(self):
        """Test user cache base on Windows."""
        with patch('app.platform.system', return_value='Windows'), \
             patch('app.os.environ.get') as mock_getenv, \
             patch('app.Path.home') as mock_home:

            mock_home.return_value = Path("C:/Users/test")
            mock_getenv.side_effect = lambda key: (
                "C:/Users/test/AppData/Roaming" if key == "APPDATA" else None
            )

            result = _user_cache_base()
            expected = Path("C:/Users/test/AppData/Roaming/acme-cli")
            assert result == expected

    def test_user_cache_base_darwin(self):
        """Test user cache base on macOS."""
        with patch('app.platform.system', return_value='Darwin'), \
             patch('app.Path.home') as mock_home:

            mock_home.return_value = Path("/Users/test")

            result = _user_cache_base()
            expected = Path("/Users/test/Library/Application Support/acme-cli")
            assert result == expected

    def test_user_cache_base_linux(self):
        """Test user cache base on Linux."""
        with patch('app.platform.system', return_value='Linux'), \
             patch('app.os.environ.get') as mock_getenv, \
             patch('app.Path.home') as mock_home:

            mock_home.return_value = Path("/home/test")
            mock_getenv.return_value = None  # No XDG_CACHE_HOME set

            result = _user_cache_base()
            expected = Path("/home/test/.cache/acme-cli")
            assert result == expected

    def test_resolve_db_path_env_var(self):
        """Test DB path resolution with environment variable."""
        with patch('app.os.environ.get') as mock_getenv:
            mock_getenv.return_value = "/custom/db/path.sqlite"

            result = _resolve_db_path()
            assert result == Path("/custom/db/path.sqlite")

    def test_resolve_db_path_project_root(self):
        """Test DB path resolution with project root."""
        with patch('app.os.environ.get', return_value=None), \
             patch('app._find_project_root') as mock_find_root, \
             patch('app.Path.cwd') as mock_cwd:

            mock_cwd.return_value = Path("/project/subdir")
            mock_find_root.return_value = Path("/project")

            result = _resolve_db_path()
            expected = Path("/project/.acme/state.sqlite")
            assert result == expected

    def test_resolve_db_path_user_cache(self):
        """Test DB path resolution falling back to user cache."""
        with patch('app.os.environ.get', return_value=None), \
             patch('app._find_project_root', return_value=None), \
             patch('app._user_cache_base') as mock_user_cache:

            mock_user_cache.return_value = Path("/home/user/.cache/acme-cli")

            result = _resolve_db_path()
            expected = Path("/home/user/.cache/acme-cli/state.sqlite")
            assert result == expected

    def test_ensure_path_secure(self):
        """Test path security enforcement."""
        with tempfile.TemporaryDirectory() as temp_dir:
            test_file = Path(temp_dir) / "subdir" / "test.db"

            # Should create parent directories and file
            _ensure_path_secure(test_file)

            assert test_file.parent.exists()
            assert test_file.exists()

            # Check file permissions (on systems that support it)
            try:
                file_stat = test_file.stat()
                # Should be readable and writable by user only
                assert file_stat.st_mode & stat.S_IRUSR
                assert file_stat.st_mode & stat.S_IWUSR
            except (AttributeError, OSError):
                # Skip permission check on systems that don't support it
                pass

    def test_ensure_path_secure_exception_handling(self):
        """Test that _ensure_path_secure handles exceptions gracefully."""
        # This should not raise an exception even if chmod fails
        with patch('pathlib.Path.chmod', side_effect=OSError("Permission denied")):
            with tempfile.TemporaryDirectory() as temp_dir:
                test_file = Path(temp_dir) / "test.db"
                # Should not raise
                _ensure_path_secure(test_file)
                # File should still be created
                assert test_file.exists()


class TestReadUrlsEdgeCases:
    """Additional edge case tests for read_urls function."""

    def test_read_urls_with_whitespace(self):
        """Test reading URLs with leading/trailing whitespace."""
        test_content = (
            "  https://huggingface.co/bert-base-uncased  \n"
            "\t https://github.com/pytorch/pytorch \t\n"
            "   \n"
            "https://example.com/model   \n"
        )

        with patch('builtins.open', mock_open(read_data=test_content)):
            urls = read_urls("test_urls.txt")

            assert len(urls) == 3
            assert "https://huggingface.co/bert-base-uncased" in urls
            assert "https://github.com/pytorch/pytorch" in urls
            assert "https://example.com/model" in urls
            # No URLs should have whitespace
            for url in urls:
                assert url == url.strip()

    def test_read_urls_unicode_handling(self):
        """Test reading URLs with unicode characters."""
        test_content = "https://huggingface.co/model-with-unicode-名前\n"

        with patch('builtins.open', mock_open(read_data=test_content)):
            urls = read_urls("test_urls.txt")

            assert len(urls) == 1
            assert urls[0] == "https://huggingface.co/model-with-unicode-名前"

    def test_read_urls_very_long_url(self):
        """Test reading very long URLs."""
        long_url = "https://huggingface.co/" + "very-long-model-name" * 100
        test_content = f"{long_url}\n"

        with patch('builtins.open', mock_open(read_data=test_content)):
            urls = read_urls("test_urls.txt")

            assert len(urls) == 1
            assert urls[0] == long_url
