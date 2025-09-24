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

from app import (main, read_urls, _build_context_for_url, persist_context, 
                 _canon_for, _resolve_db_path, _ensure_path_secure, 
                 _find_project_root, _user_cache_base)
from repo_context import RepoContext
from url_router import UrlType


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
            category, context = _build_context_for_url("https://huggingface.co/bert-base-uncased")
            
            # Verify
            assert category == "MODEL"
            assert context == mock_context
            mock_build_model.assert_called_once_with("https://huggingface.co/bert-base-uncased")

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
            category, context = _build_context_for_url("https://huggingface.co/datasets/squad")
            
            # Verify
            assert category == "DATASET"
            assert context == mock_context
            mock_build_dataset.assert_called_once_with("https://huggingface.co/datasets/squad")

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
            category, context = _build_context_for_url("https://github.com/pytorch/pytorch")
            
            # Verify
            assert category == "CODE"
            assert context == mock_context
            mock_build_code.assert_called_once_with("https://github.com/pytorch/pytorch")

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
    def test_user_cache_base_windows(self, mock_system):
        """Test user cache base path on Windows."""
        mock_system.return_value = "Windows"
        
        with patch.dict(os.environ, {'APPDATA': 'C:\\Users\\test\\AppData\\Roaming'}):
            result = _user_cache_base()
            expected = Path('C:\\Users\\test\\AppData\\Roaming\\acme-cli')
            assert result == expected

    @patch('app.platform.system')
    def test_user_cache_base_darwin(self, mock_system):
        """Test user cache base path on macOS."""
        mock_system.return_value = "Darwin"
        
        with patch('app.Path.home') as mock_home:
            mock_home.return_value = Path('/Users/test')
            result = _user_cache_base()
            expected = Path('/Users/test/Library/Application Support/acme-cli')
            assert result == expected

    @patch('app.platform.system')
    def test_user_cache_base_linux(self, mock_system):
        """Test user cache base path on Linux."""
        mock_system.return_value = "Linux"
        
        with patch.dict(os.environ, {}, clear=True):
            with patch('app.Path.home') as mock_home:
                mock_home.return_value = Path('/home/test')
                result = _user_cache_base()
                expected = Path('/home/test/.cache/acme-cli')
                assert result == expected

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
             patch('app._evaluate_and_persist'), \
             patch('sys.argv', ['app.py', 'test_urls.txt']):
            
            mock_read_urls.return_value = [
                "https://huggingface.co/bert-base-uncased",
                "https://huggingface.co/gpt2"
            ]
            mock_db_path.return_value = Path("test.db")
            
            mock_ctx1 = MagicMock(spec=RepoContext)
            mock_ctx2 = MagicMock(spec=RepoContext)
            mock_build_context.side_effect = [
                ("MODEL", mock_ctx1),
                ("MODEL", mock_ctx2)
            ]
            mock_persist.side_effect = [1, 2]
            
            result = main()
            
            # Should return 0 because all URLs succeeded
            assert result == 0


if __name__ == "__main__":
    pytest.main([__file__])
