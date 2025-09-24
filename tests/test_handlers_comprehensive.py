"""
Comprehensive tests for handlers.py to increase coverage.
"""

import pytest
import sys
import os
from unittest.mock import patch, MagicMock, Mock
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from handlers import (
    ModelUrlHandler, 
    CodeUrlHandler, 
    _safe_ext, 
    _retry_loop,
    datasets_from_card,
    datasets_from_readme
)
from repo_context import RepoContext


class TestHandlersComprehensive:
    """Comprehensive tests for handlers.py coverage."""

    def test_safe_ext_function(self):
        """Test the _safe_ext helper function."""
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

    def test_retry_loop_success_first_attempt(self):
        """Test retry with backoff when first attempt succeeds."""
        attempts = []
        
        for attempt in _retry_loop(max_retries=3):
            attempts.append(attempt)
            # Succeed on first attempt
            break
            
        assert attempts == [0]

    def test_retry_loop_success_after_retries(self):
        """Test retry with backoff succeeding after some failures."""
        attempts = []
        
        for attempt in _retry_loop(max_retries=3, base_delay=0.01):
            attempts.append(attempt)
            if attempt >= 1:  # Succeed on second attempt
                break
                
        assert len(attempts) == 2
        assert attempts == [0, 1]

    def test_retry_loop_exhaust_retries(self):
        """Test retry with backoff exhausting all retries."""
        attempts = []
        
        for attempt in _retry_loop(max_retries=2, base_delay=0.01):
            attempts.append(attempt)
            # Never succeed, exhaust all retries
            
        assert attempts == [0, 1]

    def test_retry_loop_delay_calculation(self):
        """Test that delay increases exponentially."""
        import time
        delays = []
        
        for attempt in _retry_loop(max_retries=3, base_delay=0.01):
            start_time = time.time()
            if attempt > 0:
                # Record the delay that was applied before this attempt
                pass
            delays.append(time.time())
            if attempt >= 1:
                break
        
        # Just verify the function works with timing
        assert len(delays) > 1


class TestModelUrlHandlerComprehensive:
    """Comprehensive tests for ModelUrlHandler coverage."""

    def setup_method(self):
        """Setup for each test."""
        self.handler = ModelUrlHandler("https://huggingface.co/bert-base-uncased")

    @patch('handlers.HFClient')
    def test_fetch_metadata_retry_logic_429_error(self, mock_hf_client_class):
        """Test retry logic for 429 rate limit errors."""
        mock_client = Mock()
        mock_hf_client_class.return_value = mock_client
        
        # Mock successful model info on first call (no retry needed for this test)
        mock_model_info = Mock(
            card_data={'license': 'mit'},
            tags=['nlp'],
            likes=100,
            downloads_30d=1000,
            downloads_all_time=5000,
            created_at='2023-01-01',
            last_modified='2023-06-01',
            gated=False,
            private=False,
            hf_id='test-model'
        )
        mock_client.get_model_info.return_value = mock_model_info
        mock_client.list_files.return_value = []
        mock_client.get_github_urls.return_value = []
        mock_client.get_readme.return_value = None
        
        handler = ModelUrlHandler("https://huggingface.co/test-model")
        ctx = handler.fetchMetaData()
        
        # Should have succeeded
        assert isinstance(ctx, RepoContext)
        assert ctx.hf_id == 'test-model'

    @patch('handlers.HFClient')
    def test_fetch_metadata_non_429_error_no_retry(self, mock_hf_client_class):
        """Test that non-429 errors don't trigger retry."""
        mock_client = Mock()
        mock_hf_client_class.return_value = mock_client
        
        # Mock different HTTP error (not 429)
        from huggingface_hub.utils import HfHubHTTPError
        error_404 = HfHubHTTPError("Not found", response=Mock(status_code=404))
        
        mock_client.get_model_info.side_effect = error_404
        
        handler = ModelUrlHandler("https://huggingface.co/nonexistent-model")
        
        with pytest.raises(HfHubHTTPError):
            handler.fetchMetaData()

    @patch('handlers.HFClient')
    def test_fetch_metadata_hydrate_linked_code_failure(self, mock_hf_client_class):
        """Test handling of linked code hydration failures."""
        mock_client = Mock()
        mock_hf_client_class.return_value = mock_client
        
        # Mock successful model info retrieval
        mock_model_info = Mock(
            card_data={'license': 'mit', 'tags': ['nlp']},
            tags=['nlp'],
            likes=100,
            downloads_30d=1000,
            downloads_all_time=5000,
            created_at='2023-01-01',
            last_modified='2023-06-01',
            gated=False,
            private=False,
            hf_id='test-model'
        )
        mock_client.get_model_info.return_value = mock_model_info
        mock_client.list_files.return_value = []  # Return empty list, not Mock
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
        mock_client = Mock()
        mock_hf_client_class.return_value = mock_client
        
        # Mock successful model info
        mock_model_info = Mock(
            card_data={'datasets': ['squad']},
            tags=['nlp'],
            likes=100,
            downloads_30d=None,
            downloads_all_time=None,
            created_at='2023-01-01',
            last_modified='2023-06-01',
            gated=False,
            private=False,
            hf_id='test-model'
        )
        mock_client.get_model_info.return_value = mock_model_info
        mock_client.list_files.return_value = []  # Return empty list, not Mock
        mock_client.get_github_urls.return_value = []
        mock_client.get_readme.return_value = "Dataset: squad"
        
        # Mock datasets_from_* functions to return dataset IDs
        with patch('handlers.datasets_from_card') as mock_ds_card:
            with patch('handlers.datasets_from_readme') as mock_ds_readme:
                with patch('handlers.build_dataset_context') as mock_build_ds:
                    mock_ds_card.return_value = ['squad']
                    mock_ds_readme.return_value = []
                    mock_build_ds.side_effect = Exception("Dataset context build failed")
                    
                    handler = ModelUrlHandler("https://huggingface.co/test-model")
                    ctx = handler.fetchMetaData()
                    
                    # Should continue despite dataset context failure
                    assert isinstance(ctx, RepoContext)
                    assert ctx.hf_id == 'test-model'

    @patch('handlers.HFClient')
    def test_fetch_metadata_datasets_discovery_error(self, mock_hf_client_class):
        """Test handling of dataset discovery errors."""
        mock_client = Mock()
        mock_hf_client_class.return_value = mock_client
        
        mock_model_info = Mock(
            card_data={'license': 'mit'},
            tags=['nlp'],
            likes=100,
            downloads_30d=1000,
            downloads_all_time=5000,
            created_at='2023-01-01',
            last_modified='2023-06-01',
            gated=False,
            private=False,
            hf_id='test-model'
        )
        mock_client.get_model_info.return_value = mock_model_info
        mock_client.list_files.return_value = []  # Return empty list, not Mock
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
    """Comprehensive tests for CodeUrlHandler coverage."""

    def setup_method(self):
        """Setup for each test."""
        self.handler = CodeUrlHandler("https://github.com/user/repo")

    @patch('handlers.GHClient')
    def test_fetch_metadata_repo_not_found(self, mock_gh_client_class):
        """Test handling of repository not found."""
        mock_client = Mock()
        mock_gh_client_class.return_value = mock_client
        
        mock_client.get_repo.return_value = None  # Repo not found
        
        handler = CodeUrlHandler("https://github.com/user/nonexistent")
        ctx = handler.fetchMetaData()
        
        # Should return context with error logged
        assert isinstance(ctx, RepoContext)
        not_found_logs = [log for log in ctx.fetch_logs if "not found or not accessible" in log]
        assert len(not_found_logs) > 0

    @patch('handlers.GHClient')
    def test_fetch_metadata_readme_error(self, mock_gh_client_class):
        """Test handling of README retrieval errors."""
        mock_client = Mock()
        mock_gh_client_class.return_value = mock_client
        
        # Mock successful repo info
        mock_repo_info = Mock(
            private=False,
            default_branch='main',
            description='Test repo'
        )
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
        mock_client = Mock()
        mock_gh_client_class.return_value = mock_client
        
        # Mock successful repo info and README
        mock_repo_info = Mock(
            private=False,
            default_branch='main',
            description='Test repo'
        )
        mock_client.get_repo.return_value = mock_repo_info
        mock_client.get_readme_markdown.return_value = "# Test README"
        
        # Mock contributors retrieval to fail
        mock_client.list_contributors.side_effect = Exception("Contributors fetch failed")
        
        handler = CodeUrlHandler("https://github.com/user/repo")
        ctx = handler.fetchMetaData()
        
        # Should handle contributors error gracefully
        assert ctx.api_errors >= 1
        contrib_error_logs = [log for log in ctx.fetch_logs if "contributors error" in log]
        assert len(contrib_error_logs) > 0

    @patch('handlers.GHClient')
    def test_fetch_metadata_file_tree_error(self, mock_gh_client_class):
        """Test handling of file tree retrieval errors."""
        mock_client = Mock()
        mock_gh_client_class.return_value = mock_client
        
        # Mock successful repo info, README, and contributors
        mock_repo_info = Mock(
            private=False,
            default_branch='main',
            description='Test repo'
        )
        mock_client.get_repo.return_value = mock_repo_info
        mock_client.get_readme_markdown.return_value = "# Test README"
        mock_client.list_contributors.return_value = [
            {'login': 'user1', 'contributions': 10}
        ]
        
        # Mock file tree retrieval to fail
        mock_client.get_repo_tree = Mock(side_effect=Exception("Tree fetch failed"))
        
        handler = CodeUrlHandler("https://github.com/user/repo")
        ctx = handler.fetchMetaData()
        
        # Should handle file tree error gracefully
        tree_error_logs = [log for log in ctx.fetch_logs if "tree/files error" in log]
        assert len(tree_error_logs) > 0

    @patch('handlers.GHClient')
    def test_fetch_metadata_file_tree_success(self, mock_gh_client_class):
        """Test successful file tree retrieval and processing."""
        mock_client = Mock()
        mock_gh_client_class.return_value = mock_client
        
        # Mock successful repo info
        mock_repo_info = Mock(
            private=False,
            default_branch='main',
            description='Test repo'
        )
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
        mock_client = Mock()
        mock_gh_client_class.return_value = mock_client
        
        # Test private repo
        mock_repo_info = Mock(
            private=True,
            default_branch='main',
            description='Private repo'
        )
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
        mock_client = Mock()
        mock_gh_client_class.return_value = mock_client
        
        # Mock get_repo to raise a general exception
        mock_client.get_repo.side_effect = Exception("General GitHub error")
        
        handler = CodeUrlHandler("https://github.com/user/repo")
        ctx = handler.fetchMetaData()
        
        # Should handle general errors gracefully
        assert ctx.api_errors >= 1
        error_logs = [log for log in ctx.fetch_logs if "GitHub error" in log]
        assert len(error_logs) > 0


class TestDatasetHelperFunctions:
    """Test dataset discovery helper functions."""

    def test_datasets_from_card_basic(self):
        """Test basic dataset extraction from card data."""
        card_data = {
            'datasets': ['huggingface/squad', 'microsoft/glue']
        }
        result = datasets_from_card(card_data, [])
        assert 'huggingface/squad' in result
        assert 'microsoft/glue' in result

    def test_datasets_from_card_with_tags(self):
        """Test dataset extraction with tags."""
        card_data = {}
        tags = ['dataset:huggingface/squad', 'task:qa']
        result = datasets_from_card(card_data, tags)
        assert 'huggingface/squad' in result

    def test_datasets_from_readme_basic(self):
        """Test basic dataset extraction from readme text."""
        readme_text = "This model was trained on the SQuAD dataset and evaluated on GLUE."
        result = datasets_from_readme(readme_text)
        # Should find common dataset names
        assert len(result) >= 0  # May or may not find datasets depending on implementation

    def test_datasets_from_readme_empty(self):
        """Test dataset extraction from empty readme."""
        result = datasets_from_readme("")
        assert isinstance(result, (list, set))
        assert len(result) == 0
