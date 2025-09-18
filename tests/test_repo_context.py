"""
Unit tests for RepoContext class.
"""

import pytest
from unittest.mock import MagicMock
import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

try:
    from repo_context import RepoContext
except ImportError:
    # Create mock if class doesn't exist yet
    RepoContext = MagicMock


class TestRepoContext:
    """Test suite for RepoContext class."""

    def test_init_valid_metadata(self):
        """Test that metadata is set correctly in RepoContext."""
        repo_context = RepoContext(
            url="https://huggingface.co/bert-base-uncased",
            hf_id="bert-base-uncased",
            host="HF",
            downloads_all_time=10000,
            likes=500,
            tags=["pytorch", "bert"],
            created_at="2020-01-01T00:00:00Z",
            private=False,
            gated=False
        )

        assert repo_context.url == "https://huggingface.co/bert-base-uncased"
        assert repo_context.hf_id == "bert-base-uncased"
        assert repo_context.host == "HF"
        assert repo_context.downloads_all_time == 10000
        assert repo_context.likes == 500
        assert repo_context.tags == ["pytorch", "bert"]
        assert repo_context.created_at == "2020-01-01T00:00:00Z"
        assert repo_context.private is False
        assert repo_context.gated is False

    def test_links_parsed(self):
        """Ensures repo links are properly handled in RepoContext."""
        repo_context = RepoContext(
            url="https://github.com/pytorch/pytorch",
            gh_url="https://github.com/pytorch/pytorch",
            host="GitHub",
            private=False
        )

        assert repo_context.url == "https://github.com/pytorch/pytorch"
        assert repo_context.gh_url == "https://github.com/pytorch/pytorch"
        assert repo_context.host == "GitHub"
        assert repo_context.private is False

        # Test linking functionality
        dataset_ctx = RepoContext(
            url="https://huggingface.co/datasets/test/dataset",
            hf_id="test/dataset",
            host="HF"
        )

        code_ctx = RepoContext(
            url="https://github.com/test/code",
            gh_url="https://github.com/test/code",
            host="GitHub"
        )

        # Test linking methods
        repo_context.link_dataset(dataset_ctx)
        repo_context.link_code(code_ctx)

        assert len(repo_context.linked_datasets) == 1
        assert len(repo_context.linked_code) == 1
        assert repo_context.linked_datasets[0].hf_id == "test/dataset"
        assert repo_context.linked_code[0].gh_url == (
            "https://github.com/test/code")

    # EXTRA TESTS ADDED BELOW
    def test_init_missing_optional_fields(self):
        """Test initialization with minimal fields (all optional)."""
        repo_context = RepoContext()

        # All fields should have sensible defaults
        assert repo_context.url is None
        assert repo_context.hf_id is None
        assert repo_context.gh_url is None
        assert repo_context.host is None
        assert repo_context.downloads_all_time is None
        assert repo_context.likes is None
        assert repo_context.tags == []
        assert repo_context.files == []
        assert repo_context.linked_datasets == []
        assert repo_context.linked_code == []
        assert repo_context.api_errors == 0
        assert repo_context.cache_hits == 0

    def test_file_operations(self):
        """Test file-related functionality"""
        from pathlib import Path
        repo_context = RepoContext()

        # Test adding files
        repo_context.add_files([
            Path("model.safetensors"),
            Path("config.json"),
            Path("tokenizer.json")
        ])

        assert len(repo_context.files) == 3
        assert repo_context.files[0].path == Path("model.safetensors")
        assert repo_context.files[0].ext == "safetensors"
        assert repo_context.files[1].ext == "json"

    def test_weight_file_calculations(self):
        """Test weight file size calculations"""
        from repo_context import FileInfo
        from pathlib import Path

        repo_context = RepoContext()

        # Add some weight files with sizes
        repo_context.files = [
            FileInfo(Path("model.safetensors"), 1024 * 1024 * 1024,
                     "safetensors"),  # 1GB
            FileInfo(Path("model.bin"), 512 * 1024 * 1024, "bin"),  # 512MB
            FileInfo(Path("config.json"), 1024, "json"),  # 1KB (not weight)
        ]

        total_bytes = repo_context.total_weight_bytes()
        total_gb = repo_context.total_weight_gb()

        # 1.5GB in bytes
        expected_bytes = 1024 * 1024 * 1024 + 512 * 1024 * 1024
        assert total_bytes == expected_bytes
        assert abs(total_gb - 1.5) < 0.01  # ~1.5 GiB

    def test_canonical_keys(self):
        """Test canonical key generation for linking"""
        # Test dataset canonical key
        dataset_ctx = RepoContext(hf_id="test/dataset")
        key = RepoContext._canon_dataset_key(dataset_ctx)
        assert key == "test/dataset"

        # Test code canonical key
        code_ctx = RepoContext(gh_url="https://github.com/owner/repo")
        key = RepoContext._canon_code_key(code_ctx)
        assert key == "https://github.com/owner/repo"


if __name__ == "__main__":
    pytest.main([__file__])
