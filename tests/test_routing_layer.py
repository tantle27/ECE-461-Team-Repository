"""
Unit tests for URL Router functionality.
"""

import pytest
import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from url_router import UrlRouter, UrlType  # noqa: E402


class TestUrlRouter:
    """Test suite for UrlRouter class."""

    def setup_method(self):
        """Setup test fixtures."""
        self.router = UrlRouter()

    def test_parse_model_url(self):
        """HuggingFace model URL -> returns MODEL type."""
        url = "https://huggingface.co/bert-base-uncased"

        result = self.router.parse(url)

        assert result.type == UrlType.MODEL
        assert result.hf_id == "bert-base-uncased"
        assert result.raw == url

    def test_parse_dataset_url(self):
        """HuggingFace dataset URL -> returns DATASET type."""
        url = "https://huggingface.co/datasets/squad"

        result = self.router.parse(url)

        assert result.type == UrlType.DATASET
        assert result.hf_id == "squad"
        assert result.raw == url

    def test_parse_github_url(self):
        """GitHub repository URL -> returns CODE type."""
        url = "https://github.com/pytorch/pytorch"

        result = self.router.parse(url)

        assert result.type == UrlType.CODE
        assert result.gh_owner_repo == ("pytorch", "pytorch")
        assert result.raw == url

    def test_parse_invalid_url(self):
        """Invalid URL -> returns UNKNOWN type."""
        url = "https://invalid-domain.com/model"

        result = self.router.parse(url)

        assert result.type == UrlType.UNKNOWN
        assert result.raw == url

    def test_url_router_initialization(self):
        """Test UrlRouter can be instantiated."""
        router = UrlRouter()
        assert router is not None

    def test_classify_method(self):
        """Test classify method returns correct UrlType."""
        url = "https://huggingface.co/bert-base-uncased"

        url_type = self.router.classify(url)

        assert url_type == UrlType.MODEL

    def test_parse_hf_model_id(self):
        """Test parse_hf_model_id method."""
        url = "https://huggingface.co/bert-base-uncased"

        model_id = self.router.parse_hf_model_id(url)

        assert model_id == "bert-base-uncased"

    def test_parse_hf_dataset_id(self):
        """Test parse_hf_dataset_id method."""
        url = "https://huggingface.co/datasets/squad"

        dataset_id = self.router.parse_hf_dataset_id(url)

        assert dataset_id == "squad"

    def test_parse_github_owner_repo(self):
        """Test parse_github_owner_repo method."""
        url = "https://github.com/pytorch/pytorch"

        owner_repo = self.router.parse_github_owner_repo(url)

        assert owner_repo == ("pytorch", "pytorch")

    def test_strip_query_method(self):
        """Test strip_query static method."""
        url = "https://huggingface.co/bert-base-uncased?tab=model-card"

        clean_url = UrlRouter.strip_query(url)

        assert clean_url == "https://huggingface.co/bert-base-uncased"

    def test_parse_url_with_parameters(self):
        """URL with query parameters -> extracts base URL correctly."""
        url = "https://huggingface.co/bert-base-uncased?tab=model-card"

        result = self.router.parse(url)

        assert result.type == UrlType.MODEL
        assert result.hf_id == "bert-base-uncased"

    def test_parse_gitlab_url(self):
        """GitLab repository URL -> returns UNKNOWN type (not supported)."""
        url = "https://gitlab.com/group/project"

        result = self.router.parse(url)

        # Current implementation doesn't support GitLab, should return UNKNOWN
        assert result.type == UrlType.UNKNOWN

    def test_parse_malformed_url(self):
        """Malformed URL -> returns UNKNOWN type."""
        url = "not-a-valid-url"

        result = self.router.parse(url)

        assert result.type == UrlType.UNKNOWN

    def test_parse_huggingface_spaces_url(self):
        """HuggingFace Spaces URL -> returns UNKNOWN type (not a model)."""
        url = "https://huggingface.co/spaces/gradio/hello"

        result = self.router.parse(url)

        # Spaces URLs should not be classified as models
        assert result.type == UrlType.UNKNOWN

    def test_parse_case_insensitive(self):
        """URL parsing should be case insensitive."""
        url = "HTTPS://HUGGINGFACE.CO/BERT-BASE-UNCASED"

        result = self.router.parse(url)

        assert result.type == UrlType.MODEL
        assert result.hf_id == "bert-base-uncased"  # Should be lowercased


if __name__ == "__main__":
    pytest.main([__file__])
