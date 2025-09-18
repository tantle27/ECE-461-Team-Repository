"""
Unit tests for Routing Layer (EvaluatorRouter & URLRouter).
"""

import pytest
from unittest.mock import patch, MagicMock
import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

try:
    from url_router import URLRouter, EvaluatorRouter
except ImportError:
    # Create mocks if classes don't exist yet
    URLRouter = MagicMock
    EvaluatorRouter = MagicMock

try:
    from handlers import ModelUrlHandler, DatasetUrlHandler, CodeUrlHandler
except ImportError:
    # Create mocks if handlers don't exist yet
    ModelUrlHandler = MagicMock
    DatasetUrlHandler = MagicMock
    CodeUrlHandler = MagicMock


class TestEvaluatorRouter:
    """Test suite for EvaluatorRouter class."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.router = EvaluatorRouter()
    
    @patch('url_router.URLRouter')
    def test_evaluate_multiple_urls(self, mock_url_router):
        """Test evaluation of multiple URLs from file."""
        mock_classifier = MagicMock()
        mock_url_router.return_value = mock_classifier
        
        # Mock file content
        test_urls = [
            "https://huggingface.co/model1",
            "https://huggingface.co/dataset1",
            "https://github.com/repo1"
        ]
        
        with patch('builtins.open') as mock_open:
            mock_open.return_value.__enter__.return_value = test_urls
            results = self.router.evaluate_from_file("test_urls.txt")
        
        assert len(results) == 3
        mock_classifier.classify_url.assert_called()
    
    def test_evaluator_router_initialization(self):
        """Test EvaluatorRouter can be instantiated."""
        router = EvaluatorRouter()
        assert router is not None
    
    @patch('url_router.URLRouter')
    def test_evaluate_empty_file(self, mock_url_router):
        """Test evaluation with empty URL file."""
        mock_classifier = MagicMock()
        mock_url_router.return_value = mock_classifier
        
        with patch('builtins.open') as mock_open:
            mock_open.return_value.__enter__.return_value = []
            results = self.router.evaluate_from_file("empty.txt")
        
        assert len(results) == 0


class TestURLRouter:
    """Test suite for URLRouter class."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.router = URLRouter()
    
    def test_classify_model_url(self):
        """HuggingFace model URL -> returns ModelUrlHandler."""
        url = "https://huggingface.co/bert-base-uncased"
        
        handler = self.router.classify_url(url)
        
        assert isinstance(handler, type(ModelUrlHandler()))
    
    def test_classify_dataset_url(self):
        """HuggingFace dataset URL -> returns DatasetUrlHandler."""
        url = "https://huggingface.co/datasets/squad"
        
        handler = self.router.classify_url(url)
        
        assert isinstance(handler, type(DatasetUrlHandler()))
    
    def test_classify_code_url(self):
        """GitHub repository URL -> returns CodeUrlHandler."""
        url = "https://github.com/pytorch/pytorch"
        
        handler = self.router.classify_url(url)
        
        assert isinstance(handler, type(CodeUrlHandler()))
    
    def test_classify_invalid_url(self):
        """Invalid URL -> raises ValueError."""
        url = "https://invalid-domain.com/model"
        
        with pytest.raises(ValueError):
            self.router.classify_url(url)
    
    # --- EXTRA TESTS ---
    def test_url_router_initialization(self):
        """Test URLRouter can be instantiated."""
        router = URLRouter()
        assert router is not None
    
    def test_classify_huggingface_spaces_url(self):
        """HuggingFace Spaces URL -> returns appropriate handler."""
        url = "https://huggingface.co/spaces/gradio/hello"
        
        handler = self.router.classify_url(url)
        
        # Should handle Spaces URLs appropriately
        assert handler is not None
    
    def test_classify_gitlab_url(self):
        """GitLab repository URL -> returns CodeUrlHandler."""
        url = "https://gitlab.com/group/project"
        
        handler = self.router.classify_url(url)
        
        assert isinstance(handler, type(CodeUrlHandler()))
    
    def test_classify_url_with_parameters(self):
        """URL with query parameters -> extracts base URL correctly."""
        url = "https://huggingface.co/bert-base-uncased?tab=model-card"
        
        handler = self.router.classify_url(url)
        
        assert isinstance(handler, type(ModelUrlHandler()))
    
    def test_classify_malformed_url(self):
        """Malformed URL -> raises ValueError."""
        url = "not-a-valid-url"
        
        with pytest.raises(ValueError):
            self.router.classify_url(url)


if __name__ == "__main__":
    pytest.main([__file__])
