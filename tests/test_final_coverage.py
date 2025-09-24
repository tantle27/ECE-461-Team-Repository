"""
Additional targeted tests to reach 80% coverage.
"""

import pytest
import sys
import os
from unittest.mock import patch, Mock, MagicMock
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))


class TestRepoContextCoverage:
    """Tests to increase repo_context.py coverage."""

    def test_repo_context_creation_basic(self):
        """Test basic RepoContext creation."""
        from repo_context import RepoContext
        
        ctx = RepoContext(url="https://example.com", hf_id="test-model", host="HF")
        assert ctx.url == "https://example.com"
        assert ctx.hf_id == "test-model"
        assert ctx.host == "HF"
        assert ctx.files == []
        assert ctx.fetch_logs == []

    def test_repo_context_link_methods(self):
        """Test RepoContext linking methods."""
        from repo_context import RepoContext
        
        ctx = RepoContext(url="https://example.com", hf_id="test-model", host="HF")
        
        # Test link_dataset
        dataset_ctx = RepoContext(url="https://example.com/dataset", hf_id="test-dataset", host="HF")
        ctx.link_dataset(dataset_ctx)
        assert len(ctx.linked_datasets) == 1
        assert ctx.linked_datasets[0] == dataset_ctx
        
        # Test link_code
        code_ctx = RepoContext(url="https://github.com/user/repo", hf_id="", host="GitHub")
        ctx.link_code(code_ctx)
        assert len(ctx.linked_code) == 1
        assert ctx.linked_code[0] == code_ctx

    def test_repo_context_file_info(self):
        """Test RepoContext FileInfo functionality."""
        from repo_context import RepoContext, FileInfo
        
        # Test FileInfo creation
        file_info = FileInfo(path=Path("test.py"), size_bytes=1024, ext="py")
        assert str(file_info.path) == "test.py"
        assert file_info.size_bytes == 1024
        assert file_info.ext == "py"
        
        # Test adding files to context
        ctx = RepoContext(url="https://example.com", hf_id="test-model", host="HF")
        ctx.files = [file_info]
        assert len(ctx.files) == 1


class TestGHClientCoverage:
    """Tests to increase gh_client.py coverage."""

    @patch('api.gh_client.os.getenv')
    @patch('api.gh_client.requests.Session')
    def test_gh_client_initialization(self, mock_session_class, mock_getenv):
        """Test GHClient initialization paths."""
        from api.gh_client import GHClient
        
        mock_getenv.return_value = "test-token"
        mock_session = Mock()
        mock_session_class.return_value = mock_session
        
        client = GHClient()
        # Client should have been initialized with session
        assert hasattr(client, '_http')
        assert hasattr(client, '_etag_cache')
        assert client._etag_cache == {}
        mock_getenv.assert_called_with("GITHUB_TOKEN")

    @patch('api.gh_client.os.getenv')
    @patch('api.gh_client.requests.Session')
    def test_gh_client_no_token(self, mock_session_class, mock_getenv):
        """Test GHClient with no token."""
        from api.gh_client import GHClient
        
        mock_getenv.return_value = None
        mock_session = Mock()
        mock_session_class.return_value = mock_session
        
        client = GHClient()
        # Client should have been initialized even without token
        assert hasattr(client, '_http')
        assert hasattr(client, '_etag_cache')
        mock_getenv.assert_called_with("GITHUB_TOKEN")

    @patch('api.gh_client.requests.Session')
    @patch('api.gh_client.os.getenv')
    def test_gh_client_get_json_404(self, mock_getenv, mock_session_class):
        """Test GHClient._get_json with 404 response."""
        from api.gh_client import GHClient
        
        mock_getenv.return_value = None
        mock_session = Mock()
        mock_session_class.return_value = mock_session
        
        mock_response = Mock()
        mock_response.status_code = 404
        mock_session.get.return_value = mock_response
        
        client = GHClient()
        result = client._get_json("https://api.github.com/repos/user/notfound")
        assert result is None

    @patch('api.gh_client.requests.Session')
    @patch('api.gh_client.os.getenv')
    def test_gh_client_get_json_success(self, mock_getenv, mock_session_class):
        """Test GHClient._get_json with successful response."""
        from api.gh_client import GHClient
        
        mock_getenv.return_value = None
        mock_session = Mock()
        mock_session_class.return_value = mock_session
        
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"key": "value"}
        mock_response.raise_for_status.return_value = None
        mock_session.get.return_value = mock_response
        
        client = GHClient()
        result = client._get_json("https://api.github.com/repos/user/repo")
        assert result == {"key": "value"}


class TestHFClientAdditionalCoverage:
    """Additional tests for HF Client missing lines."""

    @patch('api.hf_client.os.getenv')
    @patch('api.hf_client.HfApi')
    @patch('api.hf_client._create_session')
    def test_hf_client_to_info_method(self, mock_create_session, mock_hf_api, mock_getenv):
        """Test HFClient._to_info method with various scenarios."""
        from api.hf_client import HFClient
        
        mock_getenv.return_value = None
        mock_session = Mock()
        mock_create_session.return_value = mock_session
        
        client = HFClient()
        
        # Mock info object with various attributes
        mock_info = Mock()
        mock_info.cardData = {"license": "mit"}
        mock_info.tags = ["nlp", "pytorch"]
        mock_info.downloads = 1000
        mock_info.downloadsAllTime = 5000
        mock_info.created_at = "2023-01-01"
        mock_info.last_modified = "2023-06-01"
        mock_info.gated = False
        mock_info.private = True
        mock_info.likes = 42
        
        result = client._to_info("test-model", mock_info, repo_type="model")
        
        assert result.hf_id == "test-model"
        assert result.tags == ["nlp", "pytorch"]
        assert result.downloads_30d == 1000
        assert result.downloads_all_time == 5000
        assert result.likes == 42
        assert result.gated is False
        assert result.private is True

    @patch('api.hf_client.os.getenv')
    @patch('api.hf_client.HfApi')
    @patch('api.hf_client._create_session')
    def test_hf_client_to_info_none_values(self, mock_create_session, mock_hf_api, mock_getenv):
        """Test HFClient._to_info method with None values."""
        from api.hf_client import HFClient
        
        mock_getenv.return_value = None
        mock_session = Mock()
        mock_create_session.return_value = mock_session
        
        client = HFClient()
        
        # Mock info object with None values
        mock_info = Mock()
        mock_info.cardData = None
        mock_info.tags = None
        mock_info.downloads = None
        mock_info.downloadsAllTime = None
        mock_info.created_at = None
        mock_info.last_modified = None
        mock_info.gated = None
        mock_info.private = None
        mock_info.likes = None
        
        result = client._to_info("test-model", mock_info, repo_type="model")
        
        assert result.hf_id == "test-model"
        assert result.tags == []
        assert result.downloads_30d is None
        assert result.downloads_all_time is None
        assert result.likes is None
        assert result.gated is None
        assert result.private is None

    @patch('api.hf_client.os.getenv')
    @patch('api.hf_client.HfApi')
    @patch('api.hf_client._create_session')
    def test_hf_client_list_files_attribute_error(self, mock_create_session, mock_hf_api, mock_getenv):
        """Test HFClient.list_files with AttributeError handling."""
        from api.hf_client import HFClient
        
        mock_getenv.return_value = None
        mock_session = Mock()
        mock_create_session.return_value = mock_session
        mock_api = Mock()
        mock_hf_api.return_value = mock_api
        
        # Mock list_repo_files to return paths
        mock_api.list_repo_files.return_value = ["file1.py", "file2.txt"]
        
        # Mock get_paths_info to raise AttributeError (testing the except AttributeError path)
        mock_api.get_paths_info.side_effect = AttributeError("No size info")
        
        client = HFClient()
        result = client.list_files("test-model", repo_type="model")
        
        # Should return files with None sizes due to AttributeError
        assert len(result) == 2
        assert result[0].path == "file1.py"
        assert result[0].size is None
        assert result[1].path == "file2.txt"
        assert result[1].size is None

    def test_github_urls_from_readme_function(self):
        """Test the backward compatibility function."""
        from api.hf_client import github_urls_from_readme
        
        hf_id = "test-model"
        readme = "Check out our code: https://github.com/user/repo"
        
        # This should call GitHubMatcher.extract_urls
        result = github_urls_from_readme(hf_id, readme)
        assert isinstance(result, list)


class TestHandlersAdditionalCoverage:
    """Additional tests for handlers.py missing lines."""

    def test_build_functions(self):
        """Test build_*_context functions."""
        from handlers import build_model_context, build_dataset_context, build_code_context
        
        # These functions create handlers and call fetchMetaData
        # Test that they return RepoContext objects
        
        with patch('handlers.ModelUrlHandler') as mock_handler_class:
            mock_handler = Mock()
            mock_handler_class.return_value = mock_handler
            mock_handler.fetchMetaData.return_value = Mock()
            
            build_model_context("https://huggingface.co/bert-base-uncased")
            mock_handler_class.assert_called_once_with("https://huggingface.co/bert-base-uncased")
            mock_handler.fetchMetaData.assert_called_once()

        with patch('handlers.DatasetUrlHandler') as mock_handler_class:
            mock_handler = Mock()
            mock_handler_class.return_value = mock_handler
            mock_handler.fetchMetaData.return_value = Mock()
            
            build_dataset_context("https://huggingface.co/datasets/squad")
            mock_handler_class.assert_called_once_with("https://huggingface.co/datasets/squad")
            mock_handler.fetchMetaData.assert_called_once()

        with patch('handlers.CodeUrlHandler') as mock_handler_class:
            mock_handler = Mock()
            mock_handler_class.return_value = mock_handler
            mock_handler.fetchMetaData.return_value = Mock()
            
            build_code_context("https://github.com/user/repo")
            mock_handler_class.assert_called_once_with("https://github.com/user/repo")
            mock_handler.fetchMetaData.assert_called_once()

    def test_uniq_keep_order(self):
        """Test _uniq_keep_order function."""
        from handlers import _uniq_keep_order
        
        # Test with duplicates
        input_list = ["a", "b", "a", "c", "b", "d"]
        result = _uniq_keep_order(input_list)
        assert result == ["a", "b", "c", "d"]
        
        # Test with no duplicates
        input_list = ["x", "y", "z"]
        result = _uniq_keep_order(input_list)
        assert result == ["x", "y", "z"]
        
        # Test with empty list
        result = _uniq_keep_order([])
        assert result == []

    def test_norm_id_function(self):
        """Test _norm_id function."""
        from handlers import _norm_id
        
        assert _norm_id("Test-Dataset_Name") == "test-dataset_name"
        assert _norm_id("BERT Base") == "bert base"
        assert _norm_id("model.v2") == "model.v2"


class TestMetricEvalAdditionalCoverage:
    """Additional tests for metric_eval.py coverage."""

    def test_metric_eval_class_creation(self):
        """Test MetricEval class instantiation."""
        from metric_eval import MetricEval
        from metrics.base_metric import BaseMetric
        
        # Create mock metrics
        mock_metric1 = Mock(spec=BaseMetric)
        mock_metric1.name = "TestMetric1"
        mock_metric1.evaluate.return_value = 0.8
        
        mock_metric2 = Mock(spec=BaseMetric)
        mock_metric2.name = "TestMetric2"
        mock_metric2.evaluate.return_value = 0.6
        
        metrics = [mock_metric1, mock_metric2]
        weights = {"TestMetric1": 0.7, "TestMetric2": 0.3}
        
        evaluator = MetricEval(metrics, weights)
        assert evaluator.metrics == metrics
        assert evaluator.weights == weights

    def test_metric_eval_evaluate_all(self):
        """Test MetricEval.evaluateAll method."""
        from metric_eval import MetricEval
        from metrics.base_metric import BaseMetric
        
        mock_metric1 = Mock(spec=BaseMetric)
        mock_metric1.name = "TestMetric1"
        mock_metric1.evaluate.return_value = 0.8
        
        mock_metric2 = Mock(spec=BaseMetric)
        mock_metric2.name = "TestMetric2"
        mock_metric2.evaluate.return_value = 0.6
        
        metrics = [mock_metric1, mock_metric2]
        weights = {"TestMetric1": 0.7, "TestMetric2": 0.3}
        
        evaluator = MetricEval(metrics, weights)
        mock_repo_ctx = Mock()
        
        scores = evaluator.evaluateAll(mock_repo_ctx)
        
        assert scores["TestMetric1"] == 0.8
        assert scores["TestMetric2"] == 0.6

    def test_metric_eval_evaluate_all_with_error(self):
        """Test MetricEval.evaluateAll with metric error."""
        from metric_eval import MetricEval
        from metrics.base_metric import BaseMetric
        
        mock_metric1 = Mock(spec=BaseMetric)
        mock_metric1.name = "TestMetric1"
        mock_metric1.evaluate.side_effect = Exception("Evaluation failed")
        
        metrics = [mock_metric1]
        weights = {"TestMetric1": 1.0}
        
        evaluator = MetricEval(metrics, weights)
        mock_repo_ctx = Mock()
        
        with patch('builtins.print'):  # Suppress error print
            scores = evaluator.evaluateAll(mock_repo_ctx)
            assert scores["TestMetric1"] == -1

    def test_metric_eval_aggregate_scores(self):
        """Test MetricEval.aggregateScores method."""
        from metric_eval import MetricEval
        
        metrics = []
        weights = {"TestMetric1": 0.6, "TestMetric2": 0.4}
        
        evaluator = MetricEval(metrics, weights)
        
        # Test normal aggregation
        scores = {"TestMetric1": 0.8, "TestMetric2": 0.6}
        result = evaluator.aggregateScores(scores)
        expected = (0.8 * 0.6 + 0.6 * 0.4) / (0.6 + 0.4)
        assert abs(result - expected) < 0.001

    def test_metric_eval_aggregate_scores_zero_weight(self):
        """Test MetricEval.aggregateScores with zero total weight."""
        from metric_eval import MetricEval
        
        metrics = []
        weights = {}
        
        evaluator = MetricEval(metrics, weights)
        
        scores = {"TestMetric1": 0.8}
        result = evaluator.aggregateScores(scores)
        assert result == 0.0

    def test_init_weights_function(self):
        """Test init_weights function."""
        from metric_eval import init_weights
        
        weights = init_weights()
        assert isinstance(weights, dict)
        assert len(weights) > 0
        assert all(isinstance(v, (int, float)) for v in weights.values())
        assert all(v >= 0 for v in weights.values())
