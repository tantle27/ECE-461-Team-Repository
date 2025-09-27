import pytest
from src.metrics.dataset_quality_metric import DatasetQualityMetric
from unittest.mock import MagicMock, patch

def make_fake_repo_context():
    return {
        "tags": ["validation", "diversity", "completeness"],
        "card_data": {"schema": True, "classes": 5, "splits": ["train", "test"]},
        "readme_text": "# Dataset\nA diverse dataset with validation."
    }

def test_dataset_quality_basic():
    metric = DatasetQualityMetric(use_llm=False)
    repo_context = make_fake_repo_context()
    score = metric.evaluate(repo_context)
    assert 0.0 <= score <= 1.0

def test_dataset_quality_no_ctx():
    metric = DatasetQualityMetric(use_llm=False)
    repo_context = {}
    score = metric.evaluate(repo_context)
    assert 0.0 <= score <= 1.0

def test_dataset_quality_with_llm():
    metric = DatasetQualityMetric(use_llm=True)
    repo_context = make_fake_repo_context()
    with patch.object(metric, "_llm", create=True):
        score = metric.evaluate(repo_context)
        assert 0.0 <= score <= 1.0
