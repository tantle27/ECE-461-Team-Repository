import pytest
from src.metrics.code_quality_metric import CodeQualityMetric
from src.repo_context import RepoContext
from unittest.mock import MagicMock, patch

def make_fake_ctx():
    ctx = MagicMock(spec=RepoContext)
    ctx.readme_text = "# Project\nSome code."
    ctx.files = ["main.py", "utils.py"]
    return ctx

def test_code_quality_basic():
    metric = CodeQualityMetric(use_llm=False)
    repo_context = {"_ctx_obj": make_fake_ctx()}
    score = metric.evaluate(repo_context)
    assert 0.0 <= score <= 1.0

def test_code_quality_no_ctx():
    metric = CodeQualityMetric(use_llm=False)
    repo_context = {}
    score = metric.evaluate(repo_context)
    assert score == 0.0

def test_code_quality_with_llm():
    metric = CodeQualityMetric(use_llm=True)
    repo_context = {"_ctx_obj": make_fake_ctx()}
    with patch.object(metric, "_llm", create=True):
        score = metric.evaluate(repo_context)
        assert 0.0 <= score <= 1.0
