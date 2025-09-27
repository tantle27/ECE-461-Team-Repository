import pytest
from src.metrics.code_quality_metric import CodeQualityMetric
from src.repo_context import RepoContext
from unittest.mock import MagicMock, patch

def make_ctx(files=None, readme=None):
    ctx = MagicMock(spec=RepoContext)
    ctx.readme_text = readme if readme is not None else "# Project\nSome code."
    ctx.files = files if files is not None else ["main.py", "utils.py", "tests/test_main.py"]
    return ctx

def test_code_quality_signals_various():
    metric = CodeQualityMetric(use_llm=False)
    # No files, short readme
    ctx = make_ctx(files=[], readme="Short")
    repo_context = {"_ctx_obj": ctx}
    score = metric.evaluate(repo_context)
    assert score == 0.0
    # No files, long readme
    ctx = make_ctx(files=[], readme="A"*1300)
    repo_context = {"_ctx_obj": ctx}
    score = metric.evaluate(repo_context)
    assert score == 0.0
    # Files with tests, ci, lint, etc.
    files = [
        "main.py", "src/module.py", "tests/test_main.py", "pyproject.toml", ".github/workflows", ".flake8", "mypy.ini"
    ]
    ctx = make_ctx(files=files)
    repo_context = {"_ctx_obj": ctx}
    score = metric.evaluate(repo_context)
    assert 0.0 <= score <= 1.0

def test_code_quality_llm_exception():
    metric = CodeQualityMetric(use_llm=True)
    ctx = make_ctx()
    repo_context = {"_ctx_obj": ctx}
    # Patch _llm_score to raise
    with patch.object(metric, "_llm", create=True):
        with patch.object(metric, "_llm_score", side_effect=Exception("fail")):
            score = metric.evaluate(repo_context)
            assert 0.0 <= score <= 1.0

def test_code_quality_get_description():
    metric = CodeQualityMetric()
    desc = metric.get_description()
    assert "code quality" in desc.lower()
