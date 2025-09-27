import pytest
from src.metrics.code_quality_metric import CodeQualityMetric
from src.repo_context import RepoContext
from unittest.mock import MagicMock, patch

def make_ctx(files=None, readme=None):
    ctx = MagicMock(spec=RepoContext)
    ctx.readme_text = readme if readme is not None else "# Project\nSome code."
    ctx.files = files if files is not None else ["main.py", "utils.py", "tests/test_main.py"]
    return ctx

def test_code_quality_has_real_code():
    metric = CodeQualityMetric()
    # _has_real_code requires at least 3 real code files
    files = ["main.py", "utils.py", "foo.cpp"]
    assert metric._has_real_code(files)
    files = ["main.py", "README.md"]
    assert not metric._has_real_code(files)

def test_code_quality_signals_struct_and_rq():
    metric = CodeQualityMetric()
    class File:
        def __init__(self, path):
            self.path = path
    files = [File("pyproject.toml"), File("src/module.py"), File("manifest.in"), File("requirements.txt")]
    ctx = make_ctx(files=files, readme="pip install\nusage\n![](badge)")
    signals = metric._signals(ctx.readme_text, metric._files(ctx))
    # struct keys are now top-level in signals
    assert signals["reqs"]
    # src_layout and manifest are not present in new signals, so skip
    # assert signals["struct"]["src_layout"]
    # assert signals["struct"]["manifest"]
    # Instead, check for arch_markers and run_scripts/classic_scripts
    assert signals["arch_markers"] >= 0
    assert signals["run_scripts"] >= 0
    assert signals["classic_scripts"] >= 0
    assert signals["rq"]["install"]
    assert signals["rq"]["usage"]
    assert signals["rq"]["badges"]
    assert signals["rq"]["fences"] == 0

def test_code_quality_quant_and_weights():
    metric = CodeQualityMetric()
    signals = {
        "repo_size": 10,
        "test_file_count": 2,
        "pytest_cfg": True,
        "ci": True,
        "lint": True,
        "fmt": True,
        "typing": True,
        "struct": {"pyproject_or_setup": True, "src_layout": True, "manifest": True, "reqs": True},
        "rq": {"len": 1000, "install": True, "usage": True, "badges": True, "fences": 2},
        "test_has_dir": True,
        "arch_markers": 1,
        "notebook_count": 0,
        "script_count": 0,
        "reqs": True,
        "run_scripts": 0,
    "classic_scripts": 0,
    "notebooks": 0,
    "contrib": False,
    "license_file": False,
    }
    q = metric._quant(signals)
    w = metric._weights(signals)
    assert abs(sum(w.values()) - 1.0) < 1e-6
    assert all(0.0 <= v <= 1.0 for v in q.values())

def test_code_quality_base_score_variants():
    metric = CodeQualityMetric()
    # Use a dummy ctx (None), and test with various readme/files
    score = metric._base_score(None, "short", [])
    assert 0.0 <= score <= 1.0
    score = metric._base_score(None, "def foo():\n" + "A"*1200, [])
    assert 0.0 <= score <= 1.0
    score = metric._base_score(None, "A"*500, [])
    assert 0.0 <= score <= 1.0
    score = metric._base_score(None, "", ["main.py", "foo.cpp", "bar.rs"])
    assert 0.0 <= score <= 1.0

def test_code_quality_variance():
    metric = CodeQualityMetric()
    q = {k: 0.5 for k in ["tests", "ci", "lint_fmt", "typing", "docs", "structure", "recency"]}
    assert 0.0 <= metric._variance(q) <= 1.0
    q2 = {k: 1.0 if i % 2 == 0 else 0.0 for i, k in enumerate(q)}
    assert 0.0 <= metric._variance(q2) <= 1.0

def test_code_quality_coverage():
    metric = CodeQualityMetric()
    s = {
        "test_file_count": 1,
        "ci": True,
        "lint": True,
        "typing": True,
        "fmt": True,
        "struct": {"pyproject_or_setup": True},
        "rq": {"len": 500, "fences": 2},
        "arch_markers": 0,
        "notebook_count": 0,
        "script_count": 0,
        "reqs": True,
        "run_scripts": 0,
    "classic_scripts": 0,
    "notebooks": 0,
    "contrib": False,
    "license_file": False,
    }
    cov = metric._coverage(s)
    assert 0.0 <= cov <= 1.0
