def test_code_quality_best_code_ctx_none_and_richness():
    from src.metrics.code_quality_metric import CodeQualityMetric
    from repo_context import RepoContext, FileInfo
    from pathlib import Path
    m = CodeQualityMetric()
    ctx = RepoContext()
    # No linked_code
    ctx.linked_code = None
    assert m._best_code_ctx(ctx) is None
    # Empty linked_code
    ctx.linked_code = []
    assert m._best_code_ctx(ctx) is None
    # Multiple candidates, test richness

    c1 = RepoContext()
    c1.files = [
        FileInfo(path=Path("a.py"), size_bytes=1, ext=".py"),
        FileInfo(path=Path("b.py"), size_bytes=1, ext=".py")
    ]
    c1.readme_text = "abc"

    c2 = RepoContext()
    c2.files = [FileInfo(path=Path("a.py"), size_bytes=1, ext=".py")]
    c2.readme_text = "abcd"

    ctx.linked_code = [c1, c2]
    # c1 is richer (more files)
    assert m._best_code_ctx(ctx) == c1

def test_code_quality_signals_empty_files():
    from src.metrics.code_quality_metric import CodeQualityMetric
    m = CodeQualityMetric()
    s = m._signals("", [])
    assert isinstance(s, dict)
    assert s["test_file_count"] == 0
    assert not s["test_has_dir"]

def test_code_quality_quant_pytest_cfg_and_typing():
    from src.metrics.code_quality_metric import CodeQualityMetric
    m = CodeQualityMetric()
    s = {
        "repo_size": 10,
        "test_file_count": 2,
        "test_has_dir": True,
        "pytest_cfg": True,
        "ci": False,
        "lint": False,
        "fmt": False,
        "typing": True,
        "struct": {"pyproject_or_setup": False, "src_layout": False, "manifest": False, "reqs": False},
        "rq": {"len": 0, "install": False, "usage": False, "badges": False, "fences": 0},
        "arch_markers": 0,
        "notebook_count": 0,
        "script_count": 0,
        "reqs": False,
        "run_scripts": 0,
    "classic_scripts": 0,
    "notebooks": 0,
    "contrib": False,
    "license_file": False,
    }
    q = m._quant(s)
    assert q["tests"] > 0
    assert q["typing"] == 1.0

def test_code_quality_weights_varied_cases():
    from src.metrics.code_quality_metric import CodeQualityMetric
    m = CodeQualityMetric()
    # Minimal repo
    s1 = {
        "repo_size": 1,
        "test_file_count": 0,
        "test_has_dir": False,
        "pytest_cfg": False,
        "ci": False,
        "lint": False,
        "fmt": False,
        "typing": False,
        "struct": {"pyproject_or_setup": False, "src_layout": False, "manifest": False, "reqs": False},
        "rq": {"len": 0, "install": False, "usage": False, "badges": False, "fences": 0},
        "arch_markers": 0,
        "notebook_count": 0,
        "script_count": 0,
        "reqs": False,
        "run_scripts": 0,
    "classic_scripts": 0,
    "notebooks": 0,
    "contrib": False,
    "license_file": False,
    }
    w1 = m._weights(s1)
    assert abs(sum(w1.values()) - 1.0) < 1e-6
    # Large repo
    s2 = dict(s1)
    s2["repo_size"] = 2000
    w2 = m._weights(s2)
    assert abs(sum(w2.values()) - 1.0) < 1e-6

def test_code_quality_base_score_popularity_and_floors():
    from src.metrics.code_quality_metric import CodeQualityMetric
    from src.repo_context import RepoContext
    m = CodeQualityMetric()
    files = ["src/a.py", "tests/test_a.py", "pyproject.toml"]
    readme = "Install: pip install x\nUsage: example\n```python\nprint('x')\n```"
    ctx = RepoContext(); ctx.files = files; ctx.readme_text = readme
    ctx.downloads_all_time = 2_000_000; ctx.likes = 300
    base = m._base_score(ctx, readme, files)
    assert base >= 0.72
    # Lint/ci/test_has_dir floor
    ctx.likes = 0; ctx.downloads_all_time = 0
    base2 = m._base_score(ctx, readme, files)
    assert base2 >= 0.60
def test_code_quality_has_real_code():
    from src.metrics.code_quality_metric import CodeQualityMetric
    m = CodeQualityMetric()
    # Fewer than 3 code files
    assert not m._has_real_code(["a.py", "b.js"])
    # Exactly 3 code files
    assert m._has_real_code(["a.py", "b.js", "c.cpp"])
    # More than 3
    assert m._has_real_code(["a.py", "b.js", "c.cpp", "d.rs"])

def test_code_quality_files_and_readme():
    from src.metrics.code_quality_metric import CodeQualityMetric
    from src.repo_context import RepoContext
    m = CodeQualityMetric()
    ctx = RepoContext()
    ctx.files = [type("F", (), {"path": "src/a.py"})()]
    ctx.readme_text = "README"
    files = m._files(ctx)
    assert files == ["src/a.py"]
    readme = m._readme(ctx)
    assert readme == "README"

def test_code_quality_signals_and_quant():
    from src.metrics.code_quality_metric import CodeQualityMetric
    m = CodeQualityMetric()
    files = ["src/a.py", "tests/test_a.py", "pyproject.toml", ".flake8", ".github/workflows/ci.yml"]
    readme = "Install: pip install x\nUsage: example\n```python\nprint('x')\n```"
    s = m._signals(readme, files)
    q = m._quant(s)
    assert set(q.keys()) == {"tests", "ci", "lint_fmt", "typing", "docs", "structure", "recency", "arch", "notebooks", "scripts"}
    for v in q.values():
        assert 0.0 <= v <= 1.0

def test_code_quality_weights_and_base_score():
    from src.metrics.code_quality_metric import CodeQualityMetric
    from src.repo_context import RepoContext
    m = CodeQualityMetric()
    files = ["src/a.py", "tests/test_a.py", "pyproject.toml"]
    readme = "Install: pip install x\nUsage: example\n```python\nprint('x')\n```"
    s = m._signals(readme, files)
    w = m._weights(s)
    assert abs(sum(w.values()) - 1.0) < 1e-6
    ctx = RepoContext()
    ctx.files = files
    ctx.readme_text = readme
    ctx.downloads_all_time = 2_000_000
    ctx.likes = 300
    base = m._base_score(ctx, readme, files)
    assert 0.0 <= base <= 1.0

def test_code_quality_llm_score_and_exceptions():
    from src.metrics.code_quality_metric import CodeQualityMetric
    m = CodeQualityMetric(use_llm=True)
    class DummyLLM:
        def ask_json(self, sys_p, prompt, max_tokens=380):
            class R: pass
            r = R()
            r.ok = True
            r.data = {"maintainability": 0.8, "readability": 0.7, "documentation": 0.6, "reusability": 0.5}
            return r
    m._llm = DummyLLM()
    readme = "README"
    files = ["a.py", "b.py", "c.py"]
    signals = m._signals(readme, files)
    score, parts = m._llm_score(readme, files, signals)
    assert 0.0 <= score <= 1.0
    # Exception path
    class BadLLM:
        def ask_json(self, *a, **k):
            class R: pass
            r = R()
            r.ok = False
            r.data = None
            r.error = "fail"
            return r
    m._llm = BadLLM()
    with pytest.raises(RuntimeError):
        m._llm_score(readme, files, signals)

def test_code_quality_coverage_and_variance():
    from src.metrics.code_quality_metric import CodeQualityMetric
    m = CodeQualityMetric()
    s = {
        "repo_size": 10,
        "test_file_count": 2,
        "test_has_dir": True,
        "pytest_cfg": True,
        "ci": True,
        "lint": True,
        "fmt": True,
        "typing": True,
        "struct": {"pyproject_or_setup": True, "src_layout": True, "manifest": True, "reqs": True},
        "rq": {"len": 500, "install": True, "usage": True, "badges": True, "fences": 3},
        "arch_markers": 2,
        "notebook_count": 1,
        "script_count": 1,
        "reqs": True,
        "run_scripts": 1,
    "classic_scripts": 1,
    "notebooks": 0,
    "contrib": False,
    "license_file": False,
    }
    q = m._quant(s)
    cov = m._coverage(s)
    var = m._variance(q)
    assert 0.0 <= cov <= 1.0
    assert 0.0 <= var <= 1.0
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
