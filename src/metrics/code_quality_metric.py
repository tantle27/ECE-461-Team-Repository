from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, Optional, Tuple

from api.llm_client import LLMClient
from repo_context import RepoContext

from .base_metric import BaseMetric


# -------------------- Weights & limits --------------------


@dataclass(frozen=True)
class HeuristicW:
    tests: float = 0.25
    automation: float = 0.15
    standards: float = 0.20  # linting/typing/formatting
    docs: float = 0.15
    maintenance: float = 0.15
    structure: float = 0.10


@dataclass(frozen=True)
class LLmW:
    tests: float = 0.25
    guidance: float = 0.15
    linting: float = 0.20
    complexity_ok: float = 0.20  # 1 = low risk
    maintainability: float = 0.20


LLM_MAX_TOKENS = 700


# ==================== Metric ====================


class CodeQualityMetric(BaseMetric):
    """
    Hybrid code-quality score:
      • Heuristics from repo signals
      • Optional LLM rubric over README + signals (adaptively blended)

    For MODELs, prefers linked GitHub repo (gh_url match > GitHub > has code
    > file count).
    """

    def __init__(self, weight: float = 0.2, use_llm: bool = True):
        super().__init__(name="CodeQuality", weight=weight)
        self.hw = HeuristicW()
        self.lw = LLmW()
        self._use_llm = use_llm
        self._llm = LLMClient()

    # ---------- public ----------

    def evaluate(self, repo_context: dict) -> float:
        ctx = self._ctx(repo_context)
        if not ctx:
            return 0.0

        # Baselines so datasets / model cards don't crater NetScore
        cat = repo_context.get("category", "").upper()
        if cat == "DATASET":
            return 0.5
        if cat == "MODEL":
            code_ctx = self._best_code_ctx(ctx)
            if code_ctx:
                repo_context["_evaluated_code_repo"] = (
                    getattr(code_ctx, "gh_url", None)
                    or getattr(code_ctx, "url", None)
                )
                ctx = code_ctx
            elif not getattr(ctx, "linked_code", None):
                return 0.4

        signals = self._extract_signals(ctx)
        heuristic = self._heuristic(signals)

        if not (self._use_llm and getattr(self._llm, "provider", None)):
            return heuristic

        try:
            cov = self._signal_coverage(signals)  # 0..1
            llm_val, parts = self._llm_score(ctx, signals)
            repo_context["_code_quality_llm_parts"] = parts
            # adaptive blend: lean more on LLM if signals are sparse
            llm_w = min(0.8, 0.3 + (1.0 - cov) * 0.7)  # 0.3..0.8
            return self._clamp01(
                (1.0 - llm_w) * heuristic + llm_w * llm_val
            )
        except Exception:
            return heuristic

    def get_description(self) -> str:
        return (
            "Hybrid code-quality: tests/CI/standards/docs/maintenance + "
            "optional LLM rubric"
        )

    # ---------- LLM ----------

    def _llm_score(
        self, ctx: RepoContext, signals: Dict[str, Any]
    ) -> Tuple[float, Dict[str, float]]:
        sys = (
            "You are a strict software-quality rater. Return ONLY one JSON "
            "object. No prose."
        )
        prompt = self._make_prompt(ctx, signals)
        res = self._llm.ask_json(sys, prompt, max_tokens=LLM_MAX_TOKENS)
        if not res.ok or not isinstance(res.data, dict):
            raise RuntimeError(res.error or "LLM returned no data")

        d = res.data
        parts = {
            "tests_presence": self._clamp01(d.get("tests_presence", 0.0)),
            "test_guidance": self._clamp01(d.get("test_guidance", 0.0)),
            "linting_presence": self._clamp01(d.get("linting_presence", 0.0)),
            # 1 = low risk
            "complexity_risk": self._clamp01(d.get("complexity_risk", 0.0)),
            "maintainability": self._clamp01(d.get("maintainability", 0.0)),
        }
        w = self.lw
        score = (
            w.tests * parts["tests_presence"]
            + w.guidance * parts["test_guidance"]
            + w.linting * parts["linting_presence"]
            + w.complexity_ok * parts["complexity_risk"]
            + w.maintainability * parts["maintainability"]
        )
        return self._clamp01(score), parts

    def _make_prompt(
        self, ctx: RepoContext, signals: Dict[str, Any]
    ) -> str:
        readme = (getattr(ctx, "readme_text", "") or "")[:6000]
        # keep template lines short for flake8
        template_top = (
            "Rate repository *code quality* for engineering reuse. "
            "Use FACTS as ground truth. If a signal is missing, infer "
            "conservatively from README (avoid zero unless clearly absent)."
        )
        template_expect = (
            "Return ONLY JSON with [0,1] values and ≤60-word \"rationale\":\n"
            "{\n"
            '  "tests_presence": 0.0,\n'
            '  "test_guidance": 0.0,\n'
            '  "linting_presence": 0.0,\n'
            '  "complexity_risk": 0.0,\n'
            '  "maintainability": 0.0,\n'
            '  "rationale": ""\n'
            "}\n"
        )
        facts = f"FACTS:\n{signals}\n"
        excerpt = (
            "README (excerpt):\n---\n"
            f"{readme}\n"
            "---"
        )
        return "\n".join([template_top, "", template_expect, facts, excerpt])

    # ---------- context helpers ----------

    def _ctx(self, repo_context: dict) -> Optional[RepoContext]:
        obj = repo_context.get("_ctx_obj")
        return obj if isinstance(obj, RepoContext) else None

    def _best_code_ctx(self, ctx: RepoContext) -> Optional[RepoContext]:
        linked = [
            c for c in getattr(ctx, "linked_code", [])
            if isinstance(c, RepoContext)
        ]
        if not linked:
            return None

        gh_url = (getattr(ctx, "gh_url", "") or "").lower()

        def key(c: RepoContext) -> tuple[int, int, int, int]:
            url = (
                getattr(c, "url", "") or getattr(c, "gh_url", "") or ""
            ).lower()
            is_gh = 1 if (
                "github.com" in url
                or (getattr(c, "host", "") or "").lower() == "github.com"
            ) else 0
            same = 1 if (gh_url and (gh_url in url or url in gh_url)) else 0
            has_code = 1 if self._has_code_files(c) else 0
            nfiles = len(getattr(c, "files", []) or [])
            return (same, is_gh, has_code, nfiles)

        return max(linked, key=key)

    def _has_code_files(self, ctx: RepoContext) -> bool:
        files = getattr(ctx, "files", []) or []
        code_ext = {
            ".py", ".js", ".jsx", ".ts", ".tsx", ".java", ".cpp", ".c",
            ".rs", ".go", ".rb", ".php", ".cs",
        }
        for f in files:
            p = str(getattr(f, "path", "")).lower()
            if any(p.endswith(ext) for ext in code_ext):
                return True
        return False

    # ---------- signals ----------

    def _extract_signals(self, ctx: RepoContext) -> Dict[str, Any]:
        files = [
            str(f.path).replace("\\", "/").lstrip("./").lower()
            for f in (getattr(ctx, "files", []) or [])
        ]
        fs = set(files)

        def has_any(names: tuple[str, ...]) -> bool:
            return any(n in fs for n in names)

        test_struct = {
            "has_tests_dir": any(
                p.startswith(("tests/", "test/")) for p in fs
            ),
            "has_test_files": any(
                p.endswith("_test.py")
                or p.endswith("test.py")
                or "/test_" in p
                or p.startswith("test_")
                for p in fs
            ),
            "has_pytest_config": has_any(
                ("pytest.ini", "tox.ini", "pyproject.toml")
            ),
        }
        test_files = sum(
            1
            for p in fs
            if (
                p.endswith("_test.py")
                or (p.endswith("test.py") and not p.endswith("_test.py"))
                or (p.startswith(("tests/", "test/")) and p.endswith(".py"))
                or ("/tests/" in p and p.endswith(".py"))
                or p.endswith(".test.js")
                or p.endswith(".spec.js")
            )
        )

        ci = {
            "github_actions": any(".github/workflows/" in p for p in fs),
            "gitlab_ci": ".gitlab-ci.yml" in fs,
            "circle_ci": ".circleci/config.yml" in fs,
            "travis_ci": ".travis.yml" in fs,
            "azure_pipelines": "azure-pipelines.yml" in fs,
        }
        ci["has_ci"] = any(ci.values())

        cfg = self._config(ctx)
        lint = {
            "flake8": ".flake8" in fs or ("flake8" in cfg),
            "ruff": any(
                n in fs for n in ("ruff.toml", ".ruff.toml")
            ) or ("ruff" in cfg),
            "pylint": (".pylintrc" in fs) or ("pylint" in cfg),
            "eslint": any(
                n in fs
                for n in (
                    ".eslintrc",
                    ".eslintrc.json",
                    ".eslintrc.js",
                    "eslint.config.js",
                )
            ),
        }
        typing = {
            "mypy": ("mypy.ini" in fs) or ("mypy" in cfg),
            "pyright": ("pyrightconfig.json" in fs) or ("pyright" in cfg),
            "typescript": "tsconfig.json" in fs,
        }
        fmt = {
            "black": ("black" in cfg) or ("pyproject.toml" in fs),
            "isort": ("isort" in cfg) or (".isort.cfg" in fs),
            "prettier": (".prettierrc" in fs) or ("prettier" in cfg),
        }

        docs = {
            "has_docs_dir": any(p.startswith("docs/") for p in fs),
            "has_doc_generator": (
                "docs/conf.py" in fs
                or ("sphinx" in str(cfg).lower())
                or any(n in fs for n in ("mkdocs.yml", "mkdocs.yaml"))
            ),
            "doc_file_count": sum(
                1 for p in fs if p.endswith((".md", ".rst", ".mdx"))
            ),
        }
        readme_text = getattr(ctx, "readme_text", None) or ""
        rl = readme_text.lower()
        readme = {
            "length": len(readme_text),
            "has_installation": ("install" in rl) or ("pip install" in rl),
            "has_usage": (
                "usage" in rl or "example" in rl or "getting started" in rl
            ),
            "has_badges": (
                ("![" in readme_text and "badge" in rl) or ("shields.io" in rl)
            ),
        }

        last_modified = getattr(ctx, "last_modified", None)
        days = (
            self._days_since(last_modified) if last_modified else float("inf")
        )
        maintenance = {
            "days_since_update": None if not math.isfinite(days) else days,
            "recently_updated": days < 90,
            "actively_maintained": days < 30,
        }
        contribs = getattr(ctx, "contributors", []) or []

        structure = {
            "has_src_dir": any(p.startswith("src/") for p in fs),
            "has_lib_structure": any(p.startswith("lib/") for p in fs),
            "has_setup_py": "setup.py" in fs,
            "has_pyproject": "pyproject.toml" in fs,
            "has_manifest": "manifest.in" in fs,
        }
        has_examples = any(
            p.startswith(("examples/", "example/"))
            or "examples" in p
            or "demo" in p
            for p in fs
        )

        return {
            "repo_size": len(fs),
            "test_structure": test_struct,
            "test_file_count": test_files,
            "ci_config": ci,
            "linting_config": lint,
            "typing_config": typing,
            "formatting_config": fmt,
            "docs_quality": docs,
            "readme_quality": readme,
            "maintenance_signals": maintenance,
            "contributor_activity": {"contributor_count": len(contribs)},
            "project_structure": structure,
            "examples_present": has_examples,
        }

    def _signal_coverage(self, s: Dict[str, Any]) -> float:
        bits = [
            (
                1.0
                if s.get("test_file_count", 0) > 0
                or s.get("test_structure", {}).get("has_tests_dir")
                else 0.0
            ),
            1.0 if s.get("ci_config", {}).get("has_ci") else 0.0,
            1.0 if any(s.get("linting_config", {}).values()) else 0.0,
            1.0 if any(s.get("typing_config", {}).values()) else 0.0,
            1.0 if any(s.get("formatting_config", {}).values()) else 0.0,
            (
                1.0
                if (
                    s.get("docs_quality", {}).get("has_docs_dir")
                    or (s.get("readme_quality", {}).get("length", 0) > 300)
                )
                else 0.0
            ),
            (
                1.0
                if s.get("contributor_activity", {}).get(
                    "contributor_count", 0
                )
                > 0
                else 0.0
            ),
        ]
        return sum(bits) / len(bits)

    # ---------- heuristic scoring ----------

    def _heuristic(self, s: Dict[str, Any]) -> float:
        w = self.hw
        return self._clamp01(
            w.tests * self._score_tests(s)
            + w.automation * self._score_automation(s)
            + w.standards * self._score_standards(s)
            + w.docs * self._score_docs(s)
            + w.maintenance * self._score_maintenance(s)
            + w.structure * self._score_structure(s)
        )

    def _score_tests(self, s: Dict[str, Any]) -> float:
        ts = s["test_structure"]
        ntests = int(s["test_file_count"] or 0)
        nfiles = max(1, int(s["repo_size"] or 1))
        base = 0.0
        if ts.get("has_tests_dir") or ts.get("has_test_files"):
            base += 0.55
        if ts.get("has_pytest_config"):
            base += 0.15
        # full credit for ~1 test per 20 files (saturating)
        dens = min(ntests / max(1.0, nfiles / 20.0), 1.0)
        return min(1.0, base + 0.30 * dens)

    def _score_automation(self, s: Dict[str, Any]) -> float:
        ci = s["ci_config"]
        fs = 0.0
        if ci["has_ci"]:
            fs += 0.4
        # infer from presence of common files stored in signals (cheap proxy)
        has_precommit = any(s.get("formatting_config", {}).values())
        if has_precommit:
            fs += 0.2
        # slight positive bias (we don't track deps/locks explicitly here)
        has_deps = True
        if has_deps:
            fs += 0.2
        has_locks = any(s.get("linting_config", {}).values())
        if has_locks:
            fs += 0.2
        return min(1.0, fs)

    def _score_standards(self, s: Dict[str, Any]) -> float:
        lint = s["linting_config"]
        typ = s["typing_config"]
        fmt = s["formatting_config"]
        return min(
            1.0,
            (0.4 if any(lint.values()) else 0.0)
            + (0.3 if any(typ.values()) else 0.0)
            + (0.3 if any(fmt.values()) else 0.0),
        )

    def _score_docs(self, s: Dict[str, Any]) -> float:
        docs = s["docs_quality"]
        r = s["readme_quality"]
        length = max(0, int(r.get("length") or 0))
        len_score = min(length / 1500.0, 1.0) * 0.25
        struct = (0.25 if docs.get("has_docs_dir") else 0.0)
        struct += 0.15 if docs.get("has_doc_generator") else 0.0
        struct += 0.10 if (docs.get("doc_file_count") or 0) > 3 else 0.0
        guidance = (0.15 if r.get("has_installation") else 0.0)
        guidance += 0.20 if r.get("has_usage") else 0.0
        guidance += 0.05 if r.get("has_badges") else 0.0
        return min(1.0, len_score + struct + guidance)

    def _score_maintenance(self, s: Dict[str, Any]) -> float:
        m = s["maintenance_signals"]
        c = s["contributor_activity"]
        days = m.get("days_since_update")
        # 0 at ~1y, 1 at fresh
        activity = (
            0.0 if days is None else (1.0 - self._sigmoid(days, k=0.04, x0=90))
        )
        contribs = self._sigmoid(
            max(0, int(c.get("contributor_count") or 0)), k=0.9, x0=2
        )
        return self._clamp01(0.6 * activity + 0.4 * contribs)

    def _score_structure(self, s: Dict[str, Any]) -> float:
        st = s["project_structure"]
        ex = bool(s["examples_present"])
        score = 0.0
        if st.get("has_pyproject") or st.get("has_setup_py"):
            score += 0.35
        if st.get("has_src_dir") or st.get("has_lib_structure"):
            score += 0.35
        if st.get("has_manifest"):
            score += 0.10
        if ex:
            score += 0.20
        return min(1.0, score)

    # ---------- tiny utils ----------

    def _config(self, ctx: RepoContext) -> Dict[str, Any]:
        cfg = getattr(ctx, "config_json", {}) or {}
        if isinstance(cfg, dict):
            return {str(k).lower(): v for k, v in cfg.items()}
        return {}

    def _days_since(self, ts: str) -> float:
        try:
            dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
            delta = datetime.now(timezone.utc) - dt
            return delta.total_seconds() / 86400.0
        except Exception:
            return float("inf")

    @staticmethod
    def _clamp01(x: Any) -> float:
        try:
            xf = float(x)
            if xf < 0:
                return 0.0
            if xf > 1:
                return 1.0
            return xf
        except Exception:
            return 0.0

    @staticmethod
    def _sigmoid(x: float, k: float = 1.0, x0: float = 0.0) -> float:
        try:
            return 1.0 / (1.0 + math.exp(-k * (x - x0)))
        except Exception:
            return 0.0
