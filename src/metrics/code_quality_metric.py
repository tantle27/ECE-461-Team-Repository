from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict, Tuple

from repo_context import RepoContext
from .base_metric import BaseMetric

try:
    from api.llm_client import LLMClient
except Exception:
    LLMClient = None
import logging

logger = logging.getLogger("acme-cli")


def _c01(x: Any) -> float:
    try:
        xf = float(x)
    except Exception:
        return 0.0
    if xf < 0:
        return 0.0
    if xf > 1:
        return 1.0
    return xf


@dataclass(frozen=True)
class _Blend:
    max_llm_w: float = 0.60
    base_llm_w: float = 0.30
    cov_gain: float = 0.40
    var_gain: float = 0.20


class CodeQualityMetric(BaseMetric):
    """
    Scores code quality from README + in-context file list only (no gh_url).
    """

    def __init__(self, weight: float = 0.2, use_llm: bool = True) -> None:
        super().__init__(name="CodeQuality", weight=weight)
        self._use_llm = use_llm
        self._llm = LLMClient() if LLMClient else None
        self._blend = _Blend()

    # -------- public --------

    def evaluate(self, repo_context: dict) -> float:
        ctx = repo_context.get("_ctx_obj")
        if not isinstance(ctx, RepoContext):
            return 0.0

        files = self._files(ctx)
        readme = self._readme(ctx)
        base = self._base_score(readme, files)
        llm_ready = (
            self._use_llm
            and self._llm
            and getattr(self._llm, "provider", None)
        )
        if not llm_ready:
            logger.info("CodeQuality LLM not ready, using base score only.")
            return base

        signals = self._signals(readme, files)
        q = self._quant(signals)
        cov = self._coverage(signals)
        var = self._variance(q)
        llm_w = min(
            self._blend.max_llm_w,
            self._blend.base_llm_w
            + (1 - cov) * self._blend.cov_gain
            + var * self._blend.var_gain,
        )

        try:
            llm, parts = self._llm_score(readme, files, signals)
            repo_context["_code_quality_llm_parts"] = parts
            logger.info(
                f"CodeQuality LLM parts: {parts}, cov={cov:.3f}, "
                f"var={var:.3f}, llm_w={llm_w:.3f}"
            )
            return _c01((1 - llm_w) * base + llm_w * llm)
        except Exception:
            return base

    def get_description(self) -> str:
        return (
            "Quality from README+FILE_LIST only; tests/CI/lint/typing/format/"
            "docs/structure, generous LLM blend."
        )

    # -------- internals --------

    def _files(self, ctx: RepoContext) -> list[str]:
        out: list[str] = []
        for f in getattr(ctx, "files", []) or []:
            p = str(getattr(f, "path", "")).replace("\\", "/").lstrip("./")
            p = p.lower()
            if p:
                out.append(p)
        return out

    def _readme(self, ctx: RepoContext) -> str:
        return (getattr(ctx, "readme_text", "") or "")[:8000]

    def _has_code_hints(self, readme: str) -> bool:
        r = (readme or "").lower()
        keys = (
            "```",
            "import ",
            "from ",
            "def ",
            "class ",
            "pytorch",
            "transformers",
        )
        return any(k in r for k in keys)

    def _signals(self, readme: str, files: list[str]) -> Dict[str, Any]:
        fs = set(files)
        rl = (readme or "").lower()

        def has(*names: str) -> bool:
            return any(name in fs for name in names)

        tests_dir = any(p.startswith(("tests/", "test/")) for p in fs)
        test_cnt = sum(
            1
            for p in fs
            if p.endswith(("_test.py", ".test.js", ".spec.js"))
            or "/tests/" in p
            or "/test/" in p
        )
        nb_cnt = sum(1 for p in fs if p.endswith(".ipynb"))
        examples = any(p.startswith(("examples/", "example/")) for p in fs)

        if not fs:
            tests_dir = tests_dir or any(
                w in rl for w in ("pytest", "unittest", "unit test")
            )
            if any(w in rl for w in ("pytest", "unit test")):
                test_cnt = max(test_cnt, 1)

        return {
            "repo_size": len(fs),
            "test_file_count": test_cnt + nb_cnt,
            "test_has_dir": tests_dir or examples,
            "pytest_cfg": has("pytest.ini", "tox.ini"),
            "ci": (
                has(
                    ".github/workflows",
                    ".gitlab-ci.yml",
                    ".circleci/config.yml",
                    "azure-pipelines.yml",
                    ".travis.yml",
                )
                or any(w in rl for w in ("github actions", "workflow", "ci/"))
            ),
            "lint": has(
                ".flake8",
                ".pylintrc",
                "ruff.toml",
                ".eslintrc",
                ".eslintrc.json",
                ".eslintrc.js",
                "eslint.config.js",
            ),
            "fmt": has("pyproject.toml", ".prettierrc", ".isort.cfg"),
            "typing": has("mypy.ini", "pyrightconfig.json", "tsconfig.json"),
            "struct": {
                "pyproject_or_setup": any(
                    x in fs for x in ("pyproject.toml", "setup.py")
                ),
                "src_layout": any(p.startswith("src/") for p in fs),
                "manifest": "manifest.in" in fs,
                "reqs": ("requirements.txt" in fs or "environment.yml" in fs),
            },
            "rq": {
                "len": len(readme or ""),
                "install": any(
                    k in rl
                    for k in (
                        "pip install",
                        "conda install",
                        "poetry add",
                        "pipx",
                        "pip install -e",
                    )
                ),
                "usage": any(
                    k in rl
                    for k in (
                        "usage",
                        "example",
                        "inference",
                        "getting started",
                        "quickstart",
                        "quick start",
                    )
                ),
                "badges": ("![" in (readme or ""))
                and ("badge" in rl or "shields.io" in rl),
                "fences": (readme or "").count("```"),
            },
        }

    def _quant(self, s: Dict[str, Any]) -> Dict[str, float]:
        n = max(1, int(s["repo_size"]))
        tests = min((s["test_file_count"] * 1000.0 / n) / 10.0, 1.0)
        if s["pytest_cfg"]:
            tests = min(1.0, tests + 0.15)

        ci = 1.0 if s["ci"] else 0.0
        lint = 1.0 if s["lint"] else 0.0
        fmt = 1.0 if s["fmt"] else 0.0
        lint_fmt = _c01(0.6 * lint + 0.4 * fmt)
        typing = 1.0 if s["typing"] else 0.0

        rq = s["rq"]
        rd_len = _c01(rq["len"] / 900.0)
        guidance = 0.0
        guidance += 0.30 if rq["install"] else 0.0
        guidance += 0.40 if rq["usage"] else 0.0
        guidance += 0.05 if rq["badges"] else 0.0
        code_blocks = min(rq["fences"] / 3.0, 1.0) * 0.25
        docs = _c01(0.35 * rd_len + 0.40 * guidance + code_blocks)

        st = s["struct"]
        structure = 0.0
        structure += 0.30 if st["pyproject_or_setup"] else 0.0
        structure += 0.25 if st["src_layout"] else 0.0
        structure += 0.10 if st["manifest"] else 0.0
        structure += 0.15 if st["reqs"] else 0.0
        structure = min(1.0, structure)

        q = {
            "tests": tests,
            "ci": ci,
            "lint_fmt": lint_fmt,
            "typing": typing,
            "docs": docs,
            "structure": structure,
            "recency": 0.6,  # neutral
        }
        return q

    def _weights(self, s: Dict[str, Any]) -> Dict[str, float]:
        w = {
            "tests": 0.18,
            "ci": 0.12,
            "lint_fmt": 0.16,
            "typing": 0.12,
            "docs": 0.26,
            "structure": 0.10,
            "recency": 0.06,
        }
        if s["test_file_count"] > 0 or s["test_has_dir"]:
            w["tests"] += 0.05
            w["ci"] += 0.02
            w["docs"] -= 0.03
        if s["typing"]:
            w["typing"] += 0.05
            w["lint_fmt"] += 0.02
            w["docs"] -= 0.02

        n = max(1, int(s["repo_size"]))
        if n < 120:
            w["docs"] += 0.03
            w["tests"] -= 0.02
        elif n > 1000:
            w["tests"] += 0.03
            w["ci"] += 0.02
            w["docs"] -= 0.03

        ssum = sum(w.values())
        return {k: v / ssum for k, v in w.items()}

    def _base_score(self, readme: str, files: list[str]) -> float:
        s = self._signals(readme, files)
        q = self._quant(s)
        w = self._weights(s)
        base = _c01(sum(q[k] * w[k] for k in w))
        if not files:
            rlen = len(readme or "")
            if rlen >= 1200 or self._has_code_hints(readme):
                base = max(base, 0.55)
            elif rlen >= 400:
                base = max(base, 0.50)
            else:
                base = max(base, 0.45)
        return base

    def _variance(self, q: Dict[str, float]) -> float:
        vals = list(q.values())
        m = sum(vals) / len(vals)
        var = sum((v - m) ** 2 for v in vals) / len(vals)
        return _c01(var / 0.25)

    # -------- LLM --------

    def _llm_score(
        self,
        readme: str,
        files: list[str],
        signals: Dict[str, Any],
    ) -> Tuple[float, Dict[str, float]]:
        sys_p = (
            "You are a software-quality rater. " "Return ONLY one JSON object."
        )
        sample_files = "\n".join(files[:3500])
        prompt = (
            "Rate 0..1 for each key; ground in FACTS/README/FILE_LIST.\n"
            "{\n"
            '  "maintainability": 0.0,\n'
            '  "readability": 0.0,\n'
            '  "documentation": 0.0,\n'
            '  "reusability": 0.0\n'
            "}\n\nFACTS:\n```json\n"
            f"{json.dumps(signals, indent=2)}\n```\n\nREADME:\n---\n"
            f"{readme[:3500]}\n---\n\nFILE_LIST:\n---\n"
            f"{sample_files}\n---\n"
        )
        res = self._llm.ask_json(sys_p, prompt, max_tokens=380)
        ok = getattr(res, "ok", False)
        if not ok or not isinstance(res.data, dict):
            raise RuntimeError(getattr(res, "error", "LLM error"))

        def g(key: str) -> float:
            return _c01(res.data.get(key, 0.0))

        parts = {
            k: g(k)
            for k in (
                "maintainability",
                "readability",
                "documentation",
                "reusability",
            )
        }
        llm = (
            0.33 * parts["maintainability"]
            + 0.27 * parts["readability"]
            + 0.25 * parts["documentation"]
            + 0.15 * parts["reusability"]
        )
        return _c01(llm), parts

    def _coverage(self, s: Dict[str, Any]) -> float:
        bits = [
            s["test_file_count"] > 0,
            s["ci"],
            s["lint"],
            s["typing"],
            s["fmt"],
            s["struct"]["pyproject_or_setup"],
            (s["rq"]["len"] > 400) or (s["rq"]["fences"] >= 2),
        ]
        return sum(1.0 if b else 0.0 for b in bits) / len(bits)
