# metrics/ramp_up_time_metric.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

from api.llm_client import LLMClient
from repo_context import RepoContext

from .base_metric import BaseMetric


@dataclass(frozen=True)
class HeuristicWeights:
    """Heuristic ramp-up sub-weights (sum ~= 1)."""

    quickstart: float = 0.35  # clear install + basic usage
    examples: float = 0.30  # runnable examples / demo notebooks
    docs_depth: float = 0.20  # API/reference/troubleshooting/config
    ext_docs: float = 0.15  # external docs/wiki/readthedocs


@dataclass(frozen=True)
class BlendCfg:
    """Final blend between LLM and heuristics."""

    llm_weight: float = 0.6  # lean on LLM when available
    heu_weight: float = 0.4


class RampUpTimeMetric(BaseMetric):
    """
    Measures how quickly a developer can become productive:
    - Quickstart clarity (install + minimal working example)
    - Examples/demos (copy-pastable, runnable)
    - Depth of docs (API/ref/troubleshooting/config)
    - External docs or wiki
    Optional: LLM rubric over README (+ linked code README when present).
    """

    def __init__(self, weight: float = 0.15, use_llm: bool = True):
        super().__init__(name="RampUpTime", weight=weight)
        self._use_llm = use_llm
        self._llm = LLMClient()
        self._hw = HeuristicWeights()
        self._blend = BlendCfg()

    # ---------- public API ----------

    def evaluate(self, repo_context: dict) -> float:
        ctx = repo_context.get("_ctx_obj")
        readme = self._gather_readme(repo_context, ctx)

        # Heuristic baseline (always available)
        heur_parts = self._heuristic_parts(readme)
        heuristic_score = self._combine_heuristics(**heur_parts)

        # LLM rubric (optional)
        if (
            self._use_llm
            and getattr(self._llm, "provider", None)
            and readme.strip()
        ):
            try:
                llm_score, parts = self._score_with_llm(readme)
                repo_context["_rampup_llm_parts"] = parts
                return self._clamp01(
                    self._blend.llm_weight * llm_score
                    + self._blend.heu_weight * heuristic_score
                )
            except Exception:
                # fall back silently
                pass

        return heuristic_score

    def get_description(self) -> str:
        return (
            "Evaluates ease of getting started via quickstart, docs depth, "
            "and external docs (+ LLM rubric)."
        )

    # ---------- LLM integration ----------

    def _score_with_llm(self, readme: str) -> Tuple[float, Dict[str, float]]:
        """
        Ask LLM for structured ramp-up sub-scores in [0,1]:
          - quickstart_clarity
          - examples_quality
          - docs_depth
          - external_docs_quality
          - setup_friction (1 = low friction)
        Then compute final LLM score as weighted sum.
        """
        sys = (
            "You are a strict engineering documentation rater. "
            "Return ONLY a single JSON object."
        )
        prompt = self._make_llm_prompt(readme)
        res = self._llm.ask_json(sys, prompt, max_tokens=600)

        if not res.ok or not isinstance(res.data, dict):
            raise RuntimeError(res.error or "LLM returned no JSON")

        d = res.data

        def g(k: str) -> float:
            try:
                v = float(d.get(k, 0.0))
            except Exception:
                v = 0.0
            return self._clamp01(v)

        parts = {
            "quickstart_clarity": g("quickstart_clarity"),
            "examples_quality": g("examples_quality"),
            "docs_depth": g("docs_depth"),
            "external_docs_quality": g("external_docs_quality"),
            "setup_friction": g("setup_friction"),
        }

        llm_score = (
            self._hw.quickstart * parts["quickstart_clarity"]
            + self._hw.examples * parts["examples_quality"]
            + self._hw.docs_depth * parts["docs_depth"]
            + self._hw.ext_docs * parts["external_docs_quality"]
        )
        # small bump for setup friction (0..1) -> up to +0.05
        llm_score = self._clamp01(llm_score + 0.05 * parts["setup_friction"])

        return llm_score, parts

    def _make_llm_prompt(self, readme: str) -> str:
        excerpt = readme[:8000]
        return f"""
Read the README excerpt and rate each item in [0,1]. Be strict; reward
copy-pastable steps and runnable examples.

Return ONLY JSON with these keys:
{{
  "quickstart_clarity": 0.0,  // install + minimal working example
  "examples_quality": 0.0,    // runnable, complete examples/notebooks
  "docs_depth": 0.0,          // API/reference, config, troubleshooting
  "external_docs_quality": 0.0, // links to docs/wiki/readthedocs
  "setup_friction": 0.0       // 1=easy, 0=hard
}}

README excerpt:
---
{excerpt}
---
""".strip()

    # ---------- Heuristic path ----------

    def _gather_readme(
        self, repo_context: dict, ctx: Optional[RepoContext]
    ) -> str:
        """
        Prefer the current README. If this is a MODEL with a linked code repo,
        concatenate a short excerpt of the code README to improve signal.
        """
        base = (repo_context.get("readme_text") or "").strip()

        if isinstance(ctx, RepoContext) and ctx.linked_code:
            best = max(
                (c for c in ctx.linked_code if isinstance(c, RepoContext)),
                key=lambda c: len(c.readme_text or ""),
                default=None,
            )
            if best and best.readme_text:
                extra = best.readme_text.strip()[:4000]
                if extra and extra not in base:
                    base = (
                        f"{base}\n\n---\n# Linked code README excerpt\n{extra}"
                    )
        return base

    def _heuristic_parts(self, readme: str) -> Dict[str, float]:
        r = (readme or "").lower()

        # Quickstart
        has_install = any(
            k in r
            for k in (
                "pip install",
                "conda install",
                "poetry add",
                "npm i ",
                "npm install",
            )
        )
        has_quick = any(
            k in r
            for k in ("getting started", "quickstart", "quick start", "setup")
        )
        has_usage = any(
            k in r
            for k in (
                "usage",
                "example",
                "run",
                "inference",
                "train",
                "predict",
            )
        )
        quickstart = 0.0
        if has_install and (has_quick or has_usage):
            quickstart = (
                0.9 if ("```" in r or "import " in r or "from " in r) else 0.6
            )
        elif has_install or has_quick or has_usage:
            quickstart = 0.35

        # Examples
        has_code_fence = "```" in r
        has_notebook = ".ipynb" in r or "colab" in r
        has_demo = any(
            k in r for k in ("demo", "example", "samples", "notebook")
        )
        examples = 0.0
        if has_code_fence and (has_notebook or has_demo):
            examples = 0.9
        elif has_code_fence or has_demo or has_notebook:
            examples = 0.55

        # Docs depth
        has_api = any(
            k in r for k in ("api reference", "api docs", "reference")
        )
        has_config = any(
            k in r
            for k in ("configuration", "config", "settings", "parameters")
        )
        has_trbl = "troubleshooting" in r or "faq" in r
        docs_depth = 0.0
        if has_api and (has_config or has_trbl):
            docs_depth = 0.8
        elif has_api or has_config or has_trbl:
            docs_depth = 0.45

        # External docs
        has_ext = any(
            k in r
            for k in (
                "readthedocs",
                "https://docs.",
                "http://docs.",
                "wiki",
                "mkdocs",
            )
        )
        ext_docs = 0.8 if has_ext else 0.0

        return dict(
            quickstart=quickstart,
            examples=examples,
            docs_depth=docs_depth,
            ext_docs=ext_docs,
        )

    def _combine_heuristics(
        self,
        *,
        quickstart: float,
        examples: float,
        docs_depth: float,
        ext_docs: float,
    ) -> float:
        w = self._hw
        score = (
            w.quickstart * self._clamp01(quickstart)
            + w.examples * self._clamp01(examples)
            + w.docs_depth * self._clamp01(docs_depth)
            + w.ext_docs * self._clamp01(ext_docs)
        )
        return self._clamp01(score)

    # ---------- utils ----------

    @staticmethod
    def _clamp01(x: Any) -> float:
        try:
            xf = float(x)
        except Exception:
            return 0.0
        if xf < 0.0:
            return 0.0
        if xf > 1.0:
            return 1.0
        return xf
