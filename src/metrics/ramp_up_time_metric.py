# metrics/ramp_up_time_metric.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

from api.llm_client import LLMClient
from repo_context import RepoContext
from .base_metric import BaseMetric


@dataclass(frozen=True)
class HeuristicWeights:
    """Heuristic sub-weights (sum ~= 1)."""

    quickstart: float = 0.30
    examples: float = 0.30
    docs_depth: float = 0.25
    ext_docs: float = 0.15


@dataclass(frozen=True)
class BlendCfg:
    """Blend LLM and heuristics (LLM-forward but fair)."""

    llm_weight: float = 0.70
    heu_weight: float = 0.30
    fairness_cap: float = 0.35


class RampUpTimeMetric(BaseMetric):
    """
    Ramp-up: quickstart, examples, docs depth, external docs.
    LLM-forward with fairness: confidence-aware scoring + bounded override.
    """

    def __init__(self, weight: float = 0.15, use_llm: bool = True):
        super().__init__(name="RampUpTime", weight=weight)
        self._use_llm, self._llm = use_llm, LLMClient()
        self._hw, self._blend = HeuristicWeights(), BlendCfg()

    # ---------- public API ----------

    def evaluate(self, repo_context: dict) -> float:
        ctx: Optional[RepoContext] = repo_context.get("_ctx_obj")
        readme = self._gather_readme(repo_context, ctx)
        heur = self._combine_heuristics(**self._heuristic_parts(readme))

        if not (self._use_llm and getattr(self._llm, "provider", None)):
            return heur

        try:
            llm, conf, parts = self._score_with_llm(readme)
            repo_context["_rampup_llm_parts"] = parts

            base_w = self._blend.llm_weight
            conf_w = min(0.25, 0.5 * max(0.0, conf - 0.5))
            w_llm = min(0.9, base_w + conf_w)
            w_heu = 1.0 - w_llm
            raw = w_llm * llm + w_heu * heur

            cap = self._blend.fairness_cap
            final = heur + max(-cap, min(cap, raw - heur))

            if parts.get("any_signal"):
                final = max(final, 0.25)

            return self._clamp01(final)
        except Exception:
            return heur

    def get_description(self) -> str:
        return (
            "Measures the ease of getting started: quickstarts, examples, "
            "setup docs, and external documentation signals. "
            "LLM-forward ramp-up score with confidence and bounded override."
        )

    # ---------- LLM integration ----------

    def _score_with_llm(
        self, readme: str
    ) -> Tuple[float, float, Dict[str, float]]:
        sys = (
            "You are an engineering documentation rater. "
            "Return ONLY one JSON object."
        )
        res = self._llm.ask_json(
            sys, self._make_llm_prompt(readme), max_tokens=700
        )
        if not res.ok or not isinstance(res.data, dict):
            raise RuntimeError(res.error or "LLM returned no JSON")

        def g(k: str) -> float:
            try:
                return self._clamp01(float(res.data.get(k, 0.0)))
            except Exception:
                return 0.0

        parts = {
            "quickstart_clarity": g("quickstart_clarity"),
            "examples_quality": g("examples_quality"),
            "docs_depth": g("docs_depth"),
            "external_docs_quality": g("external_docs_quality"),
            "setup_friction": g("setup_friction"),
            "confidence": g("confidence"),
        }
        # Detect "any signal" for a fair floor
        parts["any_signal"] = any(
            parts[k] > 0.0
            for k in (
                "quickstart_clarity",
                "examples_quality",
                "docs_depth",
                "external_docs_quality",
            )
        )

        hw = self._hw
        llm = (
            hw.quickstart * parts["quickstart_clarity"]
            + hw.examples * parts["examples_quality"]
            + hw.docs_depth * parts["docs_depth"]
            + hw.ext_docs * parts["external_docs_quality"]
        )
        llm = self._clamp01(llm + 0.10 * parts["setup_friction"])

        if llm > 0.0:
            llm = self._clamp01(0.10 + 0.90 * llm)

        return llm, parts["confidence"], parts

    def _make_llm_prompt(self, readme: str) -> str:
        excerpt = readme[:8000]
        return (
            "Read the README excerpt and rate each item in [0,1]. "
            "Reward copy-pastable steps, runnable examples, actionable docs. "
            "Include confidence in [0,1] based on evidence quality.\n\n"
            "Return ONLY JSON with keys:\n"
            "{\n"
            '  "quickstart_clarity": 0.0,\n'
            '  "examples_quality": 0.0,\n'
            '  "docs_depth": 0.0,\n'
            '  "external_docs_quality": 0.0,\n'
            '  "setup_friction": 0.0,\n'
            '  "confidence": 0.0\n'
            "}\n\n"
            "README excerpt:\n---\n"
            f"{excerpt}\n---"
        )

    # ---------- Heuristics (kept simple & generous) ----------

    def _gather_readme(
        self, repo_context: dict, ctx: Optional[RepoContext]
    ) -> str:
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
                    base = f"{base}\n\n---\n# Linked code README\n{extra}"
        return base

    def _heuristic_parts(self, readme: str) -> Dict[str, float]:
        r = (readme or "").lower()
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
            k in r for k in ("usage", "example", "inference", "train", "run")
        )
        quickstart = (
            (0.85 if ("```" in r or "import " in r) else 0.65)
            if (has_install and (has_quick or has_usage))
            else (0.45 if (has_install or has_quick or has_usage) else 0.0)
        )
        has_code_fence = "```" in r
        has_notebook = ".ipynb" in r or "colab" in r
        has_demo = any(k in r for k in ("demo", "example", "samples", "nb"))
        examples = (
            0.9 if (has_code_fence and (has_notebook or has_demo)) else 0.0
        )
        has_api = any(
            k in r for k in ("api reference", "api docs", "reference")
        )
        has_cfg = any(k in r for k in ("configuration", "config", "settings"))
        has_trb = "troubleshooting" in r or "faq" in r
        docs_depth = (
            0.85
            if (has_api and (has_cfg or has_trb))
            else (0.55 if (has_api or has_cfg or has_trb) else 0.0)
        )
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
        ext_docs = 0.6 if has_ext else 0.0
        return {
            "quickstart": quickstart,
            "examples": examples,
            "docs_depth": docs_depth,
            "ext_docs": ext_docs,
        }

    def _combine_heuristics(
        self,
        *,
        quickstart: float,
        examples: float,
        docs_depth: float,
        ext_docs: float,
    ) -> float:
        hw = self._hw
        score = (
            hw.quickstart * self._clamp01(quickstart)
            + hw.examples * self._clamp01(examples)
            + hw.docs_depth * self._clamp01(docs_depth)
            + hw.ext_docs * self._clamp01(ext_docs)
        )
        return self._clamp01(score)

    # ---------- utils ----------

    @staticmethod
    def _clamp01(x: Any) -> float:
        try:
            xf = float(x)
        except Exception:
            return 0.0
        return 0.0 if xf < 0.0 else 1.0 if xf > 1.0 else xf
