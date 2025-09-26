# metrics/dataset_quality_metric.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

from api.llm_client import LLMClient
from repo_context import RepoContext

from .base_metric import BaseMetric


@dataclass(frozen=True)
class HeuristicWeights:
    """Heuristic weights (sum ~= 1)."""

    validation: float = 0.40
    diversity: float = 0.30
    completeness: float = 0.30


@dataclass(frozen=True)
class LLMBlend:
    """How to blend LLM and heuristics."""

    llm_weight: float = 0.60
    heu_weight: float = 0.40


class DatasetQualityMetric(BaseMetric):
    """
    Evaluates dataset quality via heuristics + optional LLM rubric.

    Heuristics use HF tags/card/README to approximate:
      - validation (schema/checks/quality gates)
      - diversity (classes/languages/domains/size signals)
      - completeness (splits/labels/metadata for training/eval)

    If LLM is enabled and a dataset RepoContext is available,
    we ask for structured sub-scores and blend them conservatively.
    """

    def __init__(self, weight: float = 0.15, use_llm: bool = True) -> None:
        super().__init__(name="DatasetQuality", weight=weight)
        self._use_llm = use_llm
        self._llm = LLMClient()
        self._hw = HeuristicWeights()
        self._blend = LLMBlend()

    # ---------------- public API ----------------

    def evaluate(self, repo_context: dict) -> float:
        """
        1) Build a dataset context
        2) Score heuristics from tags/card/README.
        3) If LLM is available, ask for JSON sub-scores and blend heuristics.
        """
        ds = self._pick_dataset_ctx(repo_context)
        if not ds:
            heur = self._heuristics_from_repo_context(repo_context)
            return self._combine_heuristics(**heur)

        heur = self._heuristics_from_dataset(ds)
        heuristic_score = self._combine_heuristics(**heur)

        if self._use_llm and getattr(self._llm, "provider", None):
            try:
                llm_score, parts = self._score_with_llm(ds)
                repo_context["_dataset_quality_llm_parts"] = parts
                return self._clamp01(
                    self._blend.llm_weight * llm_score
                    + self._blend.heu_weight * heuristic_score
                )
            except Exception:
                pass

        return heuristic_score

    def get_description(self) -> str:
        return (
            "Evaluates dataset quality via validation/diversity/"
            "completeness + optional LLM rubric"
        )

    # ---------------- context selection ----------------

    def _pick_dataset_ctx(self, repo_context: dict) -> Optional[RepoContext]:
        """
        If scoring a dataset URL, return that context.
        If scoring a model, try the first linked dataset (already hydrated).
        Otherwise, None.
        """
        ctx = repo_context.get("_ctx_obj")
        if not isinstance(ctx, RepoContext):
            return None

        # If the context itself is clearly a dataset
        if ctx.hf_id and ctx.hf_id.startswith("datasets/"):
            return ctx

        # If it's a model with linked datasets, use the first one
        if ctx.linked_datasets:
            return ctx.linked_datasets[0]

        return None

    # ---------------- heuristics ----------------

    def _heuristics_from_repo_context(self, rc: dict) -> Dict[str, float]:
        """Heuristics from a generic repo_context (when no ds context)."""
        readme = (rc.get("readme_text") or "").lower()
        tags = [str(t).lower() for t in (rc.get("tags") or [])]
        card = rc.get("card_data") or {}

        return self._compute_heuristics(tags, card, readme)

    def _heuristics_from_dataset(self, ds: RepoContext) -> Dict[str, float]:
        readme = (getattr(ds, "readme_text", "") or "").lower()
        tags = [str(t).lower() for t in (getattr(ds, "tags", []) or [])]
        card = getattr(ds, "card_data", {}) or {}
        return self._compute_heuristics(tags, card, readme)

    def _compute_heuristics(
        self, tags: list[str], card: Dict[str, Any], readme_low: str
    ) -> Dict[str, float]:
        """
        Turn HF tags/card/readme into three signals in [0,1]:
          - has_validation
          - data_diversity
          - data_completeness
        """

        valida_words = (
            "validation",
            "validator",
            "schema",
            "quality check",
            "data audit",
            "data checks",
        )
        has_validation = any(w in readme_low for w in valida_words)

        if (
            "sphinx" in readme_low
            or "mkdocs" in readme_low
            or "checks" in readme_low
        ):
            has_validation = True or has_validation

        split_words = ("train", "validation", "dev", "test")
        has_split_terms = sum(1 for w in split_words if w in readme_low)

        card_has_splits = False
        for k in ("train-eval-index", "splits", "configs", "tasks"):
            if k in (card or {}):
                card_has_splits = True
                break

        has_col_map = False
        try:
            tei = card.get("train-eval-index")
            if isinstance(tei, list) and tei and isinstance(tei[0], dict):
                cm = tei[0].get("col_mapping") or {}
                has_col_map = bool(cm)
        except Exception:
            pass

        # Score completeness
        completeness = 0.0
        completeness += (
            0.45
            if has_split_terms >= 2
            else 0.20 if has_split_terms >= 1 else 0.0
        )
        completeness += 0.35 if card_has_splits else 0.0
        completeness += 0.20 if has_col_map else 0.0
        completeness = self._clamp01(completeness)

        diversity = 0.0
        if any(
            t.startswith("multilinguality:") and "multilingual" in t
            for t in tags
        ):
            diversity += 0.35
        if sum(1 for t in tags if t.startswith("language:")) >= 2:
            diversity += 0.20
        elif any(t.startswith("language:") for t in tags):
            diversity += 0.10
        if any(t.startswith("size_categories:") for t in tags):
            diversity += 0.15
        modality_cnt = sum(1 for t in tags if t.startswith("modality:"))
        if modality_cnt >= 2:
            diversity += 0.20
        elif modality_cnt == 1:
            diversity += 0.10
        if any(
            w in readme_low
            for w in (
                "multiple domains",
                "varied sources",
                "heterogeneous",
                "diverse",
            )
        ):
            diversity += 0.10

        diversity = self._clamp01(diversity)

        validation = 1.0 if has_validation else 0.0

        return dict(
            has_validation=validation,
            data_diversity=diversity,
            data_completeness=completeness,
        )

    def _combine_heuristics(
        self,
        *args,
        has_validation: float | None = None,
        data_diversity: float | None = None,
        data_completeness: float | None = None,
    ) -> float:
        if args and (
            has_validation is None
            and data_diversity is None
            and data_completeness is None
        ):
            try:
                has_validation, data_diversity, data_completeness = args[:3]
            except Exception:
                pass
        has_validation = 0.0 if has_validation is None else has_validation
        data_diversity = 0.0 if data_diversity is None else data_diversity
        data_completeness = (
            0.0 if data_completeness is None else data_completeness
        )

        w = self._hw
        score = (
            w.validation * self._clamp01(has_validation)
            + w.diversity * self._clamp01(data_diversity)
            + w.completeness * self._clamp01(data_completeness)
        )
        return self._clamp01(score)

    # ---------------- LLM integration ----------------

    def _score_with_llm(
        self, ds: RepoContext
    ) -> Tuple[float, Dict[str, float]]:
        system = "You are a strict dataset-quality rater. Return ONLY JSON."
        prompt = self._make_llm_prompt(ds)
        res = self._llm.ask_json(system, prompt, max_tokens=700)

        if not res.ok or not isinstance(res.data, dict):
            raise RuntimeError(res.error or "LLM returned no data")

        d = res.data

        def g(key: str) -> float:
            try:
                v = float(d.get(key, 0.0))
            except Exception:
                v = 0.0
            return self._clamp01(v)

        parts = {
            "has_validation": g("has_validation"),
            "data_diversity": g("data_diversity"),
            "data_completeness": g("data_completeness"),
            "documentation": g("documentation"),
            "ethical_considerations": g("ethical_considerations"),
        }

        base = (
            0.40 * parts["has_validation"]
            + 0.30 * parts["data_diversity"]
            + 0.30 * parts["data_completeness"]
        )
        bonus = (
            0.06 * parts["documentation"]
            + 0.04 * parts["ethical_considerations"]
        )
        return self._clamp01(base + bonus), parts

    def _make_llm_prompt(self, ds: RepoContext) -> str:
        readme = (getattr(ds, "readme_text", "") or "")[:6000]
        tags_list = list(getattr(ds, "tags", []) or [])
        tags = ", ".join(tags_list)
        card = getattr(ds, "card_data", {}) or {}
        return f"""
    You are evaluating dataset quality for ML reuse.

    Rate the following sub-criteria strictly in [0,1]:
    - has_validation: Evidence of schema/validation/checks/quality gates.
    - data_diversity: Diversity across classes/languages/domains/demographics.
    - data_completeness: Splits/labels/metadata sufficient for normal training/eval
    - documentation: Clarity of README/card (task, license, use, limits, issues).
    - ethical_considerations: Bias/safety notes, provenance, consent.

    Return ONLY JSON with keys:
    {{
    "has_validation": 0.0,
    "data_diversity": 0.0,
    "data_completeness": 0.0,
    "documentation": 0.0,
    "ethical_considerations": 0.0
    }}

    Context (Hugging Face tags/card & README excerpt):
    Tags: {tags}
    Card: {card}
    ---
    {readme}
    ---
    """.strip()

    # ---------------- utils ----------------

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
