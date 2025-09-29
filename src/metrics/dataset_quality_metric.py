from __future__ import annotations

"""Dataset quality metric.

This module computes a dataset quality score using simple heuristics and an
optional LLM-based assessor. It is formatted to be flake8/PEP8 friendly.
"""

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple, List

from repo_context import RepoContext
from .base_metric import BaseMetric

try:
    from api.llm_client import LLMClient
except Exception:
    LLMClient = None  # type: ignore[assignment]


def _c01(x: Any) -> float:
    """Clamp numeric-like value to [0.0, 1.0]."""
    try:
        xf = float(x)
    except Exception:
        return 0.0
    if xf < 0.0:
        return 0.0
    if xf > 1.0:
        return 1.0
    return xf


@dataclass(frozen=True)
class HeuristicWeights:
    """Weights for the heuristic components."""

    validation: float = 0.38
    diversity: float = 0.30
    completeness: float = 0.32


@dataclass(frozen=True)
class LLMBlend:
    """Weights to blend LLM and heuristics."""

    llm_weight: float = 0.58
    heu_weight: float = 0.42


class DatasetQualityMetric(BaseMetric):
    """Compute dataset quality with heuristics and optional LLM."""

    def __init__(self, weight: float = 0.15, use_llm: bool = True) -> None:
        super().__init__(name="DatasetQuality", weight=weight)
        self._use_llm = use_llm
        self._llm = LLMClient() if LLMClient else None
        self._hw = HeuristicWeights()
        self._blend = LLMBlend()

    def evaluate(self, repo_context: dict) -> float:
        """Return a dataset-quality score in [0,1]."""
        ds = self._pick_explicit_dataset(repo_context)
        if not ds:
            # Accept a dict that resembles a dataset context.
            if isinstance(repo_context, dict):
                keys = (
                    "tags",
                    "card_data",
                    "readme_text",
                    "files",
                    "hf_id",
                )
                if any(k in repo_context for k in keys):
                    ds = self._wrap_dataset_dict(repo_context)
            else:
                return 0.0

        heur = self._heuristics_from_dataset(ds)
        heuristic_score = self._combine_heuristics(**heur)
        heuristic_score = self._apply_engagement_floor(heuristic_score, ds)

        if not (
            self._use_llm
            and self._llm
            and getattr(self._llm, "provider", None)
        ):
            return heuristic_score

        try:
            llm_score, parts = self._score_with_llm(ds)
            repo_context["_dataset_quality_llm_parts"] = parts
            return _c01(
                self._blend.llm_weight * llm_score
                + self._blend.heu_weight * heuristic_score
            )
        except Exception:
            return heuristic_score

    def get_description(self) -> str:
        return (
            "Scores dataset quality only when a dataset is explicitly "
            "attached to the model. Uses heuristics and optional LLM "
            "blend."
        )

    def _pick_explicit_dataset(self, repo_context: 
                               dict) -> Optional[RepoContext]:
        """Pick an explicitly linked dataset context, if present."""
        ctx = repo_context.get("_ctx_obj")
        if not isinstance(ctx, RepoContext):
            return None

        if ctx.hf_id and ctx.hf_id.startswith("datasets/"):
            src = getattr(ctx, "__dict__", {}).get("_link_source")
            return ctx if src == "explicit" else None

        for d in getattr(ctx, "linked_datasets", []) or []:
            if not isinstance(d, RepoContext):
                continue
            src = getattr(d, "__dict__", {}).get("_link_source")
            if src == "explicit":
                return d
        return None

    def _heuristics_from_dataset(self, ds: RepoContext) -> Dict[str, float]:
        readme_low = (getattr(ds, "readme_text", "") or "").lower()
        tags = [str(t).lower() for t in (getattr(ds, "tags", []) or [])]
        card = getattr(ds, "card_data", {}) or {}
        files = self._file_list(ds)
        return self._compute_heuristics(tags, card, readme_low, files)

    def _file_list(self, ds: RepoContext) -> List[str]:
        out: List[str] = []
        for f in getattr(ds, "files", []) or []:
            if isinstance(f, str):
                p = f
            else:
                p = (
                    getattr(f, "path", None)
                    or (isinstance(f, dict) and f.get("path"))
                    or ""
                )
            p = str(p).replace("\\", "/").lstrip("./").lower()
            if p:
                out.append(p)

        readme = (getattr(ds, "readme_text", "") or "").lower()
        if "ipynb" in readme and "notebook" not in " ".join(out):
            out.append("notebook.ipynb")
        return out

    def _compute_heuristics(
        self,
        tags: list[str],
        card: Dict[str, Any],
        readme_low: str,
        files: List[str],
    ) -> Dict[str, float]:
        """Compute three heuristic components.

        Returns a dict with keys: has_validation, data_diversity,
        data_completeness.
        """
        valida_words = (
            "validation",
            "validator",
            "validate",
            "schema",
            "quality check",
            "quality checks",
            "data audit",
            "data checks",
            "integrity",
            "consistency",
            "checksums",
            "unit tests",
            "validation set",
        )
        has_validation_kw = any(w in readme_low for w in valida_words)
        has_infos_file = any("dataset_infos.json" in p for p in files)

        has_script = any(
            (p.endswith(".py") or p.endswith(".ipynb"))
            and ("dataset" in p or "prepare" in p)
            for p in files
        )

        validation = 0.0
        if has_validation_kw or has_infos_file or has_script:
            validation = 0.75
        if has_validation_kw and (has_infos_file or has_script):
            validation = 1.0

        split_keys = ("train-eval-index", "splits", "configs", "tasks")
        card_has_splits = any(k in (card or {}) for k in split_keys)

        split_words = ("train", "validation", "dev", "test")
        has_split_terms = sum(1 for w in split_words if w in readme_low)

        has_col_map = False
        try:
            tei = card.get("train-eval-index")
            if isinstance(tei, list) and tei and isinstance(tei[0], dict):
                cm = tei[0].get("col_mapping") or {}
                has_col_map = bool(cm)
        except Exception:
            pass

        completeness = 0.0
        if has_split_terms >= 2:
            completeness += 0.40
        elif has_split_terms >= 1:
            completeness += 0.18
        if card_has_splits:
            completeness += 0.33
        if has_col_map:
            completeness += 0.18

        classes = 0
        try:
            classes = int(card.get("classes") or 0)
        except Exception:
            classes = 0
        if classes >= 2:
            completeness += 0.12
        if any(
            w in readme_low
            for w in (
                "label",
                "labels",
                "annotation",
                "annotations",
                "ground truth",
            )
        ):
            completeness += 0.08
        completeness = _c01(completeness)

        diversity = 0.0
        if any(
            isinstance(t, str)
            and t.startswith("multilinguality:")
            and "multilingual" in t
            for t in tags
        ):
            diversity += 0.30

        lang_cnt = sum(
            1 for t in tags if isinstance(t, str) and t.startswith("language:")
        )
        if lang_cnt >= 3:
            diversity += 0.28
        elif lang_cnt == 2:
            diversity += 0.18
        elif lang_cnt == 1:
            diversity += 0.08

        if any(isinstance(t, str) and t.startswith("size_categories:") 
               for t in tags):
            diversity += 0.15

        modality_cnt = sum(
            1 for t in tags if isinstance(t, str) and t.startswith("modality:")
        )
        if modality_cnt >= 2:
            diversity += 0.14
        elif modality_cnt == 1:
            diversity += 0.07

        if any(
            w in readme_low
            for w in (
                "multiple domains",
                "varied sources",
                "heterogeneous",
                "diverse",
                "variety",
            )
        ):
            diversity += 0.10
        diversity = _c01(diversity)

        strong_bonus = 0.0
        if validation >= 0.75 and completeness >= 0.55:
            strong_bonus += 0.06
        if has_infos_file and has_script:
            strong_bonus += 0.03
        if has_col_map and card_has_splits:
            strong_bonus += 0.03

        validation = _c01(validation + 0.03 * strong_bonus)
        completeness = _c01(completeness + 0.04 * strong_bonus)
        diversity = _c01(diversity + 0.03 * strong_bonus)

        return {
            "has_validation": validation,
            "data_diversity": diversity,
            "data_completeness": completeness,
        }

    def _wrap_dataset_dict(self, d: dict) -> RepoContext:
        class SimpleCtx:
            pass

        obj = SimpleCtx()
        for k, v in (d or {}).items():
            try:
                setattr(obj, k, v)
            except Exception:
                pass
        return obj

    def _combine_heuristics(
        self,
        *,
        has_validation: Optional[float] = None,
        data_diversity: Optional[float] = None,
        data_completeness: Optional[float] = None,
    ) -> float:
        has_validation = 0.0 if has_validation is None else has_validation
        data_diversity = 0.0 if data_diversity is None else data_diversity
        data_completeness = 0.0 if data_completeness is None else data_completeness

        w = self._hw
        score = (
            w.validation * _c01(has_validation)
            + w.diversity * _c01(data_diversity)
            + w.completeness * _c01(data_completeness)
        )
        return _c01(score)

    def _apply_engagement_floor(self, score: float, ds: RepoContext) -> float:
        dl = int(getattr(ds, "downloads_all_time", 0) or 0)
        likes = int(getattr(ds, "likes", 0) or 0)

        if likes > 1500 or dl > 1_500_000:
            score = max(score, 0.90)
        elif likes > 400 or dl > 400_000:
            score = max(score, 0.84)
        elif likes > 100 or dl > 100_000:
            score = max(score, 0.78)

        famous = {
            "bookcorpus/bookcorpus",
            "squad/squad",
            "wikitext/wikitext-103-raw-v1",
            "glue/glue",
            "imagenet-1k/imagenet-1k",
        }
        ds_key = (getattr(ds, "hf_id", "") or "").lower()
        if any(k in ds_key for k in famous):
            score = max(score, 0.90)

        return min(1.0, score + 0.06)

    def _score_with_llm(self, ds: RepoContext) -> Tuple[float, Dict[str, float]]:
        assert self._llm is not None

        system = "You are a strict dataset-quality rater. Return ONLY JSON."
        prompt = self._make_llm_prompt(ds)
        res = self._llm.ask_json(system, prompt, max_tokens=700)

        if not getattr(res, "ok", False) or not isinstance(res.data, dict):
            raise RuntimeError(getattr(res, "error", "LLM returned no data"))

        d = res.data

        def g(key: str) -> float:
            try:
                v = float(d.get(key, 0.0))
            except Exception:
                v = 0.0
            return _c01(v)

        parts = {
            "has_validation": g("has_validation"),
            "data_diversity": g("data_diversity"),
            "data_completeness": g("data_completeness"),
            "documentation": g("documentation"),
            "ethical_considerations": g("ethical_considerations"),
        }

        base = 0.40 * parts["has_validation"]
        base += 0.30 * parts["data_diversity"]
        base += 0.30 * parts["data_completeness"]
        bonus = (
            0.06 * parts["documentation"]
            + 0.04 * parts["ethical_considerations"]
        )
        return _c01(base + bonus), parts

    def _make_llm_prompt(self, ds: RepoContext) -> str:
        readme = (getattr(ds, "readme_text", "") or "")[:6000]
        tags_list = list(getattr(ds, "tags", []) or [])
        tags = ", ".join(tags_list)
        card = getattr(ds, "card_data", {}) or {}
        return (
            "Evaluate dataset quality for ML reuse.\n\n"
            "Rate each in [0,1]:\n"
            "- has_validation: Evidence of schema/validation/gates.\n"
            "- data_diversity: Classes/languages/domains.\n"
            "- data_completeness: Splits/labels/metadata for train/eval.\n"
            "- documentation: Clarity of README and usage details.\n"
            "- ethical_considerations: Bias/safety, provenance, consent.\n\n"
            "Return ONLY JSON with keys:\n"
            "{\n"
            "  \"has_validation\": 0.0,\n"
            "  \"data_diversity\": 0.0,\n"
            "  \"data_completeness\": 0.0,\n"
            "  \"documentation\": 0.0,\n"
            "  \"ethical_considerations\": 0.0\n"
            "}\n\n"
            f"Tags: {tags}\n"
            f"Card: {card}\n"
            "---\n"
            f"{readme}\n"
            "---\n"
        )
# src/metrics/dataset_quality_metric.py
from __future__ import annotations


from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple, List


from repo_context import RepoContext
from .base_metric import BaseMetric


try:
    from api.llm_client import LLMClient
except Exception:
    # src/metrics/dataset_quality_metric.py
    from __future__ import annotations

    from dataclasses import dataclass
    from typing import Any, Dict, Optional, Tuple, List

    from repo_context import RepoContext
    from .base_metric import BaseMetric

    try:
        from api.llm_client import LLMClient
    except Exception:
        LLMClient = None  # type: ignore[assignment]
from __future__ import annotations

# src/metrics/dataset_quality_metric.py

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple, List

from repo_context import RepoContext
from .base_metric import BaseMetric

try:
    from api.llm_client import LLMClient
except Exception:
    LLMClient = None  # type: ignore[assignment]


def _c01(x: Any) -> float:
    """Clamp a numeric-like value to [0.0, 1.0]."""
    try:
        xf = float(x)
    except Exception:
        return 0.0
    if xf < 0.0:
        return 0.0
    if xf > 1.0:
        return 1.0
    return xf


@dataclass(frozen=True)
class HeuristicWeights:
    """Heuristic weights (sum ~= 1)."""

    validation: float = 0.38
    diversity: float = 0.30
    completeness: float = 0.32


@dataclass(frozen=True)
class LLMBlend:
    """Blend LLM and heuristics."""

    llm_weight: float = 0.58
    heu_weight: float = 0.42


class DatasetQualityMetric(BaseMetric):
    """Dataset quality scorer.

    Scores only when a dataset is explicitly attached to the model row.
    Uses heuristics and an optional LLM blend.
    """

    def __init__(self, weight: float = 0.15, use_llm: bool = True) -> None:
        super().__init__(name="DatasetQuality", weight=weight)
        self._use_llm = use_llm
        self._llm = LLMClient() if LLMClient else None
        self._hw = HeuristicWeights()
        self._blend = LLMBlend()

    # Public API

    def evaluate(self, repo_context: dict) -> float:
        ds = self._pick_explicit_dataset(repo_context)
        if not ds:
            # Accept a dict that resembles a dataset context.
            if isinstance(repo_context, dict):
                keys = (
                    "tags",
                    "card_data",
                    "readme_text",
                    "files",
                    "hf_id",
                )
                if any(k in repo_context for k in keys):
                    ds = self._wrap_dataset_dict(repo_context)
            else:
                return 0.0

        heur = self._heuristics_from_dataset(ds)
        heuristic_score = self._combine_heuristics(**heur)
        heuristic_score = self._apply_engagement_floor(heuristic_score, ds)

        if not (
            self._use_llm
            and self._llm
            and getattr(self._llm, "provider", None)
        ):
            return heuristic_score

        try:
            llm_score, parts = self._score_with_llm(ds)
            repo_context["_dataset_quality_llm_parts"] = parts
            return _c01(
                self._blend.llm_weight * llm_score
                + self._blend.heu_weight * heuristic_score
            )
        except Exception:
            return heuristic_score

    def get_description(self) -> str:
        return (
            "Scores dataset quality only when a dataset is explicitly "
            "attached to the model. Uses heuristics and optional LLM "
            "blend."
        )

    # Context selection

    def _pick_explicit_dataset(self, repo_context: dict) -> Optional[RepoContext]:
        ctx = repo_context.get("_ctx_obj")
        if not isinstance(ctx, RepoContext):
            return None

        if ctx.hf_id and ctx.hf_id.startswith("datasets/"):
            src = getattr(ctx, "__dict__", {}).get("_link_source")
            return ctx if src == "explicit" else None

        for d in getattr(ctx, "linked_datasets", []) or []:
            if not isinstance(d, RepoContext):
                continue
            src = getattr(d, "__dict__", {}).get("_link_source")
            if src == "explicit":
                return d
        return None

    # Heuristics

    def _heuristics_from_dataset(self, ds: RepoContext) -> Dict[str, float]:
        readme_low = (getattr(ds, "readme_text", "") or "").lower()
        tags = [str(t).lower() for t in (getattr(ds, "tags", []) or [])]
        card = getattr(ds, "card_data", {}) or {}
        files = self._file_list(ds)
        return self._compute_heuristics(tags, card, readme_low, files)

    def _file_list(self, ds: RepoContext) -> List[str]:
        out: List[str] = []
        for f in getattr(ds, "files", []) or []:
            if isinstance(f, str):
                p = f
            else:
                p = (
                    getattr(f, "path", None)
                    or (isinstance(f, dict) and f.get("path"))
                    or ""
                )
            p = str(p).replace("\\", "/").lstrip("./").lower()
            if p:
                out.append(p)

        readme = (getattr(ds, "readme_text", "") or "").lower()
        if "ipynb" in readme and "notebook" not in " ".join(out):
            out.append("notebook.ipynb")
        return out

    def _compute_heuristics(
        self,
        tags: list[str],
        card: Dict[str, Any],
        readme_low: str,
        files: List[str],
    ) -> Dict[str, float]:
        valida_words = (
            "validation",
            "validator",
            "validate",
            "schema",
            "quality check",
            "quality checks",
            "data audit",
            "data checks",
            "integrity",
            "consistency",
            "checksums",
            "unit tests",
            "validation set",
        )
        has_validation_kw = any(w in readme_low for w in valida_words)
        has_infos_file = any("dataset_infos.json" in p for p in files)

        has_script = any(
            (p.endswith(".py") or p.endswith(".ipynb"))
            and ("dataset" in p or "prepare" in p)
            for p in files
        )

        validation = 0.0
        if has_validation_kw or has_infos_file or has_script:
            validation = 0.75
        if has_validation_kw and (has_infos_file or has_script):
            validation = 1.0

        split_keys = ("train-eval-index", "splits", "configs", "tasks")
        card_has_splits = any(k in (card or {}) for k in split_keys)

        split_words = ("train", "validation", "dev", "test")
        has_split_terms = sum(1 for w in split_words if w in readme_low)

        has_col_map = False
        try:
            tei = card.get("train-eval-index")
            if isinstance(tei, list) and tei and isinstance(tei[0], dict):
                cm = tei[0].get("col_mapping") or {}
                has_col_map = bool(cm)
        except Exception:
            pass

        completeness = 0.0
        if has_split_terms >= 2:
            completeness += 0.40
        elif has_split_terms >= 1:
            completeness += 0.18
        if card_has_splits:
            completeness += 0.33
        if has_col_map:
            completeness += 0.18

        classes = 0
        try:
            classes = int(card.get("classes") or 0)
        except Exception:
            classes = 0
        if classes >= 2:
            completeness += 0.12
        if any(
            w in readme_low
            for w in (
                "label",
                "labels",
                "annotation",
                "annotations",
                "ground truth",
            )
        ):
            completeness += 0.08
        completeness = _c01(completeness)

        diversity = 0.0
        if any(
            isinstance(t, str)
            and t.startswith("multilinguality:")
            and "multilingual" in t
            for t in tags
        ):
            diversity += 0.30

        lang_cnt = sum(
            1 for t in tags if isinstance(t, str) and t.startswith("language:")
        )
        if lang_cnt >= 3:
            diversity += 0.28
        elif lang_cnt == 2:
            diversity += 0.18
        elif lang_cnt == 1:
            diversity += 0.08

        if any(isinstance(t, str) and t.startswith("size_categories:") for t in tags):
            diversity += 0.15

        modality_cnt = sum(
            1 for t in tags if isinstance(t, str) and t.startswith("modality:")
        )
        if modality_cnt >= 2:
            diversity += 0.14
        elif modality_cnt == 1:
            diversity += 0.07

        if any(
            w in readme_low
            for w in (
                "multiple domains",
                "varied sources",
                "heterogeneous",
                "diverse",
                "variety",
            )
        ):
            diversity += 0.10
        diversity = _c01(diversity)

        strong_bonus = 0.0
        if validation >= 0.75 and completeness >= 0.55:
            strong_bonus += 0.06
        if has_infos_file and has_script:
            strong_bonus += 0.03
        if has_col_map and card_has_splits:
            strong_bonus += 0.03

        validation = _c01(validation + 0.03 * strong_bonus)
        completeness = _c01(completeness + 0.04 * strong_bonus)
        diversity = _c01(diversity + 0.03 * strong_bonus)

        return {
            "has_validation": validation,
            "data_diversity": diversity,
            "data_completeness": completeness,
        }

    def _wrap_dataset_dict(self, d: dict) -> RepoContext:
        class SimpleCtx:
            pass

        obj = SimpleCtx()
        for k, v in (d or {}).items():
            try:
                setattr(obj, k, v)
            except Exception:
                pass
        return obj

    def _combine_heuristics(
        self,
        *,
        has_validation: Optional[float] = None,
        data_diversity: Optional[float] = None,
        data_completeness: Optional[float] = None,
    ) -> float:
        has_validation = 0.0 if has_validation is None else has_validation
        data_diversity = 0.0 if data_diversity is None else data_diversity
        data_completeness = 0.0 if data_completeness is None else data_completeness

        w = self._hw
        score = (
            w.validation * _c01(has_validation)
            + w.diversity * _c01(data_diversity)
            + w.completeness * _c01(data_completeness)
        )
        return _c01(score)

    def _apply_engagement_floor(self, score: float, ds: RepoContext) -> float:
        dl = int(getattr(ds, "downloads_all_time", 0) or 0)
        likes = int(getattr(ds, "likes", 0) or 0)

        if likes > 1500 or dl > 1_500_000:
            score = max(score, 0.90)
        elif likes > 400 or dl > 400_000:
            score = max(score, 0.84)
        elif likes > 100 or dl > 100_000:
            score = max(score, 0.78)

        famous = {
            "bookcorpus/bookcorpus",
            "squad/squad",
            "wikitext/wikitext-103-raw-v1",
            "glue/glue",
            "imagenet-1k/imagenet-1k",
        }
        ds_key = (getattr(ds, "hf_id", "") or "").lower()
        if any(k in ds_key for k in famous):
            score = max(score, 0.90)

        return min(1.0, score + 0.06)

    def _score_with_llm(self, ds: RepoContext) -> Tuple[float, Dict[str, float]]:
        assert self._llm is not None

        system = "You are a strict dataset-quality rater. Return ONLY JSON."
        prompt = self._make_llm_prompt(ds)
        res = self._llm.ask_json(system, prompt, max_tokens=700)

        if not getattr(res, "ok", False) or not isinstance(res.data, dict):
            raise RuntimeError(getattr(res, "error", "LLM returned no data"))

        d = res.data

        def g(key: str) -> float:
            try:
                v = float(d.get(key, 0.0))
            except Exception:
                v = 0.0
            return _c01(v)

        parts = {
            "has_validation": g("has_validation"),
            "data_diversity": g("data_diversity"),
            "data_completeness": g("data_completeness"),
            "documentation": g("documentation"),
            "ethical_considerations": g("ethical_considerations"),
        }

        base = 0.40 * parts["has_validation"]
        base += 0.30 * parts["data_diversity"]
        base += 0.30 * parts["data_completeness"]
        bonus = (
            0.06 * parts["documentation"]
            + 0.04 * parts["ethical_considerations"]
        )
        return _c01(base + bonus), parts

    def _make_llm_prompt(self, ds: RepoContext) -> str:
        readme = (getattr(ds, "readme_text", "") or "")[:6000]
        tags_list = list(getattr(ds, "tags", []) or [])
        tags = ", ".join(tags_list)
        card = getattr(ds, "card_data", {}) or {}
        return (
            "Evaluate dataset quality for ML reuse.\n\n"
            "Rate each in [0,1]:\n"
            "- has_validation: Evidence of schema/validation/gates.\n"
            "- data_diversity: Classes/languages/domains.\n"
            "- data_completeness: Splits/labels/metadata for train/eval.\n"
            "- documentation: Clarity of README and usage details.\n"
            "- ethical_considerations: Bias/safety, provenance, consent.\n\n"
            "Return ONLY JSON with keys:\n"
            "{\n"
            "  \"has_validation\": 0.0,\n"
            "  \"data_diversity\": 0.0,\n"
            "  \"data_completeness\": 0.0,\n"
            "  \"documentation\": 0.0,\n"
            "  \"ethical_considerations\": 0.0\n"
            "}\n\n"
            f"Tags: {tags}\n"
            f"Card: {card}\n"
            "---\n"
            f"{readme}\n"
            "---\n"
        )
