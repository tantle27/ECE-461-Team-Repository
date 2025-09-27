"""
Bus Factor Metric for evaluating project sustainability.
More generous: blends dominance (fairness) with team size, and applies a floor for non-empty teams.
"""

from __future__ import annotations
from .base_metric import BaseMetric


class BusFactorMetric(BaseMetric):
    def __init__(self, weight: float = 0.15):
        super().__init__(name="BusFactor", weight=weight)

    def _extract_contributors(self, repo_context: dict) -> list[dict]:
        """
        Prefer code repo contributors when scoring a MODEL (linked_code),
        otherwise fall back to contributors on the current context/dict.
        """
        contributors = repo_context.get("contributors", []) or []

        if (repo_context.get("category", "").upper() == "MODEL"):
            ctx = repo_context.get("_ctx_obj")
            linked = getattr(ctx, "linked_code", []) if ctx else []
            if linked:
                # pick the richest code context by (#contributors, #files)
                def richness(c):
                    try:
                        files = len(getattr(c, "files", []) or [])
                    except Exception:
                        files = 0
                    try:
                        ppl = len(getattr(c, "contributors", []) or [])
                    except Exception:
                        ppl = 0
                    return (ppl, files)

                code_ctx = max(linked, key=richness)
                if getattr(code_ctx, "contributors", None):
                    return code_ctx.contributors

        return contributors

    def evaluate(self, repo_context: dict) -> float:
        contributors = self._extract_contributors(repo_context)

        if not contributors:
            return 0.0

        counts: list[int] = []
        for c in contributors:
            try:
                v = c.get(
                    "contributions", 0) if isinstance(c, dict) else getattr(c, "contributions", 0)
                v = int(v) if v is not None else 0
            except Exception:
                v = 0
            if v > 0:
                counts.append(v)

        n = len(counts)
        if n == 0:
            return 0.0

        total = sum(counts)
        if total <= 0:
            return 0.0

        max_share = max(counts) / float(total)
        fairness = 1.0 - max_share  # 0..1

        size_norm = min(1.0, max(0.0, (n - 1) / 9.0))

        blended = 0.6 * fairness + 0.4 * size_norm

        score = max(0.33, blended)

        return float(score)

    def get_description(self) -> str:
        return "Evaluates project sustainability from contributor dominance and team size"
