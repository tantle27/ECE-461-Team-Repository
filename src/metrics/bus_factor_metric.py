"""
Bus Factor Metric for evaluating project sustainability.
"""

from .base_metric import BaseMetric


class BusFactorMetric(BaseMetric):
    def __init__(self, weight: float = 0.15):
        super().__init__(name="BusFactor", weight=weight)

    def evaluate(self, repo_context: dict) -> float:
        contributors = repo_context.get("contributors", [])
        if repo_context.get("category", "").upper() == "MODEL":
            linked = (
                getattr(repo_context.get("_ctx_obj", None), "linked_code", [])
                or []
            )

            def richness(c):
                files = len(getattr(c, "files", []) or [])
                ppl = len(getattr(c, "contributors", []) or [])
                return (ppl, files)

            if linked:
                code_ctx = max(linked, key=richness)
                if getattr(code_ctx, "contributors", None):
                    contributors = code_ctx.contributors
        if not contributors:
            return 0.0

        # gather contribution counts
        counts = []
        for c in contributors:
            if isinstance(c, dict):
                counts.append(int(c.get("contributions", 0)))
            else:
                counts.append(int(getattr(c, "contributions", 0)))

        total = sum(counts)
        if total <= 0:
            return 0.0

        max_share = max(counts) / total
        return max(0.0, 1.0 - max_share)

    def get_description(self) -> str:
        return (
            "Evaluates project sustainability based on contributor diversity"
        )
