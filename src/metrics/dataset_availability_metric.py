from __future__ import annotations
from typing import Iterable, Optional
from repo_context import RepoContext
from .base_metric import BaseMetric


class DatasetAvailabilityMetric(BaseMetric):
    def __init__(self, weight: float = 0.10):
        super().__init__(name="DatasetAvailability", weight=weight)

    # ----- helpers -----

    def _as_ctx(self, repo_dict: dict) -> Optional[RepoContext]:
        ctx = repo_dict.get("_ctx_obj")
        return ctx if isinstance(ctx, RepoContext) else None

    @staticmethod
    def _has_with_source(objs: Iterable[RepoContext], source: str) -> bool:
        for o in objs or []:
            if getattr(o, "__dict__", {}).get("_link_source") == source:
                return True
        return False

    # ----- scoring policy (strict) -----
    # - 1.00 if BOTH dataset and code are provided explicitly in urls.txt
    # - 0.50 if EXACTLY ONE of (dataset, code) is provided explicitly
    # - 0.00 otherwise
    #
    # No bonus for README "signals" or card hints; provenance rules everything.

    def evaluate(self, repo_context: dict) -> float:
        ctx = self._as_ctx(repo_context)

        linked_datasets = (
            ctx.linked_datasets if ctx else repo_context.get("linked_datasets") or []) or []
        linked_code = (
            ctx.linked_code if ctx else repo_context.get("linked_code") or []) or []

        # Provenance booleans
        explicit_ds = self._has_with_source(linked_datasets, "explicit")
        explicit_code = self._has_with_source(linked_code, "explicit")

        if explicit_ds and explicit_code:
            return 1.0
        if (explicit_ds ^ explicit_code):  # exactly one explicit
            return 0.5
        return 0.0

    def get_description(self) -> str:
        return (
            "Strict availability: 1.0 if both dataset and code were provided "
            "explicitly in the input file; 0.5 if exactly one is explicit; "
            "0.2 if only linked in the README; 0.0 otherwise."
        )
