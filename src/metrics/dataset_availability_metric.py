from __future__ import annotations

from typing import Iterable, Optional

from repo_context import RepoContext
from .base_metric import BaseMetric


class DatasetAvailabilityMetric(BaseMetric):
    def __init__(self, weight: float = 0.10):
        super().__init__(name="DatasetAvailability", weight=weight)

    def _as_ctx(self, repo_dict: dict) -> Optional[RepoContext]:
        ctx = repo_dict.get("_ctx_obj")
        return ctx if isinstance(ctx, RepoContext) else None

    def _is_public_dataset(self, ds: RepoContext) -> bool:
        if ds.gated is True or ds.private is True:
            return False
        if ds.gated is None and ds.private is None:
            return bool(ds.readme_text or ds.tags)
        return True

    def _any_public_dataset(self, datasets: Iterable[RepoContext]) -> bool:
        return any(self._is_public_dataset(ds) for ds in (datasets or []))

    def _is_public_code(self, code: RepoContext) -> bool:
        if code.private is True:
            return False
        has_signals = bool(
            getattr(code, "readme_text", None)
            or getattr(code, "contributors", [])
            or getattr(code, "files", [])
        )
        return bool(getattr(code, "gh_url", None)) and (
            code.private is False or has_signals
        )

    def _any_public_code(
        self, codes: Iterable[RepoContext], gh_url: Optional[str]
    ) -> bool:
        if any(self._is_public_code(c) for c in (codes or [])):
            return True
        return bool(gh_url)

    def _doc_signals(self, readme: str) -> tuple[bool, bool, bool]:
        text = (readme or "").lower()
        mentions_dataset = any(
            k in text for k in ["dataset", "datasets/", "corpus", "training data"])
        mentions_training = any(
            k in text
            for k in [
                "training", "fine-tuning", "finetuning",
                "train step", "training setup", "training procedure"]
        )
        mentions_code = any(
            k in text
            for k in ["code", "repo", "github.com", "install", "pip install", "usage", "example"]
        )
        return mentions_dataset, mentions_training, mentions_code

    def evaluate(self, repo_context: dict) -> float:
        ctx = self._as_ctx(repo_context)
        readme_text = (
            repo_context.get("readme_text") or "") if ctx is None else (ctx.readme_text or "")

        if ctx and ctx.hf_id and ctx.hf_id.startswith("datasets/"):
            return 1.0 if self._is_public_dataset(ctx) else 0.0

        linked_datasets = (
            ctx.linked_datasets if ctx else repo_context.get("linked_datasets") or []) or []
        linked_code = (ctx.linked_code if ctx else repo_context.get("linked_code") or []) or []
        gh_url = ctx.gh_url if ctx else repo_context.get("gh_url")

        has_public_dataset = self._any_public_dataset(linked_datasets)
        has_public_code = self._any_public_code(linked_code, gh_url)

        if not has_public_dataset and not has_public_code:
            return 0.0
        if has_public_dataset ^ has_public_code:
            return 0.33

        md, mt, mc = self._doc_signals(readme_text)
        docs_good = md and mt and mc
        return 1.0 if docs_good else 0.67

    def get_description(self) -> str:
        return (
            "Evaluates availability of required training assets. "
            "For models: checks public access to linked dataset(s)"
            "and code repo, plus README signals. "
            "For datasets: checks public accessibility."
        )
