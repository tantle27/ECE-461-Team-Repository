"""
Bus Factor metric: prefer Dulwich git history, else contributor heuristic.
"""

from __future__ import annotations

import hashlib
import os
import time
from typing import Any, Dict
from .base_metric import BaseMetric
from repo_context import RepoContext

try:
    from dulwich import porcelain as dl_p
    from dulwich.repo import Repo as DlRepo

    _HAS_DULWICH = True
except Exception:  # pragma: no cover
    _HAS_DULWICH = False

# Tunables
_CACHE_ROOT = os.path.expanduser("~/.cache/acme-cli/git")
_SINCE_DAYS = 365 * 5
_MAX_COMMITS = 5000
_DEPTH = 2000
_AUTHORS_BONUS = 5


class BusFactorMetric(BaseMetric):
    def __init__(self, weight: float = 0.15) -> None:
        super().__init__(name="BusFactor", weight=weight)

    def evaluate(self, repo_context: dict) -> float:
        ctx = repo_context.get("_ctx_obj")
        gh_url = getattr(ctx, "gh_url", None)

        # Prefer linked code repo for models
        if repo_context.get("category", "").upper() == "MODEL" and getattr(
            ctx, "linked_code", None
        ):

            def rich(c):
                return (
                    len(getattr(c, "contributors", []) or []),
                    len(getattr(c, "files", []) or []),
                )

            best = max(
                (c for c in (getattr(ctx, "linked_code", []) or [])),
                key=rich,
                default=None,
            )
            if best and getattr(best, "gh_url", None):
                gh_url = best.gh_url

        # Git history path
        if gh_url and _HAS_DULWICH:
            try:
                st = _git_stats(gh_url)
                tot = max(0, st["total_commits"])
                top = st["top_share"] if tot > 0 else 1.0
                authors = st["unique_authors"]
                act = min(1.0, tot / 200.0)
                pen = max(0.0, (top - 0.5) / 0.5)
                score = max(0.0, act * (1.0 - 0.6 * pen))
                if authors >= _AUTHORS_BONUS:
                    score += 0.05
                return _c01(score)
            except Exception:
                pass  # fall back

        # Heuristic fallback
        contribs = repo_context.get("contributors", [])
        if repo_context.get("category", "").upper() == "MODEL" and getattr(
            ctx, "linked_code", None
        ):

            def rich(c):
                return (
                    len(getattr(c, "contributors", []) or []),
                    len(getattr(c, "files", []) or []),
                )

            code_ctx = max(
                (getattr(ctx, "linked_code", []) or []), key=rich, default=None
            )
            if code_ctx and getattr(code_ctx, "contributors", None):
                contribs = code_ctx.contributors

        if not contribs:
            return 0.0

        counts = [
            (
                int(c.get("contributions", 0))
                if isinstance(c, dict)
                else int(getattr(c, "contributions", 0))
            )
            for c in contribs
        ]
        tot = sum(counts)
        return 0.0 if tot <= 0 else max(0.0, 1.0 - max(counts) / tot)

    def get_description(self) -> str:
        return (
            "Sustainability via Dulwich git history (if available) or "
            "contributor diversity."
        )


# ---------------- helpers ----------------


def _c01(x: Any) -> float:
    try:
        xf = float(x)
    except Exception:
        return 0.0
    return 0.0 if xf < 0 else 1.0 if xf > 1 else xf


def _since_cutoff() -> int:
    return int(time.time()) - _SINCE_DAYS * 24 * 3600


def _cache_dir(url: str) -> str:
    h = hashlib.sha1(url.encode("utf-8")).hexdigest()[:16]
    return os.path.join(_CACHE_ROOT, h)


def _git_stats(gh_url: str) -> Dict[str, Any]:
    tgt = _cache_dir(gh_url)
    os.makedirs(_CACHE_ROOT, exist_ok=True)
    return _dulwich_stats(gh_url, tgt)


def _dulwich_stats(gh_url: str, tgt: str) -> Dict[str, Any]:
    since = _since_cutoff()
    git_dir = os.path.join(tgt, ".git")
    if not os.path.isdir(git_dir):
        os.makedirs(tgt, exist_ok=True)
        dl_p.clone(gh_url, tgt, depth=_DEPTH, checkout=False)
    else:
        dl_p.fetch(tgt, gh_url)

    repo = DlRepo(tgt)
    counts: Dict[str, int] = {}
    total = 0
    for w in repo.get_walker(max_entries=_MAX_COMMITS):
        c = w.commit
        if c.commit_time < since:
            break
        author = (c.author or b"").decode(errors="ignore").strip().lower()
        author = author or "unknown"
        counts[author] = counts.get(author, 0) + 1
        total += 1

    top = max(counts.values(), default=0)
    uniq = len(counts)
    return {
        "total_commits": total,
        "unique_authors": uniq,
        "top_share": (top / total) if total else 1.0,
    }
