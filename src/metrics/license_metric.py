# metrics/license_metric.py
from __future__ import annotations

import re
from typing import Any, Iterable, List, Tuple

from repo_context import RepoContext
from .base_metric import BaseMetric

LICENSE_FILE_RE = re.compile(
    r"""(?ix)
        ^ (?:license|licence|copying|notice) (?: \.[a-z0-9._-]+ )? $
    """
)

COMPATIBLE_MAP = {
    "public-domain": 1.0,
    "public domain": 1.0,
    "unlicense": 1.0,
    "mit": 1.0,
    "bsd-3-clause": 0.8,
    "bsd-2-clause": 0.8,
    "apache-2.0": 0.6,
    "apache2.0": 0.6,
    "apache": 0.6,
    "mpl-2.0": 0.4,
    "mpl2.0": 0.4,
    "mozilla": 0.4,
    "lgpl-2.1-only": 0.2,
    "lgpl-2.1": 0.2,
}

INCOMP_KEYS = {
    "gpl-3.0", "gpl-3", "gplv3", "gpl-2.0", "gpl-2", "gplv2",
    "agpl-3.0", "agpl-3", "agplv3", "agpl",
    "lgpl-3.0", "lgpl-3", "lgplv3", "lgpl-2.1-or-later", "lgpl-2.1+",
    "copyleft", "proprietary", "allrightsreserved", "all-rights-reserved",
    "cc-by", "cc-by-sa", "cc-by-nc", "cc-by-nd", "cc0-nd",
}


def _lower(s: str) -> str:
    return s.strip().lower()


def _to_list(x: Any) -> List[str]:
    """Normalize license-like values to a list of lowercase strings."""
    if x is None:
        return []
    if isinstance(x, str):
        return [_lower(x)]
    if isinstance(x, (list, tuple, set)):
        out: List[str] = []
        for v in x:
            if isinstance(v, str):
                out.append(_lower(v))
            elif isinstance(v, dict):
                sid = v.get("id") or v.get("spdx_id") or v.get("slug") \
                      or v.get("name")
                if isinstance(sid, str):
                    out.append(_lower(sid))
        return out
    if isinstance(x, dict):
        sid = x.get("id") or x.get("spdx_id") or x.get("slug") \
              or x.get("name")
        return [_lower(sid)] if isinstance(sid, str) else []
    try:
        s = str(x)
        return [_lower(s)] if s else []
    except Exception:
        return []


def _from_tags(tags: Iterable[str] | None) -> List[str]:
    if not tags:
        return []
    out = []
    for t in tags:
        if isinstance(t, str) and t.lower().startswith("license:"):
            out.append(t.split(":", 1)[1].strip().lower())
    return out


def _collect_candidates(repo_dict: dict) -> List[str]:
    """Collect license candidates from repo_context dict fields."""
    card = repo_dict.get("card_data") or {}
    tags = repo_dict.get("tags") or []
    mindex = repo_dict.get("model_index") or {}
    cfg = repo_dict.get("config_json") or {}
    files = repo_dict.get("files") or []

    cands: List[str] = []
    if isinstance(card, dict):
        cands += _to_list(card.get("license"))
        if "github_license" in card:
            cands += _to_list(card["github_license"])
    cands += _from_tags(tags)
    if isinstance(mindex, dict):
        cands += _to_list(mindex.get("license"))
    if isinstance(cfg, dict):
        cands += _to_list(cfg.get("license"))

    for fi in files:
        name = str(getattr(fi, "path", "")).split("/")[-1]
        if LICENSE_FILE_RE.match(name):
            cands.append("present-file")

    seen, uniq = set(), []
    for s in cands:
        if s not in seen:
            seen.add(s)
            uniq.append(s)
    return uniq


def _classify(slugs: Iterable[str]) -> Tuple[bool, float, str]:
    """
    Returns (is_compatible, permissiveness_if_compat_else_0, detected_slug).
    """
    best, detected, saw_file = 0.0, "", False
    for raw in slugs:
        s = raw.replace("_", "-").strip().lower()
        if s == "present-file":
            saw_file = True
            continue

        if any(k in s for k in INCOMP_KEYS):
            return False, 0.0, s
        if "lgpl" in s and ("or-later" in s or s.endswith("+")):
            return False, 0.0, s

        if s in COMPATIBLE_MAP and COMPATIBLE_MAP[s] > best:
            best, detected = COMPATIBLE_MAP[s], s
            continue

        # partials
        def _bump(key: str, label: str) -> None:
            nonlocal best, detected
            val = COMPATIBLE_MAP[label]
            if val > best:
                best, detected = val, label

        if s.startswith("apache-2"):
            _bump(s, "apache-2.0")
        elif s.startswith("mpl-2"):
            _bump(s, "mpl-2.0")
        elif s.startswith("bsd-3"):
            _bump(s, "bsd-3-clause")
        elif s.startswith("bsd-2"):
            _bump(s, "bsd-2-clause")
        elif s == "cc0" or s.startswith("cc0-"):
            if 1.0 > best:
                best, detected = 1.0, "public-domain"

    if best > 0.0:
        return True, best, detected
    return False, 0.0, "unknown-present-file" if saw_file else "unknown"


class LicenseMetric(BaseMetric):
    """
    Binary license test vs ACME policy (LGPL-2.1-only compatible).
    Stores details in repo_context for diagnostics.
    """

    def __init__(self, weight: float = 0.1, expose_parts: bool = True):
        super().__init__(name="License", weight=weight)
        self._expose_parts = expose_parts

    def evaluate(self, repo_context: dict) -> float:
        ctx = repo_context.get("_ctx_obj")
        if isinstance(ctx, RepoContext):
            base = {
                "card_data": ctx.card_data,
                "tags": ctx.tags,
                "model_index": ctx.model_index,
                "config_json": ctx.config_json,
                "files": ctx.files,
            }
            slugs = _collect_candidates(base)
        else:
            slugs = _collect_candidates(repo_context)

        ok, perm, detected = _classify(slugs)

        if self._expose_parts:
            repo_context["_license_detected"] = detected
            repo_context["_license_perm_score"] = perm
            repo_context["_license_candidates"] = slugs

        return 1.0 if ok else 0.0

    def get_description(self) -> str:
        return (
            "Binary license compatibility with ACME's LGPL-2.1 (1.0 or 0.0). "
            "Also computes permissiveness score (MIT/PD=1.0 > BSD=0.8 > "
            "Apache=0.6 > MPL=0.4 > LGPL-2.1-only=0.2) in aux."
        )
