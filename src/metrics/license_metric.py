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


def _lower(s: str) -> str:
    return s.strip().lower()


def _to_list(x: Any) -> List[str]:
    """
    Normalize license-like values to a list of lowercased strings.
    Accepts str | list | dict | None and returns [] when unknown.
    """
    if x is None:
        return []
    if isinstance(x, str):
        return [_lower(x)]
    out: List[str] = []
    if isinstance(x, (list, tuple, set)):
        for v in x:
            if isinstance(v, str):
                out.append(_lower(v))
            elif isinstance(v, dict):
                sid = (
                    v.get("id")
                    or v.get("spdx_id")
                    or v.get("slug")
                    or v.get("name")
                )
                if isinstance(sid, str):
                    out.append(_lower(sid))
        return out
    if isinstance(x, dict):
        sid = x.get("id") or x.get("spdx_id") or x.get("slug") or x.get("name")
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
        if not isinstance(t, str):
            continue
        tl = t.lower()
        # HF commonly uses "license:apache-2.0"
        if tl.startswith("license:"):
            out.append(tl.split(":", 1)[1].strip())
    return out


def _collect_candidates(repo_dict: dict) -> List[str]:
    """
    Collect license candidates from multiple places in repo_context(dict form).
    """
    cands: List[str] = []
    card = repo_dict.get("card_data") or {}
    tags = repo_dict.get("tags") or []
    model_index = repo_dict.get("model_index") or {}
    config_json = repo_dict.get("config_json") or {}
    files = repo_dict.get("files") or []

    if isinstance(card, dict):
        cands += _to_list(card.get("license"))
        # if you stash GH license in card_data as {"spdx_id": "..."} or similar
        if "github_license" in card:
            cands += _to_list(card["github_license"])

    cands += _from_tags(tags)

    if isinstance(model_index, dict):
        cands += _to_list(model_index.get("license"))
    if isinstance(config_json, dict):
        cands += _to_list(config_json.get("license"))

    # LICENSE/COPYING presence (only a presence signal; not compatibility)
    for fi in files:
        name = str(getattr(fi, "path", "")).split("/")[-1]
        if LICENSE_FILE_RE.match(name):
            cands.append("present-file")

    # dedupe preserve order
    seen = set()
    uniq = []
    for s in cands:
        if s not in seen:
            seen.add(s)
            uniq.append(s)
    return uniq


COMPATIBLE_MAP = {
    "public-domain": 1.0,
    "public domain": 1.0,
    "unlicense": 1.0,
    "mit": 1.0,
    "bsd-3-clause": 0.8,
    "bsd-2-clause": 0.8,
    "apache-2.0": 0.6,
    "apache2.0": 0.6,
    "apache": 0.6,  # fallback partial
    "mpl-2.0": 0.4,
    "mpl2.0": 0.4,
    "mozilla": 0.4,  # fallback partial (assume MPL2 family)
    "lgpl-2.1-only": 0.2,  # explicitly only
    "lgpl-2.1": 0.2,  # treat plain 2.1 as "only" unless string says "or-later"
}

# Incompatible families (any substring match means incompatible)
INCOMPATIBLE_KEYS = {
    "gpl-3.0",
    "gpl-3",
    "gplv3",
    "gpl-2.0",
    "gpl-2",
    "gplv2",
    "agpl-3.0",
    "agpl-3",
    "agplv3",
    "agpl",
    "lgpl-3.0",
    "lgpl-3",
    "lgplv3",
    "lgpl-2.1-or-later",
    "lgpl-2.1+",  # 2.1+ explicitly incompatible
    "copyleft",
    "proprietary",
    "allrightsreserved",
    "all-rights-reserved",
    "cc-by",
    "cc-by-sa",
    "cc-by-nc",
    "cc-by-nd",
    "cc0-nd",  # code-incompatible CCs
}


def _classify(slugs: Iterable[str]) -> Tuple[bool, float, str]:
    """
    Returns:
      (is_compatible, permissiveness_score_if_compatible_else_0, detected_slug)
    Rules:
      - If any candidate is in incompatible families ⇒ incompatible.
      - Else if candidate matches map ⇒ compatible (choose highest).
      - Else unknown/only 'present-file' ⇒ incompatible (binary policy).
    """
    best_perm = 0.0
    detected = ""
    saw_present = False

    for raw in slugs:
        s = raw.replace("_", "-").strip().lower()

        if s == "present-file":
            saw_present = True
            continue

        # Incompatibility (substring check)
        if any(k in s for k in INCOMPATIBLE_KEYS):
            return (False, 0.0, s)

        # Special case: "or-later" on LGPL-2.1 is incompatible
        if "lgpl" in s and ("or-later" in s or s.endswith("+")):
            return (False, 0.0, s)

        # Compatible mapping (direct or partials)
        if s in COMPATIBLE_MAP:
            if COMPATIBLE_MAP[s] > best_perm:
                best_perm = COMPATIBLE_MAP[s]
                detected = s
            continue

        # partials
        if s.startswith("apache-2"):
            if COMPATIBLE_MAP["apache-2.0"] > best_perm:
                best_perm = COMPATIBLE_MAP["apache-2.0"]
                detected = "apache-2.0"
        elif s.startswith("mpl-2"):
            if COMPATIBLE_MAP["mpl-2.0"] > best_perm:
                best_perm = COMPATIBLE_MAP["mpl-2.0"]
                detected = "mpl-2.0"
        elif s.startswith("bsd-3"):
            if COMPATIBLE_MAP["bsd-3-clause"] > best_perm:
                best_perm = COMPATIBLE_MAP["bsd-3-clause"]
                detected = "bsd-3-clause"
        elif s.startswith("bsd-2"):
            if COMPATIBLE_MAP["bsd-2-clause"] > best_perm:
                best_perm = COMPATIBLE_MAP["bsd-2-clause"]
                detected = "bsd-2-clause"
        elif s == "cc0" or s.startswith("cc0-"):
            # treat CC0 as public domain for code-compat scoring as requested
            if 1.0 > best_perm:
                best_perm = 1.0
                detected = "public-domain"

    if best_perm > 0.0:
        return (True, best_perm, detected)

    # Unknown license (or only presence of LICENSE file) ⇒ incompatible policy
    return (
        False,
        0.0,
        "unknown" if not saw_present else "unknown-present-file",
    )


class LicenseMetric(BaseMetric):
    """
    Binary license metric:
      - Returns 1.0 if license is compatible with LGPL-2.1 (exact/only).
      - Returns 0.0 if incompatible or unknown (LGPL-2.1+, GPL/AGPL, etc.).
    Also computes a permissiveness score
    and exposes it via repo_context['_license_perm_score'] for debugging.
    """

    def __init__(self, weight: float = 0.1, expose_parts: bool = True):
        super().__init__(name="License", weight=weight)
        self._expose_parts = expose_parts

    def evaluate(self, repo_context: dict) -> float:
        # Prefer the object if available, but fall back to dict robustly
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

        is_compat, perm_score, detected = _classify(slugs)

        if self._expose_parts:
            repo_context["_license_detected"] = detected
            repo_context["_license_perm_score"] = perm_score
            repo_context["_license_candidates"] = slugs

        # Binary output per policy
        return 1.0 if is_compat else 0.0

    def get_description(self) -> str:
        return (
            "Binary license compatibility with ACME's LGPLv2.1 ("
            "compatible=1.0, else 0.0). "
            "Also computes a permissiveness score ("
            "Public domain/MIT=1.0 > BSD=0.8 > Apache=0.6 "
            "> MPL=0.4 > LGPL-2.1-only=0.2) exposed in aux for diagnostics."
        )
