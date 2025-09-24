"""Thin URL handlers that build RepoContext via public API clients."""

from __future__ import annotations

import re
import time
from pathlib import Path
from typing import Iterable, Optional

from api.gh_client import GHClient
from api.hf_client import HFClient
from repo_context import FileInfo, RepoContext
from url_router import UrlRouter, UrlType


# ---------------- Utilities ----------------


def _retry_loop(max_retries: int = 3, base_delay: float = 1.0):
    delay = base_delay
    for attempt in range(max_retries):
        yield attempt
        if attempt < max_retries - 1:
            time.sleep(delay)
            delay *= 2


def _safe_ext(path_str: str) -> str:
    sfx = Path(path_str).suffix
    return sfx[1:].lower() if sfx.startswith(".") else sfx.lower()


# ---------------- Base ----------------


class UrlHandler:
    def __init__(self, url: Optional[str] = None):
        self.url = url

    def fetchMetaData(self) -> RepoContext:  # noqa: N802
        raise NotImplementedError


# ---------------- MODEL (HF) ----------------


class ModelUrlHandler(UrlHandler):
    def __init__(self, url: Optional[str] = None):
        super().__init__(url)
        self.hf_client = HFClient()
        self.gh_client = GHClient()

    def fetchMetaData(self) -> RepoContext:  # noqa: N802
        if not self.url:
            raise ValueError("URL is required")

        parsed = UrlRouter().parse(self.url)
        if parsed.type is not UrlType.MODEL or not parsed.hf_id:
            raise ValueError("Not an HF model URL")

        ctx = RepoContext(url=self.url, hf_id=parsed.hf_id, host="HF")

        for attempt in _retry_loop(3, 1.0):
            try:
                info = self.hf_client.get_model_info(parsed.hf_id)
                ctx.card_data = info.card_data
                ctx.tags = info.tags
                ctx.downloads_30d = info.downloads_30d
                ctx.downloads_all_time = info.downloads_all_time
                ctx.likes = info.likes
                ctx.created_at = info.created_at
                ctx.last_modified = info.last_modified
                ctx.gated = info.gated
                ctx.private = info.private

                files = self.hf_client.list_files(
                    parsed.hf_id, repo_type="model"
                )
                ctx.files = [
                    FileInfo(
                        path=Path(fi.path),
                        size_bytes=int(fi.size or 0),
                        ext=_safe_ext(fi.path),
                    )
                    for fi in (files or [])
                ]

                ctx.readme_text = self.hf_client.get_readme(parsed.hf_id) or ""
                ctx.model_index = self.hf_client.get_model_index_json(
                    parsed.hf_id
                )

                # Linked code (GitHub)
                try:
                    ctx.hydrate_code_links(self.hf_client, self.gh_client)
                    if ctx.gh_url:
                        code_ctx = build_code_context(ctx.gh_url)
                        ctx.link_code(code_ctx)
                        if code_ctx.contributors:
                            ctx.contributors = code_ctx.contributors
                except Exception as e:
                    ctx.api_errors += 1
                    ctx.fetch_logs.append(f"Linked code hydrate failed: {e}")

                # Linked datasets (HF)
                try:
                    ds_ids = set()
                    ds_ids |= set(
                        datasets_from_card(ctx.card_data or {}, ctx.tags or [])
                    )
                    ds_ids |= set(datasets_from_readme(ctx.readme_text or ""))
                    for did in ds_ids:
                        ds_url = f"https://huggingface.co/datasets/{did}"
                        try:
                            ds_ctx = build_dataset_context(ds_url)
                            ctx.link_dataset(ds_ctx)
                        except Exception as de:
                            ctx.api_errors += 1
                            ctx.fetch_logs.append(
                                f"Dataset hydrate failed for {did}: {de}"
                            )
                except Exception as e:
                    ctx.api_errors += 1
                    ctx.fetch_logs.append(f"Dataset discovery error: {e}")

                break  # success

            except FileNotFoundError as e:
                ctx.api_errors += 1
                ctx.fetch_logs.append(f"HF not found: {e}")
                raise
            except Exception as e:
                # 429/backoff is handled by HTTP retries; keep one more loop
                ctx.api_errors += 1
                ctx.fetch_logs.append(f"HF error: {e}")
                if attempt >= 2:
                    break

        return ctx


# ---------------- DATASET (HF) ----------------


class DatasetUrlHandler(UrlHandler):
    def __init__(self, url: Optional[str] = None):
        super().__init__(url)
        self.hf_client = HFClient()

    def fetchMetaData(self) -> RepoContext:
        if not self.url:
            raise ValueError("URL is required")

        parsed = UrlRouter().parse(self.url)
        if parsed.type is not UrlType.DATASET or not parsed.hf_id:
            raise ValueError("Not an HF dataset URL")

        ctx = RepoContext(url=self.url, hf_id=parsed.hf_id, host="HF")

        for attempt in _retry_loop(3, 1.0):
            try:
                info = self.hf_client.get_dataset_info(parsed.hf_id)
                ctx.card_data = info.card_data
                ctx.tags = info.tags
                ctx.downloads_30d = info.downloads_30d
                ctx.downloads_all_time = info.downloads_all_time
                ctx.likes = info.likes
                ctx.created_at = info.created_at
                ctx.last_modified = info.last_modified
                ctx.gated = info.gated
                ctx.private = info.private

                ctx.readme_text = self.hf_client.get_readme(parsed.hf_id) or ""
                break
            except FileNotFoundError as e:
                ctx.api_errors += 1
                ctx.fetch_logs.append(f"HF not found: {e}")
                raise
            except Exception as e:
                ctx.api_errors += 1
                ctx.fetch_logs.append(f"HF error: {e}")
                if attempt >= 2:
                    break

        return ctx


# ---------------- CODE (GitHub) ----------------


class CodeUrlHandler(UrlHandler):
    def __init__(self, url: Optional[str] = None):
        super().__init__(url)
        self.gh_client = GHClient()

    def fetchMetaData(self) -> RepoContext:
        if not self.url:
            raise ValueError("URL is required")

        parsed = UrlRouter().parse(self.url)
        if not parsed.gh_owner_repo:
            raise ValueError("Not a GitHub repo URL")

        owner, repo = parsed.gh_owner_repo
        ctx = RepoContext(
            url=self.url,
            gh_url=f"https://github.com/{owner}/{repo}",
            host="GitHub",
        )

        try:
            info = self.gh_client.get_repo(owner, repo)
            if not info:
                ctx.api_errors += 1
                ctx.fetch_logs.append(f"GitHub repo {owner}/{repo} not found")
                return ctx

            ctx.private = (
                bool(getattr(info, "private", None))
                if info.private is not None
                else None
            )
            ctx.card_data = {
                "default_branch": getattr(info, "default_branch", None),
                "description": getattr(info, "description", None),
            }

            # README
            try:
                readme = self.gh_client.get_readme_markdown(owner, repo)
                if readme:
                    ctx.readme_text = readme
            except Exception as re_err:
                ctx.api_errors += 1
                ctx.fetch_logs.append(f"GitHub readme error: {re_err}")

            # Contributors
            try:
                contribs = self.gh_client.list_contributors(owner, repo)
                ctx.contributors = [
                    {
                        "login": c.get("login"),
                        "contributions": int(c.get("contributions", 0)),
                    }
                    for c in (contribs or [])
                ]
            except Exception as ce:
                ctx.api_errors += 1
                ctx.fetch_logs.append(f"GitHub contributors error: {ce}")

            # Files tree (best-effort)
            try:
                branch = getattr(info, "default_branch", "main") or "main"
                tree = None
                if hasattr(self.gh_client, "get_repo_tree"):
                    tree = self.gh_client.get_repo_tree(
                        owner, repo, branch, recursive=True
                    )
                elif hasattr(self.gh_client, "list_tree"):
                    tree = self.gh_client.list_tree(
                        owner, repo, branch, recursive=True
                    )
                if tree:
                    files = []
                    for n in tree:
                        if n.get("type") == "blob" and n.get("path"):
                            p = n["path"]
                            files.append(
                                FileInfo(
                                    path=Path(p),
                                    size_bytes=int(n.get("size") or 0),
                                    ext=_safe_ext(p),
                                )
                            )
                    ctx.files = files
            except Exception as te:
                ctx.fetch_logs.append(f"GitHub tree error: {te}")

            return ctx

        except Exception as e:
            ctx.api_errors += 1
            ctx.fetch_logs.append(f"GitHub error: {e}")
            return ctx


# ---------------- Dataset discovery helpers ----------------

_DATASET_RE = re.compile(
    r"""
    (?:
        https?://(?:www\.)?huggingface\.co/datasets/ | \bdatasets?\/
    )
    (?P<org>[A-Za-z0-9_.-]+)\/(?P<name>[A-Za-z0-9_.-]+)
    """,
    re.IGNORECASE | re.VERBOSE,
)


def _norm_id(s: str) -> str:
    return s.strip().strip("/").lower()


def _uniq_keep_order(xs: Iterable[str]) -> list[str]:
    seen, out = set(), []
    for x in xs:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out


def datasets_from_card(
    card_data: dict | None, tags: list[str] | None
) -> list[str]:
    out: list[str] = []
    if isinstance(card_data, dict):
        ds = card_data.get("datasets")
        if isinstance(ds, list):
            for d in ds:
                if isinstance(d, str) and "/" in d:
                    out.append(_norm_id(d.removeprefix("datasets/")))
    if tags:
        for t in tags:
            if isinstance(t, str) and t.lower().startswith("dataset:"):
                did = t.split(":", 1)[1]
                if "/" in did:
                    out.append(_norm_id(did.removeprefix("datasets/")))
    return _uniq_keep_order(out)


def datasets_from_readme(readme_text: str | None) -> list[str]:
    if not readme_text:
        return []
    out: list[str] = []
    for m in _DATASET_RE.finditer(readme_text):
        org = _norm_id(m.group("org"))
        name = _norm_id(m.group("name"))
        out.append(f"{org}/{name}")
    return _uniq_keep_order(out)


# ---------------- Builders ----------------


def build_model_context(url: str) -> RepoContext:
    return ModelUrlHandler(url).fetchMetaData()


def build_dataset_context(url: str) -> RepoContext:
    return DatasetUrlHandler(url).fetchMetaData()


def build_code_context(url: str) -> RepoContext:
    return CodeUrlHandler(url).fetchMetaData()
