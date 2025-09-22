#!/usr/bin/env python3
import os
import sys
import stat
import platform
from pathlib import Path
from typing import Literal

import db
from repo_context import RepoContext
from url_router import UrlRouter, UrlType
from handlers import (
    build_model_context,
    build_dataset_context,
    build_code_context,
)

Category = Literal["MODEL", "DATASET", "CODE"]


def main() -> int:
    if len(sys.argv) != 2:
        print("usage: ./run <URL_FILE>", file=sys.stderr)
        return 1

    url_file = sys.argv[1]
    urls = read_urls(url_file)

    db_path = _resolve_db_path()
    _ensure_path_secure(db_path)

    total = 0
    succeeded = 0
    for url in urls:
        total += 1
        try:
            rid, cat = ingest_url(db_path, url)
            print(f"[OK] {cat:<7} saved id={rid} url={url}")
            succeeded += 1
        except Exception as e:
            print(f"[ERR] url={url} error={e}", file=sys.stderr)

    return 0 if succeeded == total else 1


# ----------------
# helpers
# ----------------

def read_urls(arg: str) -> list[str]:
    try:
        with open(arg, "r", encoding="ascii") as f:
            return [line.strip() for line in f if line.strip()]
    except FileNotFoundError:
        print(f"Error: File '{arg}' not found.", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error reading file '{arg}': {e}", file=sys.stderr)
        sys.exit(1)


def ingest_url(db_path: Path, url: str) -> tuple[int, Category]:
    """
    Build RepoContext from URL, persist it into SQLite, and return 
    (resource_id, category). If a dataset/code was ingested earlier, 
    model ingestion will link to it automatically.
    """
    parsed = UrlRouter().parse(url)

    if parsed.type is UrlType.MODEL:
        ctx = build_model_context(url)
        rid = persist_context(db_path, ctx, "MODEL")
        return rid, "MODEL"

    if parsed.type is UrlType.DATASET:
        ctx = build_dataset_context(url)
        rid = persist_context(db_path, ctx, "DATASET")
        return rid, "DATASET"

    if parsed.type is UrlType.CODE:
        ctx = build_code_context(url)
        rid = persist_context(db_path, ctx, "CODE")
        return rid, "CODE"

    raise ValueError("Unsupported URL type")


def persist_context(db_path: Path, ctx: RepoContext, category: Category) -> int:
    """
    Idempotently upsert the resource + files, and (if a model) link to any datasets/code
    that are already present in the DB.
    """
    canon = _canon_for(ctx, category)
    if not canon:
        raise ValueError("Cannot derive canonical key for persistence")

    base = {
        "url": ctx.url,
        "hf_id": ctx.hf_id,
        "gh_url": ctx.gh_url,
        "host": ctx.host,
        "repo_path": str(ctx.repo_path) if ctx.repo_path else None,
        "readme_text": ctx.readme_text,
        "card_data": ctx.card_data,
        "config_json": ctx.config_json,
        "model_index": ctx.model_index,
        "tags": ctx.tags,
        "downloads_30d": ctx.downloads_30d,
        "downloads_all_time": ctx.downloads_all_time,
        "likes": ctx.likes,
        "created_at": ctx.created_at,
        "last_modified": ctx.last_modified,
        "gated": ctx.gated,
        "private": ctx.private,
        "contributors": ctx.contributors,
        "commit_history": ctx.commit_history,
        "fetch_logs": ctx.fetch_logs,
        "cache_hits": ctx.cache_hits,
        "api_errors": ctx.api_errors,
    }

    files = [
        {"path": str(fi.path), "ext": fi.ext, "size_bytes": int(fi.size_bytes)}
        for fi in ctx.files
    ]

    conn = db.open_db(db_path)
    try:
        rid = db.upsert_resource(
            conn,
            category=category,
            canonical_key=canon,
            base=base,
            files=files,
        )

        # Auto-link if this is a model and the referenced dataset/code already exists
        if category == "MODEL":
            for ds in ctx.linked_datasets:
                ds_key = RepoContext._canon_dataset_key(ds)
                if ds_key:
                    ds_id = db.resource_id_by_key(
                        conn, category="DATASET", canonical_key=ds_key
                    )
                    if ds_id is not None:
                        db.link_resources(conn, rid, ds_id, "MODEL_TO_DATASET")

            for code in ctx.linked_code:
                code_key = RepoContext._canon_code_key(code)
                if code_key:
                    code_id = db.resource_id_by_key(
                        conn, category="CODE", canonical_key=code_key
                    )
                    if code_id is not None:
                        db.link_resources(conn, rid, code_id, "MODEL_TO_CODE")

        conn.commit()
        return rid
    finally:
        conn.close()


def _canon_for(ctx: RepoContext, category: Category) -> str:
    if category == "MODEL":
        return ctx.hf_id.lower() if ctx.hf_id else (ctx.url or "").lower()
    if category == "DATASET":
        return RepoContext._canon_dataset_key(ctx)
    # CODE
    return RepoContext._canon_code_key(ctx)


def _find_project_root(start: Path) -> Path | None:
    """Walk up to find a .git folder; return its parent as project root, else None."""
    cur = start.resolve()
    for _ in range(10):  # don’t walk past sanity
        if (cur / ".git").exists():
            return cur
        if cur.parent == cur:
            break
        cur = cur.parent
    return None


def _user_cache_base() -> Path:
    system = platform.system()
    home = Path.home()
    if system == "Darwin":
        return home / "Library" / "Application Support" / "acme-cli"
    elif system == "Windows":
        base = os.environ.get("APPDATA")
        return Path(base) / "acme-cli" if base else home / "AppData" / "Roaming" / "acme-cli"
    else:
        xdg = os.environ.get("XDG_CACHE_HOME")
        return Path(xdg) / "acme-cli" if xdg else home / ".cache" / "acme-cli"


def _resolve_db_path() -> Path:
    # 1) explicit override
    env_db = os.environ.get("ACME_DB")
    if env_db:
        return Path(env_db)

    # 2) project-local if inside a git repo
    proj = _find_project_root(Path.cwd())
    if proj:
        return proj / ".acme" / "state.sqlite"

    # 3) user cache dir (OS-native)
    return _user_cache_base() / "state.sqlite"


def _ensure_path_secure(p: Path) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    try:
        # Create the file if it doesn't exist so we can chmod it.
        if not p.exists():
            p.touch()
        p.chmod(stat.S_IRUSR | stat.S_IWUSR)
    except Exception:
        pass


if __name__ == "__main__":
    sys.exit(main())