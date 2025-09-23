#!/usr/bin/env python3
import os
import platform
import stat
import sys
from pathlib import Path
from typing import Dict, Literal

import db
from handlers import (build_code_context, build_dataset_context,
                      build_model_context)
from metric_eval import MetricEval, init_metrics, init_weights
from net_scorer import NetScorer
from repo_context import RepoContext
from url_router import UrlRouter, UrlType

# ----------------
# Types & constants
# ----------------

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
            category, ctx = _build_context_for_url(url)
            rid = persist_context(db_path, ctx, category)

            print(f"[OK] {category:<7} saved id={rid} url={url}")
            succeeded += 1

            _evaluate_and_persist(db_path, rid, category, ctx)

        except Exception as e:
            print(f"[ERR] url={url} error={e}", file=sys.stderr)

    return 0 if succeeded == total else 1


# ----------------
# Context building
# ----------------


def _build_context_for_url(url: str) -> tuple[Category, RepoContext]:
    """Route URL to the correct builder and return (category, RepoContext)."""
    parsed = UrlRouter().parse(url)
    if parsed.type is UrlType.MODEL:
        return "MODEL", build_model_context(url)
    if parsed.type is UrlType.DATASET:
        return "DATASET", build_dataset_context(url)
    if parsed.type is UrlType.CODE:
        return "CODE", build_code_context(url)
    raise ValueError("Unsupported URL type")


# ----------------
# Metric selection & scoring
# ----------------


def _evaluate_and_persist(
    db_path: Path, rid: int, category: Category, ctx: RepoContext
) -> None:
    """
    Run metrics via MetricEval.evaluateAll
    and print NDJSON & summary lines.
    """
    weights = init_weights()
    metrics = init_metrics()

    print("[METRICS]", [m.name for m in metrics])

    evaluator = MetricEval(metrics, weights)

    repo_ctx_dict = dict(ctx.__dict__)
    repo_ctx_dict["_ctx_obj"] = ctx
    repo_ctx_dict["category"] = category

    per_metric_scores: Dict[str, float] = evaluator.evaluateAll(repo_ctx_dict)

    for name, val in per_metric_scores.items():
        print(f"[METRIC] id={rid} {name}={float(val):.3f}")

    # Compute NetScore with the same MetricEval instance
    net = evaluator.aggregateScores(per_metric_scores)
    print(f"[SCORE] id={rid} category={category} net_score={net:.3f}")

    # ---- Persist metric rows ----
    conn = db.open_db(db_path)
    try:
        metric_version = "v1"
        fp_parts = [
            category,
            (ctx.hf_id or ctx.gh_url or ctx.url or "unknown"),
        ]

        # No latencies from evaluateAll; record zeros to keep schema happy
        zero_latencies: Dict[str, int] = {
            name: 0 for name in per_metric_scores.keys()
        }

        for m in metrics:
            value = float(per_metric_scores.get(m.name, -1.0))
            aux = {}

            # Attach any detailed LLM pieces or metric errors for visibility
            if (
                m.name == "CodeQuality"
                and "_code_quality_llm_parts" in repo_ctx_dict
            ):
                aux["llm_parts"] = repo_ctx_dict["_code_quality_llm_parts"]
            if (
                m.name == "DatasetQuality"
                and "_dataset_quality_llm_parts" in repo_ctx_dict
            ):
                aux["llm_parts"] = repo_ctx_dict["_dataset_quality_llm_parts"]
            if (
                "_metric_errors" in repo_ctx_dict
                and m.name in repo_ctx_dict["_metric_errors"]
            ):
                aux["error"] = repo_ctx_dict["_metric_errors"][m.name]

            db.upsert_metric(
                conn,
                resource_id=rid,
                metric_name=m.name,
                value=value,
                latency_ms=zero_latencies.get(m.name, 0),
                aux=aux,
                metric_version=metric_version,
                input_fingerprint_parts=fp_parts,
            )

        db.upsert_metric(
            conn,
            resource_id=rid,
            metric_name="NetScore",
            value=net,
            latency_ms=sum(zero_latencies.values()),
            aux={},
            metric_version=metric_version,
            input_fingerprint_parts=fp_parts,
        )
        conn.commit()
    finally:
        conn.close()

    # ---- NetScorer output (NDJSON + human-readable) ----
    latencies_with_net = {
        "NetScore": 0,
        **{k: 0 for k in per_metric_scores.keys()},
    }
    # Let NetScorer compute NetScore from the per-metric dict
    ns = NetScorer(
        scores=per_metric_scores,
        weights=weights,
        url=(ctx.hf_id or ctx.gh_url or ctx.url or ""),
        latencies=latencies_with_net,
    )
    print(ns.to_ndjson_string())
    print(str(ns))


# ----------------
# Persistence & path helpers
# ----------------


def persist_context(
    db_path: Path, ctx: RepoContext, category: Category
) -> int:
    """
    Upsert resource and files into DB, and link model→(dataset/code).
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
        {
            "path": str(fi.path),
            "ext": fi.ext,
            "size_bytes": int(fi.size_bytes or 0),
        }
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

        # If it's a model, link to any previously ingested datasets/code
        if category == "MODEL":
            # ---- Datasets: upsert and link MODEL -> DATASET
            for ds in ctx.linked_datasets:
                ds_key = RepoContext._canon_dataset_key(ds)
                if not ds_key:
                    continue

                ds_id = db.resource_id_by_key(
                    conn, category="DATASET", canonical_key=ds_key
                )
                if ds_id is None:
                    # hydrate + insert now so the link never dangles
                    ds_url = (
                        getattr(ds, "url", None)
                        or f"https://huggingface.co/datasets/{ds_key}"
                    )
                    try:
                        ds_ctx = build_dataset_context(ds_url)
                        ds_base = {
                            "url": ds_ctx.url,
                            "hf_id": ds_ctx.hf_id,
                            "gh_url": ds_ctx.gh_url,
                            "host": ds_ctx.host,
                            "repo_path": (
                                str(ds_ctx.repo_path)
                                if ds_ctx.repo_path
                                else None
                            ),
                            "readme_text": ds_ctx.readme_text,
                            "card_data": ds_ctx.card_data,
                            "config_json": ds_ctx.config_json,
                            "model_index": ds_ctx.model_index,
                            "tags": ds_ctx.tags,
                            "downloads_30d": ds_ctx.downloads_30d,
                            "downloads_all_time": ds_ctx.downloads_all_time,
                            "likes": ds_ctx.likes,
                            "created_at": ds_ctx.created_at,
                            "last_modified": ds_ctx.last_modified,
                            "gated": ds_ctx.gated,
                            "private": ds_ctx.private,
                            "contributors": ds_ctx.contributors,
                            "commit_history": ds_ctx.commit_history,
                            "fetch_logs": ds_ctx.fetch_logs,
                            "cache_hits": ds_ctx.cache_hits,
                            "api_errors": ds_ctx.api_errors,
                        }
                        ds_files = [
                            {
                                "path": str(fi.path),
                                "ext": fi.ext,
                                "size_bytes": int(fi.size_bytes or 0),
                            }
                            for fi in ds_ctx.files
                        ]
                        ds_id = db.upsert_resource(
                            conn,
                            category="DATASET",
                            canonical_key=RepoContext._canon_dataset_key(
                                ds_ctx
                            ),
                            base=ds_base,
                            files=ds_files,
                        )
                    except Exception as e:
                        ctx.fetch_logs.append(
                            f"Linked dataset hydrate failed for {ds_key}: {e}"
                        )

                if ds_id is not None:
                    db.link_resources(conn, rid, ds_id, "MODEL_TO_DATASET")

            for code in ctx.linked_code:
                code_key = RepoContext._canon_code_key(code)
                if not code_key:
                    continue
                code_id = db.resource_id_by_key(
                    conn, category="CODE", canonical_key=code_key
                )
                if code_id is None:
                    code_url = getattr(code, "gh_url", None) or getattr(
                        code, "url", None
                    )
                    if code_url:
                        try:
                            code_ctx = build_code_context(code_url)
                            code_base = {
                                "url": code_ctx.url,
                                "hf_id": code_ctx.hf_id,
                                "gh_url": code_ctx.gh_url,
                                "host": code_ctx.host,
                                "repo_path": (
                                    str(code_ctx.repo_path)
                                    if code_ctx.repo_path
                                    else None
                                ),
                                "readme_text": code_ctx.readme_text,
                                "card_data": code_ctx.card_data,
                                "config_json": code_ctx.config_json,
                                "model_index": code_ctx.model_index,
                                "tags": code_ctx.tags,
                                "downloads_30d": code_ctx.downloads_30d,
                                "downloads_all_time": (
                                    code_ctx.downloads_all_time
                                ),
                                "likes": code_ctx.likes,
                                "created_at": code_ctx.created_at,
                                "last_modified": code_ctx.last_modified,
                                "gated": code_ctx.gated,
                                "private": code_ctx.private,
                                "contributors": code_ctx.contributors,
                                "commit_history": code_ctx.commit_history,
                                "fetch_logs": code_ctx.fetch_logs,
                                "cache_hits": code_ctx.cache_hits,
                                "api_errors": code_ctx.api_errors,
                            }
                            code_files = [
                                {
                                    "path": str(fi.path),
                                    "ext": fi.ext,
                                    "size_bytes": int(fi.size_bytes or 0),
                                }
                                for fi in code_ctx.files
                            ]
                            code_id = db.upsert_resource(
                                conn,
                                category="CODE",
                                canonical_key=RepoContext._canon_code_key(
                                    code_ctx
                                ),
                                base=code_base,
                                files=code_files,
                            )
                        except Exception as e:
                            ctx.fetch_logs.append(
                                f"Linked code failed for {code_url}: {e}"
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
    return RepoContext._canon_code_key(ctx)


# ----------------
# CLI utilities
# ----------------


def read_urls(arg: str) -> list[str]:
    """Read newline-delimited URLs from a file, filtering blank lines."""
    try:
        with open(arg, "r", encoding="ascii") as f:
            return [line.strip() for line in f if line.strip()]
    except FileNotFoundError:
        print(f"Error: File '{arg}' not found.", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error reading file '{arg}': {e}", file=sys.stderr)
        sys.exit(1)


# ----------------
# Path helpers
# ----------------


def _find_project_root(start: Path) -> Path | None:
    """Walk up to find a .git folder; return its parent as project root"""
    cur = start.resolve()
    for _ in range(10):
        if (cur / ".git").exists():
            return cur
        if cur.parent == cur:
            break
        cur = cur.parent
    return None


def _user_cache_base() -> Path:
    """OS-native cache base for default DB location."""
    system = platform.system()
    home = Path.home()
    if system == "Darwin":
        return home / "Library" / "Application Support" / "acme-cli"
    elif system == "Windows":
        base = os.environ.get("APPDATA")
        return (
            Path(base) / "acme-cli"
            if base
            else home / "AppData" / "Roaming" / "acme-cli"
        )
    else:
        xdg = os.environ.get("XDG_CACHE_HOME")
        return Path(xdg) / "acme-cli" if xdg else home / ".cache" / "acme-cli"


def _resolve_db_path() -> Path:
    """ACME_DB env → project .acme → user cache."""
    env_db = os.environ.get("ACME_DB")
    if env_db:
        return Path(env_db)
    proj = _find_project_root(Path.cwd())
    if proj:
        return proj / ".acme" / "state.sqlite"
    return _user_cache_base() / "state.sqlite"


def _ensure_path_secure(p: Path) -> None:
    """Create parent dirs, touch file if needed,"""
    p.parent.mkdir(parents=True, exist_ok=True)
    try:
        if not p.exists():
            p.touch()
        p.chmod(stat.S_IRUSR | stat.S_IWUSR)
    except Exception:
        pass


if __name__ == "__main__":
    sys.exit(main())
