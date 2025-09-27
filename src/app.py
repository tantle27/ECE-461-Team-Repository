#!/usr/bin/env python3
import logging
import os
import platform
import stat
import sys
import time
from pathlib import Path
from typing import Dict, Literal
import math

import db
from handlers import (
    build_code_context,
    build_dataset_context,
    build_model_context,
)
from metric_eval import MetricEval, init_metrics, init_weights
from net_scorer import _emit_ndjson
from repo_context import RepoContext
from url_router import UrlRouter, UrlType
import json
from typing import Optional
Category = Literal["MODEL", "DATASET", "CODE"]


# ---------------- Logging ----------------
def _validate_log_file_env() -> str:
    """
    Validate LOG_FILE env var per spec:
      - Must be set
      - File must already exist (do NOT create it)
      - Must be writable (we must be able to open for writing)
    Exit(1) on failure.
    """
    log_file = os.getenv("LOG_FILE")
    if not log_file:
        sys.exit(1)

    p = Path(log_file)

    if not p.exists() or not p.is_file():
        sys.exit(1)

    # Check basic write access
    if not os.access(p, os.W_OK):
        sys.exit(1)

    # Try to open for append and write a test byte, then remove it
    try:
        with open(p, "a+b") as f:
            pos = f.tell()
            f.write(b"\0")
            f.flush()
            f.seek(pos)
            f.truncate()
    except Exception:
        sys.exit(1)

    return log_file


def setup_logging() -> None:
    """
    Configure logging via env. REQUIRED:
      LOG_FILE must be set and writable.
      LOG_LEVEL -> 0=silent, 1=info, 2=debug, default "0.3" ~ WARNING
    On any failure here, exit(1).
    """
    log_file = os.getenv("LOG_FILE")
    if not log_file:
        sys.exit(1)

    try:
        p = Path(log_file)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "a", encoding="utf-8"):
            pass
    except Exception:
        sys.exit(1)

    try:
        val = float(os.getenv("LOG_LEVEL", "0.3"))
    except ValueError:
        val = 0.3

    if val <= 0:
        level = logging.CRITICAL + 1
    elif int(val) == 1:
        level = logging.INFO
    elif int(val) == 2:
        level = logging.DEBUG
    else:
        level = logging.CRITICAL + 1

    logging.basicConfig(
        filename=log_file,
        filemode="a",
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        force=True,
    )

    logging.info("logging initialized: file=%s level=%s", log_file, logging.getLevelName(level))


def _require_valid_github_token() -> str:
    """
    Enforces presence of a plausibly valid GitHub token in $GITHUB_TOKEN.
    We can't check server-side validity here, but we can reject obviously bad values.
    Exits(1) on failure.
    """
    tok = os.environ.get("GITHUB_TOKEN", "")
    if not tok:
        # print("ERROR: GITHUB_TOKEN not set; refusing to run.", file=sys.stderr)
        sys.exit(1)

    # Accept common formats: legacy 'ghp_' or modern 'github_pat_'
    if not (tok.startswith("ghp_") or tok.startswith("github_pat_")):
        # print("ERROR: GITHUB_TOKEN appears invalid (unexpected format).", file=sys.stderr)
        sys.exit(1)

    return tok


# ---------------- CLI + Paths ----------------

def read_urls(path: str) -> list[tuple[str, str, str]]:
    """Read file of CSV triples: code,dataset,model (blanks allowed)."""
    rows: list[tuple[str, str, str]] = []
    with open(path, "r", encoding="ascii") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                rows.append(("", "", ""))
                continue
            parts = [p.strip() for p in line.split(",", 2)]
            while len(parts) < 3:
                parts.append("")
            code_url, dataset_url, model_url = parts[0], parts[1], parts[2]
            rows.append((code_url, dataset_url, model_url))
    return rows


def _attach_explicit_links(
    model_ctx: RepoContext,
    code_url: str,
    dataset_url: str,
) -> None:
    """
    If the row provided code_url and/or dataset_url, build those contexts,
    tag them as explicit, and attach to the model context so metrics see them.
    """
    if code_url:
        try:
            cat_c, code_ctx = _build_context_for_url(code_url)
            if cat_c == "CODE" and code_ctx:
                code_ctx.__dict__["_link_source"] = "explicit"
                model_ctx.link_code(code_ctx)
        except Exception as e:
            model_ctx.fetch_logs.append(f"explicit code attach failed: {e}")

    if dataset_url:
        try:
            cat_d, ds_ctx = _build_context_for_url(dataset_url)
            if cat_d == "DATASET" and ds_ctx:
                ds_ctx.__dict__["_link_source"] = "explicit"
                model_ctx.link_dataset(ds_ctx)
        except Exception as e:
            model_ctx.fetch_logs.append(f"explicit dataset attach failed: {e}")


def _find_project_root(start: Path) -> Optional[Path]:
    cur = start.resolve()
    for _ in range(10):
        if (cur / ".git").exists():
            return cur
        if cur.parent == cur:
            break
        cur = cur.parent
    return None


def _user_cache_base() -> Path:
    sysname = platform.system()
    home = Path.home()
    if sysname == "Darwin":
        return home / "Library" / "Application Support" / "acme-cli"
    if sysname == "Windows":
        base = os.environ.get("APPDATA")
        return Path(base) / "acme-cli" if base else \
            home / "AppData" / "Roaming" / "acme-cli"
    xdg = os.environ.get("XDG_CACHE_HOME")
    return Path(xdg) / "acme-cli" if xdg else home / ".cache" / "acme-cli"


def _resolve_db_path() -> Path:
    env_db = os.environ.get("ACME_DB")
    if env_db:
        return Path(env_db)
    proj = _find_project_root(Path.cwd())
    return proj / ".acme" / "state.sqlite" if proj else \
        _user_cache_base() / "state.sqlite"


def _ensure_path_secure(p: Path) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    try:
        if not p.exists():
            p.touch()
        p.chmod(stat.S_IRUSR | stat.S_IWUSR)
    except Exception:
        pass


# ---------------- Pipeline ----------------

def _ctx_summary(ctx: RepoContext) -> dict:
    return {
        "url": ctx.url,
        "hf_id": ctx.hf_id,
        "gh_url": ctx.gh_url,
        "host": ctx.host,
        "files": len(ctx.files or []),
        "tags": len(ctx.tags or []),
        "contributors": len(ctx.contributors or []),
        "has_readme": bool(ctx.readme_text),
        "repo_path": str(ctx.repo_path) if ctx.repo_path else None,
    }


def _build_context_for_url(url: str) -> tuple[Category, RepoContext]:
    parsed = UrlRouter().parse(url)
    if parsed.type is UrlType.MODEL:
        return "MODEL", build_model_context(url)
    if parsed.type is UrlType.DATASET:
        return "DATASET", build_dataset_context(url)
    if parsed.type is UrlType.CODE:
        return "CODE", build_code_context(url)
    raise ValueError("Unsupported URL type")


def _evaluate_and_persist(
    db_path: Path, rid: int, category: Category, ctx: RepoContext
) -> None:
    if category != "MODEL":
        return

    url_disp = ctx.hf_id or ctx.gh_url or ctx.url or ""
    logging.info("eval: id=%s category=%s url=%s", rid, category, url_disp)

    weights = init_weights()
    metrics = init_metrics()
    logging.debug("metrics=%s", [m.name for m in metrics])

    evaluator = MetricEval(metrics, weights)
    repo_ctx = {**ctx.__dict__, "_ctx_obj": ctx, "category": category}

    def _to_01(x: object) -> float:
        try:
            v = float(x)
        except Exception:
            return 0.0
        if not math.isfinite(v) or v < 0.0:
            return 0.0
        return 1.0 if v > 1.0 else v

    scores: Dict[str, float] = {}
    lats_ms: Dict[str, int] = {}

    eval_t0 = time.perf_counter_ns()

    # -------- Parallel evaluation --------
    raw_results = {}
    durations: Dict[str, int] = {}

    def timed_eval(metric):
        t0 = time.perf_counter_ns()
        try:
            val = metric.evaluate(repo_ctx)
        except Exception as e:
            repo_ctx.setdefault("_metric_errors", {})[metric.name] = str(e)
            val = 0.0
        dur_ms = max(1, int((time.perf_counter_ns() - t0) // 1_000_000))
        return metric.name, val, dur_ms

    from concurrent.futures import ThreadPoolExecutor, as_completed
    with ThreadPoolExecutor() as executor:
        futures = {executor.submit(timed_eval, m): m for m in metrics}
        for fut in as_completed(futures):
            name, raw, dur = fut.result()
            raw_results[name] = raw
            durations[name] = dur
            scores[name] = _to_01(raw)
            lats_ms[name] = dur
            logging.info("[METRIC] id=%s %s=%.3f (%d ms)", rid, name, scores[name], dur)

    net = _to_01(evaluator.aggregateScores(scores))

    total_eval_ms = max(
        1, int((time.perf_counter_ns() - eval_t0) // 1_000_000)
    )
    net_lat = total_eval_ms

    logging.info(
        "[SCORE] id=%s category=%s net=%.3f (%d ms)",
        rid,
        category,
        net,
        net_lat,
    )

    conn = db.open_db(db_path)
    try:
        ver = "v1"
        fp = [category, url_disp or "unknown"]

        for m in metrics:
            aux = {}
            if (m.name == "CodeQuality"
                    and "_code_quality_llm_parts" in repo_ctx):
                aux["llm_parts"] = repo_ctx["_code_quality_llm_parts"]
            if (m.name == "DatasetQuality"
                    and "_dataset_quality_llm_parts" in repo_ctx):
                aux["llm_parts"] = repo_ctx["_dataset_quality_llm_parts"]
            if "_metric_errors" in repo_ctx and m.name in repo_ctx["_metric_errors"]:
                aux["error"] = repo_ctx["_metric_errors"][m.name]

            db.upsert_metric(
                conn,
                resource_id=rid,
                metric_name=m.name,
                value=float(scores.get(m.name, 0.0)),
                latency_ms=int(lats_ms.get(m.name, 1)),
                aux=aux,
                metric_version=ver,
                input_fingerprint_parts=fp,
            )

        db.upsert_metric(
            conn,
            resource_id=rid,
            metric_name="NetScore",
            value=float(net),
            latency_ms=int(net_lat),
            aux={},
            metric_version=ver,
            input_fingerprint_parts=fp,
        )
        conn.commit()
    finally:
        conn.close()

    if category == "MODEL":
        _emit_ndjson(ctx, category, scores, net, lats_ms, net_lat)

    logging.info("eval done: id=%s metrics=%d", rid, len(scores))

def _canon_for(ctx: RepoContext, category: Category) -> str:
    if category == "MODEL":
        return ctx.hf_id.lower() if ctx.hf_id else (ctx.url or "").lower()
    if category == "DATASET":
        return RepoContext._canon_dataset_key(ctx)
    return RepoContext._canon_code_key(ctx)


def persist_context(
    db_path: Path, ctx: RepoContext, category: Category
) -> int:
    canon = _canon_for(ctx, category)
    if not canon:
        raise ValueError("Cannot derive canonical key")

    logging.info("persist: category=%s key=%s url=%s", category, canon,
                (ctx.hf_id or ctx.gh_url or ctx.url or ""))

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

    files = [{
        "path": str(fi.path),
        "ext": fi.ext,
        "size_bytes": int(fi.size_bytes or 0),
    } for fi in ctx.files]

    conn = db.open_db(db_path)
    try:
        rid = db.upsert_resource(
            conn, category=category, canonical_key=canon, base=base,
            files=files,
        )
        logging.info("persisted: id=%s category=%s files=%d",
                    rid, category, len(ctx.files or []))

        if category == "MODEL":
            # datasets
            for ds in ctx.linked_datasets:
                ds_key = RepoContext._canon_dataset_key(ds)
                if not ds_key:
                    continue
                ds_id = db.resource_id_by_key(
                    conn, category="DATASET", canonical_key=ds_key
                )
                if ds_id is None:
                    ds_url = getattr(ds, "url", None) or \
                        f"https://huggingface.co/datasets/{ds_key}"
                    try:
                        ds_ctx = build_dataset_context(ds_url)
                        ds_base = {
                            "url": ds_ctx.url,
                            "hf_id": ds_ctx.hf_id,
                            "gh_url": ds_ctx.gh_url,
                            "host": ds_ctx.host,
                            "repo_path": (
                                str(ds_ctx.repo_path)
                                if ds_ctx.repo_path else None
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
                        ds_files = [{
                            "path": str(fi.path),
                            "ext": fi.ext,
                            "size_bytes": int(fi.size_bytes or 0),
                        } for fi in ds_ctx.files]
                        ds_id = db.upsert_resource(
                            conn, category="DATASET",
                            canonical_key=RepoContext._canon_dataset_key(
                                ds_ctx),
                            base=ds_base, files=ds_files,
                        )
                    except Exception as e:
                        ctx.fetch_logs.append(
                            f"Linked dataset hydrate failed for {ds_key}: {e}"
                        )
                        logging.warning("dataset hydrate failed: %s", e)

                if ds_id is not None:
                    db.link_resources(conn, rid, ds_id, "MODEL_TO_DATASET")

            # code
            for code in ctx.linked_code:
                code_key = RepoContext._canon_code_key(code)
                if not code_key:
                    continue
                code_id = db.resource_id_by_key(
                    conn, category="CODE", canonical_key=code_key
                )
                if code_id is None:
                    code_url = getattr(code, "gh_url", None) or \
                        getattr(code, "url", None)
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
                                    if code_ctx.repo_path else None
                                ),
                                "readme_text": code_ctx.readme_text,
                                "card_data": code_ctx.card_data,
                                "config_json": code_ctx.config_json,
                                "model_index": code_ctx.model_index,
                                "tags": code_ctx.tags,
                                "downloads_30d": code_ctx.downloads_30d,
                                "downloads_all_time":
                                    code_ctx.downloads_all_time,
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
                            code_files = [{
                                "path": str(fi.path),
                                "ext": fi.ext,
                                "size_bytes": int(fi.size_bytes or 0),
                            } for fi in code_ctx.files]
                            code_id = db.upsert_resource(
                                conn, category="CODE",
                                canonical_key=RepoContext._canon_code_key(
                                    code_ctx),
                                base=code_base, files=code_files,
                            )
                        except Exception as e:
                            ctx.fetch_logs.append(
                                f"Linked code failed for {code_url}: {e}"
                            )
                            logging.warning("code hydrate failed: %s", e)

                if code_id is not None:
                    db.link_resources(conn, rid, code_id, "MODEL_TO_CODE")

        conn.commit()
        logging.info("persist done: id=%s category=%s key=%s",
                    rid, category, canon)
        return rid
    finally:
        conn.close()


# ---------------- Entry ----------------

def main() -> int:
    if len(sys.argv) != 2:
        sys.exit(1)
    url_file = sys.argv[1]

    rows = read_urls(url_file)
    db_path = _resolve_db_path()
    _ensure_path_secure(db_path)

    total = len(rows)
    succeeded = 0

    for code_url, dataset_url, model_url in rows:
        code_id = None
        dataset_id = None

        try:
            if code_url:
                cat, ctx = _build_context_for_url(code_url)
                if cat == "CODE":
                    code_id = persist_context(db_path, ctx, cat)
                else:
                    logging.warning("code column parsed as %s, skipping: %s", cat, code_url)
        except Exception as e:
            logging.warning("code persist failed: %s", e)

        try:
            if dataset_url:
                cat, ctx = _build_context_for_url(dataset_url)
                if cat == "DATASET":
                    dataset_id = persist_context(db_path, ctx, cat)
                else:
                    logging.warning("dataset column parsed as %s, skipping: %s", cat, dataset_url)
        except Exception as e:
            logging.warning("dataset persist failed: %s", e)

        try:
            url = (model_url or "").strip()
            if not url:
                _emit_error_ndjson("unknown", "MODEL")
                continue

            category, ctx = _build_context_for_url(url)
            if category == "MODEL":
                _attach_explicit_links(ctx, code_url, dataset_url)
            model_id = persist_context(db_path, ctx, category)
            if category == "MODEL":
                _evaluate_and_persist(db_path, model_id, category, ctx)

                if code_id is not None or dataset_id is not None:
                    conn = db.open_db(db_path)
                    try:
                        if dataset_id is not None:
                            db.link_resources(conn, model_id, dataset_id, "MODEL_TO_DATASET")
                        if code_id is not None:
                            db.link_resources(conn, model_id, code_id, "MODEL_TO_CODE")
                        conn.commit()
                    finally:
                        conn.close()
            else:
                _emit_error_ndjson(url, category)

        except Exception as e:
            logging.error("model pipeline failed: url=%s err=%s", model_url, e, exc_info=True)
            _emit_error_ndjson(model_url or "unknown", "MODEL")

    sys.exit(0 if succeeded == total else 1)


def _emit_error_ndjson(name_hint: str = "unknown", category: str = "MODEL") -> None:
    name = (name_hint.rstrip("/").split("/")[-1] or "unknown") if name_hint else "unknown"
    nd = {
        "name": name,
        "category": category,
        "net_score": 0.0, "net_score_latency": 1,
        "ramp_up_time": 0.0, "ramp_up_time_latency": 1,
        "bus_factor": 0.0, "bus_factor_latency": 1,
        "performance_claims": 0.0, "performance_claims_latency": 1,
        "license": 0.0, "license_latency": 1,
        "size_score": {
            "raspberry_pi": 0.0, "jetson_nano": 0.0,
            "desktop_pc": 0.0, "aws_server": 0.0
        },
        "size_score_latency": 1,
        "dataset_and_code_score": 0.0, "dataset_and_code_score_latency": 1,
        "dataset_quality": 0.0, "dataset_quality_latency": 1,
        "code_quality": 0.0, "code_quality_latency": 1,
    }
    print(json.dumps(nd, separators=(",", ":"), ensure_ascii=False), flush=True)


if __name__ == "__main__":
    _require_valid_github_token()
    _validate_log_file_env()
    setup_logging()
    sys.exit(main())
