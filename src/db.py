# db.py
from __future__ import annotations
import hashlib, json, sqlite3, time
from pathlib import Path
from typing import Any, Iterable, Literal, Optional

Category = Literal["MODEL","DATASET","CODE"]

# ---------- basics ----------
def _now_ms() -> int: return int(time.time() * 1000)
def _j(x: Any) -> str: return json.dumps(x, ensure_ascii=False, separators=(",", ":"))
def _fp(*parts: str) -> str:
    h = hashlib.sha256()
    for p in parts: h.update(p.encode("utf-8", "ignore"))
    return h.hexdigest()

def open_db(db_path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys=ON")
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    _ensure_schema(conn)
    return conn

# ---------- schema ----------
_SCHEMA = """
CREATE TABLE IF NOT EXISTS resources (
  id INTEGER PRIMARY KEY,
  category TEXT NOT NULL CHECK (category IN ('MODEL','DATASET','CODE')),
  canonical_key TEXT NOT NULL UNIQUE,
  url TEXT, hf_id TEXT, gh_url TEXT, host TEXT, repo_path TEXT,
  readme_text TEXT,
  card_data_json TEXT, config_json TEXT, model_index_json TEXT, tags_json TEXT,
  downloads_30d INTEGER, downloads_all_time INTEGER, likes INTEGER,
  created_at TEXT, last_modified TEXT, gated INTEGER, private INTEGER,
  contributors_json TEXT, commit_history_json TEXT, fetch_logs_json TEXT,
  cache_hits INTEGER DEFAULT 0, api_errors INTEGER DEFAULT 0,
  created_ts INTEGER NOT NULL, updated_ts INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_res_cat ON resources(category);

CREATE TABLE IF NOT EXISTS files (
  id INTEGER PRIMARY KEY,
  resource_id INTEGER NOT NULL REFERENCES resources(id) ON DELETE CASCADE,
  path TEXT NOT NULL, ext TEXT, size_bytes INTEGER,
  UNIQUE(resource_id, path)
);

CREATE TABLE IF NOT EXISTS links (
  id INTEGER PRIMARY KEY,
  src_resource_id INTEGER NOT NULL REFERENCES resources(id) ON DELETE CASCADE,
  dst_resource_id INTEGER NOT NULL REFERENCES resources(id) ON DELETE CASCADE,
  link_type TEXT NOT NULL CHECK (link_type IN ('MODEL_TO_DATASET','MODEL_TO_CODE')),
  UNIQUE(src_resource_id, dst_resource_id, link_type)
);

CREATE TABLE IF NOT EXISTS metric_results (
  id INTEGER PRIMARY KEY,
  resource_id INTEGER NOT NULL REFERENCES resources(id) ON DELETE CASCADE,
  metric_name TEXT NOT NULL,
  value_float REAL,
  aux_json TEXT,
  latency_ms INTEGER,
  input_fingerprint TEXT,
  metric_version TEXT,
  created_ts INTEGER NOT NULL, updated_ts INTEGER NOT NULL,
  UNIQUE(resource_id, metric_name, metric_version, input_fingerprint)
);
CREATE INDEX IF NOT EXISTS idx_metric_main ON metric_results(resource_id, metric_name);
"""

def _ensure_schema(conn: sqlite3.Connection) -> None:
    for stmt in _SCHEMA.strip().split(";\n"):
        s = stmt.strip()
        if s: conn.execute(s)

# ---------- canonical keys (re-uses your RepoContext logic at call site) ----------
def canonical_for(category: Category, *, hf_id: Optional[str], url: Optional[str], gh_owner_repo: Optional[tuple[str,str]]=None) -> str:
    if category == "DATASET":
        if hf_id: return hf_id.lower().removeprefix("datasets/")
        if url:   return url.lower()
        return ""
    if category == "CODE":
        if gh_owner_repo: return f"https://github.com/{gh_owner_repo[0]}/{gh_owner_repo[1]}"
        if url: return url.lower()
        return ""
    # MODEL
    if hf_id: return hf_id.lower()
    if url:   return url.lower()
    return ""

# ---------- upserts ----------
def upsert_resource(
    conn: sqlite3.Connection,
    *,
    category: Category,
    canonical_key: str,
    base: dict[str, Any],
    files: Iterable[dict[str, Any]] = ()
) -> int:
    ts = _now_ms()
    conn.execute(
        """
        INSERT INTO resources
        (category, canonical_key, url, hf_id, gh_url, host, repo_path,
         readme_text, card_data_json, config_json, model_index_json, tags_json,
         downloads_30d, downloads_all_time, likes, created_at, last_modified,
         gated, private, contributors_json, commit_history_json, fetch_logs_json,
         cache_hits, api_errors, created_ts, updated_ts)
        VALUES
        (:category, :canonical_key, :url, :hf_id, :gh_url, :host, :repo_path,
         :readme_text, :card_data_json, :config_json, :model_index_json, :tags_json,
         :downloads_30d, :downloads_all_time, :likes, :created_at, :last_modified,
         :gated, :private, :contributors_json, :commit_history_json, :fetch_logs_json,
         :cache_hits, :api_errors, :created_ts, :updated_ts)
        ON CONFLICT(canonical_key) DO UPDATE SET
          url=excluded.url, hf_id=excluded.hf_id, gh_url=excluded.gh_url, host=excluded.host,
          repo_path=excluded.repo_path, readme_text=excluded.readme_text,
          card_data_json=excluded.card_data_json, config_json=excluded.config_json,
          model_index_json=excluded.model_index_json, tags_json=excluded.tags_json,
          downloads_30d=excluded.downloads_30d, downloads_all_time=excluded.downloads_all_time,
          likes=excluded.likes, created_at=excluded.created_at, last_modified=excluded.last_modified,
          gated=excluded.gated, private=excluded.private,
          contributors_json=excluded.contributors_json, commit_history_json=excluded.commit_history_json,
          fetch_logs_json=excluded.fetch_logs_json, cache_hits=excluded.cache_hits,
          api_errors=excluded.api_errors, updated_ts=excluded.updated_ts
        """,
        {
            "category": category,
            "canonical_key": canonical_key,
            "url": base.get("url"), "hf_id": base.get("hf_id"), "gh_url": base.get("gh_url"),
            "host": base.get("host"), "repo_path": base.get("repo_path"),
            "readme_text": base.get("readme_text"),
            "card_data_json": _j(base.get("card_data") or {}),
            "config_json": _j(base.get("config_json") or {}),
            "model_index_json": _j(base.get("model_index") or {}),
            "tags_json": _j(base.get("tags") or []),
            "downloads_30d": base.get("downloads_30d"),
            "downloads_all_time": base.get("downloads_all_time"),
            "likes": base.get("likes"),
            "created_at": base.get("created_at"),
            "last_modified": base.get("last_modified"),
            "gated": int(bool(base["gated"])) if base.get("gated") is not None else None,
            "private": int(bool(base["private"])) if base.get("private") is not None else None,
            "contributors_json": _j(base.get("contributors") or []),
            "commit_history_json": _j(base.get("commit_history") or []),
            "fetch_logs_json": _j(base.get("fetch_logs") or []),
            "cache_hits": base.get("cache_hits", 0),
            "api_errors": base.get("api_errors", 0),
            "created_ts": ts, "updated_ts": ts,
        },
    )
    rid = conn.execute("SELECT id FROM resources WHERE canonical_key=?", (canonical_key,)).fetchone()["id"]

    if files:
        conn.execute("DELETE FROM files WHERE resource_id=?", (rid,))
        conn.executemany(
            "INSERT INTO files(resource_id, path, ext, size_bytes) VALUES (?,?,?,?)",
            [(rid, str(f["path"]), f.get("ext"), int(f.get("size_bytes", 0))) for f in files],
        )
    return int(rid)

def link_resources(conn: sqlite3.Connection, src_id: int, dst_id: int, link_type: str) -> None:
    conn.execute(
        "INSERT OR IGNORE INTO links(src_resource_id, dst_resource_id, link_type) VALUES (?,?,?)",
        (src_id, dst_id, link_type),
    )

def upsert_metric(
    conn: sqlite3.Connection,
    *, resource_id: int, metric_name: str,
    value: float, latency_ms: int, aux: dict[str, Any],
    metric_version: str, input_fingerprint_parts: list[str]
) -> None:
    ts = _now_ms()
    fp = _fp(metric_version, *input_fingerprint_parts)
    conn.execute(
        """
        INSERT INTO metric_results(resource_id, metric_name, value_float, aux_json,
                                   latency_ms, input_fingerprint, metric_version,
                                   created_ts, updated_ts)
        VALUES (?,?,?,?,?,?,?, ?,?)
        ON CONFLICT(resource_id, metric_name, metric_version, input_fingerprint)
        DO UPDATE SET value_float=excluded.value_float,
                      aux_json=excluded.aux_json,
                      latency_ms=excluded.latency_ms,
                      updated_ts=excluded.updated_ts
        """,
        (resource_id, metric_name, float(value), _j(aux or {}),
         int(latency_ms), fp, metric_version, ts, ts),
    )

# ---------- lookups ----------
def resource_id_by_key(conn: sqlite3.Connection, *, category: Category, canonical_key: str) -> Optional[int]:
    row = conn.execute(
        "SELECT id FROM resources WHERE category=? AND canonical_key=?",
        (category, canonical_key)
    ).fetchone()
    return int(row["id"]) if row else None

def latest_metric(conn: sqlite3.Connection, *, resource_id: int, metric_name: str) -> Optional[sqlite3.Row]:
    return conn.execute(
        "SELECT * FROM metric_results WHERE resource_id=? AND metric_name=? ORDER BY updated_ts DESC LIMIT 1",
        (resource_id, metric_name),
    ).fetchone()