import pytest
from src.db import open_db, upsert_resource, link_resources, upsert_metric, resource_id_by_key, latest_metric
import sqlite3
from pathlib import Path

def test_open_db(tmp_path):
    db_path = tmp_path / "test.sqlite"
    conn = open_db(db_path)
    assert isinstance(conn, sqlite3.Connection)
    conn.close()

def test_upsert_and_resource_id(tmp_path):
    db_path = tmp_path / "test.sqlite"
    conn = open_db(db_path)
    category = "MODEL"
    canonical_key = "foo/bar"
    base = {"url": "http://example.com", "gated": False, "private": False}
    rid = upsert_resource(conn, category=category, canonical_key=canonical_key, base=base)
    assert isinstance(rid, int)
    found_id = resource_id_by_key(conn, category=category, canonical_key=canonical_key)
    assert found_id == rid
    conn.close()

def test_link_resources(tmp_path):
    db_path = tmp_path / "test.sqlite"
    conn = open_db(db_path)
    base = {"url": "http://a.com", "gated": False, "private": False}
    rid1 = upsert_resource(conn, category="MODEL", canonical_key="a", base=base)
    rid2 = upsert_resource(conn, category="DATASET", canonical_key="b", base=base)
    link_resources(conn, rid1, rid2, "MODEL_TO_DATASET")
    # Check that the link exists
    row = conn.execute("SELECT * FROM links WHERE src_resource_id=? AND dst_resource_id=?", (rid1, rid2)).fetchone()
    assert row is not None
    conn.close()

def test_upsert_metric_and_latest(tmp_path):
    db_path = tmp_path / "test.sqlite"
    conn = open_db(db_path)
    base = {"url": "http://a.com", "gated": False, "private": False}
    rid = upsert_resource(conn, category="MODEL", canonical_key="a", base=base)
    upsert_metric(
        conn,
        resource_id=rid,
        metric_name="accuracy",
        value=0.9,
        latency_ms=123,
        aux={"foo": "bar"},
        metric_version="v1",
        input_fingerprint_parts=["abc"]
    )
    row = latest_metric(conn, resource_id=rid, metric_name="accuracy")
    assert row is not None
    assert row["value_float"] == 0.9
    conn.close()
