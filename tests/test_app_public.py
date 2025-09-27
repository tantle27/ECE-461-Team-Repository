
import os
import sys
import tempfile
import pytest
from pathlib import Path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
import app

def test_validate_log_file_env_success(tmp_path):
    log_file = tmp_path / "log.txt"
    log_file.write_text("")
    os.environ["LOG_FILE"] = str(log_file)
    assert app._validate_log_file_env() == str(log_file)

def test_validate_log_file_env_missing(monkeypatch):
    monkeypatch.delenv("LOG_FILE", raising=False)
    with pytest.raises(SystemExit):
        app._validate_log_file_env()

def test_validate_log_file_env_not_exists(tmp_path):
    log_file = tmp_path / "nope.txt"
    os.environ["LOG_FILE"] = str(log_file)
    with pytest.raises(SystemExit):
        app._validate_log_file_env()

def test_validate_log_file_env_not_writable(tmp_path):
    log_file = tmp_path / "log.txt"
    log_file.write_text("")
    log_file.chmod(0o400)  # read-only
    os.environ["LOG_FILE"] = str(log_file)
    with pytest.raises(SystemExit):
        app._validate_log_file_env()
    log_file.chmod(0o600)  # restore perms

def test_require_valid_github_token(monkeypatch):
    monkeypatch.setenv("GITHUB_TOKEN", "ghp_" + "a" * 40)
    assert app._require_valid_github_token().startswith("ghp_")
    monkeypatch.setenv("GITHUB_TOKEN", "github_pat_" + "b" * 40)
    assert app._require_valid_github_token().startswith("github_pat_")
    monkeypatch.setenv("GITHUB_TOKEN", "badtoken")
    with pytest.raises(SystemExit):
        app._require_valid_github_token()
    monkeypatch.delenv("GITHUB_TOKEN", raising=False)
    with pytest.raises(SystemExit):
        app._require_valid_github_token()

def test_read_urls(tmp_path):
    f = tmp_path / "urls.txt"
    f.write_text("http://a.com\nhttp://b.com\n")
    urls = app.read_urls(str(f))
    assert urls == [("http://a.com", "", ""), ("http://b.com", "", "")]
    f.write_text("http://a.com,http://b.com\n")
    urls = app.read_urls(str(f))
    assert urls == [("http://a.com", "http://b.com", "")]

def test_read_urls_file_not_found():
    with pytest.raises(FileNotFoundError):
        app.read_urls("not_a_file.txt")
