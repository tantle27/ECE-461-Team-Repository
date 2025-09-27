def test_get_text_absolute_and_get_json(monkeypatch):
    from src.api.gh_client import GHClient
    import requests
    client = object.__new__(GHClient)
    client._http = MagicMock()
    client._etag_cache = {}
    # _github_get returns 404
    resp = MagicMock(); resp.status_code = 404; resp.text = ""; resp.headers = {}
    client._github_get = lambda url, use_etag=True: resp
    assert client._get_text_absolute("url") is None
    # _github_get returns 200
    resp.status_code = 200; resp.text = "abc"
    assert client._get_text_absolute("url") == "abc"
    # _get_json returns None for 404
    resp.status_code = 404
    assert client._get_json("/foo") is None
    # _get_json returns JSON for 200
    import json as pyjson
    resp.status_code = 200; resp.text = "{""a"": 1}"
    resp.json.side_effect = lambda: {"a": 1}
    assert client._get_json("/foo") == {"a": 1}
    # _get_json handles ValueError
    resp.json.side_effect = lambda: (_ for _ in ()).throw(ValueError())
    assert client._get_json("/foo") is None

def test_get_branch_sha():
    from src.api.gh_client import GHClient
    client = object.__new__(GHClient)
    client._http = MagicMock()
    client._etag_cache = {}
    # _get_json returns dict with commit/sha
    client._get_json = lambda path: {"commit": {"sha": "abc123"}}
    sha = client._get_branch_sha("owner", "repo", "main")
    assert sha == "abc123"
    # _get_json returns dict without sha
    client._get_json = lambda path: {"commit": {}}
    assert client._get_branch_sha("owner", "repo", "main") is None
    # _get_json returns not a dict
    client._get_json = lambda path: None
    assert client._get_branch_sha("owner", "repo", "main") is None
import types

def test_is_token_valid():
    from src.api.gh_client import GHClient
    assert GHClient._is_token_valid("ghp_" + "a" * 40)
    assert GHClient._is_token_valid("github_pat_abc")
    assert not GHClient._is_token_valid("")
    assert not GHClient._is_token_valid("badtoken")

def test_get_repo_success(monkeypatch):
    from src.api.gh_client import GHClient
    client = object.__new__(GHClient)
    client._http = MagicMock()
    client._etag_cache = {}
    def fake_get_json(path):
        if path == "/repos/owner/repo":
            return {"private": False, "default_branch": "main", "description": "desc"}
        return None
    client._get_json = fake_get_json
    info = client.get_repo("owner", "repo")
    assert info is not None
    assert info.owner == "owner"
    assert info.repo == "repo"
    assert info.private is False
    assert info.default_branch == "main"
    assert info.description == "desc"

def test_get_repo_not_found(monkeypatch):
    from src.api.gh_client import GHClient
    client = object.__new__(GHClient)
    client._http = MagicMock()
    client._etag_cache = {}
    client._get_json = lambda path: None
    info = client.get_repo("owner", "repo")
    assert info is None

def test_get_readme_markdown_success(monkeypatch):
    from src.api.gh_client import GHClient
    client = object.__new__(GHClient)
    client._http = MagicMock()
    client._etag_cache = {}
    client._get_json = lambda path: {"download_url": "http://foo"}
    client._get_text_absolute = lambda url: "README content"
    result = client.get_readme_markdown("owner", "repo")
    assert result == "README content"

def test_get_readme_markdown_not_found(monkeypatch):
    from src.api.gh_client import GHClient
    client = object.__new__(GHClient)
    client._http = MagicMock()
    client._etag_cache = {}
    client._get_json = lambda path: {}
    result = client.get_readme_markdown("owner", "repo")
    assert result is None

def test_list_contributors_paged(monkeypatch):
    from src.api.gh_client import GHClient
    client = object.__new__(GHClient)
    client._http = MagicMock()
    client._etag_cache = {}
    # Simulate 2 pages, then empty
    def fake_get_json(path):
        if "page=1" in path:
            return [{"login": "a"}] * 100
        if "page=2" in path:
            return [{"login": "b"}] * 100
        return []
    client._get_json = fake_get_json
    items = client.list_contributors("owner", "repo", max_pages=2)
    assert len(items) == 200

def test_list_contributors_unexpected_shape(monkeypatch):
    from src.api.gh_client import GHClient
    client = object.__new__(GHClient)
    client._http = MagicMock()
    client._etag_cache = {}
    client._get_json = lambda path: {"unexpected": True}
    items = client.list_contributors("owner", "repo")
    assert items == []

def test_get_repo_tree_and_list_tree(monkeypatch):
    from src.api.gh_client import GHClient
    client = object.__new__(GHClient)
    client._http = MagicMock()
    client._etag_cache = {}
    # get_repo returns default_branch
    client.get_repo = lambda owner, repo: types.SimpleNamespace(default_branch="main")
    client._get_branch_sha = lambda owner, repo, branch: "sha123"
    def fake_get_json(path):
        if "git/trees" in path:
            return {"tree": [{"path": "foo.py", "type": "blob", "size": 123}, {"path": "bar", "type": "tree"}]}
        return None
    client._get_json = fake_get_json
    out = client.get_repo_tree("owner", "repo")
    assert any(x["path"] == "foo.py" for x in out)
    out2 = client.list_tree("owner", "repo")
    assert out2 == out

def test_get_repo_tree_branch_sha_not_found(monkeypatch):
    from src.api.gh_client import GHClient
    client = object.__new__(GHClient)
    client._http = MagicMock()
    client._etag_cache = {}
    client.get_repo = lambda owner, repo: types.SimpleNamespace(default_branch="main")
    client._get_branch_sha = lambda owner, repo, branch: None
    out = client.get_repo_tree("owner", "repo")
    assert out == []
import pytest
from src.api.gh_client import GHClient
from unittest.mock import patch, MagicMock

def test_ghclient_init_env_missing(monkeypatch):
    monkeypatch.setenv("GITHUB_TOKEN", "")
    with patch("src.api.gh_client.requests.Session") as mock_session:
        mock_session.return_value = MagicMock()
        try:
            GHClient()
        except SystemExit:
            pass

def test_ghclient_init_env_present(monkeypatch):
    # Use a valid-looking token to pass _is_token_valid
    monkeypatch.setenv("GITHUB_TOKEN", "ghp_" + "a" * 40)
    with patch("src.api.gh_client.requests.Session") as mock_session:
        mock_session.return_value = MagicMock()
        # Patch the get method to return a mock with status_code 200
        mock_session.return_value.get.return_value.status_code = 200
        client = GHClient()
        assert client is not None
