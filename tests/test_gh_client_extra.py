import pytest
from src.api.gh_client import GHClient
from unittest.mock import patch, MagicMock

@pytest.fixture
def ghclient(monkeypatch):
    monkeypatch.setenv("GITHUB_TOKEN", "ghp_" + "a" * 40)
    with patch("src.api.gh_client.requests.Session") as mock_session:
        mock_session.return_value = MagicMock()
        mock_session.return_value.get.return_value.status_code = 200
        client = GHClient()
        yield client

def test_get_repo_success(ghclient):
    with patch.object(ghclient, "_get_json", return_value={
        "private": False, "default_branch": "main", "description": "desc"
    }):
        repo = ghclient.get_repo("owner", "repo")
        assert repo is not None
        assert repo.owner == "owner"
        assert repo.repo == "repo"
        assert repo.private is False
        assert repo.default_branch == "main"
        assert repo.description == "desc"

def test_get_repo_none(ghclient):
    with patch.object(ghclient, "_get_json", return_value=None):
        repo = ghclient.get_repo("owner", "repo")
        assert repo is None

def test_get_readme_markdown_success(ghclient):
    with patch.object(ghclient, "_get_json", return_value={"download_url": "http://foo"}):
        with patch.object(ghclient, "_get_text_absolute", return_value="# README"):
            readme = ghclient.get_readme_markdown("owner", "repo")
            assert readme == "# README"

def test_get_readme_markdown_none(ghclient):
    with patch.object(ghclient, "_get_json", return_value={}):
        readme = ghclient.get_readme_markdown("owner", "repo")
        assert readme is None

def test_list_contributors(ghclient):
    # Simulate two pages, but only the first page has <100 items, so it should stop after first
    def fake_get_json(path):
        if "page=1" in path:
            return [{"login": "a"}]
        elif "page=2" in path:
            return [{"login": "b"}]
        return []
    with patch.object(ghclient, "_get_json", side_effect=fake_get_json):
        contributors = ghclient.list_contributors("owner", "repo", max_pages=2)
        # Only the first page is returned because it has <100 items
        assert contributors == [{"login": "a"}]

def test_list_contributors_empty(ghclient):
    with patch.object(ghclient, "_get_json", return_value=[]):
        contributors = ghclient.list_contributors("owner", "repo")
        assert contributors == []
