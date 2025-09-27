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
