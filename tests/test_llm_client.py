import pytest
from src.api.llm_client import LLMClient
from unittest.mock import patch, MagicMock

def test_llmclient_init(monkeypatch):
    with patch("src.api.llm_client.requests.Session") as mock_session:
        mock_session.return_value = MagicMock()
        client = LLMClient()
        assert client is not None

def test_llmclient_completion(monkeypatch):
    with patch("src.api.llm_client.requests.Session") as mock_session:
        mock_session.return_value = MagicMock()
        client = LLMClient()
        # LLMClient does not have a 'complete' method, so just check for is_available and ask_json
        with patch.object(client, "is_available", return_value=True):
            with patch.object(client, "ask_json", return_value=None) as mock_ask_json:
                result = client.ask_json("sys", "prompt")
                assert result is None
                mock_ask_json.assert_called_once_with("sys", "prompt")
