import pytest
from src.api.hf_client import HFClient, HFFileInfo
from unittest.mock import patch, MagicMock

def make_fake_api():
    api = MagicMock()
    api.list_repo_files.return_value = ["file1.txt", "file2.bin"]
    # get_paths_info should return a list of objects with 'path' and 'size' attributes
    info1 = MagicMock()
    info1.path = "file1.txt"
    info1.size = 123
    info2 = MagicMock()
    info2.path = "file2.bin"
    info2.size = 456
    api.get_paths_info.return_value = [info1, info2]
    return api

def test_list_files_with_sizes(monkeypatch):
    client = HFClient()
    monkeypatch.setattr(client, "api", make_fake_api())
    files = client.list_files("some/model")
    assert len(files) == 2
    assert files[0].path == "file1.txt"
    assert files[0].size == 123
    assert files[1].path == "file2.bin"
    assert files[1].size == 456

def test_list_files_no_sizes(monkeypatch):
    api = MagicMock()
    api.list_repo_files.return_value = ["file1.txt"]
    api.get_paths_info.side_effect = Exception()
    client = HFClient()
    monkeypatch.setattr(client, "api", api)
    files = client.list_files("some/model")
    assert len(files) == 1
    assert files[0].path == "file1.txt"
    assert files[0].size is None

def test_api_model_json(monkeypatch):
    client = HFClient()
    with patch("src.api.hf_client._json_get", return_value={"id": "foo"}):
        data = client._api_model_json("foo")
        assert data == {"id": "foo"}

def test_api_dataset_json(monkeypatch):
    client = HFClient()
    with patch("src.api.hf_client._json_get", return_value={"id": "bar"}):
        data = client._api_dataset_json("bar")
        assert data == {"id": "bar"}

def test_api_tree_json_success(monkeypatch):
    client = HFClient()
    fake_resp = MagicMock()
    fake_resp.status_code = 200
    fake_resp.json.return_value = [{"file": "a"}]
    monkeypatch.setattr(client._session, "get", lambda url: fake_resp)
    data = client._api_tree_json("models/foo")
    assert data == [{"file": "a"}]

def test_api_tree_json_fail(monkeypatch):
    client = HFClient()
    fake_resp = MagicMock()
    fake_resp.status_code = 404
    monkeypatch.setattr(client._session, "get", lambda url: fake_resp)
    data = client._api_tree_json("models/foo")
    assert data is None
