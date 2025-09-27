import pytest
from src.api.hf_client import HFClient
from unittest.mock import patch, MagicMock

def test_hfclient_list_models(monkeypatch):
    with patch('src.api.hf_client.HfApi') as mock_hfapi:
        mock_api = MagicMock()
        mock_api.list_models.return_value = [MagicMock(modelId='m1'), MagicMock(modelId='m2')]
        mock_hfapi.return_value = mock_api
        client = HFClient()
        models = client.api.list_models()
        assert len(models) == 2
        assert models[0].modelId == 'm1'

def test_hfclient_list_datasets(monkeypatch):
    with patch('src.api.hf_client.HfApi') as mock_hfapi:
        mock_api = MagicMock()
        mock_api.list_datasets.return_value = [MagicMock(datasetId='d1'), MagicMock(datasetId='d2')]
        mock_hfapi.return_value = mock_api
        client = HFClient()
        datasets = client.api.list_datasets()
        assert len(datasets) == 2
        assert datasets[0].datasetId == 'd1'

def test_hfclient_get_model_info(monkeypatch):
    with patch('src.api.hf_client.HfApi') as mock_hfapi:
        mock_api = MagicMock()
        mock_api.model_info.return_value = MagicMock(modelId='m1', tags=['tag'], likes=1, downloads=2, cardData={'license': 'mit'}, createdAt='2020', lastModified='2021', gated=False, private=False)
        mock_hfapi.return_value = mock_api
        client = HFClient()
        info = client.api.model_info('m1')
        assert info.modelId == 'm1'
        assert info.cardData['license'] == 'mit'
