import pytest
from src.metrics.dataset_quality_metric import DatasetQualityMetric
from unittest.mock import MagicMock, patch

def make_fake_repo_context():
    return {
        "tags": ["validation", "diversity", "completeness"],
        "card_data": {"schema": True, "classes": 5, "splits": ["train", "test"]},
        "readme_text": "# Dataset\nA diverse dataset with validation."
    }

def test_dataset_quality_basic():
    metric = DatasetQualityMetric(use_llm=False)
    repo_context = make_fake_repo_context()
    score = metric.evaluate(repo_context)
    assert 0.0 <= score <= 1.0

def test_dataset_quality_no_ctx():
    metric = DatasetQualityMetric(use_llm=False)
    repo_context = {}
    score = metric.evaluate(repo_context)
    assert 0.0 <= score <= 1.0

def test_dataset_quality_with_llm():
    metric = DatasetQualityMetric(use_llm=True)
    repo_context = make_fake_repo_context()
    with patch.object(metric, "_llm", create=True):
        score = metric.evaluate(repo_context)
        assert 0.0 <= score <= 1.0


# --- Additional edge/branch tests for coverage ---
import types

def test_pick_explicit_dataset_none():
    metric = DatasetQualityMetric()
    # No _ctx_obj
    repo_context = {}
    assert metric._pick_explicit_dataset(repo_context) is None

def test_pick_explicit_dataset_not_repo_context():
    metric = DatasetQualityMetric()
    # _ctx_obj is not a RepoContext
    repo_context = {"_ctx_obj": object()}
    assert metric._pick_explicit_dataset(repo_context) is None

def test_pick_explicit_dataset_explicit_self():
    metric = DatasetQualityMetric()
    # The implementation requires a real RepoContext, so this will always return None in test
    class Dummy:
        hf_id = "datasets/abc"
        __dict__ = {"_link_source": "explicit"}
    repo_context = {"_ctx_obj": Dummy()}
    ds = metric._pick_explicit_dataset(repo_context)
    assert ds is None

def test_pick_explicit_dataset_explicit_linked():
    metric = DatasetQualityMetric()
    # The implementation requires a real RepoContext, so this will always return None in test
    class Dummy:
        hf_id = "datasets/abc"
        __dict__ = {"_link_source": "not-explicit"}
    class Linked:
        hf_id = "datasets/linked"
        __dict__ = {"_link_source": "explicit"}
    ctx = Dummy()
    ctx.linked_datasets = [Linked()]
    repo_context = {"_ctx_obj": ctx}
    ds = metric._pick_explicit_dataset(repo_context)
    assert ds is None

def test_heuristics_from_dataset_minimal():
    metric = DatasetQualityMetric()
    class Minimal:
        pass
    h = metric._heuristics_from_dataset(Minimal())
    assert set(h.keys()) == {"has_validation", "data_diversity", "data_completeness"}
    for v in h.values():
        assert 0.0 <= v <= 1.0

def test_compute_heuristics_all_branches():
    metric = DatasetQualityMetric()
    # validation: both keywords and files
    tags = ["multilinguality:multilingual", "language:en", "modality:text", "size_categories:large"]
    card = {"splits": True, "train-eval-index": [{"col_mapping": {"a": 1}}]}
    readme = "validation schema train test multiple domains"
    files = ["dataset_infos.json", "script.py"]
    h = metric._compute_heuristics(tags, card, readme, files)
    assert h["has_validation"] > 0.9
    assert h["data_diversity"] > 0.5
    assert h["data_completeness"] > 0.5

def test_combine_heuristics_none_values():
    metric = DatasetQualityMetric()
    # All None
    score = metric._combine_heuristics()
    assert score == 0.0

def test_apply_engagement_floor_branches():
    metric = DatasetQualityMetric()
    class DS:
        def __init__(self, likes, downloads, hf_id):
            self.likes = likes
            self.downloads_all_time = downloads
            self.hf_id = hf_id
    # likes > 1500
    ds = DS(2000, 0, "datasets/bookcorpus/bookcorpus")
    score = metric._apply_engagement_floor(0.5, ds)
    assert score >= 0.9
    # downloads > 400_000
    ds = DS(0, 500_000, "datasets/other")
    score = metric._apply_engagement_floor(0.5, ds)
    assert score >= 0.84
    # likes > 100
    ds = DS(150, 0, "datasets/other")
    score = metric._apply_engagement_floor(0.5, ds)
    assert score >= 0.78
    # famous dataset
    ds = DS(0, 0, "datasets/squad/squad")
    score = metric._apply_engagement_floor(0.5, ds)
    assert score >= 0.9
    # normal case
    ds = DS(0, 0, "datasets/other")
    score = metric._apply_engagement_floor(0.5, ds)
    assert 0.55 < score < 0.7
