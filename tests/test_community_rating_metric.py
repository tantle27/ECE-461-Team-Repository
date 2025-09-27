import pytest
from src.metrics.community_rating_metric import CommunityRatingMetric

def test_community_rating_metric_thresholds():
    metric = CommunityRatingMetric()
    # Test likes thresholds
    assert metric.evaluate({"likes": 0, "downloads_all_time": 0}) == 0.0
    assert metric.evaluate({"likes": 3, "downloads_all_time": 0}) == 0.1
    assert metric.evaluate({"likes": 7, "downloads_all_time": 0}) == 0.2
    assert metric.evaluate({"likes": 20, "downloads_all_time": 0}) == 0.3
    assert metric.evaluate({"likes": 70, "downloads_all_time": 0}) == 0.4
    assert metric.evaluate({"likes": 150, "downloads_all_time": 0}) == 0.5
    # Test downloads thresholds
    assert metric.evaluate({"likes": 0, "downloads_all_time": 500}) == 0.0
    assert metric.evaluate({"likes": 0, "downloads_all_time": 2000}) == 0.1
    assert metric.evaluate({"likes": 0, "downloads_all_time": 7000}) == 0.2
    assert metric.evaluate({"likes": 0, "downloads_all_time": 20000}) == 0.3
    assert metric.evaluate({"likes": 0, "downloads_all_time": 70000}) == 0.4
    assert metric.evaluate({"likes": 0, "downloads_all_time": 200000}) == 0.5
    # Test combined, capped at 1.0
    assert metric.evaluate({"likes": 150, "downloads_all_time": 200000}) == 1.0

def test_community_rating_metric_negative():
    metric = CommunityRatingMetric()
    with pytest.raises(ValueError):
        metric.evaluate({"likes": -1, "downloads_all_time": 0})
    with pytest.raises(ValueError):
        metric.evaluate({"likes": 0, "downloads_all_time": -1})

def test_community_rating_metric_empty():
    metric = CommunityRatingMetric()
    assert metric.evaluate({}) == 0.0
    assert metric.evaluate(None) == 0.0

def test_community_rating_metric_description():
    metric = CommunityRatingMetric()
    desc = metric.get_description()
    assert "community engagement" in desc
